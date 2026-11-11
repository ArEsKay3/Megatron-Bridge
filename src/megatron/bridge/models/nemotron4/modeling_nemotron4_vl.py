# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nemotron4 vision-language model for processor-expanded image token sequences.

Nemotron4 uses the same media contract as Nemotron Omni: the processor emits one
image placeholder per projected image feature, and only those positions are
replaced in place. Insertion is one-for-one and length preserving, so the model
inserts media into the full token stream and then selects its own
context-parallel shard without rebuilding packed metadata.

The encoder is a Pixtral-style ViT (2D interleaved RoPE, gated SiLU MLP, RMSNorm,
no class token) whose 2x2 spatial patch merger runs inside the encoder, followed
by a two-layer non-gated MLP projector. The language stack is a ``HybridModel``
built from the provider's stack spec; the wide-residual, GDP and shortcut-MoE
layers it needs come from the Nemotron4 Megatron-LM fork.

Submodule names (``language_model``, ``vision_model``, ``vision_projection``)
match MCore ``LLaVAModel`` so the native non-MIMO training checkpoints load
unchanged and the HF weight mappings in :mod:`nemotron4_bridge` apply directly.

This forward is the production NM4 SFT path: packed THD context parallelism with
configurable linear/attention CP layouts, per-sequence zigzag outputs for
NeMo RL's loss reconstruction, optional vision sharding across language CP ranks
(``vision_dp_over_cp``), and GTP-safe vision process groups.
"""

import copy
import logging
from collections import namedtuple
from typing import Callable, Optional, Self

import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.context_parallel.layout import convert_cp_layout
from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks_dynamic_res,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_model import PatchEmbedding, ViTModel
from megatron.core.models.vision.vit_utils import dynamic_patch_grid_lists, run_vision_transformer
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

from megatron.bridge.models.logit_dtype import logit_dtype_kwarg


logger = logging.getLogger(__name__)


class Nemotron4GatedDeltaProductMixer(GatedDeltaProductMixer):
    """Materialize CP-sharded convolution inputs when their row stride is unaligned."""

    def _prepare_qkv(self, x, *args, **kwargs):
        # causal-conv1d's channel-last kernel requires the batch and sequence
        # strides after [B, L, D] -> [B, D, L] to be multiples of eight. CP8
        # can leave ``x`` as a slice of a wider projection even though its own
        # payload is aligned. A compact copy changes only the storage stride;
        # values, parameter shapes, and checkpoint keys stay unchanged.
        if x.stride(0) % 8 or x.stride(1) % 8:
            x = x.contiguous()
        return super()._prepare_qkv(x, *args, **kwargs)


class Nemotron4VisionMLP(MLP):
    """Keep TP1 GTP checkpoint shards contiguous across the GLU gate/up halves."""

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> ShardedStateDict:
        state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.config.gated_linear_unit and self.tp_group.size() == 1:
            from megatron.core.tensor_parallel.gtp_api import is_gtp_param

            if is_gtp_param(self.linear_fc1.weight):
                # The generic GLU factory treats the GTP shard as a TP shard and
                # loads [gate_0, up_0] / [gate_1, up_1]. GTP gathers row shards
                # without interleaving, so TP1 must load [gate] / [up] instead.
                metadata = ensure_metadata_has_dp_cp_group(metadata)
                fc1_prefix = f"{prefix}linear_fc1."
                fc1_state = self.linear_fc1.sharded_state_dict(fc1_prefix, sharded_offsets, metadata)
                state_dict[f"{fc1_prefix}weight"] = fc1_state[f"{fc1_prefix}weight"]
        return state_dict


def _ignore_transformer_engine_extra_state(module: torch.nn.Module, incompatible_keys: namedtuple) -> None:
    """Allow checkpoints that store no TE extra state to load."""
    del module
    for mismatch_kind, keys in incompatible_keys._asdict().items():
        ignored = [key for key in keys if "extra_state" in key]
        if not ignored:
            continue
        for key in ignored:
            keys.remove(key)
        logger.warning(
            "Ignoring %d Transformer Engine %s _extra_state entries; first entries: %s%s",
            len(ignored),
            mismatch_kind,
            ", ".join(ignored[:5]),
            " ..." if len(ignored) > 5 else "",
        )


def _build_vision_packed_seq_params(imgs_sizes: torch.Tensor, patch_dim: int, device: torch.device) -> PackedSeqParams:
    """Build one THD vision-attention sequence per input image."""
    sequence_lengths = [
        (int(height) // patch_dim) * (int(width) // patch_dim) for height, width in imgs_sizes.tolist()
    ]
    cumulative_lengths = [0]
    for sequence_length in sequence_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)

    cu_seqlens = torch.tensor(cumulative_lengths, dtype=torch.int32, device=device)
    max_seqlen = max(sequence_lengths, default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES = (1, 2)


def check_video_temporal_compression(video_temporal_patch_size: int, separate_video_embedder: bool) -> None:
    """Reject video tubelet settings the Nemotron4 vision encoder does not implement.

    ``separate_video_embedder`` has no effect with T = 1, so it is checked only for T > 1.
    """
    if video_temporal_patch_size not in SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES:
        raise ValueError(
            f"Nemotron4 supports video_temporal_patch_size in {SUPPORTED_VIDEO_TEMPORAL_PATCH_SIZES}, "
            f"got {video_temporal_patch_size!r}."
        )
    if video_temporal_patch_size > 1 and not separate_video_embedder:
        raise ValueError(
            "Nemotron4 supports only separate_video_embedder=True: images keep patch_embed and video "
            "tubelets use their own video_patch_embed."
        )


def video_patch_weight_from_image(image_weight: torch.Tensor, temporal_patch_size: int) -> torch.Tensor:
    """Derive the tubelet patch-embedding weight from the image patch-embedding weight.

    ``image_weight`` is the Conv2d weight ``[hidden, 3, P, P]``. The result is ``[hidden, 3 * T, P, P]``, and its
    input channel ``t * 3 + c`` is channel ``c`` of frame ``t`` (frame-major). Each frame gets ``W / T``, so by
    linearity a tubelet embeds as the image embedding of its mean frame, and a tubelet of ``T`` identical frames
    embeds as that frame.
    """
    return torch.cat([image_weight / temporal_patch_size] * temporal_patch_size, dim=1)


class _WithoutTEExtraState:
    """Exclude TE metadata absent from the native NM4 checkpoints, as :class:`Nemotron4VLModel` does."""

    def sharded_state_dict(self, *args, **kwargs) -> ShardedStateDict:
        from megatron.bridge.training.checkpointing import delete_extra_state

        state = super().sharded_state_dict(*args, **kwargs)
        delete_extra_state(state)
        return state


class Nemotron4MIMOVisionEncoder(_WithoutTEExtraState, ViTModel):
    """Pixtral-style ViT of the MIMO ``images`` module.

    With ``recompute_budget_gib``, each training forward sets its recompute plan (see ``recompute_plan``).

    With ``video_temporal_patch_size`` T > 1, the encoder also owns ``video_patch_embed``, a Conv2d over ``3 * T``
    channels. The frames of each video are grouped into tubelets of T consecutive frames, which go through
    ``video_patch_embed``; images keep ``patch_embed``. Everything after the embedding (``ln_pre``, 2D RoPE, the
    transformer and the merger) sees one ``imgs_sizes`` row per image or tubelet. With T = 1 the module and its
    forward are those of the plain ViT.
    """

    # Class defaults keep encoders built without ``__init__`` on the T = 1 path.
    video_temporal_patch_size: int = 1
    padded_video_count: int = 0
    warned_frames_as_images: bool = False

    def __init__(
        self,
        *args,
        recompute_budget_gib: float | None = None,
        video_temporal_patch_size: int = 1,
        separate_video_embedder: bool = True,
        temporal_ckpt_compat: bool = False,
        **kwargs,
    ) -> None:
        check_video_temporal_compression(video_temporal_patch_size, separate_video_embedder)
        super().__init__(*args, **kwargs)
        self.recompute_budget_gib = recompute_budget_gib
        self.video_temporal_patch_size = video_temporal_patch_size
        # Lazy-init flag (``--allow-checkpoint-without-temporal-compression``). The HF import applies it in
        # ``Nemotron4Bridge``; ``_fill_missing_video_patch_embed`` applies it to ``load_state_dict``.
        self.temporal_ckpt_compat = temporal_ckpt_compat
        if video_temporal_patch_size > 1:
            # Names of the T = 1 weights, recorded before adapters can wrap any module. The load hook uses them
            # to tell a full encoder state dict from an adapter-only one.
            self._base_parameter_names = tuple(name for name, _ in self.named_parameters())
            # Built last, so the modules above keep their T = 1 parameter order.
            self.video_patch_embed = self._build_video_patch_embed()
            self._register_load_state_dict_pre_hook(self._fill_missing_video_patch_embed)

    def _build_video_patch_embed(self) -> PatchEmbedding:
        """Build ``video_patch_embed``, initialized from ``patch_embed`` without drawing random numbers."""
        image_proj = self.patch_embed.proj
        if image_proj.bias is not None:
            raise ValueError("Nemotron4 video_patch_embed expects a patch embedding without bias.")
        temporal = self.video_temporal_patch_size
        # Meta construction skips the Conv2d random init, so T > 1 does not shift the RNG stream of the modules
        # built after the encoder. A checkpoint or HF import overwrites the start value.
        with torch.device("meta"):
            video = PatchEmbedding(image_proj.in_channels * temporal, image_proj.out_channels, self.patch_dim)
        video.to_empty(device=image_proj.weight.device).to(dtype=image_proj.weight.dtype)
        with torch.no_grad():
            video.proj.weight.copy_(video_patch_weight_from_image(image_proj.weight, temporal))
        return video

    def _fill_missing_video_patch_embed(self, state_dict, prefix, *args) -> None:
        """``load_state_dict`` pre-hook: apply the lazy-init rules to a state dict without ``video_patch_embed``.

        A Megatron checkpoint saved with T = 1 has no video weight. On a finetune load, Bridge's dist-ckpt filter
        (``_drop_missing_video_patch_embed_keys``) drops the key from the request, and this hook derives the weight
        from the loaded ``patch_embed`` weight. That runs in ``load_state_dict``, before
        ``optimizer.reload_model_params()`` copies the model weights into the main params (the filter rejects
        ``load_main_params_from_ckpt``, which reads them from the checkpoint dict instead). A strictness that drops missing keys, or Bridge's non-strict retry
        after a failed strict load, would otherwise keep the construction-time value, which comes from the random
        image weight. So this hook raises, and does not return, when the video key is missing and either
        ``temporal_ckpt_compat`` is off or the image weight is missing too. Only a state dict without any of this
        encoder's construction-time weights (an adapter-only partial load) is left to the caller's strictness.
        """
        del args
        video_key = f"{prefix}video_patch_embed.proj.weight"
        image_key = f"{prefix}patch_embed.proj.weight"
        if video_key in state_dict:
            return
        if image_key not in state_dict:
            present = [f"{prefix}{name}" for name in self._base_parameter_names if f"{prefix}{name}" in state_dict]
            if not present:
                return
            raise RuntimeError(
                f"The state dict has encoder weights (for example {present[0]}) but neither {video_key} nor "
                f"{image_key}, so video_temporal_patch_size={self.video_temporal_patch_size} has no video patch "
                "embedder to load or derive."
            )
        if not self.temporal_ckpt_compat:
            raise RuntimeError(
                f"The state dict has {image_key} but no {video_key}, which video_temporal_patch_size="
                f"{self.video_temporal_patch_size} needs: it was saved without temporal compression. To derive it "
                "from the image patch embedding, enable the lazy init (temporal_ckpt_compat)."
            )
        logger.warning(
            "The state dict has no %s; deriving it from %s for video_temporal_patch_size=%d (temporal_ckpt_compat).",
            video_key,
            image_key,
            self.video_temporal_patch_size,
        )
        state_dict[video_key] = video_patch_weight_from_image(state_dict[image_key], self.video_temporal_patch_size)

    def forward(
        self,
        x: torch.Tensor,
        imgs_sizes: torch.Tensor,
        packed_seq_params: PackedSeqParams,
        num_frames: torch.Tensor | None = None,
        expected_output_counts: torch.Tensor | None = None,
        media_is_video: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode the MIMO packed-media payload: per-frame patches ``x`` and one ``imgs_sizes`` row per frame.

        ``num_frames`` has one entry per media occurrence in payload order (1 for an image). With T = 1, or
        without ``num_frames``, every frame is encoded as an image and ``num_frames``, ``expected_output_counts``
        and ``media_is_video`` are not used. With T > 1, each video is encoded as tubelets, so it gives
        ``ceil(num_frames / T)`` merged patch grids. The output rows are then checked against
        ``expected_output_counts`` when the counts are on the host.

        ``media_is_video`` (optional, one bool per occurrence) marks the videos. Without it, an occurrence with
        more than one frame is a video and a one-frame occurrence is an image. With it, a one-frame video is
        padded to a T-frame tubelet ``[f, f]`` and goes through ``video_patch_embed``, as the HF and vLLM models
        do. The training data never sends a video with fewer than T frames.
        """
        # The ViT reads the patch grid on the host; MIMO payloads carry it on the GPU.
        imgs_sizes = imgs_sizes.cpu()
        if self.video_temporal_patch_size > 1 and num_frames is None and media_is_video is not None:
            raise ValueError("media_is_video needs num_frames: it has one entry per media occurrence.")
        if self.video_temporal_patch_size == 1 or num_frames is None:
            if self.video_temporal_patch_size > 1 and not self.warned_frames_as_images:
                self.warned_frames_as_images = True
                logger.warning(
                    "The Nemotron4 vision encoder has video_temporal_patch_size=%d but the payload has no "
                    "num_frames, so every frame is encoded as an image with patch_embed and video_patch_embed is "
                    "not used. Callers that send video frames must pass num_frames (logged once).",
                    self.video_temporal_patch_size,
                )
            del num_frames, expected_output_counts, media_is_video
            return self._forward_frames(x, imgs_sizes, packed_seq_params)

        frames_per_media = [int(count) for count in num_frames.reshape(-1).tolist()]
        if any(count < 1 for count in frames_per_media) or sum(frames_per_media) != imgs_sizes.shape[0]:
            raise ValueError(
                f"num_frames must be positive and sum to the {imgs_sizes.shape[0]} imgs_sizes rows, "
                f"got {frames_per_media}."
            )
        videos = self._video_flags(frames_per_media, media_is_video)
        if not any(videos):
            output = self._forward_frames(x, imgs_sizes, packed_seq_params)
        else:
            embedded, tubelet_sizes = self._embed_media(x, imgs_sizes, frames_per_media, videos)
            # Attention, RoPE and the merger work per image or tubelet, so the THD boundaries follow the tubelets.
            packed_seq_params = _build_vision_packed_seq_params(tubelet_sizes, self.patch_dim, x.device)
            if self.recompute_budget_gib is not None and self.training and torch.is_grad_enabled():
                self._fit_recompute_to_budget(embedded.shape[1])
            output = self._forward_embedded(embedded, tubelet_sizes, packed_seq_params)
        self._check_output_rows(output, expected_output_counts)
        return output

    def _forward_frames(
        self, x: torch.Tensor, imgs_sizes: torch.Tensor, packed_seq_params: PackedSeqParams
    ) -> torch.Tensor:
        """Encode every frame as an image (the plain ViT forward)."""
        if self.recompute_budget_gib is not None and self.training and torch.is_grad_enabled():
            self._fit_recompute_to_budget(x.shape[1])
        return super().forward(x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params)

    @staticmethod
    def _video_flags(frames_per_media: list[int], media_is_video: torch.Tensor | None) -> list[bool]:
        """Return one flag per media occurrence: True for a video, False for an image.

        Without ``media_is_video``, an occurrence with more than one frame is a video. An image always has one
        frame.
        """
        if media_is_video is None:
            return [count > 1 for count in frames_per_media]
        videos = [bool(flag) for flag in media_is_video.reshape(-1).tolist()]
        if len(videos) != len(frames_per_media):
            raise ValueError(
                f"media_is_video must have one entry per media occurrence ({len(frames_per_media)}), "
                f"got {len(videos)}."
            )
        images_with_frames = [media for media, count in enumerate(frames_per_media) if count > 1 and not videos[media]]
        if images_with_frames:
            raise ValueError(
                f"Media occurrences {images_with_frames} have more than one frame but media_is_video marks them as "
                "images; an image has one frame."
            )
        return videos

    def _embed_media(
        self,
        x: torch.Tensor,
        imgs_sizes: torch.Tensor,
        frames_per_media: list[int],
        videos: list[bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed images with ``patch_embed`` and video tubelets with ``video_patch_embed``, in payload order.

        A tubelet concatenates the features of T consecutive frames in time order (frame-major), which matches
        the ``video_patch_embed`` input channels. A video whose frame count is not a multiple of T repeats its
        last frame up to the next multiple. This includes a video with fewer than T frames, so a one-frame video
        is the tubelet ``[f, f]`` at T = 2. The training data path never sends such a video, so each pad logs a
        warning. ``videos`` has one flag per occurrence (see ``_video_flags``); without it, an occurrence with
        more than one frame is a video.

        Returns:
            The embedded rows ``[1, N, hidden]`` and one ``imgs_sizes`` row per image or tubelet.
        """
        temporal = self.video_temporal_patch_size
        if videos is None:
            videos = self._video_flags(frames_per_media, None)
        sizes = [tuple(size) for size in imgs_sizes.tolist()]
        _, frame_lengths = dynamic_patch_grid_lists(imgs_sizes, self.patch_dim)
        frames = torch.split(x, frame_lengths, dim=1)
        image_inputs, tubelet_inputs, is_tubelet, first_frames = [], [], [], []
        start = 0
        for media, count in enumerate(frames_per_media):
            indices = list(range(start, start + count))
            start += count
            if not videos[media]:
                image_inputs.append(frames[indices[0]])
                is_tubelet.append(False)
                first_frames.append(indices[0])
                continue
            if len({sizes[index] for index in indices}) != 1:
                raise ValueError(
                    f"All frames of video occurrence {media} must have one size, got {sizes[start - count : start]}."
                )
            if count % temporal:
                indices += [indices[-1]] * (temporal - count % temporal)
                self.padded_video_count += 1
                logger.warning(
                    "Video occurrence %d has %d frames, not a multiple of video_temporal_patch_size=%d; repeated "
                    "its last frame (%d padded videos so far).",
                    media,
                    count,
                    temporal,
                    self.padded_video_count,
                )
            for first in range(0, len(indices), temporal):
                group = indices[first : first + temporal]
                tubelet_inputs.append(torch.cat([frames[index] for index in group], dim=-1))
                is_tubelet.append(True)
                first_frames.append(group[0])
        images = self._embed_chunks(self.patch_embed, image_inputs)
        tubelets = self._embed_chunks(self.video_patch_embed, tubelet_inputs)
        embedded = torch.cat([next(tubelets if tubelet else images) for tubelet in is_tubelet], dim=1)
        return embedded, imgs_sizes[first_frames]

    @staticmethod
    def _embed_chunks(embedding: PatchEmbedding, chunks: list[torch.Tensor]):
        """Embed same-width chunks with one projection and yield them back in order."""
        if not chunks:
            return iter(())
        lengths = [chunk.shape[1] for chunk in chunks]
        return iter(torch.split(embedding.forward_patches(torch.cat(chunks, dim=1)), lengths, dim=1))

    def _forward_embedded(
        self, x: torch.Tensor, imgs_sizes: torch.Tensor, packed_seq_params: PackedSeqParams
    ) -> torch.Tensor:
        """Run the part of ``ViTModel.forward`` after the patch embedding, for packed dynamic-resolution input.

        This mirrors Megatron-LM ``ViTModel.forward`` for the layout NM4 uses: no class token, 2D RoPE and the
        optional patch merger, each computed per ``imgs_sizes`` row.
        """
        if self.add_class_token or self.pos_emb_type not in ("rope2d", "none"):
            raise NotImplementedError("Nemotron4 tubelet encoding supports only 2D RoPE and no class token.")
        patch_hw, seq_lens = dynamic_patch_grid_lists(imgs_sizes, self.patch_dim)
        if self.ln_pre is not None:
            x = self.ln_pre(x)
        rotary_pos_emb = None
        if self.pos_emb_type == "rope2d":
            rotary_pos_emb = torch.cat([self.rope(h, w, x.device) for h, w in patch_hw], dim=0)
        x = run_vision_transformer(
            self.decoder, x, attention_mask=None, rotary_pos_emb=rotary_pos_emb, packed_seq_params=packed_seq_params
        )
        if self.merger is not None:
            chunks = torch.split(x, seq_lens, dim=1)
            x = torch.cat([self.merger(chunk, int(h), int(w)) for chunk, (h, w) in zip(chunks, patch_hw)], dim=1)
        return x

    @staticmethod
    def _check_output_rows(output: torch.Tensor, expected_output_counts: torch.Tensor | None) -> None:
        """Require the output rows to equal the payload's media output counts (host counts only, no sync)."""
        if expected_output_counts is None or expected_output_counts.device.type != "cpu":
            return
        expected = int(expected_output_counts.sum())
        if output.shape[1] != expected:
            raise RuntimeError(
                f"The Nemotron4 vision encoder produced {output.shape[1]} rows, but the payload's media output "
                f"counts expect {expected}. With video_temporal_patch_size > 1 the data path must count one "
                "merged patch grid per tubelet."
            )

    def recompute_plan(self, patches: int) -> tuple[int, int]:
        """Return ``(n_full, n_mlp)`` whose estimated activations of ``patches`` fit the budget.

        Bytes kept per patch, measured on the NM4 ViT (BF16, fused SwiGLU and RoPE; hidden h, FFN f): a full
        recompute layer 2.1h, an MLP-only recompute layer 14.5h, a layer without recompute 16.5h + 6f, the rest
        of the encoder 6.5h. Layers move from full to MLP-only first (most recompute time saved per byte), then
        from MLP-only to none. If even full recompute of all layers does not fit, all layers use it.
        These estimates are calibrated only for a BF16 encoder; MXFP8 and other quantized encoders have not
        been validated. The language decoder may independently use a different precision.
        """
        if not self.config.bf16 or self.config.fp8 or self.config.fp4 or self.config.quant_recipe is not None:
            raise ValueError(
                "vision_recompute_budget_gib requires a BF16 encoder with FP8, FP4, and quant_recipe disabled; "
                "its activation estimates have not been validated for MXFP8 or other encoder precisions."
            )
        hidden, ffn = self.config.hidden_size, self.config.ffn_hidden_size
        full, mlp, none = 2.1 * hidden, 14.5 * hidden, 16.5 * hidden + 6 * ffn
        layer_budget = self.recompute_budget_gib * 2**30 / max(patches, 1) - 6.5 * hidden
        n_layers = len(self.decoder.layers)

        def fits(n_full: int, n_mlp: int) -> bool:
            return n_full * full + n_mlp * mlp + (n_layers - n_full - n_mlp) * none <= layer_budget

        n_full, n_mlp = n_layers, 0
        while n_full > 0 and fits(n_full - 1, n_mlp + 1):
            n_full, n_mlp = n_full - 1, n_mlp + 1
        while n_full == 0 and n_mlp > 0 and fits(0, n_mlp - 1):
            n_mlp -= 1
        return n_full, n_mlp

    def _fit_recompute_to_budget(self, patches: int) -> None:
        # Core reads the layer count when this forward runs and ``recompute_mlp`` whenever a layer runs. If a
        # later plan sets ``recompute_mlp`` on a layer that recomputes fully here, that backward also recomputes
        # the MLP once more: slower, same values.
        n_full, n_mlp = self.recompute_plan(patches)
        self.config.recompute_num_layers = n_full
        for index, layer in enumerate(self.decoder.layers):
            layer.recompute_mlp = n_full <= index < n_full + n_mlp


class Nemotron4MIMOProjector(_WithoutTEExtraState, MultimodalProjector):
    """Vision projector of the MIMO ``images`` module."""


class Nemotron4MIMOLanguageModel(_WithoutTEExtraState, HybridModel):
    """Hybrid decoder retaining Nemotron4's FP32 state during MIMO dtype casts.

    MIMO calls ``bfloat16()`` on the complete model. The residual controllers,
    GDP decay parameters, and router bookkeeping must keep their original FP32
    values, just as the HF model's strict FP32 module list requires.
    """

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> Self:
        retained = []
        for name, tensor in list(self.named_parameters()) + list(self.named_buffers()):
            keep_in_fp32 = getattr(tensor, "keep_in_fp32", False)
            if (
                keep_in_fp32
                or getattr(tensor, "is_residual_map_parameter", False)
                or getattr(tensor, "is_wide_residual_retention_parameter", False)
                or name.endswith((".A_log", ".expert_bias", ".qb_bin_bounds"))
            ):
                grad = tensor.grad.detach() if tensor.grad is not None else None
                retained.append((name, isinstance(tensor, torch.nn.Parameter), tensor.detach(), grad, keep_in_fp32))
        super()._apply(fn, recurse=recurse)
        # Keep parameter objects and their parallelism/optimizer attributes. A
        # detached reference retains the original storage across Module._apply.
        for name, is_parameter, original, original_grad, keep_in_fp32 in retained:
            tensor = self.get_parameter(name) if is_parameter else self.get_buffer(name)
            tensor.data = original.to(device=tensor.device, dtype=torch.float32)
            if keep_in_fp32:
                tensor.keep_in_fp32 = True
            if original_grad is not None:
                tensor.grad.data = original_grad.to(device=tensor.device, dtype=torch.float32)
        return self


class Nemotron4VLModel(MegatronModule):
    """Nemotron4 VLM: Pixtral-style ViT + MLP projector + hybrid language model.

    The input sequence already carries one placeholder per projected image
    feature. Unlike MCore ``LLaVAModel`` this model does not collapse and
    re-expand media runs; insertion is one-for-one and length preserving, so
    the model can insert media into the full token stream and then select its
    own context-parallel shard, which is what
    ``model_slices_context_parallel_inputs`` promises NeMo RL.
    """

    model_owns_packing = False
    model_owns_mtp_loss_mask_packing = False
    model_slices_context_parallel_inputs = True

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec,
        image_token_index: int,
        img_h: int,
        img_w: int,
        patch_dim: int,
        spatial_merge_size: int,
        hybrid_layer_pattern: Optional[str],
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "none",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        vision_dp_over_cp: bool = True,
        vp_stage: Optional[int] = None,
        video_temporal_patch_size: int = 1,
        separate_video_embedder: bool = True,
        temporal_ckpt_compat: bool = False,
    ) -> None:
        check_video_temporal_compression(video_temporal_patch_size, separate_video_embedder)
        super().__init__(config=language_transformer_config)
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vocab_size = language_vocab_size
        self.image_token_index = image_token_index
        self.patch_dim = patch_dim
        self.spatial_merge_size = spatial_merge_size
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.vision_dp_over_cp = vision_dp_over_cp and self.context_parallel_lm > 1
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_hidden_state = None
        self.pg_collection = pg_collection
        self.video_temporal_patch_size = video_temporal_patch_size

        self.language_model = None
        if add_decoder:
            self.language_model = HybridModel(
                config=language_transformer_config,
                hybrid_stack_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                **logit_dtype_kwarg(HybridModel, language_transformer_config.logit_dtype),
                position_embedding_type=language_position_embedding_type,
                pre_process=pre_process,
                hybrid_layer_pattern=hybrid_layer_pattern,
                post_process=post_process,
                # The caller scatters for sequence parallelism after media merge.
                scatter_embedding_sequence_parallel=False,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.language_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

        self.vision_model = None
        self.vision_projection = None
        if add_encoder:
            # Vision encodes whole images, even when images are distributed over
            # language CP ranks. RoPE must therefore see a singleton vision CP
            # group rather than select positions as if each image were sharded.
            vision_pg = copy.copy(pg_collection)
            # GTP partners consume different media and can skip the encoder.
            # Keep vision weights replicated so their gathers cannot enter the
            # language weight stream. An explicit None prevents MPU fallback.
            vision_pg.gtp_remat = None
            vision_pg.expt_gtp_remat = None
            if pg_collection.cp.size() > 1:
                vision_pg.cp = torch.distributed.new_group(
                    ranks=[torch.distributed.get_rank()],
                    use_local_synchronization=True,
                )
                vision_pg.tp_cp = vision_pg.tp
                vision_pg.hcp = None
            vision_kwargs = dict(
                transformer_config=vision_transformer_config,
                transformer_layer_spec=vision_transformer_layer_spec,
                patch_dim=patch_dim,
                img_h=img_h,
                img_w=img_w,
                add_class_token=False,
                class_token_len=0,
                pos_emb_type="rope2d",
                ln_pre=True,
                use_merger=spatial_merge_size > 1,
                spatial_merge_size=spatial_merge_size,
                pg_collection=vision_pg,
                vp_stage=vp_stage,
            )
            if video_temporal_patch_size == 1:
                self.vision_model = ViTModel(**vision_kwargs)
            else:
                # Checkpoint conversion only: the MIMO encoder owns ``video_patch_embed`` under the same name, so
                # ``vision_model.video_patch_embed.proj.weight`` matches the Nemotron4Bridge mapping. ``forward``
                # rejects videos (see ``_encode_images``).
                self.vision_model = Nemotron4MIMOVisionEncoder(
                    **vision_kwargs,
                    video_temporal_patch_size=video_temporal_patch_size,
                    separate_video_embedder=separate_video_embedder,
                    temporal_ckpt_compat=temporal_ckpt_compat,
                )
            # The merger keeps the ViT width, so the projector input is the ViT
            # hidden size rather than a pixel-shuffled multiple of it.
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                "mlp",
                getattr(self.vision_model, "out_hidden_size", vision_transformer_config.hidden_size),
                pg_collection=vision_pg,
            )
            self.vision_model.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)
            self.vision_projection.register_load_state_dict_post_hook(_ignore_transformer_engine_extra_state)

    def sharded_state_dict(self, *args, **kwargs):
        """Exclude TE metadata absent from the native NM4 checkpoints."""
        from megatron.bridge.training.checkpointing import delete_extra_state

        state = super().sharded_state_dict(*args, **kwargs)
        delete_extra_state(state)
        return state

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ) -> None:
        """Freeze selected towers."""
        modules = (
            (freeze_language_model, self.language_model),
            (freeze_vision_model, self.vision_model),
            (freeze_vision_projection, self.vision_projection),
        )
        for should_freeze, module in modules:
            if should_freeze and module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def set_input_tensor(self, input_tensor) -> None:
        """Set the pipeline input on the language model."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor must contain exactly one tensor"
        if self.language_model is not None:
            self.language_model.set_input_tensor(input_tensor[0])

    def _encode_images(self, images, imgs_sizes, vision_packed_seq_params, num_frames=None) -> torch.Tensor:
        """Return one projected row per image token.

        With ``video_temporal_patch_size`` T > 1 this model encodes images only: ``num_frames`` must be given and
        be 1 for every media occurrence. Video tubelets need the MIMO encoder payload, which this model does not
        build. ``num_frames`` is not used with T = 1.
        """
        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError("Image data was provided on a stage without the vision encoder")
        temporal = getattr(self, "video_temporal_patch_size", 1)
        if temporal > 1:
            frames = None if num_frames is None else [int(count) for count in torch.as_tensor(num_frames).reshape(-1)]
            if frames is None or any(count != 1 for count in frames):
                raise NotImplementedError(
                    f"The non-MIMO Nemotron4VLModel with video_temporal_patch_size={temporal} is for checkpoint "
                    "conversion and image-only forwards: it does not group video frames into tubelets. Pass "
                    "num_frames with 1 per image, or use the MIMO model (Nemotron4MIMOVisionEncoder) for videos. "
                    f"Got num_frames={frames}."
                )

        parameter = next(self.vision_model.parameters())
        if imgs_sizes is not None:
            imgs_sizes = imgs_sizes.cpu()
            if vision_packed_seq_params is None and imgs_sizes.numel() > 0:
                # Pixel patches from every image in a physical language pack
                # are concatenated into one THD tensor. Keep the ViT attention
                # independent per image; otherwise image i's projected tokens
                # depend on every other image that happened to share its pack.
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim, images.device)
        shard_vision = self.vision_dp_over_cp and imgs_sizes is not None and imgs_sizes.numel() > 0
        num_padded_ranks = 0
        if shard_vision:
            if self.pg_collection.cp.size() != self.context_parallel_lm:
                raise ValueError("Nemotron4 vision_dp_over_cp does not match its process group")
            (
                images,
                imgs_sizes,
                vision_packed_seq_params,
                _has_fp8_padding,
                num_padded_ranks,
                _local_num_frames,
            ) = split_to_context_parallel_ranks_dynamic_res(
                images,
                imgs_sizes,
                vision_packed_seq_params,
                patch_dim=self.patch_dim,
                fp8_enabled=False,
            )
            # MCore gives empty ranks a 1x1 dummy patch grid. The merger
            # requires a grid divisible by spatial_merge_size, even for dummies.
            if num_padded_ranks and parallel_state.get_context_parallel_rank() >= (
                self.context_parallel_lm - num_padded_ranks
            ):
                merge = self.spatial_merge_size
                images = images.new_zeros((1, merge * merge, images.shape[-1]))
                imgs_sizes = imgs_sizes.new_full((1, 2), self.patch_dim * merge)
                vision_packed_seq_params = _build_vision_packed_seq_params(imgs_sizes, self.patch_dim, images.device)
        # The Pixtral ViT adds no class token and uses the patch merger rather
        # than pixel shuffle, so the encoder output already has exactly one row
        # per placeholder and needs no per-image regrouping.
        vision_kwargs = {}
        if temporal > 1:
            # Images only (checked above): one frame per local image, so the encoder uses patch_embed.
            vision_kwargs["num_frames"] = torch.ones(imgs_sizes.shape[0], dtype=torch.int32)
        encoded = self.vision_model(
            images.to(dtype=parameter.dtype),
            imgs_sizes=imgs_sizes,
            packed_seq_params=vision_packed_seq_params,
            **vision_kwargs,
        )
        features = encoded.squeeze(0).unsqueeze(1)
        projected = self.vision_projection(features).squeeze(1)
        if shard_vision:
            # Gather only projected features; encoder and projector stay local.
            projected = gather_from_context_parallel_ranks_dynamic_res(projected, num_padded_ranks)
        return projected.contiguous()

    def _merge_projected_media(self, embeddings, input_ids, features, validity_mask) -> torch.Tensor:
        """Replace each valid media placeholder with exactly one feature row."""
        media_mask = input_ids == self.image_token_index
        if validity_mask is not None:
            if validity_mask.shape != input_ids.shape:
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(validity_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )
            media_mask = media_mask & validity_mask.bool()

        expected_features = int(media_mask.sum().item())
        if expected_features != features.shape[0]:
            raise ValueError(
                "Expanded-sequence media alignment failed: "
                f"found {expected_features} valid placeholders for {features.shape[0]} projected "
                "features. Nemotron4VLModel requires the processor to emit one placeholder "
                "for every projected patch."
            )

        # With batch size one, transpose can remain contiguous and
        # ``contiguous()`` may return the original autograd view. Media
        # insertion is in-place, so force independent storage for the merge.
        merged = embeddings.transpose(0, 1).clone()
        if expected_features:
            merged[media_mask] = features.to(dtype=merged.dtype)
        return merged.transpose(0, 1).contiguous()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        *,
        media_token_validity_mask: Optional[torch.Tensor] = None,
        inference_params=None,
        **kwargs,
    ):
        """Insert media into the expanded sequence, shard for CP, then run the hybrid LM.

        ``pixel_values``/``image_sizes`` are accepted as aliases of
        ``images``/``imgs_sizes`` because the HF Nemotron4 processor emits those
        names. Both size names may be supplied when their shapes and values match;
        in that case ``imgs_sizes`` is retained.

        Returns:
            Model output, or ``(output, local_loss_mask)`` when this model
            applied a context-parallel shard to the supervision tensors. Final
            logits, token losses, and the returned mask use per-sequence zigzag
            ordering; decoder inputs retain the configured linear CP layout.
        """
        # Frames per media occurrence; only a T > 1 model reads it (``_encode_images``).
        num_frames = kwargs.pop("num_frames", None)
        del kwargs
        if images is None:
            images = pixel_values
        if image_sizes is not None:
            if imgs_sizes is None:
                imgs_sizes = image_sizes
            elif not torch.equal(imgs_sizes, image_sizes):
                raise ValueError("imgs_sizes and image_sizes must match when both are provided.")

        lm_input_ids = input_ids
        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first Nemotron4 pipeline stage requires input_ids.")
            # An explicit mask from the caller wins: padding and attention masks
            # answer "is this a real token", which is a different question from
            # "is this a media anchor". They coincide only while every media
            # token in a valid position anchors an image. Otherwise a padding
            # mask is unambiguous for collator-owned packed inputs, and a 2-D
            # attention mask is the MBridge collators' token-validity mask;
            # MCore's 4-D causal mask cannot filter placeholders.
            if media_token_validity_mask is None:
                if padding_mask is not None:
                    media_token_validity_mask = ~padding_mask
                elif attention_mask is not None and attention_mask.dim() == input_ids.dim():
                    media_token_validity_mask = attention_mask
            if media_token_validity_mask is not None and media_token_validity_mask.shape != input_ids.shape:
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(media_token_validity_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )

            # Match LLaVAModel's execution order: embed text first, then merge.
            image_anchor_mask = input_ids == self.image_token_index
            if media_token_validity_mask is not None:
                image_anchor_mask &= media_token_validity_mask.bool()
            text_ids = input_ids.masked_fill(image_anchor_mask, 0)
            combined_embeddings = self.language_model.embedding(input_ids=text_ids, position_ids=position_ids)
            if images is not None and images.numel() > 0:
                features = self._encode_images(images, imgs_sizes, vision_packed_seq_params, num_frames=num_frames)
            else:
                features = combined_embeddings.new_empty((0, combined_embeddings.shape[-1]))
            combined_embeddings = self._merge_projected_media(
                combined_embeddings, input_ids, features, media_token_validity_mask
            )

        if packed_seq_params is not None:
            # THD boundaries are final collator output. Token-aligned tensors may
            # be CP-sharded here, but the packing metadata is never rebuilt.
            attention_mask = None

        sequence_tensors = (
            (lm_input_ids, 1),
            (combined_embeddings, 0),
            (position_ids, 1),
            (labels, 1),
            (loss_mask, 1),
            (padding_mask, 1),
        )
        full_lengths = {tensor.size(dim) for tensor, dim in sequence_tensors if tensor is not None}
        if len(full_lengths) > 1:
            raise ValueError(f"Nemotron4 token-aligned tensors have inconsistent lengths: {sorted(full_lengths)}.")
        sliced_loss_mask = False
        cp_batch = None
        output_cp_batch = None
        if full_lengths and self.context_parallel_lm > 1:
            if packed_seq_params is None:
                raise ValueError(
                    "Nemotron4 context parallelism requires caller-packed THD inputs; enable policy.sequence_packing."
                )
            cu_seqlens = packed_seq_params.cu_seqlens_q.unsqueeze(0)
            cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
            if cu_seqlens_padded is not None:
                cu_seqlens_padded = cu_seqlens_padded.unsqueeze(0)
            batch = {
                "tokens": lm_input_ids,
                "decoder_input": (
                    combined_embeddings.transpose(0, 1).contiguous() if combined_embeddings is not None else None
                ),
                "position_ids": position_ids,
                "labels": labels,
                "loss_mask": loss_mask,
                # Reuse a sequence-valued CP batch slot for the valid-token mask. The packed
                # zigzag sharder fills synthetic positions with False, so inverting the local
                # attention-layout view below marks both source and synthetic padding correctly.
                "mtp_input_mask": None if padding_mask is None else ~padding_mask.bool(),
                "cu_seqlens": cu_seqlens,
                "cu_seqlens_padded": cu_seqlens_padded,
                "max_seqlen": torch.tensor(
                    [packed_seq_params.max_seqlen_q],
                    dtype=torch.int32,
                    device=cu_seqlens.device,
                ),
            }
            additional_layouts = {
                "zigzag",
                self.config.attention_cp_layout,
                self.config.linear_cp_layout,
            }
            cp_batch = get_batches_on_this_cp_rank(
                batch,
                boundary_layout=self.config.linear_cp_layout,
                is_hybrid_cp=False,
                cp_group=self.pg_collection.cp,
                additional_layouts=additional_layouts,
                sequence_parallel=self.sequence_parallel_lm,
                tp_group=self.pg_collection.tp,
                tp_cp_group=self.pg_collection.tp_cp,
            )
            output_cp_batch = cp_batch
            if self.post_process and self.sequence_parallel_lm and cp_batch.boundary_layout != "zigzag":
                # Output logits have gathered TP sequence shards. Build a CP-only
                # route rather than reuse the decoder's combined TP/SP route.
                output_cp_batch = get_batches_on_this_cp_rank(
                    {key: value for key, value in batch.items() if key != "decoder_input"},
                    boundary_layout=self.config.linear_cp_layout,
                    is_hybrid_cp=False,
                    cp_group=self.pg_collection.cp,
                    additional_layouts={"zigzag"},
                    sequence_parallel=False,
                    tp_group=self.pg_collection.tp,
                    tp_cp_group=self.pg_collection.tp_cp,
                )
            boundary_batch = cp_batch.get_batch()
            sliced_loss_mask = loss_mask is not None
            lm_input_ids = boundary_batch["tokens"]
            combined_embeddings = boundary_batch["decoder_input"]
            if combined_embeddings is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            position_ids = boundary_batch["position_ids"]
            labels = boundary_batch["labels"]
            loss_mask = boundary_batch["loss_mask"]
            if padding_mask is not None:
                # HybridStack passes padding_mask only to Transformer/MoE consumers, whose
                # physical token order is attention_cp_layout. GDP/Mamba layers do not consume it.
                attention_valid_tokens = cp_batch.get_batch(self.config.attention_cp_layout)["mtp_input_mask"]
                padding_mask = ~attention_valid_tokens.bool()
            packed_seq_params = cp_batch.get_packed_seq_params()
            for layout_batch in cp_batch.batches_by_layout.values():
                layout_batch.pop("decoder_input", None)
                layout_batch.pop("mtp_input_mask", None)

        if combined_embeddings is not None and self.sequence_parallel_lm:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings, group=self.pg_collection.tp
            ).contiguous()
        if padding_mask is not None and self.sequence_parallel_lm:
            padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    padding_mask.transpose(0, 1).contiguous(),
                    group=self.pg_collection.tp,
                )
                .transpose(0, 1)
                .contiguous()
            )

        # Once media is merged into the decoder embeddings the language model
        # must not also receive the pre-merge token IDs. MTP is the exception:
        # its loss derives targets from input_ids.
        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        if combined_embeddings is not None and not (mtp_num_layers and mtp_num_layers > 0):
            lm_input_ids = None

        output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            inference_context=inference_context,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=packed_seq_params,
            cp_batch=cp_batch,
        )
        if output_cp_batch is not None and self.post_process:
            # NeMo RL reconstructs packed losses and logprobs in per-sequence
            # zigzag order. Keep decoder/MTP inputs in their configured layouts,
            # and adapt only the final token-aligned outputs. The LM head has
            # already gathered the TP sequence shards, so SP is disabled here.
            output = (
                convert_cp_layout(
                    output.transpose(0, 1).contiguous(),
                    source_layout=output_cp_batch.boundary_layout,
                    target_layout="zigzag",
                    cp_group=self.pg_collection.cp,
                    thd_plan=output_cp_batch.thd_plan,
                )
                .transpose(0, 1)
                .contiguous()
            )
            if sliced_loss_mask:
                loss_mask = output_cp_batch.get_batch("zigzag")["loss_mask"]
        if sliced_loss_mask:
            return output, loss_mask
        return output
