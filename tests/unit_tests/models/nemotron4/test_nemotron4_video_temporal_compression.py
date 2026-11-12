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
"""Nemotron4 video temporal compression: tubelet encoder, provider/HF config plumbing and lazy init on HF import."""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from megatron.core.models.vision.vit_model import Mistral2DRotaryEmbedding, PatchEmbedding, ViTModel
from megatron.core.packed_seq_params import PackedSeqParams


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.hf_pretrained.state import StateDict
from megatron.bridge.models.megatron_mimo.conversion import MIMOComponent, build_route_local_registry
from megatron.bridge.models.nemotron4 import (
    Nemotron4Bridge,
    Nemotron4VLModelProvider,
    modeling_nemotron4_vl,
    nemotron4_bridge,
)
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4MIMOVisionEncoder,
    _build_vision_packed_seq_params,
    video_patch_weight_from_image,
)
from megatron.bridge.models.nemotron4.nemotron4_bridge import _validate_hf_config


pytestmark = pytest.mark.unit

P = 2  # patch size: one frame patch has 3 * P * P = 12 features
HIDDEN = 8
VIDEO_PARAM = "vision_model.video_patch_embed.proj.weight"
IMAGE_PARAM = "vision_model.patch_embed.proj.weight"
HF_VIDEO = "model.video_patch_conv.weight"
HF_IMAGE = "model.vision_tower.patch_conv.weight"


# ----------------------------------------------------------------------------------------------- encoder


class _Decoder(torch.nn.Module):
    """Per-token stand-in for the ViT transformer that depends on the 2D RoPE angles it receives."""

    def __init__(self):
        super().__init__()
        self.packed_seq_params = []

    def forward(self, hidden_states, attention_mask=None, rotary_pos_emb=None, packed_seq_params=None):
        self.packed_seq_params.append(packed_seq_params)
        angles = rotary_pos_emb.reshape(rotary_pos_emb.shape[0], 1, -1).sum(-1, keepdim=True)
        return torch.tanh(hidden_states) + angles.to(hidden_states.dtype)


class _Merger(torch.nn.Module):
    """2x2 average merger with the ``MistralPatchMerger`` call signature."""

    def forward(self, x, h_patches, w_patches):
        batch, _, hidden = x.shape
        merged = x.reshape(batch, h_patches // 2, 2, w_patches // 2, 2, hidden).mean(dim=(2, 4))
        return merged.reshape(batch, -1, hidden)


def _stub_vit_init(self, transformer_config, transformer_layer_spec=None, patch_dim=P, **kwargs):
    """CPU stand-in for ``ViTModel.__init__``: real patch embedding, RMSNorm and RoPE; fake transformer and merger."""
    del transformer_layer_spec, kwargs
    torch.nn.Module.__init__(self)
    hidden = transformer_config.hidden_size
    self.patch_dim = patch_dim
    self.add_class_token = False
    self.pos_emb_type = "rope2d"
    self.patch_embed = PatchEmbedding(3, hidden, patch_dim)
    self.ln_pre = torch.nn.RMSNorm(hidden, eps=1e-5)
    self.rope = Mistral2DRotaryEmbedding(hidden // 2, 16)
    self.decoder = _Decoder()
    self.merger = _Merger()


@pytest.fixture
def make_encoder(monkeypatch):
    monkeypatch.setattr(ViTModel, "__init__", _stub_vit_init)

    def make(temporal=2, seed=0, **kwargs):
        torch.manual_seed(seed)
        encoder = Nemotron4MIMOVisionEncoder(
            transformer_config=SimpleNamespace(hidden_size=HIDDEN), video_temporal_patch_size=temporal, **kwargs
        )
        return encoder.double().eval()

    return make


def _payload(sizes, num_frames=None):
    """Random per-frame patches for ``sizes`` and the matching per-frame THD parameters."""
    imgs_sizes = torch.tensor(sizes, dtype=torch.int32)
    patches = sum((h // P) * (w // P) for h, w in sizes)
    x = torch.randn(1, patches, 3 * P * P, dtype=torch.float64)
    frames = None if num_frames is None else torch.tensor(num_frames, dtype=torch.int32)
    return x, imgs_sizes, _build_vision_packed_seq_params(imgs_sizes, P, x.device), frames


def _encode(encoder, x, imgs_sizes, num_frames=None, **kwargs):
    packed = _build_vision_packed_seq_params(imgs_sizes, P, x.device)
    return encoder(x=x, imgs_sizes=imgs_sizes, packed_seq_params=packed, num_frames=num_frames, **kwargs)


def _shapes(module):
    return [(name, tuple(parameter.shape)) for name, parameter in module.named_parameters()]


def test_t1_builds_no_new_parameters_and_t2_adds_only_the_video_embedder(make_encoder):
    plain = make_encoder(temporal=1)
    torch.manual_seed(0)
    vit = ViTModel(transformer_config=SimpleNamespace(hidden_size=HIDDEN)).double()
    assert _shapes(plain) == _shapes(vit)
    assert list(plain.state_dict()) == list(vit.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(plain.state_dict().values(), vit.state_dict().values()))

    video = make_encoder(temporal=2)
    assert _shapes(video) == _shapes(plain) + [("video_patch_embed.proj.weight", (HIDDEN, 6, P, P))]
    assert video.video_patch_embed.proj.bias is None


def test_video_embedder_starts_from_the_image_weight_without_drawing_random_numbers(make_encoder):
    plain = make_encoder(temporal=1, seed=3)
    after_plain = torch.random.get_rng_state()
    video = make_encoder(temporal=2, seed=3)
    assert torch.equal(torch.random.get_rng_state(), after_plain)
    assert torch.equal(video.patch_embed.proj.weight, plain.patch_embed.proj.weight)
    assert torch.equal(
        video.video_patch_embed.proj.weight, video_patch_weight_from_image(video.patch_embed.proj.weight, 2)
    )


@pytest.mark.parametrize("kwargs", [dict(temporal=3), dict(temporal=0), dict(separate_video_embedder=False)])
def test_encoder_rejects_unsupported_video_settings(make_encoder, kwargs):
    with pytest.raises(ValueError, match="video_temporal_patch_size|separate_video_embedder"):
        make_encoder(**kwargs)


def test_derived_weight_is_frame_major():
    image = torch.randn(HIDDEN, 3, P, P, dtype=torch.float64)
    video = video_patch_weight_from_image(image, 2)
    assert video.shape == (HIDDEN, 6, P, P)
    # Input channel t * 3 + c is channel c of frame t, and the flattened weight is [W/2 | W/2] per frame.
    assert torch.equal(video[:, 0:3], image / 2) and torch.equal(video[:, 3:6], image / 2)
    assert torch.equal(video.flatten(1), torch.cat([image.flatten(1) / 2] * 2, dim=1))


def test_init_equivalence_duplicate_frames_and_mean_frame(make_encoder):
    encoder = make_encoder()
    first, second = torch.randn(2, 1, 5, 3 * P * P, dtype=torch.float64)
    duplicated = encoder.video_patch_embed.forward_patches(torch.cat([first, first], dim=-1))
    torch.testing.assert_close(duplicated, encoder.patch_embed.forward_patches(first), rtol=1e-12, atol=1e-12)
    distinct = encoder.video_patch_embed.forward_patches(torch.cat([first, second], dim=-1))
    mean = encoder.patch_embed.forward_patches((first + second) / 2)
    torch.testing.assert_close(distinct, mean, rtol=1e-12, atol=1e-12)


def test_duplicate_frame_video_encodes_as_its_frame_image(make_encoder):
    encoder = make_encoder()
    frame, sizes, _, _ = _payload([(4, 8)])
    video = _encode(encoder, torch.cat([frame, frame], dim=1), torch.cat([sizes, sizes]), torch.tensor([2]))
    image = _encode(encoder, frame, sizes, torch.tensor([1]))
    assert video.shape == image.shape == (1, 2, HIDDEN)
    torch.testing.assert_close(video, image, rtol=1e-12, atol=1e-12)


def test_grouping_keeps_media_order_and_rebuilds_tubelet_boundaries(make_encoder):
    encoder = make_encoder()
    image_a = [(4, 8)]
    video = [(4, 4)] * 4
    image_b = [(4, 4)]
    x, sizes, _, frames = _payload(image_a + video + image_b, num_frames=[1, 4, 1])
    encoder.decoder.packed_seq_params.clear()
    output = _encode(encoder, x, sizes, frames, expected_output_counts=torch.tensor([2, 1, 1, 1]))

    # Image A: 8 patches -> 2 rows. Video: 4 frames -> 2 tubelets of 4 patches -> 2 rows. Image B: 1 row.
    assert output.shape == (1, 5, HIDDEN)
    (packed,) = encoder.decoder.packed_seq_params
    assert packed.cu_seqlens_q.tolist() == [0, 8, 12, 16, 20]
    assert packed.max_seqlen_q == 8

    # Attention is per token in the stand-in, so each media item encodes as it would alone.
    a, v, b = torch.split(x, [8, 16, 4], dim=1)
    parts = [
        _encode(encoder, a, sizes[:1], torch.tensor([1])),
        _encode(encoder, v, sizes[1:5], torch.tensor([4])),
        _encode(encoder, b, sizes[5:], torch.tensor([1])),
    ]
    torch.testing.assert_close(output, torch.cat(parts, dim=1), rtol=1e-12, atol=1e-12)

    embedded, tubelet_sizes = encoder._embed_media(x, sizes, [1, 4, 1])
    assert tubelet_sizes.tolist() == [[4, 8], [4, 4], [4, 4], [4, 4]]
    assert tubelet_sizes.dtype == sizes.dtype
    frame = torch.split(v, 4, dim=1)
    expected = torch.cat(
        [
            encoder.patch_embed.forward_patches(a),
            encoder.video_patch_embed.forward_patches(torch.cat(frame[:2], dim=-1)),
            encoder.video_patch_embed.forward_patches(torch.cat(frame[2:], dim=-1)),
            encoder.patch_embed.forward_patches(b),
        ],
        dim=1,
    )
    torch.testing.assert_close(embedded, expected, rtol=1e-12, atol=1e-12)


def _capture_warnings(monkeypatch, module):
    messages = []
    monkeypatch.setattr(module.logger, "warning", lambda message, *args: messages.append(message % args))
    return messages


def test_odd_frame_count_repeats_the_last_frame(make_encoder, monkeypatch):
    encoder = make_encoder()
    x, sizes, _, _ = _payload([(4, 4)] * 3)
    messages = _capture_warnings(monkeypatch, modeling_nemotron4_vl)
    embedded, tubelet_sizes = encoder._embed_media(x, sizes, [3])
    assert encoder.padded_video_count == 1
    assert len(messages) == 1 and "repeated its last frame" in messages[0]
    assert tubelet_sizes.tolist() == [[4, 4], [4, 4]]
    frames = torch.split(x, 4, dim=1)
    last = encoder.video_patch_embed.forward_patches(torch.cat([frames[2], frames[2]], dim=-1))
    torch.testing.assert_close(embedded[:, 4:], last, rtol=1e-12, atol=1e-12)


def test_t1_and_image_only_payloads_take_the_plain_vit_forward(make_encoder):
    x, sizes, packed, _ = _payload([(4, 4)] * 2 + [(4, 8)])
    reference = ViTModel.forward(make_encoder(temporal=1), x, imgs_sizes=sizes, packed_seq_params=packed)
    for temporal, frames in ((1, [2, 1]), (1, None), (2, None), (2, [1, 1, 1])):
        encoder = make_encoder(temporal=temporal)
        encoder.decoder.packed_seq_params.clear()
        num_frames = None if frames is None else torch.tensor(frames)
        output = encoder(x=x, imgs_sizes=sizes, packed_seq_params=packed, num_frames=num_frames)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
        # The caller's THD parameters pass through unchanged.
        assert len(encoder.decoder.packed_seq_params) == 1 and encoder.decoder.packed_seq_params[0] is packed


def test_recompute_budget_sees_the_tubelet_patch_count(make_encoder, monkeypatch):
    encoder = make_encoder(recompute_budget_gib=1.0).train()
    seen = []
    monkeypatch.setattr(encoder, "_fit_recompute_to_budget", seen.append)
    x, sizes, _, frames = _payload([(4, 4)] * 4, num_frames=[4])
    _encode(encoder, x, sizes, frames)
    assert seen == [8]


@pytest.mark.parametrize("frames", [[2], [2, 1], [0, 4]])
def test_rejects_num_frames_that_disagree_with_imgs_sizes(make_encoder, frames):
    x, sizes, _, _ = _payload([(4, 4)] * 4)
    with pytest.raises(ValueError, match="num_frames"):
        _encode(make_encoder(), x, sizes, torch.tensor(frames))


def test_rejects_videos_with_mixed_frame_sizes(make_encoder):
    x, sizes, _, _ = _payload([(4, 4), (4, 8)])
    with pytest.raises(ValueError, match="one size"):
        _encode(make_encoder(), x, sizes, torch.tensor([2]))


def test_rows_are_checked_against_per_tubelet_output_counts(make_encoder):
    x, sizes, _, frames = _payload([(4, 4)] * 4, num_frames=[4])
    # Per-frame counts (4 rows) are what a data path without tubelet counting would send.
    with pytest.raises(RuntimeError, match="per tubelet"):
        _encode(make_encoder(), x, sizes, frames, expected_output_counts=torch.ones(4, dtype=torch.int32))
    assert _encode(make_encoder(), x, sizes, frames, expected_output_counts=torch.ones(2)).shape[1] == 2


# ----------------------------------------------------------------------------------------------- provider


def _provider(**overrides):
    return Nemotron4VLModelProvider(
        hybrid_layer_pattern="M-*EME",
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        kv_channels=32,
        vocab_size=128,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_latent_size=64,
        moe_shared_expert_intermediate_size=128,
        vision_num_layers=1,
        vision_hidden_size=64,
        vision_ffn_hidden_size=128,
        vision_num_attention_heads=4,
        vision_kv_channels=16,
        **overrides,
    )


def _encoder_params(provider):
    return provider.build_mimo_modality_submodules_spec()["images"].submodules["encoders"]["pixtral"].params


def test_provider_defaults_keep_t1_and_reach_the_mimo_encoder():
    provider = _provider()
    assert (provider.video_temporal_patch_size, provider.separate_video_embedder, provider.temporal_ckpt_compat) == (
        1,
        True,
        False,
    )
    params = _encoder_params(_provider(video_temporal_patch_size=2, temporal_ckpt_compat=True))
    assert params["video_temporal_patch_size"] == 2
    assert params["separate_video_embedder"] is True
    assert params["temporal_ckpt_compat"] is True


@pytest.mark.parametrize(
    "overrides",
    [dict(video_temporal_patch_size=4), dict(video_temporal_patch_size=2, separate_video_embedder=False)],
)
def test_provider_rejects_unsupported_video_settings(overrides):
    with pytest.raises(ValueError, match="video_temporal_patch_size|separate_video_embedder"):
        _encoder_params(_provider(**overrides))


def test_separate_video_embedder_is_ignored_with_t1(make_encoder):
    """The flag has no effect without tubelets, so a T = 1 config that sets it to False still loads."""
    assert _encoder_params(_provider(separate_video_embedder=False))["separate_video_embedder"] is False
    _validate_hf_config(_hf_config(separate_video_embedder=False))
    assert not hasattr(make_encoder(temporal=1, separate_video_embedder=False), "video_patch_embed")


@pytest.mark.parametrize(
    "overrides",
    [dict(video_temporal_patch_size=3), dict(video_temporal_patch_size=2, separate_video_embedder=False)],
)
def test_non_mimo_provide_rejects_unsupported_video_settings(overrides):
    """T = 2 builds (see test_nemotron4_video_temporal_compression_export.py); unsupported settings still raise."""
    with pytest.raises(ValueError, match="video_temporal_patch_size|separate_video_embedder"):
        _provider(**overrides).provide()


# ----------------------------------------------------------------------------------------------- HF config


def _hf_config(**top_level):
    text = SimpleNamespace(
        layer_types=["linear_attention", "mlp", "shortcut_moe_attn", "shortcut_moe_gdp"],
        num_nextn_predict_layers=0,
        mtp_use_repeated_layer=False,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=16,
        num_key_value_heads=1,
        head_dim=256,
        rms_norm_eps=1e-5,
        initializer_range=0.0162,
        activation_clamp_value=16.0,
        vocab_size=512,
        max_position_embeddings=4096,
        gdp_num_heads=4,
        gdp_num_groups=4,
        gdp_head_dim=32,
        gdp_key_head_dim=32,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        gdp_norm_eps=1e-5,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        moe_latent_size=64,
        moe_use_norm_before_up_proj=True,
        routed_scaling_factor=3.16,
        moe_shortcut_post_norm=False,
        wide_residual_num_streams=3,
        wide_residual_streamwise_sigmoid_init_scale=0.01,
        wide_residual_learned_retention=True,
        wide_residual_retention_init=0.999,
        wide_residual_retention_max_forget=0.10,
        mlp_hidden_act="relu2",
        residual_in_fp32=False,
        index_topk=None,
        index_head_dim=128,
    )
    vision = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        head_dim=32,
        patch_size=14,
        image_size=56,
        hidden_act="silu",
        rope_parameters={"rope_theta": 10000.0},
    )
    return SimpleNamespace(
        text_config=text,
        vision_config=vision,
        tie_word_embeddings=False,
        image_token_id=18,
        spatial_merge_size=2,
        projector_hidden_act="gelu_fast",
        multimodal_projector_bias=False,
        projector_norm_eps=1e-5,
        vision_feature_layer=-1,
        **top_level,
    )


def _bridge(**hf_top_level):
    bridge = Nemotron4Bridge()
    bridge.hf_config = _hf_config(**hf_top_level)
    return bridge


@pytest.mark.parametrize(
    "top_level,flag,expected",
    [
        ({}, None, (1, True, False)),
        (dict(video_temporal_patch_size=None, separate_video_embedder=None), None, (1, True, False)),
        (dict(video_temporal_patch_size=2), None, (2, True, False)),
        (dict(video_temporal_patch_size=2, separate_video_embedder=True), True, (2, True, True)),
    ],
)
def test_provider_bridge_reads_the_top_level_hf_fields_and_the_import_flag(top_level, flag, expected):
    bridge = Nemotron4Bridge()
    if flag is not None:
        bridge.temporal_ckpt_compat = flag
    provider = bridge.provider_bridge(SimpleNamespace(config=_hf_config(**top_level)))
    assert (provider.video_temporal_patch_size, provider.separate_video_embedder, provider.temporal_ckpt_compat) == (
        expected
    )


@pytest.mark.parametrize(
    "top_level",
    [dict(video_temporal_patch_size=3), dict(video_temporal_patch_size=2, separate_video_embedder=False)],
)
def test_hf_config_validation_rejects_unsupported_video_settings(top_level):
    with pytest.raises(ValueError, match="video_temporal_patch_size|separate_video_embedder"):
        _validate_hf_config(_hf_config(**top_level))


# ----------------------------------------------------------------------------------------------- mappings


def test_t1_registry_has_no_video_mapping():
    registry = _bridge().mapping_registry()
    assert registry.megatron_to_hf_lookup(VIDEO_PARAM) is None
    assert not any("video" in str(mapping.hf_param) for mapping in registry.mappings)


def test_t2_maps_the_video_embedder_to_the_top_level_hf_tensor():
    bridge = _bridge(video_temporal_patch_size=2)
    registry = bridge.mapping_registry()
    mapping = registry.megatron_to_hf_lookup(VIDEO_PARAM)
    assert mapping.hf_param == HF_VIDEO
    assert mapping.allow_hf_name_mismatch
    assert registry.megatron_to_hf_lookup(IMAGE_PARAM).hf_param == HF_IMAGE
    # The MIMO images route keeps the mapping and its flag.
    route = MIMOComponent("images", "vision_model.", "modality_submodules.images.encoders.pixtral")
    local = build_route_local_registry(registry, route).megatron_to_hf_lookup("video_patch_embed.proj.weight")
    assert local.hf_param == HF_VIDEO and local.allow_hf_name_mismatch
    # A checkpoint without the tensor passes the strict key check.
    bridge._validate_conversion_mappings(registry, [IMAGE_PARAM, VIDEO_PARAM], {HF_IMAGE})


def test_video_task_declares_the_image_patch_conv_as_an_input():
    assert Nemotron4Bridge.get_hf_import_param_names(HF_VIDEO, {HF_IMAGE}) == (HF_IMAGE,)
    assert Nemotron4Bridge.get_hf_import_param_names(HF_VIDEO, {HF_IMAGE, HF_VIDEO}) == (HF_VIDEO, HF_IMAGE)
    assert Nemotron4Bridge.get_hf_import_param_names(HF_VIDEO) == (HF_VIDEO, HF_IMAGE)
    for available in (None, {HF_IMAGE}):
        assert Nemotron4Bridge.get_hf_import_param_names(HF_IMAGE, available) == (
            MegatronModelBridge.get_hf_import_param_names(HF_IMAGE, available)
        )


# ----------------------------------------------------------------------------------------------- lazy init


def _import_bridge(flag=None, source=None):
    bridge = _bridge(video_temporal_patch_size=2)
    if flag is not None:
        bridge.temporal_ckpt_compat = flag
    if source is not None:
        bridge.source_video_temporal_patch_size = source
    return bridge


def _state(with_video):
    tensors = {HF_IMAGE: torch.randn(HIDDEN, 3, 14, 14, dtype=torch.bfloat16)}
    if with_video:
        tensors[HF_VIDEO] = torch.randn(HIDDEN, 6, 14, 14, dtype=torch.bfloat16)
    return tensors


@pytest.mark.parametrize("flag,source", [(None, None), (True, 1), (False, 2), (True, 2)])
@pytest.mark.parametrize("lazy", [False, True], ids=["dict", "lazy_state_dict"])
def test_present_video_tensor_loads_as_is(flag, source, lazy):
    tensors = _state(with_video=True)
    state = StateDict(tensors) if lazy else tensors
    loaded = _import_bridge(flag, source).maybe_modify_loaded_hf_weight(HF_VIDEO, state)
    assert torch.equal(loaded, tensors[HF_VIDEO])


@pytest.mark.parametrize("source", [1, 0])
@pytest.mark.parametrize("lazy", [False, True], ids=["dict", "lazy_state_dict"])
def test_missing_video_tensor_is_derived_with_the_flag_and_a_t1_source(source, lazy, monkeypatch):
    tensors = _state(with_video=False)
    state = StateDict(tensors) if lazy else tensors
    messages = _capture_warnings(monkeypatch, nemotron4_bridge)
    loaded = _import_bridge(True, source).maybe_modify_loaded_hf_weight(HF_VIDEO, state)
    expected = torch.cat([tensors[HF_IMAGE] / 2] * 2, dim=1)
    assert loaded.dtype == expected.dtype
    assert torch.equal(loaded, expected)
    assert len(messages) == 1 and HF_VIDEO in messages[0]


@pytest.mark.parametrize(
    "flag,source,match",
    [
        (None, 1, "enable the lazy init"),
        (False, 1, "enable the lazy init"),
        (True, 2, "trained video embedder was lost"),
        (False, 2, "trained video embedder was lost"),
        # Without source_video_temporal_patch_size the HF config's own T=2 is the source T.
        (None, None, "trained video embedder was lost"),
        (True, None, "trained video embedder was lost"),
    ],
)
def test_missing_video_tensor_raises_without_the_lazy_init(flag, source, match):
    with pytest.raises(ValueError, match=match):
        _import_bridge(flag, source).maybe_modify_loaded_hf_weight(HF_VIDEO, StateDict(_state(with_video=False)))


def test_import_options_are_declared_with_defaults():
    bridge = Nemotron4Bridge()
    assert bridge.temporal_ckpt_compat is False
    assert bridge.source_video_temporal_patch_size is None
    # The provider gets the declared flag.
    provider = bridge.provider_bridge(SimpleNamespace(config=_hf_config(video_temporal_patch_size=2)))
    assert provider.temporal_ckpt_compat is False


def test_configure_video_patch_import_sets_both_options():
    bridge = _bridge(video_temporal_patch_size=2)
    bridge.configure_video_patch_import(temporal_ckpt_compat=True, source_video_temporal_patch_size=1)
    assert (bridge.temporal_ckpt_compat, bridge.source_video_temporal_patch_size) == (True, 1)
    # The lazy init then derives the missing tensor.
    tensors = _state(with_video=False)
    loaded = bridge.maybe_modify_loaded_hf_weight(HF_VIDEO, tensors)
    assert torch.equal(loaded, torch.cat([tensors[HF_IMAGE] / 2] * 2, dim=1))
    # Instances do not share the options.
    assert Nemotron4Bridge().temporal_ckpt_compat is False


@pytest.mark.parametrize("source", [0, -1, 1.0, True, "1", None])
def test_configure_video_patch_import_rejects_a_bad_source_t(source):
    with pytest.raises(ValueError, match="source_video_temporal_patch_size"):
        Nemotron4Bridge().configure_video_patch_import(
            temporal_ckpt_compat=True, source_video_temporal_patch_size=source
        )


def test_configure_video_patch_import_takes_only_the_declared_options():
    with pytest.raises(TypeError):
        Nemotron4Bridge().configure_video_patch_import(temporal_ckpt_compat=True, source_temporal_patch_size=1)


def test_other_tensors_load_unchanged():
    tensors = _state(with_video=False)
    assert _import_bridge(True).maybe_modify_loaded_hf_weight(HF_IMAGE, tensors) is tensors[HF_IMAGE]


def test_encoder_payload_without_num_frames_is_unchanged_with_t2(make_encoder):
    x, sizes, packed, _ = _payload([(4, 4)] * 2)
    encoder = make_encoder()
    output = encoder(x=x, imgs_sizes=sizes, packed_seq_params=packed)
    assert isinstance(encoder.decoder.packed_seq_params[0], PackedSeqParams)
    assert output.shape == (1, 2, HIDDEN)


# ----------------------------------------------------------------------------------------------- state-dict load


def test_t2_encoder_loads_its_own_state_dict_unchanged(make_encoder):
    source = make_encoder(seed=1)
    with torch.no_grad():
        source.video_patch_embed.proj.weight.add_(1.0)  # a trained value, not the derived one
    target = make_encoder(seed=2, temporal_ckpt_compat=True)
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.video_patch_embed.proj.weight, source.video_patch_embed.proj.weight)


def test_t1_state_dict_into_t2_encoder_raises_without_the_flag(make_encoder):
    with pytest.raises(RuntimeError, match="temporal_ckpt_compat"):
        make_encoder(seed=2).load_state_dict(make_encoder(temporal=1, seed=1).state_dict())
    # Bridge retries a failed strict load with strict=False; the hook still raises.
    with pytest.raises(RuntimeError, match="temporal_ckpt_compat"):
        make_encoder(seed=2).load_state_dict(make_encoder(temporal=1, seed=1).state_dict(), strict=False)


def test_t1_state_dict_into_t2_encoder_derives_from_the_loaded_image_weight(make_encoder, monkeypatch):
    source = make_encoder(temporal=1, seed=1)
    target = torch.nn.Module()
    target.encoder = make_encoder(seed=2, temporal_ckpt_compat=True)
    messages = _capture_warnings(monkeypatch, modeling_nemotron4_vl)
    state = {f"encoder.{name}": value for name, value in source.state_dict().items()}
    target.load_state_dict(state)  # strict: the hook fills the only missing key
    loaded = target.encoder
    assert torch.equal(loaded.patch_embed.proj.weight, source.patch_embed.proj.weight)
    assert torch.equal(
        loaded.video_patch_embed.proj.weight, video_patch_weight_from_image(source.patch_embed.proj.weight, 2)
    )
    assert len(messages) == 1 and "encoder.video_patch_embed.proj.weight" in messages[0]


def test_adapter_only_state_dict_is_left_alone(make_encoder):
    encoder = make_encoder()
    before = encoder.video_patch_embed.proj.weight.clone()
    adapter_only = {"decoder.adapter.linear_in.weight": torch.ones(HIDDEN, dtype=torch.float64)}
    result = encoder.load_state_dict(adapter_only, strict=False)
    assert "video_patch_embed.proj.weight" in result.missing_keys
    assert torch.equal(encoder.video_patch_embed.proj.weight, before)


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("strict", [True, False])
def test_encoder_state_dict_without_the_image_weight_raises(make_encoder, flag, strict):
    """No video key and no image key to derive it from: raise even with strict=False (Bridge's retry)."""
    encoder = make_encoder(temporal_ckpt_compat=flag)
    with pytest.raises(RuntimeError, match="neither"):
        encoder.load_state_dict({"ln_pre.weight": torch.ones(HIDDEN, dtype=torch.float64)}, strict=strict)


def test_t1_encoder_registers_no_load_hook(make_encoder):
    torch.manual_seed(0)
    vit = ViTModel(transformer_config=SimpleNamespace(hidden_size=HIDDEN))
    assert len(make_encoder(temporal=1)._load_state_dict_pre_hooks) == len(vit._load_state_dict_pre_hooks)
    assert len(make_encoder()._load_state_dict_pre_hooks) == len(vit._load_state_dict_pre_hooks) + 1


def test_missing_num_frames_with_t2_warns_once(make_encoder, monkeypatch):
    x, sizes, packed, _ = _payload([(4, 4)] * 2)
    messages = _capture_warnings(monkeypatch, modeling_nemotron4_vl)
    encoder = make_encoder()
    for _ in range(2):
        encoder(x=x, imgs_sizes=sizes, packed_seq_params=packed)
    assert len(messages) == 1 and "no num_frames" in messages[0]
    plain = make_encoder(temporal=1)
    plain(x=x, imgs_sizes=sizes, packed_seq_params=packed)
    encoder(x=x, imgs_sizes=sizes, packed_seq_params=packed, num_frames=torch.tensor([2]))
    assert len(messages) == 1


# ----------------------------------------------------------------------------------------------- one-frame videos


def _trained(encoder):
    """Give the video embedder a value other than the derived one, so the video and image paths differ."""
    with torch.no_grad():
        encoder.video_patch_embed.proj.weight.add_(0.25)
    return encoder


def test_one_frame_video_is_a_repeated_tubelet_through_the_video_embedder(make_encoder, monkeypatch):
    """A one-frame video is [f, f] through video_patch_embed, as in HF and vLLM; the same frame as an image is not."""
    encoder = _trained(make_encoder())
    frame, sizes, _, _ = _payload([(4, 8)])
    messages = _capture_warnings(monkeypatch, modeling_nemotron4_vl)
    one = _encode(encoder, frame, sizes, torch.tensor([1]), media_is_video=torch.tensor([True]))
    assert len(messages) == 1 and "repeated its last frame" in messages[0]
    pair = _encode(encoder, torch.cat([frame, frame], dim=1), torch.cat([sizes, sizes]), torch.tensor([2]))
    torch.testing.assert_close(one, pair, rtol=0, atol=0)
    image = _encode(encoder, frame, sizes, torch.tensor([1]), media_is_video=torch.tensor([False]))
    assert one.shape == image.shape == (1, 2, HIDDEN)
    assert not torch.allclose(one, image)
    # Without the flag a one-frame occurrence is an image, as before.
    torch.testing.assert_close(_encode(encoder, frame, sizes, torch.tensor([1])), image, rtol=0, atol=0)


def test_one_frame_video_with_the_derived_weight_equals_its_image(make_encoder):
    encoder = make_encoder()
    frame, sizes, _, _ = _payload([(4, 4)])
    video = _encode(encoder, frame, sizes, torch.tensor([1]), media_is_video=torch.tensor([True]))
    torch.testing.assert_close(video, _encode(encoder, frame, sizes, torch.tensor([1])), rtol=1e-12, atol=1e-12)


def test_flagged_payload_keeps_media_order_and_images(make_encoder):
    encoder = _trained(make_encoder())
    image_a, short, video, image_b = [(4, 8)], [(4, 4)], [(4, 4)] * 4, [(4, 4)]
    x, sizes, _, frames = _payload(image_a + short + video + image_b, num_frames=[1, 1, 4, 1])
    flags = torch.tensor([False, True, True, False])
    encoder.decoder.packed_seq_params.clear()
    output = _encode(
        encoder, x, sizes, frames, media_is_video=flags, expected_output_counts=torch.tensor([2, 1, 1, 1, 1])
    )
    # Image A: 2 rows. One-frame video: 1 tubelet, 1 row. Video: 2 tubelets, 2 rows. Image B: 1 row.
    assert output.shape == (1, 6, HIDDEN)
    (packed,) = encoder.decoder.packed_seq_params
    assert packed.cu_seqlens_q.tolist() == [0, 8, 12, 16, 20, 24]
    a, s, v, b = torch.split(x, [8, 4, 16, 4], dim=1)
    parts = [
        _encode(encoder, a, sizes[:1], torch.tensor([1])),
        _encode(encoder, torch.cat([s, s], dim=1), sizes[1:2].repeat(2, 1), torch.tensor([2])),
        _encode(encoder, v, sizes[2:6], torch.tensor([4])),
        _encode(encoder, b, sizes[6:], torch.tensor([1])),
    ]
    torch.testing.assert_close(output, torch.cat(parts, dim=1), rtol=1e-12, atol=1e-12)
    # The image rows use patch_embed, unchanged by the flags of the other media.
    embedded, tubelet_sizes = encoder._embed_media(x, sizes, [1, 1, 4, 1], [False, True, True, False])
    assert tubelet_sizes.tolist() == [[4, 8], [4, 4], [4, 4], [4, 4], [4, 4]]
    torch.testing.assert_close(embedded[:, :8], encoder.patch_embed.forward_patches(a), rtol=0, atol=0)
    torch.testing.assert_close(embedded[:, -4:], encoder.patch_embed.forward_patches(b), rtol=0, atol=0)


def test_unflagged_payload_matches_the_flag_free_rule(make_encoder):
    encoder = _trained(make_encoder())
    x, sizes, _, frames = _payload([(4, 8)] + [(4, 4)] * 4 + [(4, 4)], num_frames=[1, 4, 1])
    flags = torch.tensor([False, True, False])
    torch.testing.assert_close(
        _encode(encoder, x, sizes, frames, media_is_video=flags), _encode(encoder, x, sizes, frames), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "frames,flags,match",
    [
        ([1, 4], [True], "one entry per media occurrence"),
        ([1, 4], [True, False], "more than one frame"),
    ],
)
def test_rejects_inconsistent_video_flags(make_encoder, frames, flags, match):
    x, sizes, _, num_frames = _payload([(4, 4)] * sum(frames), num_frames=frames)
    with pytest.raises(ValueError, match=match):
        _encode(make_encoder(), x, sizes, num_frames, media_is_video=torch.tensor(flags))


def test_video_flags_need_num_frames_with_t2_and_are_ignored_with_t1(make_encoder):
    x, sizes, packed, _ = _payload([(4, 4)])
    with pytest.raises(ValueError, match="needs num_frames"):
        make_encoder()(x=x, imgs_sizes=sizes, packed_seq_params=packed, media_is_video=torch.tensor([True]))
    plain = make_encoder(temporal=1)
    reference = ViTModel.forward(plain, x, imgs_sizes=sizes, packed_seq_params=packed)
    flags = torch.tensor([True])
    for frames in (None, torch.tensor([1])):
        output = plain(x=x, imgs_sizes=sizes, packed_seq_params=packed, num_frames=frames, media_is_video=flags)
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
