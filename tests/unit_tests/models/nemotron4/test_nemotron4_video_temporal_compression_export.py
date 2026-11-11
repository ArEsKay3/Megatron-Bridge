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
"""Nemotron4 video temporal compression: HF export of a T > 1 checkpoint.

Covers the non-MIMO ``Nemotron4VLModel`` build (the SSA export pipeline loads a Megatron checkpoint into it), the
HF config that ``megatron_to_hf_config`` synthesizes, and the export guards (MIMO ``save_hf_pretrained_mimo`` and
non-MIMO ``AutoBridge.save_hf_pretrained``) that keep a non-strict save from skipping
``model.video_patch_conv.weight``.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from megatron.core.models.vision.vit_model import Mistral2DRotaryEmbedding, PatchEmbedding, ViTModel


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

import megatron.bridge.models.megatron_mimo.conversion.orchestrator as orchestrator_module
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.utils import conform_config_to_reference
from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource
from megatron.bridge.models.nemotron4 import Nemotron4Bridge, Nemotron4VLModelProvider, nemotron4_bridge
from megatron.bridge.models.nemotron4 import modeling_nemotron4_vl as nm4
from megatron.bridge.models.nemotron4 import nemotron4_provider as nm4_provider
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import Nemotron4MIMOVisionEncoder
from megatron.bridge.models.nemotron4.nemotron4_bridge import assert_no_temporal_ckpt_compat


pytestmark = pytest.mark.unit

P = 2
HIDDEN = 8
VIDEO_PARAM = "vision_model.video_patch_embed.proj.weight"
HF_VIDEO = "model.video_patch_conv.weight"


# ----------------------------------------------------------------------------------------------- helpers


def _stub_vit_init(self, transformer_config, transformer_layer_spec=None, patch_dim=P, **kwargs):
    """CPU stand-in for ``ViTModel.__init__``: real patch embedding, RMSNorm and RoPE; no transformer layers."""
    del transformer_layer_spec, kwargs
    torch.nn.Module.__init__(self)
    hidden = transformer_config.hidden_size
    self.patch_dim = patch_dim
    self.add_class_token = False
    self.pos_emb_type = "rope2d"
    self.patch_embed = PatchEmbedding(3, hidden, patch_dim)
    self.ln_pre = torch.nn.RMSNorm(hidden, eps=1e-5)
    self.rope = Mistral2DRotaryEmbedding(hidden // 2, 16)


def _stub_vit_forward(self, x, imgs_sizes=None, packed_seq_params=None):
    """Patch embedding of every frame, then 2x2 average merge, so the output has one row per placeholder."""
    rows = []
    for chunk, (h, w) in zip(
        torch.split(x, [int(h // P) * int(w // P) for h, w in imgs_sizes.tolist()], dim=1), imgs_sizes.tolist()
    ):
        embedded = self.patch_embed.forward_patches(chunk)
        hp, wp = h // P, w // P
        rows.append(
            embedded.reshape(1, hp // 2, 2, wp // 2, 2, -1).mean(dim=(2, 4)).reshape(1, -1, embedded.shape[-1])
        )
    return torch.cat(rows, dim=1)


def _groups():
    cp = MagicMock()
    cp.size.return_value = 1
    groups = SimpleNamespace(cp=cp, tp=MagicMock(), tp_cp=MagicMock(), hcp=[MagicMock()])
    groups.gtp_remat = MagicMock()
    groups.expt_gtp_remat = MagicMock()
    return groups


@pytest.fixture
def make_model(monkeypatch):
    """Build ``Nemotron4VLModel`` on CPU with the real vision module classes and stubbed language and projector."""
    monkeypatch.setattr(ViTModel, "__init__", _stub_vit_init)
    monkeypatch.setattr(ViTModel, "forward", _stub_vit_forward)

    def make(**video_kwargs):
        torch.manual_seed(0)
        with (
            patch.object(nm4, "HybridModel", return_value=torch.nn.Identity()),
            patch.object(nm4, "MultimodalProjector", side_effect=lambda *a, **k: torch.nn.Identity()),
            patch.object(nm4, "logit_dtype_kwarg", return_value={}),
        ):
            return nm4.Nemotron4VLModel(
                language_transformer_config=SimpleNamespace(
                    sequence_parallel=False, context_parallel_size=1, logit_dtype=torch.float32
                ),
                language_transformer_layer_spec=None,
                language_vocab_size=16,
                language_max_sequence_length=16,
                vision_transformer_config=SimpleNamespace(hidden_size=HIDDEN),
                vision_transformer_layer_spec=None,
                vision_projection_config=None,
                vision_projection_layer_spec=None,
                image_token_index=1,
                img_h=8,
                img_w=8,
                patch_dim=P,
                spatial_merge_size=2,
                hybrid_layer_pattern=None,
                pg_collection=_groups(),
                vision_dp_over_cp=False,
                **video_kwargs,
            )

    return make


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


def _hf_config(**top_level):
    """Smallest HF config that ``Nemotron4Bridge.mapping_registry`` reads."""
    text = SimpleNamespace(
        wide_residual_num_streams=3,
        wide_residual_learned_retention=True,
        moe_use_norm_before_up_proj=True,
        moe_shortcut_post_norm=False,
        num_nextn_predict_layers=0,
    )
    vision = SimpleNamespace(head_dim=32, num_hidden_layers=1)
    return SimpleNamespace(text_config=text, vision_config=vision, **top_level)


# ----------------------------------------------------------------------------------------------- (a) non-MIMO build


def test_t1_model_keeps_the_plain_vit(make_model):
    model = make_model()
    assert type(model.vision_model) is ViTModel
    assert model.video_temporal_patch_size == 1
    assert not any("video" in name for name in model.state_dict())


def test_t2_model_builds_the_video_embedder_under_the_bridge_name(make_model):
    model = make_model(video_temporal_patch_size=2, temporal_ckpt_compat=False)
    assert isinstance(model.vision_model, Nemotron4MIMOVisionEncoder)
    assert model.vision_model.video_temporal_patch_size == 2
    assert model.vision_model.temporal_ckpt_compat is False
    state = model.state_dict()
    assert tuple(state[VIDEO_PARAM].shape) == (HIDDEN, 6, P, P)
    # Same parameters as T = 1 plus the video embedder, which is the last one.
    names = [name for name, _ in model.named_parameters()]
    assert names == [name for name, _ in make_model().named_parameters()] + [VIDEO_PARAM]
    # The export registry built from a T = 2 HF config maps every vision parameter, the video embedder included.
    bridge = Nemotron4Bridge()
    bridge.hf_config = _hf_config(video_temporal_patch_size=2)
    registry = bridge.mapping_registry()
    vision_names = [name for name in names if name.startswith("vision_model.")]
    assert vision_names and all(registry.megatron_to_hf_lookup(name) is not None for name in vision_names)
    assert registry.megatron_to_hf_lookup(VIDEO_PARAM).hf_param == HF_VIDEO


def test_t2_model_loads_a_saved_video_weight_without_lazy_init(make_model):
    source = make_model(video_temporal_patch_size=2)
    with torch.no_grad():
        source.vision_model.video_patch_embed.proj.weight.add_(1.0)
    target = make_model(video_temporal_patch_size=2)
    target.load_state_dict(source.state_dict())
    assert torch.equal(
        target.vision_model.video_patch_embed.proj.weight, source.vision_model.video_patch_embed.proj.weight
    )
    # A T = 1 checkpoint (no video key) into the T = 2 export model raises without the lazy-init flag.
    plain = make_model()
    with pytest.raises(RuntimeError, match="temporal_ckpt_compat"):
        make_model(video_temporal_patch_size=2).load_state_dict(plain.state_dict())


@pytest.mark.parametrize("num_frames", [None, [2], [1, 2]], ids=["missing", "video", "image_and_video"])
def test_t2_model_rejects_video_forwards(make_model, num_frames):
    model = make_model(video_temporal_patch_size=2)
    count = 1 if num_frames is None else sum(num_frames)
    images = torch.randn(1, 4 * count, 3 * P * P)
    sizes = torch.tensor([[4, 4]] * count)
    frames = None if num_frames is None else torch.tensor(num_frames)
    with pytest.raises(NotImplementedError, match="MIMO model"):
        model._encode_images(images, sizes, None, num_frames=frames)


def test_t2_model_encodes_images_with_the_image_embedder(make_model):
    t1, t2 = make_model(), make_model(video_temporal_patch_size=2)
    images = torch.randn(1, 8, 3 * P * P)
    sizes = torch.tensor([[4, 4], [4, 4]])
    with torch.no_grad():
        expected = t1._encode_images(images, sizes, None)
        actual = t2._encode_images(images, sizes, None, num_frames=torch.tensor([1, 1]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # num_frames does not change the T = 1 path.
    torch.testing.assert_close(t1._encode_images(images, sizes, None, num_frames=torch.tensor([2])), expected)


def test_t2_forward_passes_num_frames_to_the_media_check(make_model):
    model = make_model(video_temporal_patch_size=2)
    language = torch.nn.Module()
    language.embedding = lambda input_ids, position_ids: torch.zeros(input_ids.shape[1], input_ids.shape[0], HIDDEN)
    model.language_model = language
    input_ids = torch.tensor([[1, 1, 1, 1, 0]])
    images = torch.randn(1, 8, 3 * P * P)
    sizes = torch.tensor([[4, 4], [4, 4]])
    with pytest.raises(NotImplementedError, match="MIMO model"):
        model(input_ids, images=images, imgs_sizes=sizes, num_frames=torch.tensor([2]))


def test_provide_passes_the_video_settings_only_for_t2(monkeypatch):
    built = []
    monkeypatch.setattr(nm4_provider, "Nemotron4VLModel", lambda **kwargs: built.append(kwargs) or Mock())
    monkeypatch.setattr(nm4_provider, "require_nemotron4_mcore", lambda: None)
    monkeypatch.setattr(nm4_provider.ProcessGroupCollection, "use_mpu_process_groups", staticmethod(lambda: Mock()))
    for name in ("_language_layer_spec", "_vision_layer_spec"):
        monkeypatch.setattr(Nemotron4VLModelProvider, name, lambda *args: None)
    monkeypatch.setattr(Nemotron4VLModelProvider, "_vision_configs", lambda self, cfg: (None, None))
    monkeypatch.setattr(
        Nemotron4VLModelProvider, "_copy_config_without_runtime_process_groups", lambda self, deep=False: self
    )

    _provider().provide()
    _provider(video_temporal_patch_size=2, temporal_ckpt_compat=True).provide()
    assert not any("video" in key or "temporal" in key for key in built[0])
    assert (built[1]["video_temporal_patch_size"], built[1]["separate_video_embedder"]) == (2, True)
    assert built[1]["temporal_ckpt_compat"] is True


# ----------------------------------------------------------------------------------------------- (b) HF config


@pytest.fixture
def base_hf_config(monkeypatch):
    """Isolate ``Nemotron4Bridge.megatron_to_hf_config`` from the generic CONFIG_MAPPING conversion."""
    base = {"text_config": {"hidden_size": 128}, "image_token_id": 18}
    monkeypatch.setattr(
        MegatronModelBridge, "megatron_to_hf_config", classmethod(lambda cls, provider: {**base, **provider.extra})
    )
    monkeypatch.setattr(nemotron4_bridge, "hf_mtp_layer_count", lambda provider: 2)
    return base


def _config_provider(extra=None, **fields):
    defaults = dict(video_temporal_patch_size=1, separate_video_embedder=True, temporal_ckpt_compat=False)
    return SimpleNamespace(
        extra=extra or {},
        experimental_attention_variant=None,
        fp32_residual_connection=False,
        **{**defaults, **fields},
    )


def test_t1_hf_config_has_no_video_fields(base_hf_config):
    cfg = Nemotron4Bridge.megatron_to_hf_config(_config_provider(temporal_ckpt_compat=True))
    assert "video_temporal_patch_size" not in cfg and "separate_video_embedder" not in cfg
    assert "temporal_ckpt_compat" not in str(cfg)


def test_t2_hf_config_declares_the_video_fields_but_never_the_lazy_init_flag(base_hf_config):
    cfg = Nemotron4Bridge.megatron_to_hf_config(
        _config_provider(video_temporal_patch_size=2, temporal_ckpt_compat=True)
    )
    assert (cfg["video_temporal_patch_size"], cfg["separate_video_embedder"]) == (2, True)
    assert "temporal_ckpt_compat" not in str(cfg)
    # A T = 2 template keeps the fields; a template without them (old remote code) drops them, and the T = 1
    # registry then has no mapping for the video weight, so the export fails instead of skipping it.
    t2_reference = {**base_hf_config, "video_temporal_patch_size": 2, "separate_video_embedder": True}
    assert conform_config_to_reference(cfg, t2_reference)["video_temporal_patch_size"] == 2
    assert "video_temporal_patch_size" not in conform_config_to_reference(cfg, base_hf_config)
    bridge = Nemotron4Bridge()
    bridge.hf_config = _hf_config()
    with pytest.raises(ValueError, match="No mapping found"):
        bridge._validate_conversion_mappings(bridge.mapping_registry(), [VIDEO_PARAM])


@pytest.mark.parametrize(
    "extra", [{"temporal_ckpt_compat": True}, {"text_config": {"temporal_ckpt_compat": False}}], ids=["top", "nested"]
)
def test_hf_config_with_the_lazy_init_flag_raises(base_hf_config, extra):
    with pytest.raises(AssertionError, match="temporal_ckpt_compat"):
        Nemotron4Bridge.megatron_to_hf_config(_config_provider(extra=extra, video_temporal_patch_size=2))


def test_hf_config_rejects_unsupported_video_settings(base_hf_config):
    with pytest.raises(ValueError, match="separate_video_embedder"):
        Nemotron4Bridge.megatron_to_hf_config(
            _config_provider(video_temporal_patch_size=2, separate_video_embedder=False)
        )


def test_assert_no_temporal_ckpt_compat_accepts_clean_configs():
    assert_no_temporal_ckpt_compat({"video_temporal_patch_size": 2, "text_config": {"a": {"b": 1}}})


# ----------------------------------------------------------------------------------------------- (c) MIMO export


def test_required_export_keys():
    t2_provider = SimpleNamespace(video_temporal_patch_size=2)
    assert Nemotron4Bridge.required_hf_export_keys(SimpleNamespace(video_temporal_patch_size=1), _hf_config()) == ()
    assert Nemotron4Bridge.required_hf_export_keys(t2_provider, _hf_config(video_temporal_patch_size=2)) == (HF_VIDEO,)
    with pytest.raises(ValueError, match="declares 1"):
        Nemotron4Bridge.required_hf_export_keys(t2_provider, _hf_config())
    with pytest.raises(AssertionError, match="temporal_ckpt_compat"):
        Nemotron4Bridge.required_hf_export_keys(
            t2_provider, _hf_config(video_temporal_patch_size=2, temporal_ckpt_compat=True)
        )


def _mimo_bridge(template_keys, yielded, temporal=2, hf_temporal=2):
    hf_pretrained = Mock()
    hf_pretrained.config = _hf_config(video_temporal_patch_size=hf_temporal)
    hf_pretrained.model_name_or_path = "/templates/t2"
    state_source = Mock(spec=SafeTensorsStateSource)
    state_source.get_all_keys.return_value = sorted(template_keys)
    state_source.has_glob.return_value = False
    # A non-strict save writes what it gets and does not check for missing keys.
    state_source.save_generator.side_effect = lambda generator, path, **kwargs: list(generator)
    hf_pretrained.state.source = state_source
    bridge = Mock()
    bridge.hf_pretrained = hf_pretrained
    bridge._model_bridge = Nemotron4Bridge()
    bridge._require_provider.return_value = SimpleNamespace(
        standard_provider=SimpleNamespace(video_temporal_patch_size=temporal, mtp_num_layers=None)
    )
    tensors = [(name, torch.zeros(1)) for name in yielded]
    return bridge, tensors


@pytest.fixture
def mimo_save(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestrator_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(orchestrator_module, "_model_omits_mtp", lambda provider: False)

    def save(bridge, tensors):
        monkeypatch.setattr(orchestrator_module, "_stream_mimo_weights_to_rank0", lambda **kwargs: iter(tensors))
        orchestrator_module.save_hf_pretrained_mimo(bridge, Mock(), [], {}, tmp_path / "hf")  # strict=False
        return bridge

    return save


def test_mimo_export_rejects_a_template_without_the_video_key(mimo_save):
    bridge, tensors = _mimo_bridge({"model.vision_tower.patch_conv.weight"}, [HF_VIDEO])
    with pytest.raises(ValueError, match="lacks"):
        mimo_save(bridge, tensors)
    bridge.hf_pretrained.save_artifacts.assert_not_called()


def test_mimo_export_rejects_a_template_config_with_another_t(mimo_save):
    bridge, tensors = _mimo_bridge({HF_VIDEO}, [HF_VIDEO], hf_temporal=1)
    with pytest.raises(ValueError, match="declares 1"):
        mimo_save(bridge, tensors)
    bridge.hf_pretrained.save_artifacts.assert_not_called()


def test_mimo_export_raises_when_the_video_key_is_not_exported(mimo_save):
    bridge, tensors = _mimo_bridge({HF_VIDEO, "lm_head.weight"}, ["lm_head.weight"])
    with pytest.raises(RuntimeError, match="did not export"):
        mimo_save(bridge, tensors)


def test_mimo_export_with_the_video_key_passes(mimo_save):
    bridge, tensors = _mimo_bridge({HF_VIDEO, "lm_head.weight"}, ["lm_head.weight", HF_VIDEO])
    mimo_save(bridge, tensors)
    state_source = bridge.hf_pretrained.state.source
    assert state_source.save_generator.call_args.kwargs["strict"] is False
    bridge.hf_pretrained.save_artifacts.assert_called_once()


def test_t1_mimo_export_requires_nothing(mimo_save):
    bridge, tensors = _mimo_bridge({"lm_head.weight"}, ["lm_head.weight"], temporal=1, hf_temporal=None)
    mimo_save(bridge, tensors)
    bridge.hf_pretrained.state.source.get_all_keys.assert_not_called()


# ------------------------------------------------------------------------- (c') non-MIMO export (run_conversion.py)


class _Config(SimpleNamespace):
    """HF config double with the ``to_dict`` that ``required_hf_export_keys`` reads."""

    def to_dict(self):
        return {key: vars(value) if isinstance(value, SimpleNamespace) else value for key, value in vars(self).items()}


def _non_mimo_bridge(template_keys, temporal=2, hf_temporal=2, **config_fields):
    """AutoBridge over a template wrapper, as ``gpu_backend.export_checkpoint`` builds it (config replaced)."""
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge
    from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

    hf_pretrained = Mock(spec=PreTrainedCausalLM)
    hf_pretrained.config = _Config(**{**vars(_hf_config(video_temporal_patch_size=hf_temporal)), **config_fields})
    hf_pretrained.model_name_or_path = "/templates/t2"
    hf_pretrained.save_artifacts.side_effect = lambda path, **_: Path(path).mkdir(parents=True, exist_ok=True)
    state_source = Mock(spec=SafeTensorsStateSource)
    state_source.get_all_keys.return_value = sorted(template_keys)
    hf_pretrained.state = Mock()
    hf_pretrained.state.source = state_source
    bridge = AutoBridge(hf_pretrained)
    model = [SimpleNamespace(config=SimpleNamespace(video_temporal_patch_size=temporal))]
    return bridge, model


@pytest.fixture
def non_mimo_save(monkeypatch, tmp_path):
    """Run ``AutoBridge.save_hf_pretrained`` (strict=False) with a weight writer that writes ``written`` only."""
    from safetensors.torch import save_file

    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(AutoBridge, "_model_bridge", property(lambda self: Nemotron4Bridge()))
    calls = []

    def save(bridge, model, written):
        def fake_save_hf_weights(self, model, path, *args, **kwargs):
            # Stands in for a non-strict save_generator: a yielded key the index lacks is skipped, not written.
            calls.append(kwargs)
            save_file({name: torch.zeros(1) for name in written}, str(Path(path) / "model.safetensors"))

        monkeypatch.setattr(AutoBridge, "save_hf_weights", fake_save_hf_weights)
        bridge.save_hf_pretrained(model, tmp_path / "hf", strict=False)
        return calls

    return save


def test_non_mimo_export_rejects_a_template_without_the_video_key(non_mimo_save):
    bridge, model = _non_mimo_bridge({"lm_head.weight"})
    with pytest.raises(ValueError, match="lacks"):
        non_mimo_save(bridge, model, ["lm_head.weight"])
    bridge.hf_pretrained.save_artifacts.assert_not_called()


def test_non_mimo_export_raises_when_the_video_key_is_not_written(non_mimo_save):
    bridge, model = _non_mimo_bridge({HF_VIDEO, "lm_head.weight"})
    with pytest.raises(RuntimeError, match="did not write"):
        non_mimo_save(bridge, model, ["lm_head.weight"])


def test_non_mimo_export_with_the_video_key_passes(non_mimo_save):
    bridge, model = _non_mimo_bridge({HF_VIDEO, "lm_head.weight"})
    calls = non_mimo_save(bridge, model, ["lm_head.weight", HF_VIDEO])
    assert len(calls) == 1
    bridge.hf_pretrained.save_artifacts.assert_called_once()


def test_non_mimo_export_rejects_a_config_with_another_t(non_mimo_save):
    # A T = 1 model with a T = 2 template: the reference merge fills T = 2 into the config that would be written.
    bridge, model = _non_mimo_bridge({HF_VIDEO, "lm_head.weight"}, temporal=1, hf_temporal=2)
    with pytest.raises(ValueError, match="declares 2"):
        non_mimo_save(bridge, model, ["lm_head.weight"])
    bridge.hf_pretrained.save_artifacts.assert_not_called()


@pytest.mark.parametrize("temporal", [1, 2])
def test_non_mimo_export_rejects_a_lazy_init_flag_filled_from_the_template(non_mimo_save, temporal):
    # megatron_to_hf_config never emits the flag, but conform_config_to_reference fills any key the template has.
    synthesized = conform_config_to_reference({"video_temporal_patch_size": 2}, {"temporal_ckpt_compat": True})
    assert synthesized["temporal_ckpt_compat"] is True
    bridge, model = _non_mimo_bridge(
        {HF_VIDEO}, temporal=temporal, hf_temporal=temporal if temporal > 1 else None, temporal_ckpt_compat=True
    )
    with pytest.raises(AssertionError, match="temporal_ckpt_compat"):
        non_mimo_save(bridge, model, [HF_VIDEO])
    bridge.hf_pretrained.save_artifacts.assert_not_called()


def test_t1_non_mimo_export_requires_nothing(non_mimo_save):
    bridge, model = _non_mimo_bridge({"lm_head.weight"}, temporal=1, hf_temporal=None)
    non_mimo_save(bridge, model, ["lm_head.weight"])
    bridge.hf_pretrained.state.source.get_all_keys.assert_not_called()
    bridge.hf_pretrained.save_artifacts.assert_called_once()
