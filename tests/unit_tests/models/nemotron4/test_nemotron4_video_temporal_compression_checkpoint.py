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

"""Nemotron4 video temporal compression on Megatron checkpoint loads (plan phase 2).

A T = 1 checkpoint has no ``video_patch_embed``. On a finetune load, Bridge drops the absent keys from the
dist-ckpt request and the encoder's load hook derives the weight from the loaded ``patch_embed`` weight.
"""

import importlib.util
import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.models.vision.vit_model import Mistral2DRotaryEmbedding, PatchEmbedding, ViTModel


if importlib.util.find_spec("megatron.core.models.hybrid.shortcut_block") is None:
    pytest.skip("Nemotron4 requires its MCore fork", allow_module_level=True)

from megatron.bridge.models.nemotron4 import modeling_nemotron4_vl
from megatron.bridge.models.nemotron4.modeling_nemotron4_vl import (
    Nemotron4MIMOVisionEncoder,
    video_patch_weight_from_image,
)
from megatron.bridge.training import checkpointing
from megatron.bridge.training.checkpointing import (
    _align_mimo_model_sharded_keys,
    _drop_missing_video_patch_embed_keys,
    _load_model_state_dict,
)


pytestmark = pytest.mark.unit

P = 2
HIDDEN = 8
# A MIMO vision component: runtime keys carry the DDP and Float16Module wrapper levels.
RUNTIME = "modality_submodules.images.module.module.encoders.vision."
STORED = "modality_submodules.images.module.encoders.vision."
VIDEO = "video_patch_embed.proj.weight"
IMAGE = "patch_embed.proj.weight"
T1_RUN_CONFIG = {"model": {"standard_provider": {"video_temporal_patch_size": 1}}}


def _stub_vit_init(self, transformer_config, transformer_layer_spec=None, patch_dim=P, **kwargs):
    """CPU stand-in for ``ViTModel.__init__``: real patch embedding and norms, no transformer."""
    del transformer_layer_spec, kwargs
    torch.nn.Module.__init__(self)
    hidden = transformer_config.hidden_size
    self.patch_dim = patch_dim
    self.add_class_token = False
    self.pos_emb_type = "rope2d"
    self.patch_embed = PatchEmbedding(3, hidden, patch_dim)
    self.ln_pre = torch.nn.RMSNorm(hidden, eps=1e-5)
    self.rope = Mistral2DRotaryEmbedding(hidden // 2, 16)


def _encoder(temporal=2, seed=0, **kwargs):
    torch.manual_seed(seed)
    return Nemotron4MIMOVisionEncoder(
        transformer_config=SimpleNamespace(hidden_size=HIDDEN), video_temporal_patch_size=temporal, **kwargs
    )


@pytest.fixture
def stub_vit(monkeypatch):
    monkeypatch.setattr(ViTModel, "__init__", _stub_vit_init)


def _model(**kwargs):
    """A model chunk that holds one vision encoder."""
    chunk = torch.nn.Module()
    chunk.vision = _encoder(**kwargs)
    return [chunk]


def _request(names, *, runtime=RUNTIME, stored=STORED):
    """A flat MIMO 'model' sharded request: runtime dict keys, stored lookup keys."""
    return {"model": {runtime + name: SimpleNamespace(key=stored + name) for name in names}}


def _metadata(names, stored=STORED):
    return {stored + name: object() for name in names}


def _capture_warnings(monkeypatch):
    messages = []
    monkeypatch.setattr(checkpointing.logger, "warning", lambda message, *args: messages.append(message % args))
    return messages


def _drop(request, model, metadata, *, finetune=True, run_config=T1_RUN_CONFIG):
    return _drop_missing_video_patch_embed_keys(
        request,
        model,
        checkpoint_name="/ckpt/iter_0000000",
        finetune=finetune,
        run_config=run_config,
        checkpoint_metadata=metadata,
    )


# ----------------------------------------------------------------------------------------------- dist-ckpt filter


def test_filter_drops_only_the_absent_video_keys_and_logs_each(stub_vit, monkeypatch):
    names = [IMAGE, VIDEO, "ln_pre.weight", "decoder.missing.weight"]
    request = _request(names)
    other = {"optimizer": {"x": SimpleNamespace(key=STORED + VIDEO)}}
    request.update(other)
    messages = _capture_warnings(monkeypatch)

    dropped = _drop(request, _model(temporal_ckpt_compat=True), _metadata([IMAGE, "ln_pre.weight"]))

    assert dropped == [RUNTIME + VIDEO]
    # The other absent key stays, for the load's strictness to report.
    assert set(request["model"]) == {RUNTIME + IMAGE, RUNTIME + "ln_pre.weight", RUNTIME + "decoder.missing.weight"}
    assert request["optimizer"] is other["optimizer"] and "x" in request["optimizer"]
    assert len(messages) == 1 and STORED + VIDEO in messages[0] and RUNTIME + VIDEO in messages[0]


def test_filter_keeps_a_stored_video_key(stub_vit, monkeypatch):
    request = _request([IMAGE, VIDEO])
    messages = _capture_warnings(monkeypatch)
    assert _drop(request, _model(temporal_ckpt_compat=True), _metadata([IMAGE, VIDEO])) == []
    assert set(request["model"]) == {RUNTIME + IMAGE, RUNTIME + VIDEO}
    assert messages == []


def test_filter_compares_the_stored_key_not_the_runtime_key(stub_vit):
    # The stored key exists under the checkpoint's wrapper depth; the runtime key never does.
    request = _request([IMAGE, VIDEO])
    assert _drop(request, _model(temporal_ckpt_compat=True), _metadata([IMAGE, VIDEO])) == []
    # Without the MIMO mapping the stored key carries the runtime prefix and is absent.
    request = _request([IMAGE, VIDEO], stored=RUNTIME)
    with pytest.raises(RuntimeError, match="matching patch_embed"):
        _drop(request, _model(temporal_ckpt_compat=True), _metadata([IMAGE, VIDEO]))


@pytest.mark.parametrize("finetune,temporal", [(False, 2), (True, 1)])
def test_filter_is_a_no_op_on_a_resume_or_without_temporal_compression(stub_vit, finetune, temporal):
    request = _request([IMAGE, VIDEO])
    with patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy:
        dropped = _drop_missing_video_patch_embed_keys(
            request,
            _model(temporal=temporal, temporal_ckpt_compat=True),
            checkpoint_name="/ckpt",
            finetune=finetune,
            run_config=T1_RUN_CONFIG,
        )
    assert dropped == []
    assert set(request["model"]) == {RUNTIME + IMAGE, RUNTIME + VIDEO}
    strategy.assert_not_called()  # no metadata read when the filter is off


def test_filter_skips_chunks_that_are_not_modules_and_does_not_walk_on_a_resume():
    # load_checkpoint unit tests pass Mock model chunks; the filter must not walk them.
    request = _request([IMAGE, VIDEO])
    with patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy:
        assert _drop(request, [Mock()], None) == []
        walked = Mock(spec=torch.nn.Module)
        assert _drop(request, [walked], None, finetune=False) == []
    walked.modules.assert_not_called()
    assert set(request["model"]) == {RUNTIME + IMAGE, RUNTIME + VIDEO}
    strategy.assert_not_called()


def test_filter_reads_the_metadata_when_not_given(stub_vit):
    request = _request([IMAGE, VIDEO])
    with patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy:
        strategy.return_value.load_sharded_metadata.return_value = _metadata([IMAGE])
        dropped = _drop_missing_video_patch_embed_keys(
            request,
            _model(temporal_ckpt_compat=True),
            checkpoint_name="/ckpt",
            finetune=True,
            run_config=None,
        )
    assert dropped == [RUNTIME + VIDEO]


def test_filter_raises_with_the_flag_off(stub_vit):
    request = _request([IMAGE, VIDEO])
    with pytest.raises(RuntimeError, match="temporal_ckpt_compat"):
        _drop(request, _model(temporal_ckpt_compat=False), _metadata([IMAGE]))
    assert RUNTIME + VIDEO in request["model"]


@pytest.mark.parametrize(
    "run_config",
    [
        {"model": {"standard_provider": {"video_temporal_patch_size": 2}}},
        {"model": {"spec": [{"params": {"video_temporal_patch_size": 2}}], "video_temporal_patch_size": 1}},
    ],
)
def test_filter_raises_when_the_checkpoint_declares_t_above_1(stub_vit, run_config):
    """A source that claims a trained video embedder but lost it is never re-derived."""
    request = _request([IMAGE, VIDEO])
    with pytest.raises(RuntimeError, match="lost it"):
        _drop(request, _model(temporal_ckpt_compat=True), _metadata([IMAGE]), run_config=run_config)


def test_filter_raises_without_the_stored_image_weight(stub_vit):
    request = _request([IMAGE, VIDEO])
    with pytest.raises(RuntimeError, match="matching patch_embed"):
        _drop(request, _model(temporal_ckpt_compat=True), _metadata(["ln_pre.weight"]))


def test_filter_raises_when_main_params_come_from_the_checkpoint(stub_vit):
    """load_main_params_from_ckpt builds the main params from the checkpoint dict, which has no derived weight."""
    request = _request([IMAGE, VIDEO])
    with pytest.raises(RuntimeError, match="load_main_params_from_ckpt"):
        _drop_missing_video_patch_embed_keys(
            request,
            _model(temporal_ckpt_compat=True),
            checkpoint_name="/ckpt",
            finetune=True,
            run_config=T1_RUN_CONFIG,
            checkpoint_metadata=_metadata([IMAGE]),
            load_main_params_from_ckpt=True,
        )
    assert RUNTIME + VIDEO in request["model"]
    # Nothing to drop: the flag does not matter.
    request = _request([IMAGE, VIDEO])
    assert (
        _drop_missing_video_patch_embed_keys(
            request,
            _model(temporal_ckpt_compat=True),
            checkpoint_name="/ckpt",
            finetune=True,
            run_config=T1_RUN_CONFIG,
            checkpoint_metadata=_metadata([IMAGE, VIDEO]),
            load_main_params_from_ckpt=True,
        )
        == []
    )


def test_load_path_passes_load_main_params_from_ckpt_to_the_filter():
    source = inspect.getsource(checkpointing._load_checkpoint_from_path)
    call = source[source.index("_drop_missing_video_patch_embed_keys(") :]
    call = call[: call.index(")\n")]
    assert "load_main_params_from_ckpt=bool(cfg.checkpoint.load_main_params_from_ckpt)" in call


def test_align_returns_the_metadata_it_read():
    request = {"model": {RUNTIME + IMAGE: SimpleNamespace(key=RUNTIME + IMAGE)}}
    metadata = _metadata([IMAGE])
    with (
        patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy,
        patch.object(checkpointing, "apply_prefix_mapping"),
    ):
        strategy.return_value.load_sharded_metadata.return_value = metadata
        assert _align_mimo_model_sharded_keys(request, checkpoint_name="/ckpt", module_name="images") is metadata
    no_component = {"model": {"language_model.weight": object()}}
    with patch.object(checkpointing, "TorchDistLoadShardedStrategy") as strategy:
        assert _align_mimo_model_sharded_keys(no_component, checkpoint_name="/ckpt", module_name="images") is None
        strategy.assert_not_called()


def test_load_path_derives_the_weight_before_the_main_params_are_rebuilt():
    """The filter, the model load (hook) and reload_model_params run in this order on the load path."""
    source = inspect.getsource(checkpointing._load_checkpoint_from_path)
    align = source.index("_align_mimo_model_sharded_keys(")
    drop = source.index("_drop_missing_video_patch_embed_keys(")
    load = source.index("_load_model_state_dict(")
    reload = source.index("optimizer.reload_model_params()")
    assert align < drop < load < reload


# ----------------------------------------------------------------------------------------------- load hook


def test_hook_derives_through_bridge_load_model_state_dict(stub_vit, monkeypatch):
    source = _encoder(temporal=1, seed=1)
    target = _model(seed=2, temporal_ckpt_compat=True)[0]
    messages = []
    monkeypatch.setattr(modeling_nemotron4_vl.logger, "warning", lambda m, *a: messages.append(m % a))
    state = {f"vision.{name}": value for name, value in source.state_dict().items()}
    _load_model_state_dict(target, state, strict=True)
    assert torch.equal(
        target.vision.video_patch_embed.proj.weight, video_patch_weight_from_image(source.patch_embed.proj.weight, 2)
    )
    assert len(messages) == 1


@pytest.mark.parametrize("strict", [True, False])
def test_hook_raises_through_bridge_non_strict_retry_with_the_flag_off(stub_vit, strict):
    """Bridge retries a failed strict load with strict=False; the hook raises on both tries."""
    state = {f"vision.{name}": value for name, value in _encoder(temporal=1, seed=1).state_dict().items()}
    with pytest.raises(RuntimeError, match="temporal_ckpt_compat"):
        _load_model_state_dict(_model(seed=2)[0], state, strict=strict)


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("compat", [True, False])
def test_hook_raises_without_the_image_weight(stub_vit, strict, compat):
    state = {f"vision.{name}": value for name, value in _encoder(temporal=1, seed=1).state_dict().items()}
    del state[f"vision.{IMAGE}"]
    with pytest.raises(RuntimeError, match="neither"):
        _load_model_state_dict(_model(seed=2, temporal_ckpt_compat=compat)[0], state, strict=strict)


# ----------------------------------------------------------------------------------------------- dist-ckpt round trip


def _sharded(module, prefix=""):
    from megatron.core.dist_checkpointing import ShardedTensor

    return {
        prefix + name: ShardedTensor.from_rank_offsets(prefix + name, tensor.detach(), replica_id=0)
        for name, tensor in module.state_dict(keep_vars=True).items()
    }


def _round_trip_worker(rank, init_method, ckpt_dir):
    import torch.distributed as dist
    from megatron.core import dist_checkpointing

    ViTModel.__init__ = _stub_vit_init
    if not torch.cuda.is_available():
        # The MCore torch_dist strategy synchronizes CUDA and reduces its failure flag on the current CUDA
        # device. This spawned worker has no GPU: no-op the sync and put the flag on the CPU (gloo).
        torch.cuda.synchronize = lambda *args, **kwargs: None
        torch.cuda.current_device = lambda: "cpu"
    dist.init_process_group("gloo", rank=rank, world_size=1, init_method=init_method)
    try:
        source = _model(temporal=1, seed=1)[0].to(torch.bfloat16)
        image = source.vision.patch_embed.proj.weight.detach().clone()
        dist_checkpointing.save({"model": _sharded(source)}, ckpt_dir)

        def request(target):
            return {"model": _sharded(target)}

        # Without the filter the default strictness fails on the absent key. The torch DCP error,
        # CheckpointException, derives from BaseException.
        target = _model(seed=2, temporal_ckpt_compat=True)[0].to(torch.bfloat16)
        with pytest.raises(BaseException, match="video_patch_embed"):
            dist_checkpointing.load(request(target), ckpt_dir, strict="assume_ok_unexpected")

        target = _model(seed=2, temporal_ckpt_compat=True)[0].to(torch.bfloat16)
        before = target.vision.video_patch_embed.proj.weight.detach().clone()
        sharded = request(target)
        dropped = _drop_missing_video_patch_embed_keys(
            sharded,
            [target],
            checkpoint_name=ckpt_dir,
            finetune=True,
            run_config={"model": {"video_temporal_patch_size": 1}},
        )
        assert dropped == [f"vision.{VIDEO}"]
        loaded = dist_checkpointing.load(sharded, ckpt_dir, strict="assume_ok_unexpected")
        _load_model_state_dict(target, loaded["model"], strict=True)

        video = target.vision.video_patch_embed.proj.weight
        assert torch.equal(target.vision.patch_embed.proj.weight, image)
        expected = torch.cat([image / 2] * 2, dim=1)
        assert video.dtype == torch.bfloat16 and torch.equal(video, expected)
        assert not torch.equal(video, before)  # not the construction-time value from the random image weight
    finally:
        dist.destroy_process_group()


def test_t1_dist_checkpoint_loads_into_t2_with_the_derived_weight(tmp_path):
    if not torch.distributed.is_gloo_available():
        pytest.skip("requires the Gloo backend")
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    torch.multiprocessing.spawn(
        _round_trip_worker,
        args=((tmp_path / "init").as_uri(), str(ckpt_dir)),
        nprocs=1,
        join=True,
    )
