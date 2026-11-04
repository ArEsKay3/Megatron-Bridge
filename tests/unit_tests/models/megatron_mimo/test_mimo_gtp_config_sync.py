# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from megatron.core.model_parallel_config import ModelParallelConfig

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)
from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import (
    MegatronMIMOProvider,
    _align_component_config_parallelism,
)


@pytest.mark.unit
@pytest.mark.parametrize("gtp,egtp", [(1, 1), (2, 1), (1, 2)])
def test_checkpoint_shard_counts_do_not_override_mimo_gtp(gtp, egtp):
    # Checkpoints contain the public shard counts as well as the legacy GTP fields.
    standard = ModelParallelConfig(
        tensor_parallel_num_weight_shards=1,
        expert_tensor_parallel_num_weight_shards=1,
    )
    provider = object.__new__(MegatronMIMOProvider)
    provider.standard_provider = standard
    provider.megatron_mimo_parallelism_config = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                gtp_weight_remat_size=gtp,
                expert_gtp_weight_remat_size=egtp,
                data_parallel_size=2 // gtp,
            )
        },
    )
    provider._sync_standard_provider_language_parallelism()
    standard.__post_init__()
    assert standard.gtp_weight_remat_size == gtp
    assert standard.expert_gtp_weight_remat_size == egtp
    assert standard.tensor_parallel_num_weight_shards == gtp
    assert standard.expert_tensor_parallel_num_weight_shards == egtp


@pytest.mark.unit
def test_encoder_does_not_inherit_language_weight_sharding():
    config = ModelParallelConfig(
        tensor_parallel_num_weight_shards=2,
        expert_tensor_parallel_num_weight_shards=2,
    )
    tp = MagicMock()
    tp.size.return_value = 1
    _align_component_config_parallelism(config, SimpleNamespace(tp=tp))
    config.__post_init__()
    assert config.gtp_weight_remat_size == 1
    assert config.expert_gtp_weight_remat_size == 1
    assert config.tensor_parallel_num_weight_shards == 1
    assert config.expert_tensor_parallel_num_weight_shards == 1
