# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""GTP must preserve sample lanes while changing only weight ownership."""

from unittest.mock import patch

import pytest
from megatron.core.hyper_comm_grid import HyperCommGrid

from megatron.bridge.models.megatron_mimo.megatron_mimo_builder import build_hypercomm_grids
from megatron.bridge.models.megatron_mimo.megatron_mimo_config import (
    MegatronMIMOParallelismConfig,
    ModuleParallelismConfig,
)


@pytest.mark.unit
@pytest.mark.parametrize("cp,encoder_ranks", [(8, 4), (32, 12)])
@pytest.mark.parametrize("gtp,egtp", [(1, 1), (2, 1), (1, 2)])
def test_weight_shards_preserve_data_lanes(monkeypatch, cp, encoder_ranks, gtp, egtp):
    world = encoder_ranks + 2 * cp
    monkeypatch.setenv("WORLD_SIZE", str(world))
    config = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "images": ModuleParallelismConfig(data_parallel_size=encoder_ranks),
            "language": ModuleParallelismConfig(
                context_parallel_size=cp,
                expert_model_parallel_size=cp,
                data_parallel_size=2 // gtp,
                gtp_weight_remat_size=gtp,
                expert_gtp_weight_remat_size=egtp,
                rank_offset=encoder_ranks,
            ),
        }
    )
    config.finalize(world)
    # Use the actual grid/view enumeration; only skip NCCL group creation.
    with patch.object(HyperCommGrid, "create_pg"):
        grids = build_hypercomm_grids(config)
    language = grids["language"]
    lanes = [list(range(encoder_ranks, encoder_ranks + cp)), list(range(encoder_ranks + cp, world))]
    assert language.get_rank_enum(["tp", "cp", "pp"]) == lanes
    assert language.get_rank_enum(["expt_tp", "ep"], view="expert") == lanes
    expected_pairs = [[encoder_ranks + i, encoder_ranks + cp + i] for i in range(cp)]
    assert language.get_rank_enum(["dp", "gtp_remat"]) == expected_pairs
    assert language.get_rank_enum(["expt_dp", "expt_gtp_remat"], view="expert") == expected_pairs
    dense_weight_groups = language.get_rank_enum("gtp_remat")
    expert_weight_groups = language.get_rank_enum("expt_gtp_remat", view="expert")
    assert all(len(group) == gtp for group in dense_weight_groups)
    assert all(len(group) == egtp for group in expert_weight_groups)
    assert sorted(r for group in dense_weight_groups for r in group) == list(range(encoder_ranks, world))
    assert sorted(r for group in expert_weight_groups for r in group) == list(range(encoder_ranks, world))
    assert config.module_parallelisms["language"].batch_parallel_size == 2


@pytest.mark.unit
@pytest.mark.parametrize("gtp,egtp", [(2, 1), (1, 2), (2, 2)])
def test_encoder_weight_sharding_rejected(gtp, egtp):
    config = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "images": ModuleParallelismConfig(
                data_parallel_size=2 // gtp,
                gtp_weight_remat_size=gtp,
                expert_gtp_weight_remat_size=egtp,
            ),
            "language": ModuleParallelismConfig(data_parallel_size=2, rank_offset=2),
        }
    )
    with pytest.raises(ValueError, match="Encoder GTP/EGTP is not supported"):
        config.finalize(4)


@pytest.mark.unit
@pytest.mark.parametrize("gtp,egtp", [(1, 1), (2, 1), (1, 2)])
def test_nccl_option_names_follow_mcore_groups(monkeypatch, gtp, egtp):
    monkeypatch.setenv("WORLD_SIZE", "4")
    config = MegatronMIMOParallelismConfig(
        module_parallelisms={
            "language": ModuleParallelismConfig(
                data_parallel_size=4 // gtp,
                gtp_weight_remat_size=gtp,
                expert_gtp_weight_remat_size=egtp,
            ),
        }
    )
    config.finalize(4)
    options = {"mp": {"cga_cluster_size": 2}}
    with (
        patch("megatron.core.parallel_state.get_nccl_options", side_effect=lambda name, cfg: name) as lookup,
        patch.object(HyperCommGrid, "create_pg") as create,
    ):
        build_hypercomm_grids(config, nccl_comm_cfgs=options)
    assert all(call.args[1] is options for call in lookup.call_args_list)
    groups = {call.kwargs["pg_options"]: (call.args[0], call.kwargs.get("view")) for call in create.call_args_list}
    assert groups["mp"] == (["tp", "gtp_remat", "pp"], None)
    assert groups["tp_dp_cp"] == (["tp", "gtp_remat", "dp", "cp"], None)
    assert groups["intra_dist_opt_instance"] == (["tp", "cp", "gtp_remat", "dp", "pp"], None)
    assert groups["gtp_remat"] == (["gtp_remat"], None)
    assert groups["gtp_remat_dp"] == (["gtp_remat", "dp"], None)
    assert groups["gtp_remat_dp_cp"] == (["gtp_remat", "dp", "cp"], None)
    assert groups["expt_gtp_remat"] == (["expt_gtp_remat"], "expert")
    assert groups["ep_gtp_remat_dp"] == (["expt_gtp_remat", "expt_dp"], "expert")
    assert groups["tp_ep_gtp_remat_pp"] == (["expt_tp", "ep", "expt_gtp_remat", "pp"], "expert")
    assert groups[None] == (["tp", "pp"], None)
