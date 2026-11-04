# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch.distributed as dist

from megatron.bridge.models.megatron_mimo.megatron_mimo_config import MegatronMIMOParallelismConfig


if TYPE_CHECKING:
    from megatron.core.hyper_comm_grid import HyperCommGrid


EXPERT_VIEW_NAME = "expert"


def build_hypercomm_grids(
    megatron_mimo_parallelism_config: MegatronMIMOParallelismConfig,
    nccl_comm_cfgs: Optional[Dict[str, dict]] = None,
) -> Dict[str, "HyperCommGrid"]:
    """Create HyperCommGrid objects per module from MegatronMIMO parallelism config.

    Creates grids on ALL ranks (required for consistent collective calls),
    but only ranks in each grid's range will participate in its operations.

    Each grid is built with a dense ``base`` view plus a registered ``expert`` view over the same
    rank span. Dense process groups (tp/cp/dp/pp) come from the base view; expert-parallel groups
    (expt_tp/ep/expt_dp) come from the expert view, matching the contract mcore's
    ``get_mimo_optimizer`` expects.

    Args:
        megatron_mimo_parallelism_config: MegatronMIMOParallelismConfig specifying parallelism per module.
        nccl_comm_cfgs: Optional per-group NCCL options. Keys are the mcore group names
            (e.g. ``ep_dp``) and apply to that group of every module.

    Returns:
        Dict mapping module names to their HyperCommGrids.
    """
    from megatron.core.hyper_comm_grid import HyperCommGrid
    from megatron.core.parallel_state import get_nccl_options

    nccl_comm_cfgs = nccl_comm_cfgs or {}

    grids: Dict[str, HyperCommGrid] = {}
    for module_name, parallelism in megatron_mimo_parallelism_config.module_parallelisms.items():
        shape = [
            parallelism.tensor_model_parallel_size,
            parallelism.context_parallel_size,
            parallelism.gtp_weight_remat_size,
            parallelism.data_parallel_size,
            parallelism.pipeline_model_parallel_size,
        ]
        grid = HyperCommGrid(
            shape=shape,
            dim_names=["tp", "cp", "gtp_remat", "dp", "pp"],
            rank_offset=parallelism.rank_offset,
            backend="nccl",
        )
        grid.register_view(
            EXPERT_VIEW_NAME,
            shape=[
                parallelism.expert_tensor_parallel_size,
                parallelism.expert_model_parallel_size,
                parallelism.expert_gtp_weight_remat_size,
                parallelism.expert_data_parallel_size,
                parallelism.pipeline_model_parallel_size,
            ],
            dim_names=["expt_tp", "ep", "expt_gtp_remat", "expt_dp", "pp"],
            shared_dims=["pp"],
        )

        # Use MCore NCCL option names, as in the communicator configuration path.
        # Optimizer/model groups include GTP; the sample-lane groups do not.
        for dims, name in (
            (["tp"], "tp"),
            (["cp"], "cp"),
            (["pp"], "pp"),
            (["dp"], "dp"),
            (["gtp_remat"], "gtp_remat"),
            (["gtp_remat", "dp"], "gtp_remat_dp"),
            (["gtp_remat", "dp", "cp"], "gtp_remat_dp_cp"),
            (["dp", "cp"], "dp_cp"),
            (["tp", "cp"], "tp_cp"),
            (["tp", "pp"], None),
            (["tp", "gtp_remat", "pp"], "mp"),
            (["tp", "gtp_remat", "dp", "cp"], "tp_dp_cp"),
            (["tp", "cp", "gtp_remat", "dp", "pp"], "intra_dist_opt_instance"),
        ):
            _ = grid.create_pg(dims, pg_options=get_nccl_options(name, nccl_comm_cfgs) if name is not None else None)

        for dims, name in (
            (["ep"], "ep"),
            (["expt_tp"], "ep_tp"),
            (["expt_dp"], "ep_dp"),
            (["expt_gtp_remat"], "expt_gtp_remat"),
            (["expt_gtp_remat", "expt_dp"], "ep_gtp_remat_dp"),
            (["expt_tp", "ep"], "tp_ep_mp"),
            (["expt_tp", "ep", "pp"], "tp_ep_pp"),
            (["expt_tp", "ep", "expt_gtp_remat", "pp"], "tp_ep_gtp_remat_pp"),
        ):
            _ = grid.create_pg(dims, view=EXPERT_VIEW_NAME, pg_options=get_nccl_options(name, nccl_comm_cfgs))

        grids[module_name] = grid

    return grids


def populate_embedding_and_position_groups(
    pp_rank_groups: list[list[int]] | None,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    """Create embedding-related process groups from globally enumerated PP ranks.

    Following MCore semantics:
    - pos_embd_pg: Only rank 0 of PP (first stage) - for position embeddings
    - embd_pg: Ranks 0 and -1 of PP (first and last stages) - for tied word embeddings

    IMPORTANT: This calls dist.new_group which is a collective operation.
    Must be called on all ranks that could participate.

    Args:
        pp_rank_groups: Every pipeline-parallel rank group in global creation order.

    Returns:
        Tuple of process groups for the current rank. Returns (None, None) when
        no pipeline-parallel rank groups are provided.
    """
    if not pp_rank_groups:
        return None, None

    current_rank = dist.get_rank()
    local_pos_embd_pg = None
    local_embd_pg = None
    for pp_rank_group in pp_rank_groups:
        pp_ranks = sorted(pp_rank_group)

        # Every global rank must create every derived group in the same order.
        pos_embd_ranks = [pp_ranks[0]]
        pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)

        embd_ranks = [pp_ranks[0]]
        if len(pp_ranks) > 1 and pp_ranks[-1] != pp_ranks[0]:
            embd_ranks.append(pp_ranks[-1])
        embd_pg = dist.new_group(ranks=embd_ranks)

        if current_rank in pos_embd_ranks:
            local_pos_embd_pg = pos_embd_pg
        if current_rank in embd_ranks:
            local_embd_pg = embd_pg

    return local_pos_embd_pg, local_embd_pg


def is_pp_first_stage(pp_group: Optional[dist.ProcessGroup]) -> bool:
    """Check if current rank is first stage in pipeline."""
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[0]


def is_pp_last_stage(pp_group: Optional[dist.ProcessGroup]) -> bool:
    """Check if current rank is last stage in pipeline."""
    if pp_group is None:
        return True
    pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
    return dist.get_rank() == pp_ranks[-1]
