# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch DSA planning with Triton Q-share query sampling."""

from __future__ import annotations

import typing

import torch

from vllm.model_executor.models.nemotron_h_dsa_query_providers import (
    SelectionQueryState,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_qshare import (
    EfficientMeanQShareProvider,
    EfficientQShareState,
)
from vllm.model_executor.models.nemotron_h_qshare_dsa_components_pytorch import (
    TorchQShareMeanChunkedDSAProviderBundle,
)


class TorchQShareEfficientQueryChunkedDSAProviderBundle(
    TorchQShareMeanChunkedDSAProviderBundle
):
    """Replace only PyTorch Q sampling with the efficient batch sampler."""

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.efficient_query_provider = EfficientMeanQShareProvider(
            group_size=self.qshare_group_size
        )

    def prepare_selection_query_batch(
        self,
        *,
        score_query_states: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor | None,
        active_seq_count: int,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
    ) -> EfficientQShareState:
        del active_seq_infos
        gpu_starts = typing.cast(torch.Tensor, query_start_loc)[: active_seq_count + 1]
        cpu_starts = typing.cast(torch.Tensor, query_start_loc_cpu)[
            : active_seq_count + 1
        ]
        cpu_lengths = cpu_starts[1:] - cpu_starts[:-1]
        total_sampled_rows = int(
            torch.div(
                cpu_lengths + self.qshare_group_size - 1,
                self.qshare_group_size,
                rounding_mode="floor",
            ).sum()
        )
        return self.efficient_query_provider(
            projected_q=score_query_states,
            query_start_loc=gpu_starts,
            query_start_loc_cpu=cpu_starts,
            total_sampled_rows=total_sampled_rows,
        )

    def build_selection_query_state_from_batch(
        self,
        *,
        selection_query_batch: typing.Any | None,
        seq_idx: int,
        q_start: int,
        q_end: int,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> SelectionQueryState:
        del score_query_states, query_positions
        batch = typing.cast(EfficientQShareState, selection_query_batch)
        sampled_start = int(batch.sampled_query_start_loc_cpu[seq_idx])
        sampled_end = int(batch.sampled_query_start_loc_cpu[seq_idx + 1])
        original_start = int(batch.original_query_start_loc_cpu[seq_idx])
        global_run_starts = batch.sampled_to_original_start[sampled_start:sampled_end]
        local_run_starts = global_run_starts - original_start
        return SelectionQueryState(
            reduced_q=batch.sampled_q[sampled_start:sampled_end, 0],
            reduced_current_chunks=current_chunks.index_select(0, local_run_starts),
            run_starts=local_run_starts,
            run_counts=batch.sampled_run_lengths[sampled_start:sampled_end],
            query_row_to_reduced_row=(
                batch.original_to_sampled[q_start:q_end] - sampled_start
            ),
        )
