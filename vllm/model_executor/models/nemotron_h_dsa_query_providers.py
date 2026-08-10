# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Selection-query providers for Nemotron-H chunked DSA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class SelectionQueryState:
    """Opaque compact query rows and their ragged expansion metadata."""

    reduced_q: torch.Tensor
    reduced_current_chunks: torch.Tensor
    run_starts: torch.Tensor
    run_counts: torch.Tensor
    query_row_to_reduced_row: torch.Tensor


class IdentityQProvider(nn.Module):
    """Describe the existing one-routing-decision-per-query-row layout."""

    def forward(
        self,
        *,
        projected_q: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        chunk_size: int,
    ) -> SelectionQueryState:
        del query_positions, chunk_size
        rows = int(projected_q.shape[0])
        row_ids = torch.arange(
            rows,
            device=current_chunks.device,
            dtype=torch.long,
        )
        return SelectionQueryState(
            reduced_q=projected_q,
            reduced_current_chunks=current_chunks,
            run_starts=row_ids,
            run_counts=torch.ones_like(row_ids),
            query_row_to_reduced_row=row_ids,
        )


class MeanQShareProvider(nn.Module):
    """Compact slice-relative query runs into floating-point means."""

    def __init__(self, *, group_size: int) -> None:
        super().__init__()
        if group_size <= 0:
            raise ValueError(f"group_size must be positive: {group_size}")
        self.group_size = group_size

    def forward(
        self,
        *,
        projected_q: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        chunk_size: int,
    ) -> SelectionQueryState:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive: {chunk_size}")
        rows = int(projected_q.shape[0])
        if (
            current_chunks.dim() != 1
            or query_positions.dim() != 1
            or int(current_chunks.shape[0]) != rows
            or int(query_positions.shape[0]) != rows
        ):
            raise ValueError(
                "projected Q, current chunks, and positions must have the "
                "same leading row count"
            )
        if rows == 0:
            empty = torch.empty(
                0,
                device=current_chunks.device,
                dtype=torch.long,
            )
            return SelectionQueryState(
                reduced_q=projected_q.float(),
                reduced_current_chunks=current_chunks,
                run_starts=empty,
                run_counts=empty,
                query_row_to_reduced_row=empty,
            )

        del query_positions
        group_size = self.group_size
        run_starts = torch.arange(
            0,
            rows,
            group_size,
            device=current_chunks.device,
            dtype=torch.long,
        )
        run_ends = (run_starts + group_size).clamp(max=rows)
        run_counts = run_ends - run_starts

        prefix = torch.cat(
            (
                torch.zeros_like(projected_q[:1], dtype=torch.float32),
                projected_q.float().cumsum(dim=0),
            ),
            dim=0,
        )
        sums = prefix.index_select(0, run_ends) - prefix.index_select(
            0,
            run_starts,
        )
        reduced_q = sums / run_counts.to(prefix.dtype).view(
            -1, *([1] * (projected_q.dim() - 1))
        )
        query_row_to_reduced_row = torch.div(
            torch.arange(rows, device=run_starts.device, dtype=torch.long),
            group_size,
            rounding_mode="floor",
        )
        return SelectionQueryState(
            reduced_q=reduced_q,
            reduced_current_chunks=current_chunks.index_select(0, run_starts),
            run_starts=run_starts,
            run_counts=run_counts,
            query_row_to_reduced_row=query_row_to_reduced_row,
        )

    def prepare_batch(
        self,
        *,
        projected_q: torch.Tensor,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        chunk_size: int,
    ) -> dict[int, SelectionQueryState]:
        states: dict[int, SelectionQueryState] = {}
        for seq_idx, q_start, q_end, _ in active_seq_infos:
            query_positions = positions[q_start:q_end].to(
                device=projected_q.device,
                dtype=torch.long,
            )
            current_chunks = torch.div(
                query_positions,
                chunk_size,
                rounding_mode="floor",
            )
            states[seq_idx] = self(
                projected_q=projected_q[q_start:q_end],
                current_chunks=current_chunks,
                query_positions=query_positions,
                chunk_size=chunk_size,
            )
        return states
