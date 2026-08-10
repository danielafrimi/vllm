# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch mean Q-share components for Nemotron-H chunked DSA."""

from __future__ import annotations

import os
import typing

import torch

from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    TorchChunkedDSAProviderBundle,
    _TorchChunkBlockSelection,
)
from vllm.model_executor.models.nemotron_h_dsa_query_providers import (
    MeanQShareProvider,
    SelectionQueryState,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


_DSA_PATH_DEBUG_COUNTS: dict[str, int] = {}


def _print_dsa_path_marker(marker: str, **fields: typing.Any) -> None:
    limit = _env_int("VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT", 0)
    if limit <= 0 or os.environ.get("RANK", "0") != "0":
        return
    count = _DSA_PATH_DEBUG_COUNTS.get(marker, 0)
    if count >= limit:
        return
    _DSA_PATH_DEBUG_COUNTS[marker] = count + 1
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"DSA_PATH_MARKER marker={marker} {details}".rstrip(), flush=True)


class TorchQShareMeanChunkedDSAProviderBundle(TorchChunkedDSAProviderBundle):
    """Use compact mean-Q rows while retaining existing score/top-k providers."""

    def __init__(
        self,
        *,
        qshare_group_size: int | None = None,
        chunk_size: int,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        if qshare_group_size is None:
            qshare_group_size = _env_int(
                "VLLM_NEMOTRON_H_DSA_QSHARE_GROUP_SIZE",
                _env_int(
                    "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE",
                    chunk_size,
                ),
            )
        if qshare_group_size <= 0:
            raise ValueError(f"qshare_group_size must be positive: {qshare_group_size}")
        share_mode = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE",
                "mean",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        share_mode = {
            "avg": "mean",
            "average": "mean",
            "noncausal_mean": "mean",
        }.get(share_mode, share_mode)
        if share_mode != "mean":
            raise ValueError(
                "TorchQShareMeanChunkedDSAProviderBundle requires mean "
                f"Q-share mode, got {share_mode!r}"
            )

        self.qshare_group_size = int(qshare_group_size)
        self.qshare_enabled = self.qshare_group_size > 1
        self.query_provider = MeanQShareProvider(group_size=self.qshare_group_size)
        _print_dsa_path_marker(
            "config",
            provider=type(self).__name__,
            qshare_group_size=self.qshare_group_size,
            qshare_mode="mean",
        )

    def build_selection_query_state(
        self,
        *,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> SelectionQueryState | None:
        if not self.qshare_enabled:
            return None
        return self.query_provider(
            projected_q=score_query_states,
            current_chunks=current_chunks,
            query_positions=query_positions,
            chunk_size=self.chunk_size,
        )

    def should_prepare_batched_representatives(self) -> bool:
        return self.qshare_enabled

    def expand_selection_state(
        self,
        *,
        selection_state: typing.Any | None,
        selection_query_state: SelectionQueryState | None,
    ) -> typing.Any | None:
        if selection_query_state is None or selection_state is None:
            return selection_state
        selected_blocks, selected_valid = self.get_selected_blocks(
            selection_state,
            device=selection_query_state.query_row_to_reduced_row.device,
        )
        row_mapping = selection_query_state.query_row_to_reduced_row
        return _TorchChunkBlockSelection(
            selected_block_indices=selected_blocks.index_select(0, row_mapping),
            selected_block_valid=selected_valid.index_select(0, row_mapping),
        )

    def selection_query_chunk_size(self, q_len: int) -> int:
        if not self.qshare_enabled:
            return super().selection_query_chunk_size(q_len)
        return q_len

    def build_page_table_plan(
        self,
        *,
        block_table: torch.Tensor,
        chunk_size: int,
        key_len: int,
        **kwargs: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        mode = kwargs.get("mode", "prefill")
        if mode == "decode" or kwargs.get("q_len") == 1:
            marker = "sparse_decode"
        elif kwargs.get("dense", False):
            marker = "dense_prefill_page_table_bucket"
        else:
            marker = "sparse_prefill_page_table_bucket"
        _print_dsa_path_marker(marker, key_len=key_len)
        return super().build_page_table_plan(
            block_table=block_table,
            chunk_size=chunk_size,
            key_len=key_len,
            **kwargs,
        )
