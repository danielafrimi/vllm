# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Threshold-based block selection for Nemotron-H chunked DSA.

This module is intentionally separate from the main DSA component files so the
selector can be dropped into an existing provider bundle for experiments.  The
selector consumes the same score state as the top-k selectors and returns the
same selection state type expected by the current block-table providers.
"""

from __future__ import annotations

import os
import typing

import torch
from torch import nn

from vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient import (
    EfficientChunkedDSAProviderBundle,
    _EfficientChunkBlockSelection,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    TorchChunkedDSAProviderBundle,
    _TorchChunkBlockSelection,
)

try:
    from vllm.triton_utils import tl, triton
except ImportError:
    tl = None
    triton = None


_THRESHOLD_BASE_ENV = "VLLM_NEMOTRON_H_DSA_THRESHOLD_BASE"
_THRESHOLD_LENGTH_SCALE_ENV = "VLLM_NEMOTRON_H_DSA_THRESHOLD_LENGTH_SCALE"
_THRESHOLD_LOG_LENGTH_SCALE_ENV = (
    "VLLM_NEMOTRON_H_DSA_THRESHOLD_LOG_LENGTH_SCALE"
)
_THRESHOLD_LENGTH_SOURCE_ENV = "VLLM_NEMOTRON_H_DSA_THRESHOLD_LENGTH_SOURCE"
_THRESHOLD_MAX_SELECTED_BLOCKS_ENV = (
    "VLLM_NEMOTRON_H_DSA_THRESHOLD_MAX_SELECTED_BLOCKS"
)
_THRESHOLD_USE_TRITON_ENV = "VLLM_NEMOTRON_H_DSA_THRESHOLD_USE_TRITON"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value == "1"


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


if triton is not None and tl is not None:

    @triton.jit
    def _compact_bool_mask_kernel(
        selected_mask,
        selected_indices,
        num_chunks: tl.constexpr,
        capacity: tl.constexpr,
        mask_stride_row,
        mask_stride_col,
        out_stride_row,
        out_stride_col,
        BLOCK_CHUNKS: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_CHUNKS)
        in_bounds = offsets < num_chunks
        mask_values = tl.load(
            selected_mask + row * mask_stride_row + offsets * mask_stride_col,
            mask=in_bounds,
            other=0,
        ).to(tl.int32)
        ranks = tl.cumsum(mask_values, 0) - 1
        store_mask = in_bounds & (mask_values != 0) & (ranks < capacity)
        tl.store(
            selected_indices + row * out_stride_row + ranks * out_stride_col,
            offsets,
            mask=store_mask,
        )


class ThresholdChunkedDSABlockSelectionProvider(nn.Module):
    """Select DSA chunks by threshold, then compact selected chunk indices.

    The output is left-packed to match the existing top-k selection contract.
    When ``block_top_k`` or ``chunk_top_k`` is supplied by the caller, it is
    treated as the maximum compacted width.  This keeps the selector compatible
    with fallback paths that assume the configured selection width.

    ``length_source`` controls the input to the threshold schedule:
      * ``sequence_chunks``: one scalar per selection call, ``max_prior_chunks+1``.
      * ``row_chunks``: one value per query row, ``current_chunks+1``.
      * ``provided``: use ``sequence_lengths``, ``key_lens``, ``seq_lens``, or
        ``effective_lengths`` from the call kwargs.
    """

    def __init__(
        self,
        *,
        base_threshold: float = 0.0,
        length_scale: float = 0.0,
        log_length_scale: float = 0.0,
        length_source: str = "sequence_chunks",
        length_multiplier: float = 1.0,
        max_selected_blocks: int | None = None,
        prefer_triton: bool = True,
    ) -> None:
        super().__init__()
        if max_selected_blocks is not None and max_selected_blocks <= 0:
            raise ValueError(
                f"max_selected_blocks must be positive: {max_selected_blocks}")
        if length_multiplier <= 0:
            raise ValueError(
                f"length_multiplier must be positive: {length_multiplier}")
        if length_source not in {"sequence_chunks", "row_chunks", "provided"}:
            raise ValueError(f"unsupported threshold length_source: {length_source}")
        self.base_threshold = float(base_threshold)
        self.length_scale = float(length_scale)
        self.log_length_scale = float(log_length_scale)
        self.length_source = length_source
        self.length_multiplier = float(length_multiplier)
        self.max_selected_blocks = max_selected_blocks
        self.prefer_triton = prefer_triton

    @classmethod
    def from_env(
        cls,
        *,
        length_multiplier: float = 1.0,
    ) -> "ThresholdChunkedDSABlockSelectionProvider":
        return cls(
            base_threshold=_env_float(_THRESHOLD_BASE_ENV, 0.0),
            length_scale=_env_float(_THRESHOLD_LENGTH_SCALE_ENV, 0.0),
            log_length_scale=_env_float(_THRESHOLD_LOG_LENGTH_SCALE_ENV, 0.0),
            length_source=os.environ.get(
                _THRESHOLD_LENGTH_SOURCE_ENV,
                "sequence_chunks",
            ),
            length_multiplier=length_multiplier,
            max_selected_blocks=_env_optional_int(
                _THRESHOLD_MAX_SELECTED_BLOCKS_ENV),
            prefer_triton=_env_bool(_THRESHOLD_USE_TRITON_ENV, True),
        )

    def forward(
        self,
        *,
        score_state: typing.Any,
        block_top_k: int | None = None,
        chunk_top_k: int | None = None,
        current_chunks: torch.Tensor | None = None,
        max_prior_chunks: int | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any:
        scores = self._materialize_scores(score_state)
        if scores is None:
            return None
        chunk_logits, chunk_valid = scores
        if chunk_logits.dim() != 2 or chunk_valid.shape != chunk_logits.shape:
            return None

        rows, num_chunks = chunk_logits.shape
        capacity = self._selection_capacity(
            block_top_k=block_top_k,
            chunk_top_k=chunk_top_k,
            num_chunks=num_chunks,
        )
        if capacity <= 0 or rows == 0 or num_chunks == 0:
            selected_indices = torch.empty(
                rows,
                0,
                device=chunk_logits.device,
                dtype=torch.long,
            )
            selected_valid = torch.empty(
                rows,
                0,
                device=chunk_logits.device,
                dtype=torch.bool,
            )
            return self._make_selection_state(
                score_state,
                selected_indices,
                selected_valid,
            )

        thresholds = self._thresholds_for_rows(
            rows=rows,
            device=chunk_logits.device,
            dtype=chunk_logits.dtype,
            current_chunks=current_chunks,
            max_prior_chunks=max_prior_chunks,
            **kwargs,
        )
        selected_mask = (chunk_logits >= thresholds[:, None]) & chunk_valid
        selected_indices, selected_valid = self._compact_selected_mask(
            selected_mask,
            capacity=capacity,
        )
        return self._make_selection_state(
            score_state,
            selected_indices,
            selected_valid,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not None

    def get_selected_blocks(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if result is None:
            return None
        if not (
            hasattr(result, "_selected_block_indices")
            and hasattr(result, "_selected_block_valid")
        ):
            raise TypeError(f"unexpected block selection result: {type(result)!r}")
        return result._selected_block_indices, result._selected_block_valid

    def _selection_capacity(
        self,
        *,
        block_top_k: int | None,
        chunk_top_k: int | None,
        num_chunks: int,
    ) -> int:
        requested = block_top_k if block_top_k is not None else chunk_top_k
        caps = [num_chunks]
        if requested is not None:
            caps.append(max(0, int(requested)))
        if self.max_selected_blocks is not None:
            caps.append(self.max_selected_blocks)
        return min(caps)

    def _thresholds_for_rows(
        self,
        *,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
        current_chunks: torch.Tensor | None,
        max_prior_chunks: int | None,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        lengths = self._lengths_for_rows(
            rows=rows,
            device=device,
            dtype=dtype,
            current_chunks=current_chunks,
            max_prior_chunks=max_prior_chunks,
            **kwargs,
        )
        thresholds = torch.full(
            (rows,),
            self.base_threshold,
            device=device,
            dtype=torch.float32,
        )
        if self.length_scale:
            thresholds = thresholds + lengths.float() * self.length_scale
        if self.log_length_scale:
            thresholds = thresholds + torch.log1p(
                lengths.float()) * self.log_length_scale
        return thresholds.to(dtype=dtype)

    def _lengths_for_rows(
        self,
        *,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
        current_chunks: torch.Tensor | None,
        max_prior_chunks: int | None,
        **kwargs: typing.Any,
    ) -> torch.Tensor:
        provided = self._provided_lengths(
            rows=rows,
            device=device,
            dtype=dtype,
            **kwargs,
        )
        if provided is not None:
            return provided * self.length_multiplier

        if self.length_source == "provided":
            raise ValueError(
                "threshold length_source='provided' requires sequence_lengths, "
                "key_lens, seq_lens, or effective_lengths")
        if self.length_source == "row_chunks":
            if current_chunks is None:
                raise ValueError(
                    "threshold length_source='row_chunks' requires current_chunks")
            lengths = current_chunks.to(device=device, dtype=dtype) + 1
        else:
            if max_prior_chunks is None:
                raise ValueError(
                    "threshold length_source='sequence_chunks' requires "
                    "max_prior_chunks")
            lengths = torch.full(
                (rows,),
                float(max_prior_chunks + 1),
                device=device,
                dtype=dtype,
            )
        if lengths.dim() == 0:
            lengths = lengths.expand(rows)
        return lengths.reshape(rows) * self.length_multiplier

    @staticmethod
    def _provided_lengths(
        *,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
        **kwargs: typing.Any,
    ) -> torch.Tensor | None:
        for name in (
            "sequence_lengths",
            "key_lens",
            "seq_lens",
            "effective_lengths",
        ):
            value = kwargs.get(name)
            if value is None:
                continue
            lengths = torch.as_tensor(value, device=device, dtype=dtype)
            if lengths.numel() == 1:
                return lengths.reshape(1).expand(rows)
            if lengths.numel() != rows:
                raise ValueError(
                    f"{name} must have one value or {rows} row values, "
                    f"got {lengths.numel()}")
            return lengths.reshape(rows)
        return None

    def _compact_selected_mask(
        self,
        selected_mask: torch.Tensor,
        *,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        counts = selected_mask.sum(dim=-1).to(torch.long).clamp(max=capacity)
        selected_valid = (
            torch.arange(
                capacity,
                device=selected_mask.device,
                dtype=torch.long,
            )[None, :]
            < counts[:, None]
        )
        compacted = self._compact_selected_mask_triton(
            selected_mask,
            capacity=capacity,
        )
        if compacted is None:
            compacted = self._compact_selected_mask_torch(
                selected_mask,
                capacity=capacity,
            )
        return compacted, selected_valid

    def _compact_selected_mask_triton(
        self,
        selected_mask: torch.Tensor,
        *,
        capacity: int,
    ) -> torch.Tensor | None:
        if (
            not self.prefer_triton
            or triton is None
            or tl is None
            or not selected_mask.is_cuda
            or selected_mask.dim() != 2
        ):
            return None
        rows, num_chunks = selected_mask.shape
        block_chunks = triton.next_power_of_2(num_chunks)
        if block_chunks > 8192:
            return None
        selected_indices = torch.zeros(
            rows,
            capacity,
            device=selected_mask.device,
            dtype=torch.long,
        )
        _compact_bool_mask_kernel[(rows,)](
            selected_mask,
            selected_indices,
            num_chunks,
            capacity,
            selected_mask.stride(0),
            selected_mask.stride(1),
            selected_indices.stride(0),
            selected_indices.stride(1),
            BLOCK_CHUNKS=block_chunks,
            num_warps=self._triton_num_warps(block_chunks),
        )
        return selected_indices

    @staticmethod
    def _triton_num_warps(block_chunks: int) -> int:
        if block_chunks <= 64:
            return 1
        if block_chunks <= 128:
            return 2
        if block_chunks <= 256:
            return 4
        return 8

    @staticmethod
    def _compact_selected_mask_torch(
        selected_mask: torch.Tensor,
        *,
        capacity: int,
    ) -> torch.Tensor:
        rows = int(selected_mask.shape[0])
        selected_indices = torch.zeros(
            rows,
            capacity,
            device=selected_mask.device,
            dtype=torch.long,
        )
        if capacity == 0 or selected_mask.numel() == 0:
            return selected_indices
        ranks = selected_mask.to(torch.long).cumsum(dim=-1) - 1
        row_ids, chunk_ids = selected_mask.nonzero(as_tuple=True)
        if row_ids.numel() == 0:
            return selected_indices
        out_cols = ranks[row_ids, chunk_ids]
        keep = out_cols < capacity
        if bool(keep.any()):
            selected_indices[row_ids[keep], out_cols[keep]] = chunk_ids[keep]
        return selected_indices

    @staticmethod
    def _materialize_scores(
        score_state: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not (
            hasattr(score_state, "_chunk_logits")
            and hasattr(score_state, "_chunk_valid")
        ):
            return None
        return score_state._chunk_logits, score_state._chunk_valid

    @staticmethod
    def _make_selection_state(
        score_state: typing.Any,
        selected_indices: torch.Tensor,
        selected_valid: torch.Tensor,
    ) -> typing.Any:
        score_module = type(score_state).__module__
        if "efficient" in score_module:
            return _EfficientChunkBlockSelection(
                selected_block_indices=selected_indices,
                selected_block_valid=selected_valid,
            )
        return _TorchChunkBlockSelection(
            selected_block_indices=selected_indices,
            selected_block_valid=selected_valid,
        )


class TorchThresholdChunkedDSAProviderBundle(TorchChunkedDSAProviderBundle):
    """Torch chunked DSA bundle with threshold block selection."""

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.block_selection_provider = (
            ThresholdChunkedDSABlockSelectionProvider.from_env(
                length_multiplier=float(kwargs.get("chunk_size", 1)),
            )
        )


class EfficientThresholdChunkedDSAProviderBundle(
    EfficientChunkedDSAProviderBundle,
):
    """Efficient chunked DSA bundle with threshold block selection."""

    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)
        self.block_selection_provider = (
            ThresholdChunkedDSABlockSelectionProvider.from_env(
                length_multiplier=float(kwargs.get("chunk_size", 1)),
            )
        )
