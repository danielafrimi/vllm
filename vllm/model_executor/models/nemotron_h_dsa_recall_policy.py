# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Recall-budget providers for Nemotron-H chunked DSA."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from vllm.distributed.parallel_state import is_global_first_rank
from vllm.logger import init_logger

logger = init_logger(__name__)

_DYNAMIC_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K"
_DYNAMIC_DENSE_TOKENS_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_DENSE_TOKENS"
_DYNAMIC_STEP_TOKENS_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_STEP_TOKENS"
_DYNAMIC_BUDGET_DIVISOR_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_BUDGET_DIVISOR"
_DYNAMIC_MIN_BUDGET_TOKENS_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_MIN_BUDGET_TOKENS"
_DYNAMIC_MAX_CHUNK_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_DYNAMIC_MAX_CHUNK_TOP_K"
_RECENT_WINDOW_PAGES_ENV = "VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES"
_RECALL_DEBUG_PRINT_LIMIT_ENV = "VLLM_NEMOTRON_H_DSA_RECALL_DEBUG_PRINT_LIMIT"

_RECALL_DEBUG_COUNTS: dict[str, int] = {}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value == "1"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True, slots=True)
class RecallPolicyProvider:
    """Request-local policy for remote recall and deterministic recency."""

    chunk_size: int
    fixed_chunk_top_k: int
    recent_window_pages: int

    @property
    def dynamic(self) -> bool:
        return False

    @property
    def dense_tokens(self) -> int:
        return self.chunk_size * self.fixed_chunk_top_k

    def top_k_for_context(self, context_len: int) -> int:
        del context_len
        return self.fixed_chunk_top_k

    def top_k_for_context_tensor(
        self,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(
            context_lens,
            self.fixed_chunk_top_k,
            dtype=torch.int32,
        )

    def dense_query_prefix_len(
        self,
        *,
        query_position_start: int,
        query_len: int,
        dense_tokens: int | None = None,
    ) -> int:
        """Return rows whose one-based context is within the dense prefix.

        ``query_position_start`` is the zero-based absolute position of the
        first query row. Therefore row ``i`` has one-based context length
        ``query_position_start + i + 1`` and is dense exactly when that value
        is at most ``dense_tokens``.
        """
        if query_position_start < 0 or query_len < 0:
            raise ValueError(
                "query_position_start and query_len must be non-negative: "
                f"{query_position_start=}, {query_len=}"
            )
        threshold = self.dense_tokens if dense_tokens is None else dense_tokens
        if threshold <= 0:
            raise ValueError(f"dense_tokens must be positive: {threshold}")
        return min(query_len, max(threshold - query_position_start, 0))

    def query_tile_end(
        self,
        *,
        query_start: int,
        query_len: int,
        first_query_position: int,
        query_chunk_size: int,
    ) -> int:
        del first_query_position
        return min(query_start + query_chunk_size, query_len)

    def top_k_segments(
        self,
        *,
        query_position_start: int,
        query_len: int,
        maximum_top_k: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Return local query-row ranges whose recall budget is constant."""
        if query_position_start < 0 or query_len < 0:
            raise ValueError(
                "query_position_start and query_len must be non-negative: "
                f"{query_position_start=}, {query_len=}"
            )
        if query_len == 0:
            return []
        top_k = self.top_k_for_context(query_position_start + 1)
        if maximum_top_k is not None:
            top_k = min(top_k, maximum_top_k)
        return [(0, query_len, top_k)]

    def remote_chunk_counts(self, current_chunks: torch.Tensor) -> torch.Tensor:
        if self.recent_window_pages == 0:
            return current_chunks
        return (current_chunks - self.recent_window_pages).clamp_min(0)

    def recent_page_counts(self, current_chunks: torch.Tensor) -> torch.Tensor:
        if self.recent_window_pages == 0:
            return torch.zeros_like(current_chunks)
        return current_chunks.clamp(min=0, max=self.recent_window_pages)

    def describe(self) -> str:
        return (
            f"mode=fixed fixed_chunk_top_k={self.fixed_chunk_top_k} "
            f"chunk_size={self.chunk_size} "
            f"recent_window_pages={self.recent_window_pages}"
        )


@dataclass(frozen=True, slots=True)
class DynamicRecallPolicyProvider(RecallPolicyProvider):
    """The original request-local, 4K-banded Nemotron-H recall policy."""

    dynamic_dense_tokens: int
    dynamic_step_tokens: int
    dynamic_budget_divisor: int
    dynamic_min_budget_tokens: int
    dynamic_max_chunk_top_k: int = 4096

    @property
    def dynamic(self) -> bool:
        return True

    @property
    def dense_tokens(self) -> int:
        return self.dynamic_dense_tokens

    def top_k_for_context(self, context_len: int) -> int:
        if context_len <= 0:
            raise ValueError(f"context_len must be positive: {context_len}")
        context_band = (
            _ceil_div(context_len, self.dynamic_step_tokens) * self.dynamic_step_tokens
        )
        budget_tokens = max(
            self.dynamic_min_budget_tokens,
            _ceil_div(context_band, self.dynamic_budget_divisor),
        )
        return min(
            _ceil_div(budget_tokens, self.chunk_size),
            self.dynamic_max_chunk_top_k,
        )

    def top_k_for_context_tensor(
        self,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        context_lens = context_lens.to(dtype=torch.int64)
        context_bands = (
            torch.div(
                context_lens + self.dynamic_step_tokens - 1,
                self.dynamic_step_tokens,
                rounding_mode="floor",
            )
            * self.dynamic_step_tokens
        )
        budget_tokens = torch.div(
            context_bands + self.dynamic_budget_divisor - 1,
            self.dynamic_budget_divisor,
            rounding_mode="floor",
        ).clamp_min(self.dynamic_min_budget_tokens)
        return (
            torch.div(
                budget_tokens + self.chunk_size - 1,
                self.chunk_size,
                rounding_mode="floor",
            )
            .clamp_max(self.dynamic_max_chunk_top_k)
            .to(dtype=torch.int32)
        )

    def query_tile_end(
        self,
        *,
        query_start: int,
        query_len: int,
        first_query_position: int,
        query_chunk_size: int,
    ) -> int:
        query_end = min(query_start + query_chunk_size, query_len)
        context_len = first_query_position + query_start + 1
        if self.top_k_for_context(context_len) == self.dynamic_max_chunk_top_k:
            return query_end
        if context_len <= self.dynamic_dense_tokens:
            policy_end_context = self.dynamic_dense_tokens
        else:
            policy_end_context = (
                _ceil_div(context_len, self.dynamic_step_tokens)
                * self.dynamic_step_tokens
            )
        policy_query_end = policy_end_context - first_query_position
        return min(query_end, max(query_start + 1, policy_query_end))

    def top_k_segments(
        self,
        *,
        query_position_start: int,
        query_len: int,
        maximum_top_k: int | None = None,
    ) -> list[tuple[int, int, int]]:
        if query_position_start < 0 or query_len < 0:
            raise ValueError(
                "query_position_start and query_len must be non-negative: "
                f"{query_position_start=}, {query_len=}"
            )
        segments: list[tuple[int, int, int]] = []
        query_start = 0
        while query_start < query_len:
            query_end = self.query_tile_end(
                query_start=query_start,
                query_len=query_len,
                first_query_position=query_position_start,
                query_chunk_size=query_len,
            )
            top_k = self.top_k_for_context(query_position_start + query_start + 1)
            if maximum_top_k is not None:
                top_k = min(top_k, maximum_top_k)
            if segments and segments[-1][2] == top_k:
                segment_start, _, _ = segments[-1]
                segments[-1] = (segment_start, query_end, top_k)
            else:
                segments.append((query_start, query_end, top_k))
            query_start = query_end
        return segments

    def describe(self) -> str:
        return (
            f"mode=dynamic fixed_chunk_top_k={self.fixed_chunk_top_k} "
            f"chunk_size={self.chunk_size} "
            f"dense_tokens={self.dynamic_dense_tokens} "
            f"step_tokens={self.dynamic_step_tokens} "
            f"budget_divisor={self.dynamic_budget_divisor} "
            f"min_budget_tokens={self.dynamic_min_budget_tokens} "
            f"max_chunk_top_k={self.dynamic_max_chunk_top_k} "
            f"recent_window_pages={self.recent_window_pages}"
        )


def make_recall_policy_provider(
    *,
    chunk_size: int,
    fixed_chunk_top_k: int,
) -> RecallPolicyProvider:
    """Build the configured provider while keeping fixed recall the default."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive: {chunk_size}")
    if fixed_chunk_top_k <= 0:
        raise ValueError(f"fixed_chunk_top_k must be positive: {fixed_chunk_top_k}")
    recent_window_pages = _env_int(_RECENT_WINDOW_PAGES_ENV, 0)
    if recent_window_pages < 0:
        raise ValueError(
            f"{_RECENT_WINDOW_PAGES_ENV} must be non-negative: {recent_window_pages}"
        )

    common = dict(
        chunk_size=chunk_size,
        fixed_chunk_top_k=fixed_chunk_top_k,
        recent_window_pages=recent_window_pages,
    )
    if not _env_bool(_DYNAMIC_TOP_K_ENV):
        return RecallPolicyProvider(**common)

    dense_tokens = _env_int(_DYNAMIC_DENSE_TOKENS_ENV, 16 * 1024)
    step_tokens = _env_int(_DYNAMIC_STEP_TOKENS_ENV, 4 * 1024)
    budget_divisor = _env_int(_DYNAMIC_BUDGET_DIVISOR_ENV, 8)
    min_budget_tokens = _env_int(
        _DYNAMIC_MIN_BUDGET_TOKENS_ENV,
        16 * 1024,
    )
    max_chunk_top_k = _env_int(_DYNAMIC_MAX_CHUNK_TOP_K_ENV, 4096)
    settings = {
        _DYNAMIC_DENSE_TOKENS_ENV: dense_tokens,
        _DYNAMIC_STEP_TOKENS_ENV: step_tokens,
        _DYNAMIC_BUDGET_DIVISOR_ENV: budget_divisor,
        _DYNAMIC_MIN_BUDGET_TOKENS_ENV: min_budget_tokens,
        _DYNAMIC_MAX_CHUNK_TOP_K_ENV: max_chunk_top_k,
    }
    for name, value in settings.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive: {value}")
    return DynamicRecallPolicyProvider(
        **common,
        dynamic_dense_tokens=dense_tokens,
        dynamic_step_tokens=step_tokens,
        dynamic_budget_divisor=budget_divisor,
        dynamic_min_budget_tokens=min_budget_tokens,
        dynamic_max_chunk_top_k=max_chunk_top_k,
    )


def log_recall_config(provider: RecallPolicyProvider, *, owner: str) -> None:
    logger.info_once(
        "DSA_RECALL_CONFIG owner=%s %s",
        owner,
        provider.describe(),
        scope="global",
    )


def log_recall_plan(marker: str, **fields: object) -> None:
    """Emit a bounded, opt-in marker without reading device tensor values."""
    limit = _env_int(_RECALL_DEBUG_PRINT_LIMIT_ENV, 0)
    if limit <= 0 or not is_global_first_rank():
        return
    count = _RECALL_DEBUG_COUNTS.get(marker, 0)
    if count >= limit:
        return
    _RECALL_DEBUG_COUNTS[marker] = count + 1
    details = " ".join(f"{name}={value}" for name, value in fields.items())
    print(
        f"DSA_RECALL_PLAN marker={marker} count={count + 1} {details}".rstrip(),
        flush=True,
    )
