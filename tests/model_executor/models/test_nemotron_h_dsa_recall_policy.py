# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state
from vllm.model_executor.models import nemotron_h_dsa_recall_policy
from vllm.model_executor.models.nemotron_h_dsa_recall_policy import (
    DynamicRecallPolicyProvider,
    RecallPolicyProvider,
    log_recall_plan,
    make_recall_policy_provider,
)

_POLICY_ENV_NAMES = (
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K",
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_DENSE_TOKENS",
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_STEP_TOKENS",
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_BUDGET_DIVISOR",
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_MIN_BUDGET_TOKENS",
    "VLLM_NEMOTRON_H_DSA_DYNAMIC_MAX_CHUNK_TOP_K",
    "VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES",
    "VLLM_NEMOTRON_H_DSA_RECALL_DEBUG_PRINT_LIMIT",
)


@pytest.fixture(autouse=True)
def _clear_recall_policy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _POLICY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    nemotron_h_dsa_recall_policy._RECALL_DEBUG_COUNTS.clear()


def _make_dynamic_policy(
    *, recent_window_pages: int = 0
) -> DynamicRecallPolicyProvider:
    return DynamicRecallPolicyProvider(
        chunk_size=16,
        fixed_chunk_top_k=64,
        recent_window_pages=recent_window_pages,
        dynamic_dense_tokens=16 * 1024,
        dynamic_step_tokens=4 * 1024,
        dynamic_budget_divisor=8,
        dynamic_min_budget_tokens=16 * 1024,
    )


@pytest.mark.parametrize(
    ("context_len", "expected_top_k"),
    [
        (16 * 1024, 1024),
        (128 * 1024, 1024),
        (128 * 1024 + 1, 1056),
        (256 * 1024, 2048),
        (512 * 1024, 4096),
        (1024 * 1024, 4096),
    ],
)
def test_dynamic_top_k_scalar_schedule(
    context_len: int,
    expected_top_k: int,
) -> None:
    policy = _make_dynamic_policy()

    assert policy.top_k_for_context(context_len) == expected_top_k


def test_dynamic_top_k_tensor_matches_scalar_schedule() -> None:
    policy = _make_dynamic_policy()
    context_lens = torch.tensor(
        [
            16 * 1024,
            128 * 1024,
            128 * 1024 + 1,
            256 * 1024,
            512 * 1024,
            1024 * 1024,
        ],
        dtype=torch.int64,
    )

    actual = policy.top_k_for_context_tensor(context_lens)

    assert actual.dtype == torch.int32
    assert actual.device == context_lens.device
    torch.testing.assert_close(
        actual,
        torch.tensor([1024, 1024, 1056, 2048, 4096, 4096], dtype=torch.int32),
    )


def test_dynamic_top_k_rejects_nonpositive_scalar_context() -> None:
    policy = _make_dynamic_policy()

    with pytest.raises(ValueError, match="context_len must be positive"):
        policy.top_k_for_context(0)


def test_dynamic_query_tiles_end_after_last_token_in_policy_band() -> None:
    policy = _make_dynamic_policy()
    # Offset zero has one-based context 128K - 1, so offsets zero and one
    # remain in the 1,024-page band. Offset two is context 128K + 1.
    first_query_position = 128 * 1024 - 2

    first_end = policy.query_tile_end(
        query_start=0,
        query_len=8,
        first_query_position=first_query_position,
        query_chunk_size=8,
    )
    second_end = policy.query_tile_end(
        query_start=first_end,
        query_len=8,
        first_query_position=first_query_position,
        query_chunk_size=8,
    )

    assert first_end == 2
    assert second_end == 8
    assert policy.top_k_for_context(first_query_position + first_end) == 1024
    assert policy.top_k_for_context(first_query_position + first_end + 1) == 1056


def test_dynamic_query_tiles_respect_dense_and_query_chunk_boundaries() -> None:
    policy = _make_dynamic_policy()

    assert (
        policy.query_tile_end(
            query_start=0,
            query_len=10,
            first_query_position=16 * 1024 - 2,
            query_chunk_size=10,
        )
        == 2
    )
    assert (
        policy.query_tile_end(
            query_start=0,
            query_len=10,
            first_query_position=128 * 1024 - 1,
            query_chunk_size=1,
        )
        == 1
    )


def test_dynamic_segments_split_an_8k_prefill_at_the_exact_policy_boundary() -> None:
    policy = _make_dynamic_policy()

    assert policy.top_k_segments(
        query_position_start=128 * 1024 - 2,
        query_len=8 * 1024,
        maximum_top_k=10000,
    ) == [
        (0, 2, 1024),
        (2, 4098, 1056),
        (4098, 8192, 1088),
    ]


def test_dynamic_segments_merge_adjacent_4k_bands_with_the_same_budget() -> None:
    policy = _make_dynamic_policy()

    assert policy.top_k_segments(
        query_position_start=64 * 1024,
        query_len=8 * 1024,
    ) == [(0, 8 * 1024, 1024)]


def test_dynamic_query_tiles_do_not_split_after_reaching_the_cap() -> None:
    policy = _make_dynamic_policy()

    assert (
        policy.query_tile_end(
            query_start=0,
            query_len=8 * 1024,
            first_query_position=512 * 1024 - 1,
            query_chunk_size=8 * 1024,
        )
        == 8 * 1024
    )
    assert policy.top_k_segments(
        query_position_start=512 * 1024 - 1,
        query_len=8 * 1024,
    ) == [(0, 8 * 1024, 4096)]


def test_fixed_policy_preserves_constant_budget_and_regular_tiles() -> None:
    policy = RecallPolicyProvider(
        chunk_size=16,
        fixed_chunk_top_k=73,
        recent_window_pages=0,
    )
    context_lens = torch.tensor([1, 16 * 1024, 1024 * 1024])

    assert not policy.dynamic
    assert policy.dense_tokens == 73 * 16
    assert policy.top_k_for_context(1) == 73
    assert policy.top_k_for_context(1024 * 1024) == 73
    torch.testing.assert_close(
        policy.top_k_for_context_tensor(context_lens),
        torch.full((3,), 73, dtype=torch.int32),
    )
    assert (
        policy.query_tile_end(
            query_start=7,
            query_len=20,
            first_query_position=128 * 1024 - 1,
            query_chunk_size=5,
        )
        == 12
    )


@pytest.mark.parametrize(
    ("recent_window_pages", "expected_remote", "expected_recent"),
    [
        (0, [0, 1, 127, 128, 129, 1000], [0, 0, 0, 0, 0, 0]),
        (128, [0, 0, 0, 0, 1, 872], [0, 1, 127, 128, 128, 128]),
    ],
)
def test_recent_window_counts_are_additive_and_disjoint(
    recent_window_pages: int,
    expected_remote: list[int],
    expected_recent: list[int],
) -> None:
    policy = _make_dynamic_policy(recent_window_pages=recent_window_pages)
    current_chunks = torch.tensor([0, 1, 127, 128, 129, 1000])

    torch.testing.assert_close(
        policy.remote_chunk_counts(current_chunks),
        torch.tensor(expected_remote),
    )
    torch.testing.assert_close(
        policy.recent_page_counts(current_chunks),
        torch.tensor(expected_recent),
    )


def test_factory_defaults_to_fixed_policy() -> None:
    policy = make_recall_policy_provider(
        chunk_size=16,
        fixed_chunk_top_k=73,
    )

    assert type(policy) is RecallPolicyProvider
    assert policy.top_k_for_context(1024 * 1024) == 73
    assert policy.recent_window_pages == 0


def test_factory_builds_default_dynamic_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K", "1")
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES", "128")

    policy = make_recall_policy_provider(
        chunk_size=16,
        fixed_chunk_top_k=73,
    )

    assert isinstance(policy, DynamicRecallPolicyProvider)
    assert policy.dynamic_dense_tokens == 16 * 1024
    assert policy.dynamic_step_tokens == 4 * 1024
    assert policy.dynamic_budget_divisor == 8
    assert policy.dynamic_min_budget_tokens == 16 * 1024
    assert policy.dynamic_max_chunk_top_k == 4096
    assert policy.recent_window_pages == 128
    assert policy.top_k_for_context(128 * 1024 + 1) == 1056


def test_factory_honors_configured_dynamic_top_k_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K", "1")
    monkeypatch.setenv(
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_MAX_CHUNK_TOP_K",
        "2048",
    )

    policy = make_recall_policy_provider(
        chunk_size=16,
        fixed_chunk_top_k=73,
    )

    assert isinstance(policy, DynamicRecallPolicyProvider)
    assert policy.dynamic_max_chunk_top_k == 2048
    assert policy.top_k_for_context(1024 * 1024) == 2048
    torch.testing.assert_close(
        policy.top_k_for_context_tensor(torch.tensor([256 * 1024, 1024 * 1024])),
        torch.tensor([2048, 2048], dtype=torch.int32),
    )


def test_factory_min_budget_is_independent_of_configured_dense_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K", "1")
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_DENSE_TOKENS", "32768")

    policy = make_recall_policy_provider(
        chunk_size=16,
        fixed_chunk_top_k=73,
    )

    assert isinstance(policy, DynamicRecallPolicyProvider)
    assert policy.dynamic_dense_tokens == 32768
    assert policy.dynamic_min_budget_tokens == 16 * 1024


@pytest.mark.parametrize(
    ("query_position_start", "query_len", "dense_tokens", "expected"),
    [
        (0, 8 * 1024, 64 * 1024, 8 * 1024),
        (60 * 1024, 8 * 1024, 64 * 1024, 4 * 1024),
        (64 * 1024 - 1, 8, 64 * 1024, 1),
        (64 * 1024, 8 * 1024, 64 * 1024, 0),
    ],
)
def test_dense_query_prefix_len_uses_each_rows_one_based_context(
    query_position_start: int,
    query_len: int,
    dense_tokens: int,
    expected: int,
) -> None:
    policy = _make_dynamic_policy()

    assert (
        policy.dense_query_prefix_len(
            query_position_start=query_position_start,
            query_len=query_len,
            dense_tokens=dense_tokens,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("chunk_size", "fixed_chunk_top_k", "message"),
    [
        (0, 1, "chunk_size must be positive"),
        (1, 0, "fixed_chunk_top_k must be positive"),
    ],
)
def test_factory_rejects_invalid_base_settings(
    chunk_size: int,
    fixed_chunk_top_k: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        make_recall_policy_provider(
            chunk_size=chunk_size,
            fixed_chunk_top_k=fixed_chunk_top_k,
        )


def test_factory_rejects_negative_recent_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES", "-1")

    with pytest.raises(
        ValueError,
        match="VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES must be non-negative",
    ):
        make_recall_policy_provider(chunk_size=16, fixed_chunk_top_k=73)


@pytest.mark.parametrize(
    "name",
    [
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_DENSE_TOKENS",
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_STEP_TOKENS",
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_BUDGET_DIVISOR",
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_MIN_BUDGET_TOKENS",
        "VLLM_NEMOTRON_H_DSA_DYNAMIC_MAX_CHUNK_TOP_K",
    ],
)
def test_factory_rejects_nonpositive_dynamic_settings(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K", "1")
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValueError, match=f"{name} must be positive"):
        make_recall_policy_provider(chunk_size=16, fixed_chunk_top_k=73)


def test_recall_plan_marker_is_disabled_without_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_if_called() -> bool:
        raise AssertionError("disabled markers must not query distributed rank")

    monkeypatch.setattr(
        nemotron_h_dsa_recall_policy,
        "is_global_first_rank",
        fail_if_called,
    )

    log_recall_plan("disabled", top_k=1024)

    assert capsys.readouterr().out == ""


def test_recall_plan_marker_only_prints_on_global_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECALL_DEBUG_PRINT_LIMIT", "2")
    monkeypatch.setattr(
        parallel_state,
        "_WORLD",
        SimpleNamespace(is_first_rank=False),
    )

    log_recall_plan("nonzero_rank", top_k=1024)

    assert capsys.readouterr().out == ""
    assert "nonzero_rank" not in nemotron_h_dsa_recall_policy._RECALL_DEBUG_COUNTS


def test_recall_plan_marker_is_bounded_on_global_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECALL_DEBUG_PRINT_LIMIT", "2")
    monkeypatch.setattr(
        parallel_state,
        "_WORLD",
        SimpleNamespace(is_first_rank=True),
    )

    log_recall_plan("bounded", top_k=1024)
    log_recall_plan("bounded", top_k=1056)
    log_recall_plan("bounded", top_k=1088)

    assert capsys.readouterr().out.splitlines() == [
        "DSA_RECALL_PLAN marker=bounded count=1 top_k=1024",
        "DSA_RECALL_PLAN marker=bounded count=2 top_k=1056",
    ]


def test_recall_plan_marker_is_safe_before_distributed_init(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECALL_DEBUG_PRINT_LIMIT", "1")
    monkeypatch.setattr(parallel_state, "_WORLD", None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    log_recall_plan("pre_init", top_k=1024)

    assert capsys.readouterr().out == (
        "DSA_RECALL_PLAN marker=pre_init count=1 top_k=1024\n"
    )
