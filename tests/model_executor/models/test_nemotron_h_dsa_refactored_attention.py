# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
import random
from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.models.nemotron_h as nemotron_h_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.models import (
    nemotron_h_chunked_dsa_components_pytorch as pytorch_components_module,
)
from vllm.model_executor.models import (
    nemotron_h_dsa_attention_refactored as refactored_attention_module,
)
from vllm.model_executor.models import (
    nemotron_h_qshare_dsa_components_efficient as efficient_qshare_components_module,
)
from vllm.model_executor.models import (
    nemotron_h_qshare_dsa_components_pytorch as qshare_components_module,
)
from vllm.model_executor.models.nemotron_h import (
    NemotronHDSASelectiveAttention,
    _split_dsa_kv_cache,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient import (
    EfficientChunkedDSAProviderBundle,
)
from vllm.model_executor.models.nemotron_h_dsa_attention_refactored import (
    NemotronHDSARefactoredAttention,
)
from vllm.model_executor.models.nemotron_h_dsa_query_providers import (
    IdentityQProvider,
    MeanQShareProvider,
)
from vllm.model_executor.models.nemotron_h_dsa_recall_policy import (
    DynamicRecallPolicyProvider,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_qshare import (
    EfficientIdentityQShareProvider,
    EfficientMeanQShareProvider,
    EfficientQShareState,
    qshare_batched_page_table_triton,
    qshare_score_metadata_triton,
)
from vllm.model_executor.models.nemotron_h_nonchunked_dsa_components_pytorch import (
    TorchNonChunkedDSAProviderBundle,
)
from vllm.model_executor.models.nemotron_h_qshare_dsa_components_efficient import (
    EfficientQShareChunkedDSAProviderBundle,
    _absolute_qshare_run_start,
    _absolute_qshare_sampled_length,
    _absolute_qshare_top_k_segments,
)
from vllm.model_executor.models.nemotron_h_qshare_dsa_components_pytorch import (
    TorchQShareMeanChunkedDSAProviderBundle,
)
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


_QUERY_PATTERNS = (
    ("decode1", lambda batch_size: [1] * batch_size),
    ("decode4", lambda batch_size: [4] * batch_size),
    ("prefill8", lambda batch_size: [8] * batch_size),
    ("prefill_many", lambda batch_size: [17 + 3 * (i % 3) for i in range(batch_size)]),
    (
        "mixed_decode_prefill",
        lambda batch_size: [(1, 4, 17, 33)[i % 4] for i in range(batch_size)],
    ),
)

_HEAD_CASES = (
    (1, 1, 4, 2),
    (1, 4, 8, 3),
    (2, 1, 4, 2),
    (2, 3, 8, 4),
    (4, 2, 8, 3),
)


def test_split_current_packed_kv_cache_with_two_kv_heads() -> None:
    key = torch.arange(3 * 5 * 2 * 4).view(3, 5, 2, 4)
    value = key + 1000
    packed = torch.cat((key, value), dim=-1).transpose(1, 2)

    actual_key, actual_value = _split_dsa_kv_cache(
        packed,
        num_kv_heads=2,
        head_dim=4,
    )

    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)


def test_split_legacy_stacked_kv_cache() -> None:
    key = torch.arange(3 * 5 * 2 * 4).view(3, 5, 2, 4)
    value = key + 1000

    actual_key, actual_value = _split_dsa_kv_cache(
        torch.stack((key, value)),
        num_kv_heads=2,
        head_dim=4,
    )

    torch.testing.assert_close(actual_key, key)
    torch.testing.assert_close(actual_value, value)

_CHUNK_CASES = (
    (4, 1, 3),
    (4, 3, 5),
    (8, 2, 4),
)

_LARGE_MIXED_BATCH_SIZES = (16, 32, 64, 128)

_LARGE_MIXED_HEAD_CASES = (
    (1, 8, 16, 4),
    (2, 4, 16, 4),
    (4, 2, 16, 6),
)

_LARGE_MIXED_CHUNK_CASES = (
    (8, 4, 8),
    (16, 4, 8),
)

_LARGE_MIXED_QUERY_PROFILES = (
    ("decode_heavy_random", (1, 1, 1, 4, 4, 64, 96)),
    ("balanced_random", (1, 4, 64, 1, 4, 96, 128, 32)),
    ("prefill_heavy_random", (1, 4, 64, 96, 128, 160, 192, 32)),
    ("long_tail_random", (1, 4, 4, 64, 128, 192, 256)),
)


def _expected_qshare_runs(
    positions: torch.Tensor,
    *,
    chunk_size: int,
    group_size: int,
) -> list[list[int]]:
    del chunk_size
    return [
        list(range(start, min(start + group_size, positions.numel())))
        for start in range(0, positions.numel(), group_size)
    ]


@pytest.mark.parametrize("q_len", range(1, 10))
@pytest.mark.parametrize("start_residue", range(4))
def test_mean_qshare_provider_ragged_contract(
    q_len: int,
    start_residue: int,
) -> None:
    chunk_size = 8
    group_size = 4
    start = chunk_size - 2 + start_residue
    positions = torch.arange(start, start + q_len, dtype=torch.long)
    current_chunks = torch.div(positions, chunk_size, rounding_mode="floor")
    projected_q = torch.arange(q_len * 3, dtype=torch.float32).view(q_len, 3)

    state = MeanQShareProvider(group_size=group_size)(
        projected_q=projected_q,
        current_chunks=current_chunks,
        query_positions=positions,
        chunk_size=chunk_size,
    )

    expected_runs = _expected_qshare_runs(
        positions,
        chunk_size=chunk_size,
        group_size=group_size,
    )
    expected_starts = torch.tensor([run[0] for run in expected_runs])
    expected_counts = torch.tensor([len(run) for run in expected_runs])
    expected_mapping = torch.repeat_interleave(
        torch.arange(len(expected_runs)), expected_counts
    )
    expected_q = torch.stack(
        [projected_q[run].float().mean(dim=0) for run in expected_runs]
    )

    torch.testing.assert_close(state.reduced_q, expected_q)
    torch.testing.assert_close(state.run_starts, expected_starts)
    torch.testing.assert_close(state.run_counts, expected_counts)
    torch.testing.assert_close(state.query_row_to_reduced_row, expected_mapping)
    torch.testing.assert_close(
        state.reduced_current_chunks,
        current_chunks.index_select(0, expected_starts),
    )
    assert int(state.run_counts.sum().item()) == q_len


def test_mean_qshare_provider_multiple_sequences_and_padding() -> None:
    provider = MeanQShareProvider(group_size=4)
    projected_q = torch.arange(12 * 2, dtype=torch.float32).view(12, 2)
    positions = torch.tensor([3, 4, 5, 6, 7, 8, 9, 14, 15, 0, 0, 0])
    active_seq_infos = [(0, 0, 3, 6), (1, 3, 5, 9), (2, 5, 9, 16)]

    states = provider.prepare_batch(
        projected_q=projected_q,
        positions=positions,
        active_seq_infos=active_seq_infos,
        chunk_size=8,
    )

    assert set(states) == {0, 1, 2}
    assert sum(int(state.run_counts.sum()) for state in states.values()) == 9
    assert all(
        int(state.query_row_to_reduced_row.numel()) == q_end - q_start
        for state, (_, q_start, q_end, _) in zip(
            states.values(), active_seq_infos, strict=True
        )
    )


def test_qshare_provider_empty_slice_and_large_group() -> None:
    provider = MeanQShareProvider(group_size=64)
    empty = provider(
        projected_q=torch.empty(0, 3),
        current_chunks=torch.empty(0, dtype=torch.long),
        query_positions=torch.empty(0, dtype=torch.long),
        chunk_size=8,
    )
    assert empty.reduced_q.shape == (0, 3)
    assert empty.run_starts.numel() == 0
    assert empty.run_counts.numel() == 0
    assert empty.query_row_to_reduced_row.numel() == 0

    positions = torch.tensor([5, 6, 7])
    q = torch.arange(9, dtype=torch.float32).view(3, 3)
    state = provider(
        projected_q=q,
        current_chunks=positions // 8,
        query_positions=positions,
        chunk_size=8,
    )
    torch.testing.assert_close(state.run_counts, torch.tensor([3]))
    torch.testing.assert_close(state.reduced_q, q.mean(dim=0, keepdim=True))


def test_mean_qshare_provider_cuda_matches_cpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Q-share provider device test")
    positions = torch.arange(5, 22, dtype=torch.long)
    chunks = positions // 8
    q = torch.randn(17, 5)
    provider = MeanQShareProvider(group_size=4)
    expected = provider(
        projected_q=q,
        current_chunks=chunks,
        query_positions=positions,
        chunk_size=8,
    )
    actual = provider.cuda()(
        projected_q=q.cuda(),
        current_chunks=chunks.cuda(),
        query_positions=positions.cuda(),
        chunk_size=8,
    )
    torch.testing.assert_close(actual.reduced_q.cpu(), expected.reduced_q)
    torch.testing.assert_close(actual.run_starts.cpu(), expected.run_starts)
    torch.testing.assert_close(actual.run_counts.cpu(), expected.run_counts)
    torch.testing.assert_close(
        actual.query_row_to_reduced_row.cpu(),
        expected.query_row_to_reduced_row,
    )


@pytest.mark.parametrize(
    "query_lengths",
    [
        (1,),
        (2, 3, 4, 5),
        (1, 8192),
        (1, 1, 4, 1, 17, 64, 257),
        (7, 32, 129, 1024, 4097),
    ],
    ids=[
        "single_decode",
        "partial_groups",
        "decode_and_8k_prefill",
        "mixed_decode_and_prefills",
        "multiple_varied_prefills",
    ],
)
@pytest.mark.parametrize("num_heads,head_dim", [(1, 8), (3, 17)])
@pytest.mark.parametrize("group_size", [2, 4, 8, 16])
def test_efficient_mean_qshare_provider_matches_pytorch(
    query_lengths: tuple[int, ...],
    num_heads: int,
    head_dim: int,
    group_size: int,
) -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    starts = [0, *itertools.accumulate(query_lengths)]
    total_sampled_rows = sum(
        (length + group_size - 1) // group_size for length in query_lengths
    )
    torch.manual_seed(1234)
    projected_q = torch.randn(starts[-1], num_heads, head_dim)
    reference_provider = MeanQShareProvider(group_size=group_size)
    reference_states = []
    for start, end in zip(starts[:-1], starts[1:], strict=True):
        rows = end - start
        reference_states.append(
            reference_provider(
                projected_q=projected_q[start:end],
                current_chunks=torch.zeros(rows, dtype=torch.long),
                query_positions=torch.arange(rows),
                chunk_size=16,
            )
        )

    actual = EfficientMeanQShareProvider(group_size=group_size).cuda()(
        projected_q=projected_q.cuda(),
        query_start_loc=torch.tensor(starts, device="cuda"),
        query_start_loc_cpu=torch.tensor(starts),
        total_sampled_rows=total_sampled_rows,
    )
    assert not actual.absolute_position_aligned
    expected_q = torch.cat([state.reduced_q for state in reference_states])
    expected_lengths = torch.tensor(
        [state.reduced_q.shape[0] for state in reference_states]
    )
    expected_starts = torch.cat(
        (torch.zeros(1, dtype=torch.long), expected_lengths.cumsum(dim=0))
    )
    expected_sampled_to_sequence = torch.repeat_interleave(
        torch.arange(len(query_lengths)), expected_lengths
    )
    expected_run_starts = []
    expected_run_lengths = []
    expected_original_to_sampled = []
    sampled_offset = 0
    for original_start, state in zip(starts[:-1], reference_states, strict=True):
        expected_run_starts.append(state.run_starts + original_start)
        expected_run_lengths.append(state.run_counts)
        expected_original_to_sampled.append(
            state.query_row_to_reduced_row + sampled_offset
        )
        sampled_offset += state.reduced_q.shape[0]

    torch.testing.assert_close(actual.sampled_q.cpu(), expected_q)
    torch.testing.assert_close(actual.sampled_query_lengths.cpu(), expected_lengths)
    torch.testing.assert_close(actual.sampled_query_start_loc.cpu(), expected_starts)
    torch.testing.assert_close(actual.sampled_query_start_loc_cpu, expected_starts)
    torch.testing.assert_close(
        actual.sampled_to_sequence.cpu(), expected_sampled_to_sequence
    )
    torch.testing.assert_close(
        actual.sampled_to_original_start.cpu(),
        torch.cat(expected_run_starts),
    )
    torch.testing.assert_close(
        actual.sampled_run_lengths.cpu(), torch.cat(expected_run_lengths)
    )
    torch.testing.assert_close(
        actual.original_to_sampled.cpu(),
        torch.cat(expected_original_to_sampled),
    )


def test_efficient_mean_qshare_provider_preserves_projected_q_dtype() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    starts = [0, 5, 6, 15]
    projected_q = torch.randn(15, 3, 17, dtype=torch.bfloat16, device="cuda")

    actual = EfficientMeanQShareProvider(group_size=4).cuda()(
        projected_q=projected_q,
        query_start_loc=torch.tensor(starts, device="cuda"),
        query_start_loc_cpu=torch.tensor(starts),
        total_sampled_rows=6,
    )
    expected = torch.cat(
        [
            torch.stack(
                [
                    projected_q[row : min(row + 4, end)].float().mean(dim=0)
                    for row in range(start, end, 4)
                ]
            ).to(projected_q.dtype)
            for start, end in zip(starts[:-1], starts[1:], strict=True)
        ]
    )

    assert actual.sampled_q.dtype == projected_q.dtype
    torch.testing.assert_close(actual.sampled_q, expected)


@pytest.mark.parametrize(
    ("query_position_start", "query_len", "expected_starts"),
    [
        (5, 0, []),
        (0, 9, [0, 4, 8]),
        (5, 9, [0, 3, 7]),
        (15, 6, [0, 1, 5]),
    ],
)
def test_absolute_qshare_run_layout(
    query_position_start: int,
    query_len: int,
    expected_starts: list[int],
) -> None:
    sampled_len = _absolute_qshare_sampled_length(
        query_position_start=query_position_start,
        query_len=query_len,
        group_size=4,
    )
    actual_starts = [
        _absolute_qshare_run_start(
            query_position_start=query_position_start,
            sampled_row=sampled_row,
            group_size=4,
        )
        for sampled_row in range(sampled_len)
    ]

    assert actual_starts == expected_starts
    for run_idx, local_start in enumerate(actual_starts):
        local_end = (
            actual_starts[run_idx + 1] if run_idx + 1 < sampled_len else query_len
        )
        assert (query_position_start + local_start) // 4 == (
            query_position_start + local_end - 1
        ) // 4


def test_absolute_qshare_dynamic_top_k_segments_split_at_policy_boundary() -> None:
    policy = DynamicRecallPolicyProvider(
        chunk_size=16,
        fixed_chunk_top_k=64,
        recent_window_pages=128,
        dynamic_dense_tokens=16 * 1024,
        dynamic_step_tokens=4 * 1024,
        dynamic_budget_divisor=8,
        dynamic_min_budget_tokens=16 * 1024,
    )

    actual = _absolute_qshare_top_k_segments(
        policy=policy,
        query_position_start=128 * 1024 - 3,
        query_len=7,
        group_size=4,
        sampled_row_start=10,
        maximum_top_k=4096,
    )

    assert actual == [(10, 11, 1024), (11, 12, 1056)]


def test_efficient_mean_qshare_provider_absolute_alignment() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    packed_starts = torch.tensor([0, 6, 15], dtype=torch.int32)
    query_position_starts = torch.tensor([5, 16], dtype=torch.int32)
    projected_q = torch.arange(15 * 2, dtype=torch.float32).view(15, 1, 2)

    actual = EfficientMeanQShareProvider(group_size=4).cuda()(
        projected_q=projected_q.cuda(),
        query_start_loc=packed_starts.cuda(),
        query_start_loc_cpu=packed_starts,
        total_sampled_rows=5,
        query_position_starts=query_position_starts.cuda(),
        query_position_starts_cpu=query_position_starts,
    )

    expected_global_starts = torch.tensor([0, 3, 6, 10, 14], dtype=torch.int32)
    expected_run_lengths = torch.tensor([3, 3, 4, 4, 1], dtype=torch.int32)
    expected_q = torch.stack(
        [
            projected_q[int(start) : int(start + length)].float().mean(dim=0)
            for start, length in zip(
                expected_global_starts,
                expected_run_lengths,
                strict=True,
            )
        ]
    )
    expected_mapping = torch.repeat_interleave(
        torch.arange(5, dtype=torch.int32), expected_run_lengths
    )

    assert actual.absolute_position_aligned
    torch.testing.assert_close(
        actual.sampled_query_start_loc.cpu(),
        torch.tensor([0, 2, 5], dtype=torch.int32),
    )
    torch.testing.assert_close(
        actual.sampled_to_original_start.cpu(), expected_global_starts
    )
    torch.testing.assert_close(actual.sampled_run_lengths.cpu(), expected_run_lengths)
    torch.testing.assert_close(actual.original_to_sampled.cpu(), expected_mapping)
    torch.testing.assert_close(actual.sampled_q.cpu(), expected_q)


def test_efficient_identity_qshare_provider_preserves_inputs() -> None:
    projected_q = torch.randn(7, 2, 8)
    gpu_starts = torch.tensor([0, 1, 7])
    cpu_starts = torch.tensor([0, 1, 7])
    state = EfficientIdentityQShareProvider()(
        projected_q=projected_q,
        query_start_loc=gpu_starts,
        query_start_loc_cpu=cpu_starts,
        total_sampled_rows=7,
    )
    assert state.sampled_q is projected_q
    assert state.sampled_query_start_loc is gpu_starts
    assert state.sampled_query_start_loc_cpu is cpu_starts
    assert state.metadata is None


def test_efficient_qshare_one_bundle_uses_identity_sampler() -> None:
    bundle = EfficientQShareChunkedDSAProviderBundle(
        qshare_group_size=1,
        q_indexer_dim=8,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=8,
        logit_scale=1.0,
        chunk_top_k=2,
        query_chunk_size=3,
    )
    projected_q = torch.randn(7, 1, 8)
    starts = torch.tensor([0, 1, 7])
    state = bundle.prepare_selection_query_batch(
        score_query_states=projected_q,
        query_start_loc=starts,
        query_start_loc_cpu=starts,
        active_seq_count=2,
    )
    assert state.sampled_q is projected_q
    assert state.metadata is None
    assert bundle.selection_query_chunk_size(7) == 3


@pytest.mark.parametrize("group_size", [0, 3, 6])
def test_efficient_qshare_bundle_rejects_non_power_of_two(
    group_size: int,
) -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        EfficientQShareChunkedDSAProviderBundle(
            qshare_group_size=group_size,
            q_indexer_dim=8,
            chunk_size=4,
            num_kv_heads=1,
            head_dim=8,
            logit_scale=1.0,
        )


@pytest.mark.parametrize("group_size", [1, 2, 4, 8, 16])
def test_efficient_qshare_table_builder_matches_pytorch(
    group_size: int,
) -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    query_lengths = (1, 3, 8, 17)
    starts = [0, *itertools.accumulate(query_lengths)]
    key_lens = [length + 20 for length in query_lengths]
    chunk_size = 4
    total_sampled_rows = sum(
        (length + group_size - 1) // group_size for length in query_lengths
    )
    sampled_state = EfficientMeanQShareProvider(group_size=group_size).cuda()(
        projected_q=torch.randn(starts[-1], 1, 8, device="cuda"),
        query_start_loc=torch.tensor(starts, device="cuda"),
        query_start_loc_cpu=torch.tensor(starts),
        total_sampled_rows=total_sampled_rows,
    )
    top_width = 2
    selected_blocks = torch.tensor(
        [[0, 1]] * total_sampled_rows,
        device="cuda",
        dtype=torch.int32,
    )
    selected_counts = torch.full(
        (total_sampled_rows,),
        top_width,
        device="cuda",
        dtype=torch.int32,
    )
    block_table = torch.stack(
        [
            torch.arange(32, dtype=torch.int32) + seq_idx * 100
            for seq_idx in range(len(query_lengths))
        ]
    ).cuda()
    max_local_pages = (group_size + chunk_size - 2) // chunk_size + 1
    table_width = top_width + max_local_pages
    actual = qshare_batched_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        state=sampled_state,
        seq_lens=torch.tensor(key_lens, device="cuda"),
        active_seq_count=len(query_lengths),
        num_requests=total_sampled_rows,
        table_width=table_width,
        max_sampled_q_len=max(
            (length + group_size - 1) // group_size for length in query_lengths
        ),
        chunk_size=chunk_size,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )
    assert actual is not None
    actual_pages, actual_seqused, actual_cu_q = actual

    reference_provider = pytorch_components_module.TorchChunkedDSABlockTableProvider()
    expected_pages = []
    expected_lens = []
    expected_seqused = []
    for seq_idx, (q_start, q_end, key_len) in enumerate(
        zip(starts[:-1], starts[1:], key_lens, strict=True)
    ):
        q_len = q_end - q_start
        query_position_start = key_len - q_len
        positions = torch.arange(query_position_start, key_len)
        current_chunks = positions // chunk_size
        reference_qshare = MeanQShareProvider(group_size=group_size)(
            projected_q=torch.randn(q_len, 8),
            current_chunks=current_chunks,
            query_positions=positions,
            chunk_size=chunk_size,
        )
        runs = int(reference_qshare.run_counts.shape[0])
        selection = pytorch_components_module._TorchChunkBlockSelection(
            selected_block_indices=torch.tensor([[0, 1]] * runs),
            selected_block_valid=torch.ones(runs, top_width, dtype=torch.bool),
        )
        reference = reference_provider(
            block_table=block_table[seq_idx].cpu(),
            chunk_size=chunk_size,
            key_len=key_len,
            q_len=q_len,
            dense=False,
            mode="prefill",
            selection_state=selection,
            current_chunks=current_chunks,
            query_position_start=query_position_start,
            selection_query_state=reference_qshare,
        )
        reference_plan = reference_provider.get_page_table(reference)
        assert reference_plan is not None
        pages, request_lens, seqused_k, _, _ = reference_plan
        expected_pages.append(
            torch.nn.functional.pad(
                pages,
                (0, table_width - int(pages.shape[1])),
            )
        )
        expected_lens.append(request_lens)
        expected_seqused.append(seqused_k)

    expected_lens_t = torch.cat(expected_lens)
    torch.testing.assert_close(actual_pages.cpu(), torch.cat(expected_pages))
    torch.testing.assert_close(actual_seqused.cpu(), torch.cat(expected_seqused))
    torch.testing.assert_close(
        actual_cu_q.cpu(),
        torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                expected_lens_t.cumsum(0, dtype=torch.int32),
            )
        ),
    )


def test_efficient_qshare_table_builder_splits_dense_sparse_crossing() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    group_size = 4
    chunk_size = 4
    query_position_start = 28
    query_len = 12
    dense_tokens = 32
    state = EfficientMeanQShareProvider(group_size=group_size).cuda()(
        projected_q=torch.randn(query_len, 1, 8, device="cuda"),
        query_start_loc=torch.tensor([0, query_len], device="cuda"),
        query_start_loc_cpu=torch.tensor([0, query_len]),
        query_position_starts=torch.tensor([query_position_start], device="cuda"),
        query_position_starts_cpu=torch.tensor([query_position_start]),
        total_sampled_rows=3,
    )
    selected_blocks = torch.tensor(
        [[0, 1], [0, 1], [0, 1]],
        device="cuda",
        dtype=torch.int32,
    )
    selected_counts = torch.full((3,), 2, device="cuda", dtype=torch.int32)
    block_table = torch.arange(100, 116, device="cuda", dtype=torch.int32).view(1, -1)

    actual = qshare_batched_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        state=state,
        seq_lens=torch.tensor([40], device="cuda"),
        active_seq_count=1,
        num_requests=3,
        table_width=8,
        max_sampled_q_len=3,
        chunk_size=chunk_size,
        dense_decode_threshold=dense_tokens,
        dense_prefill_threshold=dense_tokens,
        qshare_group_size=group_size,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    torch.testing.assert_close(
        page_table.cpu(),
        torch.tensor(
            [
                [100, 101, 102, 103, 104, 105, 106, 107],
                [100, 101, 108, 0, 0, 0, 0, 0],
                [100, 101, 109, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        seqused_k.cpu(), torch.tensor([32, 12, 12], dtype=torch.int32)
    )
    torch.testing.assert_close(
        cu_seqlens_q.cpu(), torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    )


def test_efficient_qshare_plan_sizes_dense_sparse_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = EfficientQShareChunkedDSAProviderBundle(
        qshare_group_size=4,
        q_indexer_dim=8,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=8,
        logit_scale=1.0,
        chunk_top_k=2,
    )
    sampled_state = EfficientQShareState(
        sampled_q=torch.zeros(3, 1, 8),
        original_query_start_loc=torch.tensor([0, 12], dtype=torch.int32),
        original_query_start_loc_cpu=torch.tensor([0, 12], dtype=torch.int32),
        sampled_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        sampled_query_start_loc_cpu=torch.tensor([0, 3], dtype=torch.int32),
        sampled_query_lengths=torch.tensor([3], dtype=torch.int32),
        sampled_to_sequence=torch.zeros(3, dtype=torch.int32),
        original_to_sampled=torch.repeat_interleave(torch.arange(3), 4).to(torch.int32),
        sampled_to_original_start=torch.tensor([0, 4, 8], dtype=torch.int32),
        sampled_run_lengths=torch.full((3,), 4, dtype=torch.int32),
        absolute_position_aligned=True,
    )
    selected_blocks = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32)
    selected_counts = torch.full((3,), 2, dtype=torch.int32)
    selection = (
        efficient_qshare_components_module._EfficientBatchedChunkBlockSelections(
            selected_block_indices=selected_blocks,
            selected_block_valid=None,
            selected_block_counts=selected_counts,
            seq_slices={0: (0, 3, 2)},
            chunk_top_k_by_seq={0: 2},
            row_seq_ids=torch.zeros(3, dtype=torch.int32),
            row_current_chunks=torch.tensor([7, 8, 9], dtype=torch.int32),
            row_tail_lens=torch.ones(3, dtype=torch.int32),
            per_seq={},
        )
    )
    calls = []

    def fake_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(3, 8, dtype=torch.int32),
            torch.tensor([32, 12, 12], dtype=torch.int32),
            torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_qshare_components_module,
        "qshare_batched_page_table_triton",
        fake_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=torch.arange(16, dtype=torch.int32).view(1, -1),
        active_seq_infos=[(0, 0, 12, 40)],
        block_selection_by_seq=selection,
        selection_query_batch=sampled_state,
        seq_lens=torch.tensor([40], dtype=torch.int32),
        active_seq_count=1,
        dense_decode_threshold=32,
        dense_prefill_threshold=32,
    )

    assert result is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["num_requests"] == 3
    assert call["table_width"] == 8
    assert call["max_sampled_q_len"] == 3
    assert call["qshare_group_size"] == 4
    assert result[3:] == (4, 32)


@pytest.mark.parametrize("group_size", [2, 4, 8, 16])
def test_efficient_qshare_score_metadata_matches_sampled_rows(
    group_size: int,
) -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    query_lengths = (1, 3, 8, 17)
    starts = [0, *itertools.accumulate(query_lengths)]
    key_lens = [length + 20 + seq_idx for seq_idx, length in enumerate(query_lengths)]
    total_sampled_rows = sum(
        (length + group_size - 1) // group_size for length in query_lengths
    )
    state = EfficientMeanQShareProvider(group_size=group_size).cuda()(
        projected_q=torch.randn(starts[-1], 2, 8, device="cuda"),
        query_start_loc=torch.tensor(starts, device="cuda"),
        query_start_loc_cpu=torch.tensor(starts),
        total_sampled_rows=total_sampled_rows,
    )
    metadata = qshare_score_metadata_triton(
        state=state,
        seq_lens=torch.tensor(key_lens, device="cuda"),
        active_seq_count=len(query_lengths),
        representative_group_idx=1,
        chunk_size=4,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )
    assert metadata is not None
    (
        score_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_prior_chunks,
        row_current_chunks,
        row_tail_lens,
    ) = (tensor.cpu() for tensor in metadata)
    expected_seq_ids = state.sampled_to_sequence.cpu().to(torch.int32)
    expected_positions = []
    expected_prior_chunks = []
    for seq_idx, (q_start, key_len) in enumerate(
        zip(starts[:-1], key_lens, strict=True)
    ):
        sampled_start = int(state.sampled_query_start_loc_cpu[seq_idx])
        sampled_end = int(state.sampled_query_start_loc_cpu[seq_idx + 1])
        run_starts = state.sampled_to_original_start[sampled_start:sampled_end].cpu()
        expected_positions.append(
            key_len - query_lengths[seq_idx] + run_starts - q_start
        )
        expected_prior_chunks.extend(
            [math.ceil(key_len / 4) - 1] * (sampled_end - sampled_start)
        )
    expected_positions_t = torch.cat(expected_positions).to(torch.int32)
    expected_current_chunks = expected_positions_t // 4
    torch.testing.assert_close(score_seq_ids, expected_seq_ids)
    torch.testing.assert_close(row_seq_ids, expected_seq_ids)
    torch.testing.assert_close(row_group_ids, torch.ones_like(row_group_ids))
    torch.testing.assert_close(
        row_prior_chunks,
        torch.tensor(expected_prior_chunks, dtype=torch.int32),
    )
    torch.testing.assert_close(row_current_chunks, expected_current_chunks)
    torch.testing.assert_close(
        row_tail_lens,
        expected_positions_t - expected_current_chunks * 4 + 1,
    )


def test_identity_q_provider_preserves_tensor_objects() -> None:
    q = torch.randn(5, 3)
    positions = torch.arange(5)
    chunks = positions // 4
    state = IdentityQProvider()(
        projected_q=q,
        current_chunks=chunks,
        query_positions=positions,
        chunk_size=4,
    )
    assert state.reduced_q is q
    assert state.reduced_current_chunks is chunks


def test_qshare_path_markers_are_environment_gated(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    qshare_components_module._DSA_PATH_DEBUG_COUNTS.clear()
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT", "1")
    qshare_components_module._print_dsa_path_marker("config", provider="test")
    qshare_components_module._print_dsa_path_marker("config", provider="test")
    qshare_components_module._print_dsa_path_marker("sparse_decode")
    output = capsys.readouterr().out
    assert output.count("marker=config") == 1
    assert output.count("marker=sparse_decode") == 1


def test_qshare_sparse_page_table_uses_one_request_per_ragged_run() -> None:
    positions = torch.arange(5, 13, dtype=torch.long)
    current_chunks = positions // 8
    state = MeanQShareProvider(group_size=4)(
        projected_q=torch.randn(8, 3),
        current_chunks=current_chunks,
        query_positions=positions,
        chunk_size=8,
    )
    selection = pytorch_components_module._TorchChunkBlockSelection(
        selected_block_indices=torch.zeros(2, 2, dtype=torch.long),
        selected_block_valid=torch.tensor([[False, False], [True, False]]),
    )
    provider = pytorch_components_module.TorchChunkedDSABlockTableProvider()

    result = provider(
        block_table=torch.tensor([10, 20], dtype=torch.long),
        chunk_size=8,
        key_len=13,
        q_len=8,
        dense=False,
        mode="prefill",
        selection_state=selection,
        current_chunks=current_chunks,
        query_position_start=5,
        selection_query_state=state,
    )
    page_table, request_lens, seqused_k, max_q, max_k = provider.get_page_table(result)

    torch.testing.assert_close(request_lens, torch.tensor([4, 4], dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([9, 13], dtype=torch.int32))
    torch.testing.assert_close(
        page_table,
        torch.tensor([[10, 20, 0, 0], [10, 20, 0, 0]], dtype=torch.int32),
    )
    assert max_q == 4
    assert max_k == 13


def _assert_refactored_mean_qshare_sparse_matches_vanilla(
    *,
    monkeypatch: pytest.MonkeyPatch,
    group_size: int,
    share_vanilla: bool,
    exact: bool,
    device: torch.device | None = None,
    use_page_table: bool = False,
    provider_module: str = "nemotron_h_qshare_dsa_components_pytorch",
    provider_class: str = "TorchQShareMeanChunkedDSAProviderBundle",
) -> None:
    if device is None:
        device = torch.device("cpu")
    monkeypatch.setenv(
        "VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS",
        (f"vllm.model_executor.models.{provider_module}.{provider_class}"),
    )
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_QSHARE_GROUP_SIZE", str(group_size))
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE", "mean")
    monkeypatch.setenv(
        "VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK",
        "1" if share_vanilla else "0",
    )
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE", str(group_size))
    case = {
        "batch_size": 5,
        "q_lens": [1, 3, 4, 7, 9],
        "num_kv_heads": 2,
        "group_size": 2,
        "head_dim": 8,
        "q_indexer_dim": 4,
        "chunk_size": 4,
        "chunk_top_k": 2,
        "query_chunk_size": 64,
        "cache_layout": "NHD",
        "seed": 271828 + group_size,
    }
    if share_vanilla:
        _install_single_rank_tp(monkeypatch)
        config = _make_nemotron_h_dsa_config(case)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(70001 + int(case["seed"]))
            current = nemotron_h_module.NemotronHDSASelectiveAttention(
                config,
                layer_idx=0,
                prefix="test_dsa_attention_vanilla_qshare",
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(90001 + int(case["seed"]))
            refactored = NemotronHDSARefactoredAttention(
                config,
                layer_idx=0,
                prefix="test_dsa_attention_refactored_qshare",
            )
        _configure_dsa_test_flags(
            current,
            use_triton_batched_summaries=False,
        )
        _configure_dsa_test_flags(
            refactored,
            use_triton_batched_summaries=False,
        )
        _initialize_forward_test_weights(
            current,
            seed=1000003 + int(case["seed"]),
        )
        refactored.load_state_dict(current.state_dict(), strict=True)
        current.to(device=device).eval()
        refactored.to(device=device).eval()
    else:
        current, refactored = _make_forward_attn_pair(
            case,
            device=device,
            monkeypatch=monkeypatch,
        )
    if use_page_table:
        current.q_indexer_use_page_table_fa = True
        current.q_indexer_use_prefill_page_table_fa = True
        current.q_indexer_use_shared_prefill_page_table_fa = True
        refactored.q_indexer_use_page_table_fa = True
        refactored.q_indexer_use_prefill_page_table_fa = True
        refactored.q_indexer_use_flattened_prefill_page_table_fa = True
    assert type(refactored.dsa_components).__name__ == provider_class
    assert refactored.dsa_components.qshare_enabled is (group_size > 1)
    if isinstance(
        refactored.dsa_components,
        TorchQShareMeanChunkedDSAProviderBundle,
    ):
        assert refactored.dsa_components.should_prepare_batched_representatives() is (
            group_size > 1
        )
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=case["q_lens"],
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout=case["cache_layout"],
        device=device,
        seed=case["seed"],
    )
    assert max(batch_inputs["key_lens"]) > (case["chunk_size"] * case["chunk_top_k"])
    expected, actual = _run_forward_pair(
        current,
        refactored,
        batch_inputs,
        q_lens=case["q_lens"],
        device=device,
        seed=161803 + group_size,
        monkeypatch=monkeypatch,
        direct_refactored_core=device.type == "cuda",
    )
    if exact:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_refactored_mean_qshare_sparse_matches_vanilla_cpu(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=4,
        share_vanilla=True,
        exact=False,
    )


def test_refactored_mean_qshare_sparse_page_table_matches_vanilla_cuda(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Q-share page-table oracle test")
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=4,
        share_vanilla=True,
        exact=False,
        device=torch.device("cuda"),
        use_page_table=True,
    )


def test_hybrid_efficient_qshare_sparse_page_table_matches_vanilla_cuda(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for hybrid Q-share")
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=4,
        share_vanilla=True,
        exact=False,
        device=torch.device("cuda"),
        use_page_table=True,
        provider_module="nemotron_h_qshare_dsa_components_hybrid",
        provider_class=("TorchQShareEfficientQueryChunkedDSAProviderBundle"),
    )


@pytest.mark.parametrize("group_size", [2, 4])
def test_fully_efficient_qshare_sparse_page_table_matches_vanilla_cuda(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
    group_size: int,
) -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient Q-share")
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=group_size,
        share_vanilla=True,
        exact=False,
        device=torch.device("cuda"),
        use_page_table=True,
        provider_module="nemotron_h_qshare_dsa_components_efficient",
        provider_class="EfficientQShareChunkedDSAProviderBundle",
    )


def test_refactored_qshare_one_hard_bypass_matches_existing_path_cpu(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=1,
        share_vanilla=False,
        exact=True,
    )


def test_refactored_qshare_one_hard_bypass_matches_existing_path_cuda(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Q-share-one invariant test")
    _assert_refactored_mean_qshare_sparse_matches_vanilla(
        monkeypatch=monkeypatch,
        group_size=1,
        share_vanilla=False,
        exact=True,
        device=torch.device("cuda"),
    )


def _large_mixed_decode_mtp_prefill_q_lens(
    batch_size: int,
    *,
    profile: tuple[int, ...],
    shuffle_seed: int,
) -> list[int]:
    q_lens = [profile[i % len(profile)] for i in range(batch_size)]
    random.Random(shuffle_seed).shuffle(q_lens)
    if q_lens == sorted(q_lens) or q_lens == sorted(q_lens, reverse=True):
        q_lens = q_lens[1:] + q_lens[:1]
    return q_lens


def _assert_large_mixed_decode_mtp_prefill_batch(q_lens: list[int]) -> None:
    assert 1 in q_lens
    assert 4 in q_lens
    assert sum(q_len >= 64 for q_len in q_lens) >= 2
    assert q_lens != sorted(q_lens)
    assert q_lens != sorted(q_lens, reverse=True)
    is_prefill = [q_len > 4 for q_len in q_lens]
    assert any(
        previous != current for previous, current in zip(is_prefill, is_prefill[1:])
    )


def _assert_chunked_sparse_path_is_configured(attn) -> None:
    assert not getattr(attn, "q_indexer_use_full_attention_short_seq", False)
    assert attn.q_indexer_chunk_top_k > 0
    assert attn.q_indexer_chunk_size > 0


def _assert_large_mixed_exceeds_dense_budget(
    attn,
    *,
    q_lens: list[int],
    key_lens: list[int],
) -> None:
    dense_eligible = []
    sparse_required = []
    prefill_sparse_required = []
    for seq_idx, (q_len, key_len) in enumerate(zip(q_lens, key_lens)):
        budget = attn.q_indexer_chunk_size * attn.q_indexer_chunk_top_k
        if key_len <= budget:
            dense_eligible.append(seq_idx)
        else:
            sparse_required.append(seq_idx)
            if q_len > 4:
                prefill_sparse_required.append(seq_idx)

    assert sparse_required, (
        "large mixed batch did not include any sequence longer than the "
        "dense attention budget"
    )
    assert len(sparse_required) >= 2
    assert prefill_sparse_required, (
        "large mixed batch did not include a prefill sequence longer than "
        "the dense attention budget"
    )
    assert len(dense_eligible) < len(q_lens)


def _install_single_rank_tp(monkeypatch: pytest.MonkeyPatch) -> None:
    def dispatch_unquantized_gemm():

        def apply(_layer, inputs, weight, bias):
            return torch.nn.functional.linear(inputs, weight, bias)

        return apply

    monkeypatch.setattr(
        linear_module,
        "dispatch_unquantized_gemm",
        dispatch_unquantized_gemm,
    )
    for module in (
        nemotron_h_module,
        linear_module,
        parameter_module,
        refactored_attention_module,
    ):
        if hasattr(module, "get_tensor_model_parallel_world_size"):
            monkeypatch.setattr(
                module,
                "get_tensor_model_parallel_world_size",
                lambda: 1,
            )
        if hasattr(module, "get_tensor_model_parallel_rank"):
            monkeypatch.setattr(
                module,
                "get_tensor_model_parallel_rank",
                lambda: 0,
            )


def _make_nemotron_h_dsa_config(case: dict[str, object]) -> NemotronHConfig:
    hidden_size = case["num_kv_heads"] * case["group_size"] * case["head_dim"]
    return NemotronHConfig(
        hidden_size=hidden_size,
        num_attention_heads=case["num_kv_heads"] * case["group_size"],
        num_key_value_heads=case["num_kv_heads"],
        head_dim=case["head_dim"],
        num_hidden_layers=1,
        hybrid_override_pattern="*",
        q_indexer_dim=case["q_indexer_dim"],
        q_indexer_attn_mode="chunked_topk_sparse",
        q_indexer_logit_scale=1.0,
        q_indexer_top_k=case["chunk_size"] * case["chunk_top_k"],
        q_indexer_chunk_size=case["chunk_size"],
        q_indexer_chunk_top_k=case["chunk_top_k"],
        q_indexer_chunked_query_chunk_size=case["query_chunk_size"],
    )


def test_dsa_chunk_top_k_env_override(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
) -> None:
    _install_single_rank_tp(monkeypatch)
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K", "1024")
    case = {
        "num_kv_heads": 1,
        "group_size": 1,
        "head_dim": 8,
        "q_indexer_dim": 4,
        "chunk_size": 16,
        "chunk_top_k": 128,
        "query_chunk_size": 4,
    }
    config = _make_nemotron_h_dsa_config(case)

    baseline = NemotronHDSASelectiveAttention(
        config,
        layer_idx=0,
        prefix="test_dsa_attention_moonshot_env_override",
    )
    refactored = NemotronHDSARefactoredAttention(
        config,
        layer_idx=0,
        prefix="test_dsa_attention_refactored_env_override",
    )

    assert baseline.q_indexer_chunk_top_k == 1024
    assert refactored.q_indexer_chunk_top_k == 1024


def _configure_dsa_test_flags(
    attn,
    *,
    use_triton_batched_summaries: bool,
) -> None:
    attn.q_indexer_use_page_table_fa = False
    attn.q_indexer_use_prefill_page_table_fa = False
    attn.q_indexer_use_flattened_prefill_page_table_fa = False
    attn.q_indexer_use_flattened_decode_page_table_fa = False
    attn.q_indexer_use_full_attention_short_seq = False
    attn.q_indexer_dense_prefill_kv_threshold_tokens = (
        attn.q_indexer_chunk_size * attn.q_indexer_chunk_top_k
    )
    attn.q_indexer_use_triton_batched_summaries = use_triton_batched_summaries


def _initialize_forward_test_weights(
    attn: torch.nn.Module,
    *,
    seed: int,
) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        for param in attn.parameters():
            values = torch.randn(
                param.shape,
                device="cpu",
                dtype=torch.float32,
                generator=generator,
            )
            values.mul_(0.02)
            param.copy_(values.to(device=param.device, dtype=param.dtype))


def _make_forward_attn_pair(
    case: dict[str, object],
    *,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    use_triton_batched_summaries: bool = False,
) -> tuple[NemotronHDSASelectiveAttention, NemotronHDSARefactoredAttention]:
    _install_single_rank_tp(monkeypatch)
    config = _make_nemotron_h_dsa_config(case)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(70001 + int(case["seed"]))
        baseline = NemotronHDSASelectiveAttention(
            config,
            layer_idx=0,
            prefix="test_dsa_attention_moonshot_baseline",
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(90001 + int(case["seed"]))
        refactored = NemotronHDSARefactoredAttention(
            config,
            layer_idx=0,
            prefix="test_dsa_attention_refactored",
        )
    _configure_dsa_test_flags(
        baseline,
        use_triton_batched_summaries=use_triton_batched_summaries,
    )
    _configure_dsa_test_flags(
        refactored,
        use_triton_batched_summaries=use_triton_batched_summaries,
    )
    _initialize_forward_test_weights(
        baseline,
        seed=1000003 + int(case["seed"]),
    )
    refactored.load_state_dict(baseline.state_dict(), strict=True)
    baseline.to(device=device)
    refactored.to(device=device)
    baseline.eval()
    refactored.eval()
    return baseline, refactored


def _make_forward_metadata(
    batch_inputs: dict[str, object],
    *,
    q_lens: list[int],
    device: torch.device,
):
    query_starts = [0]
    for q_len in q_lens:
        query_starts.append(query_starts[-1] + q_len)
    query_start_loc = torch.tensor(
        query_starts,
        device=device,
        dtype=torch.long,
    )
    seq_lens = torch.tensor(
        batch_inputs["key_lens"],
        device=device,
        dtype=torch.long,
    )
    return SimpleNamespace(
        num_actual_tokens=query_starts[-1],
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        block_table=batch_inputs["block_table"],
        use_cascade=False,
        dcp_context_kv_lens=None,
    )


def _install_forward_attention_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata,
    kv_cache: torch.Tensor,
) -> None:

    def get_attention_context(_layer_name):
        return metadata, None, kv_cache, None

    def unified_kv_cache_update(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        _layer_name: str,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)
        num_kv_heads = int(key_states.shape[1])
        if key_cache.shape[2] == num_kv_heads:
            block_size = int(key_cache.shape[1])
            cache_layout = "NHD"
        elif key_cache.shape[1] == num_kv_heads:
            block_size = int(key_cache.shape[2])
            cache_layout = "HND"
        else:
            raise AssertionError(f"unexpected key cache shape: {key_cache.shape}")

        query_start_loc = metadata.query_start_loc
        seq_lens = metadata.seq_lens
        block_table = metadata.block_table
        for seq_idx in range(query_start_loc.numel() - 1):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            key_len = int(seq_lens[seq_idx].item())
            q_len = q_end - q_start
            position_start = key_len - q_len
            for row_idx, token_pos in enumerate(range(position_start, key_len)):
                block_idx = token_pos // block_size
                block_offset = token_pos % block_size
                block_id = int(block_table[seq_idx, block_idx].item())
                source_idx = q_start + row_idx
                if cache_layout == "NHD":
                    key_cache[block_id, block_offset] = key_states[source_idx]
                    value_cache[block_id, block_offset] = value_states[source_idx]
                else:
                    key_cache[block_id, :, block_offset] = key_states[source_idx]
                    value_cache[block_id, :, block_offset] = value_states[source_idx]

    monkeypatch.setattr(
        nemotron_h_module,
        "get_attention_context",
        get_attention_context,
    )
    monkeypatch.setattr(
        refactored_attention_module,
        "get_attention_context",
        get_attention_context,
    )
    monkeypatch.setattr(
        nemotron_h_module,
        "unified_kv_cache_update",
        unified_kv_cache_update,
    )
    monkeypatch.setattr(
        refactored_attention_module,
        "unified_kv_cache_update",
        unified_kv_cache_update,
    )


def _run_forward_pair(
    baseline: NemotronHDSASelectiveAttention,
    refactored: NemotronHDSARefactoredAttention,
    batch_inputs: dict[str, object],
    *,
    q_lens: list[int],
    device: torch.device,
    seed: int,
    monkeypatch: pytest.MonkeyPatch,
    direct_refactored_core: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = baseline.num_heads * baseline.head_dim
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    hidden_states = torch.randn(
        sum(q_lens),
        hidden_size,
        device=device,
        generator=generator,
    )
    positions = torch.cat(batch_inputs["positions_by_seq"], dim=0)
    kv_cache = torch.stack(
        (
            batch_inputs["key_cache"],
            batch_inputs["value_cache"],
        ),
        dim=0,
    )
    metadata = _make_forward_metadata(
        batch_inputs,
        q_lens=q_lens,
        device=device,
    )
    _install_forward_attention_context(
        monkeypatch,
        metadata=metadata,
        kv_cache=kv_cache,
    )
    with torch.no_grad():
        expected = baseline.forward(hidden_states, positions)
        if direct_refactored_core:
            qkv, _ = refactored.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [refactored.q_size, refactored.kv_size, refactored.kv_size],
                dim=-1,
            )
            indexer_q, _ = refactored.indexer_q_proj(hidden_states)
            indexer_q = indexer_q.view(
                -1,
                refactored.total_num_kv_heads,
                refactored.q_indexer_dim,
            ).index_select(
                1,
                refactored._local_kv_head_indices.to(indexer_q.device),
            )
            attn_output = torch.empty_like(q)
            refactored._forward_dsa_attention_with_output(
                hidden_states=hidden_states,
                query_states=q,
                key_states=k,
                value_states=v,
                output=attn_output,
                positions=positions,
                indexer_q=indexer_q,
            )
            actual, _ = refactored.o_proj(attn_output)
        else:
            actual = refactored.forward(hidden_states, positions)
    return expected, actual


def _physical_block_table(
    *,
    batch_size: int,
    max_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for seq_idx in range(batch_size):
        row = torch.arange(
            seq_idx * max_blocks,
            (seq_idx + 1) * max_blocks,
            device=device,
            dtype=torch.long,
        )
        if max_blocks > 1:
            row = torch.roll(row, shifts=(2 * seq_idx + 1) % max_blocks)
        if seq_idx % 2 == 1:
            row = torch.flip(row, dims=(0,))
        rows.append(row)
    return torch.stack(rows)


def _pack_nhd_cache(
    states_by_seq: list[torch.Tensor],
    *,
    block_size: int,
    block_table: torch.Tensor,
) -> torch.Tensor:
    first = states_by_seq[0]
    num_physical_blocks = int(block_table.max().item()) + 1
    cache = first.new_zeros(
        num_physical_blocks,
        block_size,
        first.shape[1],
        first.shape[2],
    )
    for seq_idx, states in enumerate(states_by_seq):
        for token_idx in range(states.shape[0]):
            block_id = int(block_table[seq_idx, token_idx // block_size].item())
            cache[block_id, token_idx % block_size] = states[token_idx]
    return cache


def _key_len_for_query(
    *,
    query_len: int,
    seq_idx: int,
    chunk_size: int,
) -> int:
    prior_chunks = 1 + (seq_idx % 5)
    tail = (3 * seq_idx + query_len) % chunk_size
    return max(query_len, prior_chunks * chunk_size + tail + query_len // 2)


def _make_batch_inputs(
    *,
    batch_size: int,
    q_lens: list[int],
    num_kv_heads: int,
    num_heads: int,
    head_dim: int,
    chunk_size: int,
    q_indexer_dim: int,
    cache_layout: str,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    key_lens = [
        _key_len_for_query(
            query_len=q_len,
            seq_idx=seq_idx,
            chunk_size=chunk_size,
        )
        for seq_idx, q_len in enumerate(q_lens)
    ]
    max_blocks = max(math.ceil(key_len / chunk_size) for key_len in key_lens)
    block_table = _physical_block_table(
        batch_size=batch_size,
        max_blocks=max_blocks,
        device=device,
    )

    key_states_by_seq = [
        torch.randn(
            key_len,
            num_kv_heads,
            head_dim,
            device=device,
            generator=generator,
        )
        for key_len in key_lens
    ]
    value_states_by_seq = [
        torch.randn(
            key_len,
            num_kv_heads,
            head_dim,
            device=device,
            generator=generator,
        )
        for key_len in key_lens
    ]
    query_states_by_seq = [
        torch.randn(
            q_len,
            num_heads,
            head_dim,
            device=device,
            generator=generator,
        )
        for q_len in q_lens
    ]
    indexer_query_states_by_seq = [
        torch.randn(
            q_len,
            num_kv_heads,
            q_indexer_dim,
            device=device,
            generator=generator,
        )
        for q_len in q_lens
    ]
    positions_by_seq = [
        torch.arange(
            key_len - q_len,
            key_len,
            device=device,
            dtype=torch.long,
        )
        for key_len, q_len in zip(key_lens, q_lens)
    ]

    key_cache = _pack_nhd_cache(
        key_states_by_seq,
        block_size=chunk_size,
        block_table=block_table,
    )
    value_cache = _pack_nhd_cache(
        value_states_by_seq,
        block_size=chunk_size,
        block_table=block_table,
    )
    if cache_layout == "HND":
        key_cache = key_cache.permute(0, 2, 1, 3).contiguous()
        value_cache = value_cache.permute(0, 2, 1, 3).contiguous()
    elif cache_layout != "NHD":
        raise ValueError(f"unknown cache layout: {cache_layout}")

    return {
        "block_table": block_table,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "key_lens": key_lens,
        "query_states_by_seq": query_states_by_seq,
        "indexer_query_states_by_seq": indexer_query_states_by_seq,
        "positions_by_seq": positions_by_seq,
    }


def _pytorch_cases():
    batch_sizes = (1, 2, 4, 8)
    cache_layouts = ("NHD", "HND")
    for case_idx, (
        batch_size,
        (pattern_name, pattern_fn),
        (num_kv_heads, group_size, head_dim, q_indexer_dim),
        (chunk_size, chunk_top_k, query_chunk_size),
        cache_layout,
    ) in enumerate(
        itertools.product(
            batch_sizes,
            _QUERY_PATTERNS,
            _HEAD_CASES,
            _CHUNK_CASES,
            cache_layouts,
        )
    ):
        yield pytest.param(
            {
                "batch_size": batch_size,
                "q_lens": pattern_fn(batch_size),
                "num_kv_heads": num_kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
                "q_indexer_dim": q_indexer_dim,
                "chunk_size": chunk_size,
                "chunk_top_k": chunk_top_k,
                "query_chunk_size": query_chunk_size,
                "cache_layout": cache_layout,
                "seed": 1309 + case_idx,
            },
            id=(
                f"bs{batch_size}-{pattern_name}-kv{num_kv_heads}"
                f"-gqa{group_size}-chunk{chunk_size}-top{chunk_top_k}"
                f"-qchunk{query_chunk_size}-{cache_layout}"
            ),
        )


def _efficient_cases():
    batch_sizes = (1, 3, 6)
    patterns = (_QUERY_PATTERNS[0], _QUERY_PATTERNS[1], _QUERY_PATTERNS[4])
    head_cases = (_HEAD_CASES[0], _HEAD_CASES[2], _HEAD_CASES[4])
    for case_idx, (
        batch_size,
        (pattern_name, pattern_fn),
        (num_kv_heads, group_size, head_dim, q_indexer_dim),
        (chunk_size, chunk_top_k, query_chunk_size),
    ) in enumerate(
        itertools.product(
            batch_sizes,
            patterns,
            head_cases,
            _CHUNK_CASES,
        )
    ):
        yield pytest.param(
            {
                "batch_size": batch_size,
                "q_lens": pattern_fn(batch_size),
                "num_kv_heads": num_kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
                "q_indexer_dim": q_indexer_dim,
                "chunk_size": chunk_size,
                "chunk_top_k": chunk_top_k,
                "query_chunk_size": query_chunk_size,
                "cache_layout": "NHD",
                "seed": 9157 + case_idx,
            },
            id=(
                f"bs{batch_size}-{pattern_name}-kv{num_kv_heads}"
                f"-gqa{group_size}-chunk{chunk_size}-top{chunk_top_k}"
            ),
        )


def _large_mixed_pytorch_cases():
    cache_layouts = ("NHD", "HND")
    for case_idx, (
        batch_size,
        (profile_name, profile),
        (num_kv_heads, group_size, head_dim, q_indexer_dim),
        (chunk_size, chunk_top_k, query_chunk_size),
        cache_layout,
    ) in enumerate(
        itertools.product(
            _LARGE_MIXED_BATCH_SIZES,
            _LARGE_MIXED_QUERY_PROFILES,
            _LARGE_MIXED_HEAD_CASES,
            _LARGE_MIXED_CHUNK_CASES,
            cache_layouts,
        )
    ):
        shuffle_seed = 23117 + case_idx
        yield pytest.param(
            {
                "batch_size": batch_size,
                "q_lens": _large_mixed_decode_mtp_prefill_q_lens(
                    batch_size,
                    profile=profile,
                    shuffle_seed=shuffle_seed,
                ),
                "profile_name": profile_name,
                "num_kv_heads": num_kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
                "q_indexer_dim": q_indexer_dim,
                "chunk_size": chunk_size,
                "chunk_top_k": chunk_top_k,
                "query_chunk_size": query_chunk_size,
                "cache_layout": cache_layout,
                "seed": 53117 + case_idx,
            },
            id=(
                f"bs{batch_size}-{profile_name}"
                f"-kv{num_kv_heads}-gqa{group_size}-chunk{chunk_size}"
                f"-top{chunk_top_k}-{cache_layout}"
            ),
        )


def _large_mixed_efficient_cases():
    for case_idx, (
        batch_size,
        (profile_name, profile),
        (num_kv_heads, group_size, head_dim, q_indexer_dim),
        (chunk_size, chunk_top_k, query_chunk_size),
    ) in enumerate(
        itertools.product(
            _LARGE_MIXED_BATCH_SIZES,
            _LARGE_MIXED_QUERY_PROFILES,
            _LARGE_MIXED_HEAD_CASES,
            _LARGE_MIXED_CHUNK_CASES,
        )
    ):
        shuffle_seed = 31991 + case_idx
        yield pytest.param(
            {
                "batch_size": batch_size,
                "q_lens": _large_mixed_decode_mtp_prefill_q_lens(
                    batch_size,
                    profile=profile,
                    shuffle_seed=shuffle_seed,
                ),
                "profile_name": profile_name,
                "num_kv_heads": num_kv_heads,
                "group_size": group_size,
                "head_dim": head_dim,
                "q_indexer_dim": q_indexer_dim,
                "chunk_size": chunk_size,
                "chunk_top_k": chunk_top_k,
                "query_chunk_size": query_chunk_size,
                "cache_layout": "NHD",
                "seed": 61991 + case_idx,
            },
            id=(
                f"bs{batch_size}-{profile_name}"
                f"-kv{num_kv_heads}-gqa{group_size}-chunk{chunk_size}"
                f"-top{chunk_top_k}"
            ),
        )


@pytest.mark.parametrize("case", list(_pytorch_cases()))
def test_refactored_attention_forward_matches_moonshot_with_pytorch_representatives(
    case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    device = torch.device("cpu")
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
    )
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=case["q_lens"],
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout=case["cache_layout"],
        device=device,
        seed=case["seed"],
    )
    _assert_chunked_sparse_path_is_configured(baseline)
    _assert_chunked_sparse_path_is_configured(refactored)

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=case["q_lens"],
        device=device,
        seed=110003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("case", list(_large_mixed_pytorch_cases()))
def test_refactored_attention_forward_matches_moonshot_with_large_mixed_pytorch_batch(
    case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    q_lens = case["q_lens"]
    _assert_large_mixed_decode_mtp_prefill_batch(q_lens)

    device = torch.device("cpu")
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
    )
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=q_lens,
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout=case["cache_layout"],
        device=device,
        seed=case["seed"],
    )
    _assert_chunked_sparse_path_is_configured(baseline)
    _assert_chunked_sparse_path_is_configured(refactored)
    _assert_large_mixed_exceeds_dense_budget(
        baseline,
        q_lens=q_lens,
        key_lens=batch_inputs["key_lens"],
    )

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=q_lens,
        device=device,
        seed=120003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _assert_refactored_nonchunked_forward_matches_original_chunked_full_recall(
    *,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS",
        (
            "vllm.model_executor.models."
            "nemotron_h_nonchunked_dsa_components_pytorch."
            "TorchNonChunkedDSAProviderBundle"
        ),
    )
    case = {
        "batch_size": 4,
        "q_lens": [1, 4, 9, 17],
        "num_kv_heads": 2,
        "group_size": 2,
        "head_dim": 8,
        "q_indexer_dim": 4,
        "chunk_size": 4,
        "chunk_top_k": 16,
        "query_chunk_size": 5,
        "cache_layout": "NHD",
        "seed": 271828,
    }
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
    )
    assert isinstance(refactored.dsa_components, TorchNonChunkedDSAProviderBundle)
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=case["q_lens"],
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout=case["cache_layout"],
        device=device,
        seed=case["seed"],
    )
    assert max(batch_inputs["key_lens"]) <= (case["chunk_size"] * case["chunk_top_k"])

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=case["q_lens"],
        device=device,
        seed=150003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_refactored_nonchunked_forward_matches_original_chunked_full_recall_cpu(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    _assert_refactored_nonchunked_forward_matches_original_chunked_full_recall(
        device=torch.device("cpu"),
        monkeypatch=monkeypatch,
    )


def test_refactored_nonchunked_forward_matches_original_chunked_full_recall_cuda(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the non-chunked forward oracle test")
    _assert_refactored_nonchunked_forward_matches_original_chunked_full_recall(
        device=torch.device("cuda"),
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("case", list(_efficient_cases()))
def test_efficient_attention_forward_matches_moonshot_representatives(
    case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient DSA parity")

    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS", "efficient")
    device = torch.device("cuda")
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
        use_triton_batched_summaries=True,
    )
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=case["q_lens"],
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout="NHD",
        device=device,
        seed=case["seed"],
    )
    assert isinstance(refactored.dsa_components, EfficientChunkedDSAProviderBundle)
    _assert_chunked_sparse_path_is_configured(baseline)
    _assert_chunked_sparse_path_is_configured(refactored)

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=case["q_lens"],
        device=device,
        seed=130003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_efficient_attention_decode_forward_matches_moonshot_with_page_table(
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient DSA parity")

    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS", "efficient")
    monkeypatch.setattr(
        pytorch_components_module,
        "_get_dsa_kv_cache_layout",
        lambda: "NHD",
    )
    case = {
        "batch_size": 3,
        "q_lens": [1, 1, 1],
        "num_kv_heads": 1,
        "group_size": 2,
        "head_dim": 16,
        "q_indexer_dim": 4,
        "chunk_size": 4,
        "chunk_top_k": 3,
        "query_chunk_size": 1,
        "cache_layout": "NHD",
        "seed": 424242,
    }
    device = torch.device("cuda")
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
        use_triton_batched_summaries=True,
    )
    assert isinstance(refactored.dsa_components, EfficientChunkedDSAProviderBundle)
    refactored.dsa_components.q_indexer_use_page_table_fa = True
    refactored.dsa_components.q_indexer_use_prefill_page_table_fa = False
    refactored.dsa_components.q_indexer_use_flattened_prefill_page_table_fa = False
    refactored.dsa_components.q_indexer_use_flattened_decode_page_table_fa = True

    flash_attn_calls = 0

    def fake_flash_attn_varlen_func(
        *,
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        block_table,
        **kwargs,
    ):
        nonlocal flash_attn_calls
        assert max_seqlen_q == 1
        assert dropout_p == 0.0
        assert causal is False
        assert cu_seqlens_q.tolist() == [0, 1]
        assert k.shape[2] == 1
        assert v.shape[2] == 1
        assert max_seqlen_k >= int(seqused_k.max().item())
        for row in range(q.shape[0]):
            recalled_tokens = int(seqused_k[row].item())
            selected_k = []
            selected_v = []
            for token_idx in range(recalled_tokens):
                page_idx = token_idx // case["chunk_size"]
                page_offset = token_idx % case["chunk_size"]
                block_id = int(block_table[row, page_idx].item())
                selected_k.append(k[block_id, page_offset, 0])
                selected_v.append(v[block_id, page_offset, 0])
            selected_k_t = torch.stack(selected_k)
            selected_v_t = torch.stack(selected_v)
            logits = torch.einsum(
                "hd,kd->hk",
                q[row].float(),
                selected_k_t.float(),
            )
            weights = torch.softmax(logits * softmax_scale, dim=-1).to(q.dtype)
            out[row].copy_(torch.einsum("hk,kd->hd", weights, selected_v_t))
        flash_attn_calls += 1
        return out

    monkeypatch.setattr(
        nemotron_h_module,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )

    flattened_decode_calls = 0
    components = refactored.dsa_components
    original_flattened_decode = (
        components._forward_dsa_chunked_flattened_decode_page_table_fa_sequence
    )

    def wrapped_flattened_decode(**kwargs):
        nonlocal flattened_decode_calls
        result = original_flattened_decode(**kwargs)
        if result is not None:
            flattened_decode_calls += 1
        return result

    object.__setattr__(
        components,
        "_forward_dsa_chunked_flattened_decode_page_table_fa_sequence",
        wrapped_flattened_decode,
    )

    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=case["q_lens"],
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout="NHD",
        device=device,
        seed=case["seed"],
    )
    _assert_chunked_sparse_path_is_configured(baseline)
    _assert_chunked_sparse_path_is_configured(refactored)

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=case["q_lens"],
        device=device,
        seed=160003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    assert flattened_decode_calls == case["batch_size"]
    assert flash_attn_calls == case["batch_size"]
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("case", list(_large_mixed_efficient_cases()))
def test_efficient_attention_forward_matches_moonshot_with_large_mixed_batch(
    case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    default_vllm_config,
):
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient DSA parity")

    q_lens = case["q_lens"]
    _assert_large_mixed_decode_mtp_prefill_batch(q_lens)

    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS", "efficient")
    device = torch.device("cuda")
    baseline, refactored = _make_forward_attn_pair(
        case,
        device=device,
        monkeypatch=monkeypatch,
        use_triton_batched_summaries=True,
    )
    batch_inputs = _make_batch_inputs(
        batch_size=case["batch_size"],
        q_lens=q_lens,
        num_kv_heads=case["num_kv_heads"],
        num_heads=case["num_kv_heads"] * case["group_size"],
        head_dim=case["head_dim"],
        chunk_size=case["chunk_size"],
        q_indexer_dim=case["q_indexer_dim"],
        cache_layout="NHD",
        device=device,
        seed=case["seed"],
    )
    assert isinstance(refactored.dsa_components, EfficientChunkedDSAProviderBundle)
    _assert_chunked_sparse_path_is_configured(baseline)
    _assert_chunked_sparse_path_is_configured(refactored)
    _assert_large_mixed_exceeds_dense_budget(
        baseline,
        q_lens=q_lens,
        key_lens=batch_inputs["key_lens"],
    )

    expected, actual = _run_forward_pair(
        baseline,
        refactored,
        batch_inputs,
        q_lens=q_lens,
        device=device,
        seed=140003 + int(case["seed"]),
        monkeypatch=monkeypatch,
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
