# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models import (
    nemotron_h_chunked_dsa_components_efficient as efficient_components,
)
from vllm.model_executor.models import (
    nemotron_h_dsa_triton_decode_page_table as decode_page_table,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    TorchChunkedDSABlockTableProvider,
    _TorchChunkBlockSelection,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_qshare import (
    HAS_TRITON,
    EfficientQShareState,
    qshare_batched_page_table_triton,
)


def _require_triton_cuda() -> None:
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for page-table tests")


def test_reference_sparse_page_table_compacts_remote_before_recent_window() -> None:
    provider = TorchChunkedDSABlockTableProvider()
    selection = _TorchChunkBlockSelection(
        selected_block_indices=torch.tensor([[2, 0]], dtype=torch.long),
        selected_block_valid=torch.tensor([[False, True]]),
    )

    result = provider(
        block_table=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        chunk_size=4,
        key_len=15,
        q_len=1,
        selection_state=selection,
        current_chunks=torch.tensor([3], dtype=torch.long),
        query_position_start=14,
        recent_window_pages=2,
    )

    assert provider.is_available(result)
    plan = provider.get_page_table(result)
    assert plan is not None
    page_table, request_lens, seqused_k, max_q, max_k = plan
    torch.testing.assert_close(
        page_table,
        torch.tensor([[10, 11, 12, 13, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(request_lens, torch.ones(1, dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([15], dtype=torch.int32))
    assert max_q == 1
    assert max_k >= 15


def test_efficient_sparse_page_table_handles_window_larger_than_history() -> None:
    provider = efficient_components.EfficientChunkedDSABlockTableProvider()
    selection = efficient_components._EfficientChunkBlockSelection(
        selected_block_indices=torch.empty(1, 0, dtype=torch.long),
        selected_block_counts=torch.zeros(1, dtype=torch.int32),
    )

    result = provider(
        block_table=torch.tensor([10, 11], dtype=torch.int32),
        chunk_size=4,
        key_len=6,
        q_len=1,
        selection_state=selection,
        current_chunks=torch.tensor([1], dtype=torch.long),
        query_position_start=5,
        recent_window_pages=2,
    )

    assert provider.is_available(result)
    plan = provider.get_page_table(result)
    assert plan is not None
    page_table, request_lens, seqused_k, max_q, max_k = plan
    torch.testing.assert_close(
        page_table,
        torch.tensor([[10, 11, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(request_lens, torch.ones(1, dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([6], dtype=torch.int32))
    assert max_q == 1
    assert max_k >= 6


def test_efficient_bundle_forwards_recent_window_to_sequence_table_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_RECENT_WINDOW_PAGES", "2")
    bundle = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=1,
    )
    selection = efficient_components._EfficientChunkBlockSelection(
        selected_block_indices=torch.tensor([[0]], dtype=torch.long),
        selected_block_counts=torch.ones(1, dtype=torch.int32),
    )

    plan = bundle.build_page_table_plan(
        block_table=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        chunk_size=4,
        key_len=15,
        q_len=1,
        selection_state=selection,
        current_chunks=torch.tensor([3], dtype=torch.long),
        query_position_start=14,
    )

    assert plan is not None
    page_table, _, seqused_k, _, _ = plan
    torch.testing.assert_close(
        page_table,
        torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
    )
    torch.testing.assert_close(seqused_k, torch.tensor([15], dtype=torch.int32))


def test_decode_page_table_appends_disjoint_recent_window() -> None:
    _require_triton_cuda()
    block_table = torch.tensor(
        [101, 103, 107, 109, 113, 127],
        device="cuda",
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [[0, 2, 3, 4]],
        device="cuda",
        dtype=torch.long,
    )
    selected_valid = torch.ones_like(selected_blocks, dtype=torch.bool)

    actual = decode_page_table.dsa_decode_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_valid=selected_valid,
        current_chunk=5,
        chunk_size=4,
        tail_len=2,
        recent_window_pages=2,
    )

    assert actual is not None
    page_table, seqused_k = actual
    torch.testing.assert_close(
        page_table,
        torch.tensor(
            [[101, 107, 109, 113, 127, 0, 0]],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([18], device="cuda", dtype=torch.int32),
    )


def test_unified_page_table_appends_recent_window_per_row() -> None:
    _require_triton_cuda()
    block_table = torch.tensor(
        [[101, 103, 107, 109, 113, 127]],
        device="cuda",
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [[0, 0], [0, 0]],
        device="cuda",
        dtype=torch.int32,
    )
    selected_counts = torch.tensor([1, 1], device="cuda", dtype=torch.int32)

    actual = decode_page_table.dsa_batched_unified_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        query_start_loc=torch.tensor([0, 2], device="cuda", dtype=torch.int32),
        seq_lens=torch.tensor([24], device="cuda", dtype=torch.int32),
        num_actual_tokens=2,
        active_seq_count=1,
        num_requests=2,
        table_width=5,
        max_q_len=2,
        chunk_size=4,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
        recent_window_pages=2,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    torch.testing.assert_close(
        page_table,
        torch.tensor(
            [[101, 109, 113, 127, 0], [101, 109, 113, 127, 0]],
            device="cuda",
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([15, 16], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        cu_seqlens_q,
        torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32),
    )


def test_qshare_page_table_places_recent_before_local_pages() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for Q-share page-table tests")
    device = torch.device("cuda")
    original_query_start_loc = torch.tensor([0, 2], device=device, dtype=torch.int32)
    sampled_query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
    state = EfficientQShareState(
        sampled_q=torch.empty(1, 1, 1, device=device),
        original_query_start_loc=original_query_start_loc,
        original_query_start_loc_cpu=original_query_start_loc.cpu(),
        sampled_query_start_loc=sampled_query_start_loc,
        sampled_query_start_loc_cpu=sampled_query_start_loc.cpu(),
        sampled_query_lengths=torch.tensor([1], device=device, dtype=torch.int32),
        sampled_to_sequence=torch.tensor([0], device=device, dtype=torch.int32),
        original_to_sampled=torch.tensor([0, 0], device=device, dtype=torch.int32),
        sampled_to_original_start=torch.tensor([0], device=device, dtype=torch.int32),
        sampled_run_lengths=torch.tensor([2], device=device, dtype=torch.int32),
    )

    actual = qshare_batched_page_table_triton(
        block_table=torch.tensor(
            [[101, 103, 107, 109, 113, 127]],
            device=device,
            dtype=torch.int32,
        ),
        selected_blocks=torch.tensor([[0]], device=device, dtype=torch.int32),
        selected_counts=torch.tensor([1], device=device, dtype=torch.int32),
        state=state,
        seq_lens=torch.tensor([24], device=device, dtype=torch.int32),
        active_seq_count=1,
        num_requests=1,
        table_width=4,
        max_sampled_q_len=1,
        chunk_size=4,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
        recent_window_pages=2,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    torch.testing.assert_close(
        page_table,
        torch.tensor([[101, 109, 113, 127]], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([16], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        cu_seqlens_q,
        torch.tensor([0, 2], device=device, dtype=torch.int32),
    )
