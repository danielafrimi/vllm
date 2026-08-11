# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import (
    nemotron_h_chunked_dsa_components_efficient as efficient_components,
)
from vllm.model_executor.models import (
    nemotron_h_chunked_dsa_components_pytorch as pytorch_components,
)
from vllm.model_executor.models import (
    nemotron_h_dsa_triton_decode_page_table as decode_page_table,
)
from vllm.model_executor.models import (
    nemotron_h_dsa_triton_scoring as triton_scoring,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient import (
    EfficientChunkedDSAScoringProvider,
    TritonBatchedChunkedDSARepresentativeProvider,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    TorchChunkedDSARepresentativeProvider,
    TorchChunkedDSAScoringProvider,
)
from vllm.model_executor.models.nemotron_h_dsa_attention_legacy import (
    NemotronHDSALegacyAttention,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_summaries import HAS_TRITON
from vllm.model_executor.models.nemotron_h_nonchunked_dsa_components_pytorch import (
    TorchNonChunkedDSARepresentativeProvider,
    TorchNonChunkedDSAScoringProvider,
    TorchTopKTokenDSASelectionProvider,
)


def _make_chunked_dsa_attn() -> NemotronHDSALegacyAttention:
    attn = NemotronHDSALegacyAttention.__new__(NemotronHDSALegacyAttention)
    attn.q_indexer_chunk_size = 4
    attn.q_indexer_chunk_top_k = 1
    attn.q_indexer_chunked_query_chunk_size = 3
    attn.q_indexer_logit_scale = 1.0
    attn.q_indexer_dim = 2
    attn.num_kv_heads = 1
    attn.num_heads = 1
    attn.head_dim = 2
    attn.layer_idx = 0
    attn.q_indexer_use_full_attention_short_seq = False
    attn.attn = SimpleNamespace(
        sliding_window=None,
        impl=SimpleNamespace(
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
            sliding_window=(-1, -1),
            vllm_flash_attn_version=2,
        ),
    )
    return attn


def _pack_nhd_cache(
    value_states: torch.Tensor,
    block_size: int,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    key_len, num_kv_heads, head_dim = value_states.shape
    num_logical_blocks = math.ceil(key_len / block_size)
    if block_table is None:
        block_table = torch.arange(num_logical_blocks)
    num_blocks = int(block_table.max().item()) + 1
    cache = value_states.new_zeros(num_blocks, block_size, num_kv_heads, head_dim)
    for token in range(key_len):
        block_id = int(block_table[token // block_size].item())
        cache[block_id, token % block_size] = value_states[token]
    return cache


def test_sequence_fallback_splits_dense_prefix_at_exact_context_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = pytorch_components.TorchChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=2,
        query_chunk_size=4,
    )
    provider.q_indexer_use_full_attention_short_seq = True
    provider.q_indexer_dense_prefill_kv_threshold_tokens = 8
    calls = []

    def fake_dense(_self, **kwargs):
        calls.append(("dense", kwargs))
        return torch.ones_like(kwargs["query_states"])

    def fake_sparse(_self, **kwargs):
        calls.append(("sparse", kwargs))
        return torch.full_like(kwargs["query_states"], 2)

    monkeypatch.setattr(
        pytorch_components.ChunkedDSAAttentionProviderMixin,
        "_forward_dsa_full_page_table_fa_sequence",
        fake_dense,
    )
    monkeypatch.setattr(
        pytorch_components.ChunkedDSAAttentionProviderMixin,
        "_forward_dsa_chunked_sequence",
        fake_sparse,
    )
    query_states = torch.zeros(4, 1, 2)
    positions = torch.arange(6, 10)

    output = provider._forward_dsa_chunked_sequence_with_dense_prefix(
        query_states=query_states,
        indexer_query_states=torch.zeros(4, 1, 2),
        key_states=None,
        key_cache=torch.empty(0),
        value_cache=torch.empty(0),
        block_table=torch.empty(0, dtype=torch.int32),
        attn=SimpleNamespace(),
        attn_metadata=None,
        positions=positions,
        key_len=10,
    )

    assert [kind for kind, _ in calls] == ["dense", "sparse"]
    dense_call = calls[0][1]
    sparse_call = calls[1][1]
    assert dense_call["key_len"] == 8
    assert dense_call["allow_long_sequence"] is True
    torch.testing.assert_close(dense_call["positions"], torch.tensor([6, 7]))
    assert sparse_call["key_len"] == 10
    torch.testing.assert_close(sparse_call["positions"], torch.tensor([8, 9]))
    torch.testing.assert_close(output[:2], torch.ones_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.full_like(output[2:], 2))


def test_dynamic_dense_threshold_is_shared_by_prefill_and_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_CHUNK_TOP_K", "1")
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_DYNAMIC_DENSE_TOKENS", "32768")
    monkeypatch.setenv(
        "VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS",
        "65536",
    )
    provider = pytorch_components.TorchChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=16,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=1024,
    )

    assert provider._dsa_dense_attention_budget_tokens(query_len=1) == 32768
    assert provider._dsa_dense_attention_budget_tokens(query_len=8192) == 32768
    assert provider.recall_policy.dynamic_min_budget_tokens == 16 * 1024


def test_sequence_fallback_gathers_full_history_for_partial_key_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = pytorch_components.TorchChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=1,
    )
    full_key_states = torch.arange(16, dtype=torch.float32).view(8, 1, 2)
    calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def fake_gather(
        cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        calls.append((cache, block_table, key_len))
        return full_key_states

    def stop_after_representatives(**kwargs):
        torch.testing.assert_close(kwargs["key_states"], full_key_states)
        raise RuntimeError("representatives received full history")

    monkeypatch.setattr(provider, "gather_kv_sequence", fake_gather)
    monkeypatch.setattr(provider, "build_representative_state", stop_after_representatives)
    key_cache = torch.empty(1)
    block_table = torch.tensor([0, 1], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="representatives received full history"):
        provider._forward_dsa_chunked_sequence(
            query_states=torch.zeros(1, 1, 2),
            indexer_query_states=torch.zeros(1, 1, 2),
            key_states=torch.zeros(1, 1, 2),
            key_cache=key_cache,
            value_cache=torch.empty(1),
            block_table=block_table,
            attn=SimpleNamespace(),
            attn_metadata=None,
            positions=torch.tensor([7]),
            key_len=8,
        )

    assert calls == [(key_cache, block_table, 8)]


def test_torch_chunk_representatives_match_current_single_sequence_helper():
    attn = _make_chunked_dsa_attn()
    provider = TorchChunkedDSARepresentativeProvider(
        q_indexer_dim=attn.q_indexer_dim,
        chunk_size=attn.q_indexer_chunk_size,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
    )
    key_len = 10
    key_states = torch.arange(20, dtype=torch.float32).view(key_len, 1, 2)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, attn.q_indexer_chunk_size, block_table)

    result = provider(
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
        unused_future_argument="ignored",
    )

    assert provider.is_available(result)
    representatives = provider.get_for_sequence(result, seq_idx=123)
    expected = attn._get_indexer_chunk_representatives(
        key_states=None,
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
    )
    torch.testing.assert_close(representatives, expected)


def test_torch_chunk_representatives_can_build_active_sequence_batch():
    provider = TorchChunkedDSARepresentativeProvider(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
    )
    block_table = torch.tensor(
        [
            [2, 0, 3],
            [5, 4, 1],
        ],
        dtype=torch.int32,
    )
    seq0 = torch.arange(20, dtype=torch.float32).view(10, 1, 2)
    seq1 = torch.arange(24, dtype=torch.float32).view(12, 1, 2) + 100
    key_cache = torch.zeros(6, 4, 1, 2)
    key_cache.index_copy_(0, block_table[0].to(torch.long), _pack_nhd_cache(seq0, 4))
    key_cache.index_copy_(0, block_table[1].to(torch.long), _pack_nhd_cache(seq1, 4))

    result = provider(
        key_cache=key_cache,
        block_table=block_table,
        active_seq_infos=[(0, 0, 1, 10), (1, 1, 2, 12)],
    )

    assert provider.is_available(result)
    seq0_result = provider(
        key_states=seq0,
        key_cache=key_cache,
        block_table=block_table[0],
    )
    seq1_result = provider(
        key_states=seq1,
        key_cache=key_cache,
        block_table=block_table[1],
    )
    torch.testing.assert_close(
        provider.get_for_sequence(result, seq_idx=0),
        provider.get_for_sequence(seq0_result, seq_idx=0),
    )
    torch.testing.assert_close(
        provider.get_for_sequence(result, seq_idx=1),
        provider.get_for_sequence(seq1_result, seq_idx=1),
    )


def test_torch_chunk_scoring_returns_logits_from_opaque_representatives():
    attn = _make_chunked_dsa_attn()
    representative_provider = TorchChunkedDSARepresentativeProvider(
        q_indexer_dim=attn.q_indexer_dim,
        chunk_size=attn.q_indexer_chunk_size,
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
    )
    provider = TorchChunkedDSAScoringProvider(
        q_indexer_dim=attn.q_indexer_dim,
        logit_scale=attn.q_indexer_logit_scale,
    )
    score_query_states = torch.tensor(
        [
            [0.25, 0.5],
            [1.0, -0.25],
            [0.75, 0.125],
        ],
        dtype=torch.float32,
    )
    key_states = torch.tensor(
        [
            [[0.5, 1.0]],
            [[0.25, 0.75]],
            [[-0.5, 0.5]],
            [[0.75, -0.25]],
            [[-0.25, 0.75]],
            [[1.0, 0.25]],
            [[1.5, -0.5]],
            [[0.5, -1.0]],
            [[0.125, 0.25]],
            [[-0.125, 0.5]],
        ],
        dtype=torch.float32,
    )
    current_chunks = torch.tensor([0, 2, 4], dtype=torch.long)
    max_prior_chunks = 3
    chunk_ids = torch.arange(max_prior_chunks, dtype=torch.long)
    representative_state = representative_provider(
        key_states=key_states,
        key_len=key_states.shape[0],
    )

    result = provider(
        score_query_states=score_query_states,
        representative_state=representative_state,
        current_chunks=current_chunks,
        max_prior_chunks=max_prior_chunks,
        chunk_ids=chunk_ids,
        group_idx=0,
        ignored_future_argument="ignored",
    )

    assert provider.is_available(result)
    actual = provider.get_scores(result)
    chunk_representatives = representative_provider.get_for_sequence(
        representative_state
    )
    assert chunk_representatives is not None
    expected_representatives = chunk_representatives[:max_prior_chunks, 0]
    expected_valid = (
        chunk_ids[None, :] < current_chunks.clamp(min=0, max=max_prior_chunks)[:, None]
    )
    expected_logits = torch.matmul(
        score_query_states.float(),
        expected_representatives.transpose(0, 1),
    )
    expected_logits.mul_(attn.q_indexer_logit_scale / math.sqrt(attn.q_indexer_dim))
    expected_logits = expected_logits.masked_fill(
        ~expected_valid,
        torch.finfo(expected_logits.dtype).min,
    )
    assert actual is not None
    torch.testing.assert_close(actual[0], expected_logits)
    torch.testing.assert_close(actual[1], expected_valid)


def test_efficient_chunk_scoring_matches_torch_backend():
    torch_provider = TorchChunkedDSAScoringProvider(
        q_indexer_dim=3,
        logit_scale=1.25,
    )
    efficient_provider = EfficientChunkedDSAScoringProvider(
        q_indexer_dim=3,
        logit_scale=1.25,
    )
    score_query_states = torch.randn(5, 3)
    chunk_representatives = torch.randn(7, 3)
    current_chunks = torch.tensor([0, 1, 2, 4, 7], dtype=torch.long)

    torch_result = torch_provider(
        score_query_states=score_query_states,
        representative_state=chunk_representatives,
        current_chunks=current_chunks,
        max_prior_chunks=6,
    )
    efficient_result = efficient_provider(
        score_query_states=score_query_states,
        representative_state=chunk_representatives,
        current_chunks=current_chunks,
        max_prior_chunks=6,
    )

    expected = torch_provider.get_scores(torch_result)
    actual = efficient_provider.get_scores(efficient_result)
    assert expected is not None
    assert actual is not None
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_efficient_sparse_prefill_checks_width_without_tensor_item(monkeypatch):
    provider = efficient_components.EfficientChunkedDSABlockTableProvider()
    current_chunks = torch.tensor([1, 1, 1, 1, 2], dtype=torch.long)

    def fail_item(self):
        raise AssertionError("sparse prefill must not read a tensor scalar")

    with monkeypatch.context() as item_guard:
        item_guard.setattr(torch.Tensor, "item", fail_item)
        unavailable = provider(
            block_table=torch.tensor([7, 8], dtype=torch.int32),
            chunk_size=4,
            key_len=9,
            q_len=5,
            current_chunks=current_chunks,
            query_position_start=4,
        )
        result = provider(
            block_table=torch.tensor([7, 8, 9], dtype=torch.int32),
            chunk_size=4,
            key_len=9,
            q_len=5,
            current_chunks=current_chunks,
            query_position_start=4,
        )

    assert not provider.is_available(unavailable)
    assert provider.is_available(result)
    page_table = provider.get_page_table(result)
    assert page_table is not None
    block_table, request_lens, seqused_k, max_seqlen_q, max_seqlen_k = page_table
    torch.testing.assert_close(
        block_table,
        torch.tensor([[8], [8], [8], [8], [9]], dtype=torch.int32),
    )
    torch.testing.assert_close(request_lens, torch.ones(5, dtype=torch.int32))
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([1, 2, 3, 4, 1], dtype=torch.int32),
    )
    assert max_seqlen_q == 1
    assert max_seqlen_k == 4


def test_efficient_sparse_prefill_ignores_invalid_selection_tail():
    provider = efficient_components.EfficientChunkedDSABlockTableProvider()
    selection_state = efficient_components._EfficientChunkBlockSelection(
        selected_block_indices=torch.tensor(
            [[0, -1, -1]],
            dtype=torch.long,
        ),
        selected_block_counts=torch.tensor([1], dtype=torch.int32),
    )

    result = provider(
        block_table=torch.tensor([7, 8], dtype=torch.int32),
        chunk_size=4,
        key_len=6,
        q_len=1,
        selection_state=selection_state,
        current_chunks=torch.tensor([1], dtype=torch.long),
        query_position_start=5,
    )

    assert provider.is_available(result)
    page_table = provider.get_page_table(result)
    assert page_table is not None
    block_table, request_lens, seqused_k, max_seqlen_q, max_seqlen_k = page_table
    torch.testing.assert_close(
        block_table,
        torch.tensor([[7, 8, 0, 0]], dtype=torch.int32),
    )
    torch.testing.assert_close(request_lens, torch.ones(1, dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([6], dtype=torch.int32))
    assert max_seqlen_q == 1
    assert max_seqlen_k == 14


def test_efficient_decode_builds_fixed_width_table_without_scalar_reads(
    monkeypatch,
):
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the decode page-table builder")

    device = torch.device("cuda")
    provider = efficient_components.EfficientChunkedDSABlockTableProvider()
    selection_state = efficient_components._EfficientChunkBlockSelection(
        selected_block_indices=torch.tensor(
            [[2, 0, 1, 99]],
            device=device,
            dtype=torch.long,
        ),
        selected_block_valid=torch.tensor(
            [[True, False, True, True]],
            device=device,
        ),
    )

    def fail_item(self):
        raise AssertionError("decode must not read a tensor scalar")

    def fail_masked_select(self, mask):
        raise AssertionError("decode must not build a dynamic masked selection")

    with monkeypatch.context() as tensor_guard:
        tensor_guard.setattr(torch.Tensor, "item", fail_item)
        tensor_guard.setattr(torch.Tensor, "masked_select", fail_masked_select)
        unavailable = provider(
            block_table=torch.tensor([7, 8], device=device, dtype=torch.int32),
            chunk_size=4,
            key_len=10,
            mode="decode",
            selection_state=selection_state,
            current_chunks=torch.tensor([2], device=device, dtype=torch.long),
            query_positions=torch.tensor([9], device=device, dtype=torch.long),
        )
        result = provider(
            block_table=torch.tensor([7, 8, 9], device=device, dtype=torch.int32),
            chunk_size=4,
            key_len=10,
            mode="decode",
            selection_state=selection_state,
            current_chunks=torch.tensor([2], device=device, dtype=torch.long),
            query_positions=torch.tensor([9], device=device, dtype=torch.long),
        )

    assert not provider.is_available(unavailable)
    assert provider.is_available(result)
    page_table = provider.get_page_table(result)
    assert page_table is not None
    block_table, request_lens, seqused_k, max_seqlen_q, max_seqlen_k = page_table
    torch.testing.assert_close(
        block_table,
        torch.tensor([[8, 9, 0, 0, 0]], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        request_lens,
        torch.ones(1, device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([6], device=device, dtype=torch.int32),
    )
    assert max_seqlen_q == 1
    assert max_seqlen_k == 18


def _compact_decode_page_table_torch_reference(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor,
    current_chunk: int,
    chunk_size: int,
    tail_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_valid = (
        selected_valid & (selected_blocks >= 0) & (selected_blocks < current_chunk)
    )
    num_rows, top_width = selected_blocks.shape
    table_width = top_width + 1
    logical_pages = torch.zeros(
        num_rows,
        table_width,
        device=block_table.device,
        dtype=torch.long,
    )
    ranks = selected_valid.to(torch.long).cumsum(dim=-1) - 1
    selected_positions = torch.where(
        selected_valid,
        ranks,
        torch.full_like(ranks, top_width),
    )
    logical_pages.scatter_(
        1,
        selected_positions,
        selected_blocks.masked_fill(~selected_valid, 0),
    )
    selected_counts = selected_valid.sum(dim=-1).to(torch.long)
    logical_pages.scatter_(
        1,
        selected_counts[:, None],
        torch.full(
            (num_rows, 1),
            current_chunk,
            device=block_table.device,
            dtype=torch.long,
        ),
    )
    page_table = (
        block_table.to(torch.long)
        .gather(
            0,
            logical_pages.reshape(-1),
        )
        .to(torch.int32)
        .view(num_rows, table_width)
    )
    used_mask = (
        torch.arange(table_width, device=block_table.device)[None, :]
        <= selected_counts[:, None]
    )
    page_table.masked_fill_(~used_mask, 0)
    seqused_k = selected_counts.to(torch.int32) * chunk_size + tail_len
    return page_table, seqused_k


def _batched_decode_page_table_torch_reference(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor,
    row_seq_ids: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_parts = []
    seqused_k_parts = []
    for row in range(int(selected_blocks.shape[0])):
        seq_idx = int(row_seq_ids[row].item())
        current_chunk = int(current_chunks[row].item())
        tail_len = int(tail_lens[row].item())
        page_table, seqused_k = _compact_decode_page_table_torch_reference(
            block_table=block_table[seq_idx],
            selected_blocks=selected_blocks[row : row + 1],
            selected_valid=selected_valid[row : row + 1],
            current_chunk=current_chunk,
            chunk_size=chunk_size,
            tail_len=tail_len,
        )
        page_parts.append(page_table)
        seqused_k_parts.append(seqused_k)
    return (
        torch.cat(page_parts, dim=0),
        torch.cat(seqused_k_parts, dim=0),
        torch.arange(
            selected_blocks.shape[0] + 1,
            device=block_table.device,
            dtype=torch.int32,
        ),
    )


def _batched_mixed_page_table_torch_reference(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor,
    row_metadata: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    table_width: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    page_parts = []
    seqused_k_parts = []
    for row in range(int(row_metadata.shape[0])):
        seq_idx = int(row_metadata[row, 0].item())
        sparse_row = int(row_metadata[row, 1].item())
        dense_pages = int(row_metadata[row, 2].item())
        dense_seqused_k = int(row_metadata[row, 3].item())
        if sparse_row < 0:
            page_table = torch.zeros(
                1,
                table_width,
                device=block_table.device,
                dtype=torch.int32,
            )
            page_table[0, :dense_pages] = block_table[seq_idx, :dense_pages]
            seqused_k = torch.tensor(
                [dense_seqused_k],
                device=block_table.device,
                dtype=torch.int32,
            )
        else:
            compact_page_table, seqused_k = _compact_decode_page_table_torch_reference(
                block_table=block_table[seq_idx],
                selected_blocks=selected_blocks[sparse_row : sparse_row + 1],
                selected_valid=selected_valid[sparse_row : sparse_row + 1],
                current_chunk=int(current_chunks[sparse_row].item()),
                chunk_size=chunk_size,
                tail_len=int(tail_lens[sparse_row].item()),
            )
            page_table = torch.zeros(
                1,
                table_width,
                device=block_table.device,
                dtype=torch.int32,
            )
            page_table[:, : compact_page_table.shape[1]] = compact_page_table
        page_parts.append(page_table)
        seqused_k_parts.append(seqused_k)
    return torch.cat(page_parts, dim=0), torch.cat(seqused_k_parts, dim=0)


def _batched_unified_page_table_torch_reference(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_counts: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    active_seq_count: int,
    table_width: int,
    chunk_size: int,
    dense_decode_threshold: int,
    dense_prefill_threshold: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_parts = []
    seqused_k_parts = []
    cu_seqlens = []
    for seq_idx in range(active_seq_count):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        q_len = q_end - q_start
        key_len = int(seq_lens[seq_idx].item())
        if q_len <= 0 or key_len <= 0:
            continue
        dense_threshold = (
            dense_prefill_threshold if q_len > 1 else dense_decode_threshold
        )
        query_position_start = key_len - q_len
        dense_prefix_len = (
            min(q_len, max(dense_threshold - query_position_start, 0))
            if dense_threshold >= 0
            else 0
        )
        if dense_prefix_len > 0:
            dense_key_len = query_position_start + dense_prefix_len
            dense_pages = math.ceil(dense_key_len / chunk_size)
            page_table = torch.zeros(
                1,
                table_width,
                device=block_table.device,
                dtype=torch.int32,
            )
            page_table[0, :dense_pages] = block_table[seq_idx, :dense_pages]
            page_parts.append(page_table)
            seqused_k_parts.append(
                torch.tensor(
                    [dense_key_len], device=block_table.device, dtype=torch.int32
                )
            )
            if not cu_seqlens:
                cu_seqlens.append(q_start)
            cu_seqlens.append(q_start + dense_prefix_len)
        if dense_prefix_len == q_len:
            continue

        for token_row in range(q_start + dense_prefix_len, q_end):
            position = query_position_start + token_row - q_start
            current_chunk = position // chunk_size
            tail_len = position - current_chunk * chunk_size + 1
            selected_count = int(selected_counts[token_row].item())
            selected_count = max(0, min(selected_count, selected_blocks.shape[1]))
            page_table = torch.zeros(
                1,
                table_width,
                device=block_table.device,
                dtype=torch.int32,
            )
            for slot in range(selected_count):
                logical_page = int(selected_blocks[token_row, slot].item())
                if 0 <= logical_page < current_chunk:
                    page_table[0, slot] = block_table[seq_idx, logical_page]
            page_table[0, selected_count] = block_table[seq_idx, current_chunk]
            page_parts.append(page_table)
            seqused_k_parts.append(
                torch.tensor(
                    [selected_count * chunk_size + tail_len],
                    device=block_table.device,
                    dtype=torch.int32,
                )
            )
            if not cu_seqlens:
                cu_seqlens.append(token_row)
            cu_seqlens.append(token_row + 1)

    return (
        torch.cat(page_parts, dim=0),
        torch.cat(seqused_k_parts, dim=0),
        torch.tensor(cu_seqlens, device=block_table.device, dtype=torch.int32),
    )


def test_triton_decode_page_table_matches_compact_torch_reference():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the decode page-table builder")

    device = torch.device("cuda")
    block_table = torch.tensor([17, 13, 19], device=device, dtype=torch.int32)
    selected_blocks = torch.tensor(
        [
            [2, 0, 1, 99],
            [1, 0, 2, 0],
        ],
        device=device,
        dtype=torch.long,
    )
    selected_valid = torch.tensor(
        [
            [True, False, True, True],
            [False, True, True, False],
        ],
        device=device,
    )

    actual = decode_page_table.dsa_decode_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_valid=selected_valid,
        current_chunk=2,
        chunk_size=4,
        tail_len=2,
    )

    assert actual is not None
    page_table, seqused_k = actual
    expected_page_table, expected_seqused_k = (
        _compact_decode_page_table_torch_reference(
            block_table=block_table,
            selected_blocks=selected_blocks,
            selected_valid=selected_valid,
            current_chunk=2,
            chunk_size=4,
            tail_len=2,
        )
    )
    torch.testing.assert_close(page_table, expected_page_table)
    torch.testing.assert_close(seqused_k, expected_seqused_k)

    wide_selected_blocks = (
        torch.arange(
            300,
            device=device,
            dtype=torch.long,
        )
        .remainder(3)
        .view(1, -1)
    )
    wide_selected_valid = torch.ones_like(wide_selected_blocks, dtype=torch.bool)
    wide_block_table = torch.tensor(
        [17, 13, 19, 23],
        device=device,
        dtype=torch.int32,
    )
    wide_actual = decode_page_table.dsa_decode_page_table_triton(
        block_table=wide_block_table,
        selected_blocks=wide_selected_blocks,
        selected_valid=wide_selected_valid,
        current_chunk=3,
        chunk_size=4,
        tail_len=1,
    )
    assert wide_actual is not None
    wide_page_table, wide_seqused_k = wide_actual
    wide_expected_page_table, wide_expected_seqused_k = (
        _compact_decode_page_table_torch_reference(
            block_table=wide_block_table,
            selected_blocks=wide_selected_blocks,
            selected_valid=wide_selected_valid,
            current_chunk=3,
            chunk_size=4,
            tail_len=1,
        )
    )
    torch.testing.assert_close(wide_page_table, wide_expected_page_table)
    torch.testing.assert_close(wide_seqused_k, wide_expected_seqused_k)


def test_triton_batched_decode_page_table_matches_compact_torch_reference():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the decode page-table builder")

    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [17, 13, 19, 23],
            [31, 29, 37, 41],
            [43, 47, 53, 59],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 2, 1],
            [1, 99, 0],
            [2, 1, 0],
            [3, 1, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    selected_valid = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, True, True],
            [True, False, True],
        ],
        device=device,
    )
    row_seq_ids = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.int32)
    current_chunks = torch.tensor([2, 2, 3, 3], device=device, dtype=torch.int32)
    tail_lens = torch.tensor([1, 4, 2, 3], device=device, dtype=torch.int32)

    actual = decode_page_table.dsa_batched_decode_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_valid=selected_valid,
        row_seq_ids=row_seq_ids,
        current_chunks=current_chunks,
        tail_lens=tail_lens,
        chunk_size=4,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    expected_page_table, expected_seqused_k, expected_cu_seqlens_q = (
        _batched_decode_page_table_torch_reference(
            block_table=block_table,
            selected_blocks=selected_blocks,
            selected_valid=selected_valid,
            row_seq_ids=row_seq_ids,
            current_chunks=current_chunks,
            tail_lens=tail_lens,
            chunk_size=4,
        )
    )
    torch.testing.assert_close(page_table, expected_page_table)
    torch.testing.assert_close(seqused_k, expected_seqused_k)
    torch.testing.assert_close(cu_seqlens_q, expected_cu_seqlens_q)


def test_triton_batched_row_metadata_matches_cpu_reference():
    if not torch.cuda.is_available() or triton_scoring.triton is None:
        pytest.skip("CUDA and Triton are required for row metadata packing")

    device = torch.device("cuda")
    row_plan = torch.tensor(
        [
            [0, 3, 4, 10, 0, 7, 8],
            [3, 2, 5, 12, 1, 5, 14],
        ],
        device=device,
        dtype=torch.int32,
    )

    actual = triton_scoring.dsa_batched_row_metadata_triton(
        row_plan=row_plan,
        total_rows=5,
        chunk_size=4,
        max_q_len=3,
    )

    assert actual is not None
    (
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
    ) = actual
    torch.testing.assert_close(
        score_row_seq_ids,
        torch.tensor([4, 4, 4, 5, 5], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        row_seq_ids,
        torch.tensor([10, 10, 10, 12, 12], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        row_group_ids,
        torch.tensor([0, 0, 0, 1, 1], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        row_num_prior_chunks,
        torch.tensor([7, 7, 7, 5, 5], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        row_current_chunks,
        torch.tensor([2, 2, 2, 3, 3], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(
        row_tail_lens,
        torch.tensor([1, 2, 3, 3, 4], device=device, dtype=torch.int32),
    )


def test_triton_score_metadata_builder_matches_mixed_cpu_reference():
    if not torch.cuda.is_available() or triton_scoring.triton is None:
        pytest.skip("CUDA and Triton are required for row metadata packing")

    device = torch.device("cuda")
    (
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = triton_scoring.dsa_score_tile_plan_config()

    cases = [
        dict(
            chunk_size=4,
            q_lens=[1, 1, 3, 5, 2],
            key_lens=[1, 32, 8, 33, 10],
            dense_decode_threshold=4,
            dense_prefill_threshold=8,
            representative_group_idx=2,
            chunk_top_k=3,
            expected_sparse_plans=3,
        ),
        dict(
            chunk_size=2,
            q_lens=[1, 1, 2, 3, 5],
            key_lens=[4, 9, 4, 9, 17],
            dense_decode_threshold=4,
            dense_prefill_threshold=6,
            representative_group_idx=1,
            chunk_top_k=2,
            expected_sparse_plans=3,
        ),
    ]

    for case in cases:
        chunk_size = case["chunk_size"]
        q_lens = case["q_lens"]
        key_lens = case["key_lens"]
        dense_decode_threshold = case["dense_decode_threshold"]
        dense_prefill_threshold = case["dense_prefill_threshold"]
        representative_group_idx = case["representative_group_idx"]
        chunk_top_k = case["chunk_top_k"]
        query_start_locs = [0]
        for q_len in q_lens:
            query_start_locs.append(query_start_locs[-1] + q_len)
        total_rows = query_start_locs[-1]

        sparse_parts = []
        tile_offset = 0
        expected_score_seq_ids = torch.zeros(
            total_rows, device=device, dtype=torch.int32
        )
        expected_row_seq_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
        expected_group_ids = torch.zeros(total_rows, device=device, dtype=torch.int32)
        expected_prior_chunks = torch.zeros(
            total_rows, device=device, dtype=torch.int32
        )
        expected_current_chunks = torch.empty(
            total_rows, device=device, dtype=torch.int32
        )
        expected_tail_lens = torch.empty(total_rows, device=device, dtype=torch.int32)

        for seq_idx, (q_len, key_len) in enumerate(zip(q_lens, key_lens)):
            q_start = query_start_locs[seq_idx]
            query_position_start = key_len - q_len
            num_chunks = math.ceil(key_len / chunk_size)
            prior_chunks = max(num_chunks - 1, 0)
            is_decode = q_len == 1
            dense_threshold = (
                dense_decode_threshold if is_decode else dense_prefill_threshold
            )
            is_sparse = key_len > dense_threshold and prior_chunks > 0
            if is_sparse:
                row_plan_base = (
                    q_start,
                    q_len,
                    len(sparse_parts),
                    seq_idx,
                    representative_group_idx,
                    prior_chunks,
                    query_position_start,
                )
                tile_count, _, _, _ = triton_scoring.dsa_count_score_tile_plan_parts(
                    (row_plan_base,),
                    small_block_rows=small_block_rows,
                    large_block_rows=large_block_rows,
                    block_chunks=block_chunks,
                    decode_block_chunks=decode_block_chunks,
                )
                sparse_parts.append((*row_plan_base, tile_offset, tile_count))
                tile_offset += tile_count

            row_offsets = torch.arange(q_len, device=device, dtype=torch.int32)
            rows = q_start + row_offsets
            row_indices = rows.to(torch.long)
            positions = query_position_start + row_offsets
            expected_score_seq_ids[row_indices] = (
                len(sparse_parts) - 1 if is_sparse else 0
            )
            expected_row_seq_ids[row_indices] = seq_idx
            expected_group_ids[row_indices] = (
                representative_group_idx if is_sparse else 0
            )
            expected_prior_chunks[row_indices] = prior_chunks if is_sparse else 0
            expected_current_chunks[row_indices] = torch.div(
                positions,
                chunk_size,
                rounding_mode="floor",
            )
            expected_tail_lens[row_indices] = (
                positions - expected_current_chunks[row_indices] * chunk_size + 1
            )

        assert len(sparse_parts) == case["expected_sparse_plans"]

        actual = triton_scoring.dsa_build_score_metadata_triton(
            query_start_loc=torch.tensor(
                query_start_locs, device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor(key_lens, device=device, dtype=torch.int32),
            num_actual_tokens=total_rows,
            active_seq_count=len(q_lens),
            num_sparse_plans=len(sparse_parts),
            total_rows=total_rows,
            chunk_size=chunk_size,
            representative_group_idx=representative_group_idx,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
            chunk_top_k=chunk_top_k,
            max_q_len=max(q_lens),
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )

        assert actual is not None
        row_plan, row_metadata = actual
        torch.testing.assert_close(
            row_plan,
            torch.tensor(sparse_parts, device=device, dtype=torch.int32),
        )
        (
            score_row_seq_ids,
            row_seq_ids,
            row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        ) = row_metadata
        torch.testing.assert_close(score_row_seq_ids, expected_score_seq_ids)
        torch.testing.assert_close(row_seq_ids, expected_row_seq_ids)
        torch.testing.assert_close(row_group_ids, expected_group_ids)
        torch.testing.assert_close(row_num_prior_chunks, expected_prior_chunks)
        torch.testing.assert_close(row_current_chunks, expected_current_chunks)
        torch.testing.assert_close(row_tail_lens, expected_tail_lens)

        actual_original_ids = triton_scoring.dsa_build_score_metadata_triton(
            query_start_loc=torch.tensor(
                query_start_locs, device=device, dtype=torch.int32
            ),
            seq_lens=torch.tensor(key_lens, device=device, dtype=torch.int32),
            num_actual_tokens=total_rows,
            active_seq_count=len(q_lens),
            num_sparse_plans=len(sparse_parts),
            total_rows=total_rows,
            chunk_size=chunk_size,
            representative_group_idx=representative_group_idx,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
            chunk_top_k=chunk_top_k,
            max_q_len=max(q_lens),
            representatives_use_original_seq_ids=True,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )

        assert actual_original_ids is not None
        row_plan_original, row_metadata_original = actual_original_ids
        expected_original_plan = [
            (*part[:2], part[3], *part[3:]) for part in sparse_parts
        ]
        torch.testing.assert_close(
            row_plan_original,
            torch.tensor(expected_original_plan, device=device, dtype=torch.int32),
        )
        expected_original_score_seq_ids = torch.where(
            expected_prior_chunks > 0,
            expected_row_seq_ids,
            torch.zeros_like(expected_row_seq_ids),
        )
        (
            score_row_seq_ids_original,
            row_seq_ids_original,
            row_group_ids_original,
            row_num_prior_chunks_original,
            row_current_chunks_original,
            row_tail_lens_original,
        ) = row_metadata_original
        torch.testing.assert_close(
            score_row_seq_ids_original,
            expected_original_score_seq_ids,
        )
        torch.testing.assert_close(row_seq_ids_original, expected_row_seq_ids)
        torch.testing.assert_close(row_group_ids_original, expected_group_ids)
        torch.testing.assert_close(row_num_prior_chunks_original, expected_prior_chunks)
        torch.testing.assert_close(row_current_chunks_original, expected_current_chunks)
        torch.testing.assert_close(row_tail_lens_original, expected_tail_lens)


def test_triton_batched_decode_page_table_accepts_prefix_counts():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the decode page-table builder")

    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [17, 13, 19, 23],
            [31, 29, 37, 41],
            [43, 47, 53, 59],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 1, 2],
            [1, 0, 99],
            [2, 1, 0],
            [2, 1, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_counts = torch.tensor([2, 1, 2, 3], device=device, dtype=torch.int32)
    row_seq_ids = torch.tensor([0, 1, 1, 2], device=device, dtype=torch.int32)
    current_chunks = torch.tensor([2, 2, 3, 3], device=device, dtype=torch.int32)
    tail_lens = torch.tensor([1, 4, 2, 3], device=device, dtype=torch.int32)
    selected_valid = (
        torch.arange(
            selected_blocks.shape[1],
            device=device,
            dtype=torch.int32,
        )[None, :]
        < selected_counts[:, None]
    )

    actual = decode_page_table.dsa_batched_decode_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        row_seq_ids=row_seq_ids,
        current_chunks=current_chunks,
        tail_lens=tail_lens,
        chunk_size=4,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    expected_page_table, expected_seqused_k, expected_cu_seqlens_q = (
        _batched_decode_page_table_torch_reference(
            block_table=block_table,
            selected_blocks=selected_blocks.to(torch.long),
            selected_valid=selected_valid,
            row_seq_ids=row_seq_ids,
            current_chunks=current_chunks,
            tail_lens=tail_lens,
            chunk_size=4,
        )
    )
    torch.testing.assert_close(page_table, expected_page_table)
    torch.testing.assert_close(seqused_k, expected_seqused_k)
    torch.testing.assert_close(cu_seqlens_q, expected_cu_seqlens_q)


def test_triton_batched_mixed_page_table_matches_torch_reference():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the mixed page-table builder")

    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [17, 13, 19, 23, 29],
            [31, 37, 41, 43, 47],
            [53, 59, 61, 67, 71],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 2, 1],
            [3, 1, 0],
            [1, 99, 2],
        ],
        device=device,
        dtype=torch.long,
    )
    selected_valid = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [True, True, True],
        ],
        device=device,
    )
    row_metadata = torch.tensor(
        [
            [0, -1, 2, 8],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [2, -1, 3, 11],
            [2, 2, 0, 0],
        ],
        device=device,
        dtype=torch.int32,
    )
    current_chunks = torch.tensor([2, 4, 3], device=device, dtype=torch.int32)
    tail_lens = torch.tensor([1, 3, 2], device=device, dtype=torch.int32)
    table_width = 4

    actual = decode_page_table.dsa_batched_mixed_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_valid=selected_valid,
        row_metadata=row_metadata,
        current_chunks=current_chunks,
        tail_lens=tail_lens,
        cu_seqlens_q=torch.tensor(
            [0, 2, 3, 4, 7, 8],
            device=device,
            dtype=torch.int32,
        ),
        table_width=table_width,
        chunk_size=4,
    )

    assert actual is not None
    page_table, seqused_k = actual
    expected_page_table, expected_seqused_k = _batched_mixed_page_table_torch_reference(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_valid=selected_valid,
        row_metadata=row_metadata,
        current_chunks=current_chunks,
        tail_lens=tail_lens,
        table_width=table_width,
        chunk_size=4,
    )
    torch.testing.assert_close(page_table, expected_page_table)
    torch.testing.assert_close(seqused_k, expected_seqused_k)


def test_triton_batched_unified_page_table_matches_torch_reference():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the unified page-table builder")

    device = torch.device("cuda")
    block_table = torch.tensor(
        [
            [17, 13, 19, 23],
            [31, 37, 41, 43],
            [53, 59, 61, 67],
            [71, 73, 79, 83],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
        ],
        device=device,
        dtype=torch.int32,
    )
    selected_counts = torch.tensor(
        [0, 0, 0, 0, 1, 2, 0, 0, 0, 1],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor(
        [0, 4, 6, 9, 10],
        device=device,
        dtype=torch.int32,
    )
    seq_lens = torch.tensor([8, 12, 7, 6], device=device, dtype=torch.int32)
    table_width = 3

    actual = decode_page_table.dsa_batched_unified_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=10,
        active_seq_count=4,
        num_requests=5,
        table_width=table_width,
        max_q_len=4,
        chunk_size=4,
        dense_decode_threshold=8,
        dense_prefill_threshold=8,
    )

    assert actual is not None
    page_table, seqused_k, cu_seqlens_q = actual
    expected_page_table, expected_seqused_k, expected_cu_seqlens_q = (
        _batched_unified_page_table_torch_reference(
            block_table=block_table,
            selected_blocks=selected_blocks,
            selected_counts=selected_counts,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            active_seq_count=4,
            table_width=table_width,
            chunk_size=4,
            dense_decode_threshold=8,
            dense_prefill_threshold=8,
        )
    )
    torch.testing.assert_close(page_table, expected_page_table)
    torch.testing.assert_close(seqused_k, expected_seqused_k)
    torch.testing.assert_close(cu_seqlens_q, expected_cu_seqlens_q)


def test_triton_batched_unified_page_table_splits_dense_sparse_crossing():
    if not torch.cuda.is_available() or decode_page_table.triton is None:
        pytest.skip("CUDA and Triton are required for the unified page-table builder")

    device = torch.device("cuda")
    block_table = torch.arange(100, 108, device=device, dtype=torch.int32).view(1, -1)
    selected_blocks = torch.tensor(
        [[0, 1]] * 6,
        device=device,
        dtype=torch.int32,
    )
    selected_counts = torch.tensor(
        [0, 0, 0, 0, 2, 2],
        device=device,
        dtype=torch.int32,
    )
    query_start_loc = torch.tensor([0, 6], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([10], device=device, dtype=torch.int32)

    actual = decode_page_table.dsa_batched_unified_page_table_triton(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=6,
        active_seq_count=1,
        num_requests=3,
        table_width=4,
        max_q_len=6,
        chunk_size=2,
        dense_decode_threshold=8,
        dense_prefill_threshold=8,
    )

    assert actual is not None
    expected = _batched_unified_page_table_torch_reference(
        block_table=block_table,
        selected_blocks=selected_blocks,
        selected_counts=selected_counts,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        active_seq_count=1,
        table_width=4,
        chunk_size=2,
        dense_decode_threshold=8,
        dense_prefill_threshold=8,
    )
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_triton_decode_page_table_runtime_args_are_not_constexpr():
    if decode_page_table.tl is None or not hasattr(
        decode_page_table, "_dsa_decode_page_table_kernel"
    ):
        return

    kernel = decode_page_table._dsa_decode_page_table_kernel
    fn = getattr(kernel, "fn", kernel)
    annotations = getattr(fn, "__annotations__", {})
    for name in {
        "num_rows",
        "top_width",
        "table_width",
        "current_chunk",
        "chunk_size",
        "tail_len",
    }:
        assert name not in annotations
    constexpr_annotations = {
        decode_page_table.tl.constexpr,
        "tl.constexpr",
    }
    assert annotations["BLOCK_WIDTH"] in constexpr_annotations


def test_efficient_batched_page_table_hook_builds_whole_batch(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
    )
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 99],
            [1, 0],
            [2, 1],
        ],
        dtype=torch.int32,
    )
    selected_counts = torch.tensor([2, 1, 2], dtype=torch.int32)
    row_seq_ids = torch.tensor([0, 1, 1], dtype=torch.int32)
    row_current_chunks = torch.tensor([1, 3, 3], dtype=torch.int32)
    row_tail_lens = torch.tensor([2, 1, 2], dtype=torch.int32)
    selection_by_seq = efficient_components._EfficientBatchedChunkBlockSelections(
        selected_block_indices=selected_blocks,
        selected_block_valid=None,
        selected_block_counts=selected_counts,
        seq_slices={
            0: (0, 1, 2),
            1: (1, 3, 2),
        },
        chunk_top_k_by_seq={
            0: 2,
            1: 2,
        },
        row_seq_ids=row_seq_ids,
        row_current_chunks=row_current_chunks,
        row_tail_lens=row_tail_lens,
        per_seq={
            0: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[:1],
                selected_block_counts=selected_counts[:1],
            ),
            1: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[1:],
                selected_block_counts=selected_counts[1:],
            ),
        },
    )
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    seq_lens = torch.tensor([6, 14], dtype=torch.int32)
    calls = []

    def fake_batched_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(3, 3, dtype=torch.int32),
            torch.tensor([6, 9, 10], dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_unified_page_table_triton",
        fake_batched_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=block_table,
        active_seq_infos=[
            (0, 0, 1, 6),
            (1, 1, 3, 14),
        ],
        sparse_infos=[
            (0, 0, 1, 6, 2, 5, torch.tensor([1], dtype=torch.long)),
            (1, 1, 3, 14, 4, 12, torch.tensor([3, 3], dtype=torch.long)),
        ],
        block_selection_by_seq=selection_by_seq,
        total_rows=3,
        device=torch.device("cpu"),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=3,
        active_seq_count=2,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert result is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["block_table"] is block_table
    assert call["selected_blocks"] is selected_blocks
    assert call["selected_counts"] is selected_counts
    assert call["query_start_loc"] is query_start_loc
    assert call["seq_lens"] is seq_lens
    assert call["num_requests"] == 3
    assert call["table_width"] == 3
    assert call["max_q_len"] == 2
    assert call["chunk_size"] == 4
    page_table, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k = result
    assert tuple(page_table.shape) == (3, 3)
    torch.testing.assert_close(cu_seqlens_q, torch.arange(4, dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([6, 9, 10], dtype=torch.int32))
    assert max_seqlen_q == 1
    assert max_seqlen_k == 10


def test_efficient_batched_page_table_hook_accepts_large_plan(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=16,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
    )
    num_rows = 32737
    top_width = 1024
    selected_blocks = torch.zeros(1, top_width, dtype=torch.int32).expand(num_rows, -1)
    selected_counts = torch.full((1,), top_width, dtype=torch.int32).expand(num_rows)
    row_values = torch.zeros(1, dtype=torch.int32).expand(num_rows)
    selection = efficient_components._EfficientChunkBlockSelection(
        selected_block_indices=selected_blocks,
        selected_block_counts=selected_counts,
    )
    selection_by_seq = efficient_components._EfficientBatchedChunkBlockSelections(
        selected_block_indices=selected_blocks,
        selected_block_valid=None,
        selected_block_counts=selected_counts,
        seq_slices={0: (0, num_rows, top_width)},
        chunk_top_k_by_seq={0: top_width},
        row_seq_ids=row_values,
        row_current_chunks=row_values,
        row_tail_lens=row_values,
        per_seq={0: selection},
    )
    calls = []

    def fake_batched_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_unified_page_table_triton",
        fake_batched_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        active_seq_infos=[(0, 0, num_rows, num_rows)],
        sparse_infos=[],
        block_selection_by_seq=selection_by_seq,
        total_rows=num_rows,
        device=torch.device("cpu"),
        query_start_loc=torch.tensor([0, num_rows], dtype=torch.int32),
        seq_lens=torch.tensor([num_rows], dtype=torch.int32),
        num_actual_tokens=num_rows,
        active_seq_count=1,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert result is not None
    assert len(calls) == 1
    assert calls[0]["num_requests"] == num_rows
    assert calls[0]["table_width"] == top_width + 1
    assert num_rows * (top_width + 1) > 1 << 25


def test_efficient_batched_page_table_hook_passes_selection_counts(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
    )
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 99],
            [1, 0],
            [2, 1],
        ],
        dtype=torch.int32,
    )
    selected_counts = torch.tensor([2, 1, 2], dtype=torch.int32)
    row_seq_ids = torch.tensor([0, 1, 1], dtype=torch.int32)
    row_current_chunks = torch.tensor([1, 3, 3], dtype=torch.int32)
    row_tail_lens = torch.tensor([2, 1, 2], dtype=torch.int32)
    selection_by_seq = efficient_components._EfficientBatchedChunkBlockSelections(
        selected_block_indices=selected_blocks,
        selected_block_valid=None,
        selected_block_counts=selected_counts,
        seq_slices={
            0: (0, 1, 2),
            1: (1, 3, 2),
        },
        chunk_top_k_by_seq={
            0: 2,
            1: 2,
        },
        row_seq_ids=row_seq_ids,
        row_current_chunks=row_current_chunks,
        row_tail_lens=row_tail_lens,
        per_seq={
            0: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[:1],
                selected_block_counts=selected_counts[:1],
            ),
            1: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[1:],
                selected_block_counts=selected_counts[1:],
            ),
        },
    )
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    seq_lens = torch.tensor([6, 14], dtype=torch.int32)
    calls = []

    def fake_batched_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(3, 3, dtype=torch.int32),
            torch.tensor([6, 9, 10], dtype=torch.int32),
            torch.arange(4, dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_unified_page_table_triton",
        fake_batched_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=block_table,
        active_seq_infos=[
            (0, 0, 1, 6),
            (1, 1, 3, 14),
        ],
        sparse_infos=[
            (0, 0, 1, 6, 2, 5, torch.tensor([1], dtype=torch.long)),
            (1, 1, 3, 14, 4, 12, torch.tensor([3, 3], dtype=torch.long)),
        ],
        block_selection_by_seq=selection_by_seq,
        total_rows=3,
        device=torch.device("cpu"),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=3,
        active_seq_count=2,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert result is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["selected_blocks"] is selected_blocks
    assert call["selected_counts"] is selected_counts
    assert call["query_start_loc"] is query_start_loc
    assert call["seq_lens"] is seq_lens
    assert call["num_requests"] == 3
    assert call["table_width"] == 3
    assert call["max_q_len"] == 2


def test_efficient_unified_bucket_handles_padded_decode_query_rows(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=2,
    )
    provider.q_indexer_use_page_table_fa = True
    provider.q_indexer_use_prefill_page_table_fa = True
    provider.q_indexer_use_flattened_prefill_page_table_fa = True
    provider.q_indexer_use_full_attention_short_seq = False

    hidden_states = torch.zeros(4, 2)
    query_states = torch.zeros(4, 1, 2)
    key_cache = torch.zeros(8, 4, 1, 2)
    value_cache = torch.zeros_like(key_cache)
    block_table = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ],
        dtype=torch.int32,
    )
    active_seq_infos = [
        (0, 0, 1, 10),
        (1, 1, 2, 12),
    ]
    representatives = efficient_components._TritonBatchedChunkRepresentatives(
        representatives=torch.zeros(2, 3, 1, 2),
        local_by_seq={0: 0, 1: 1},
        num_chunks_by_seq={0: 3, 1: 3},
        seq_id_layout="original",
    )
    attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([10, 12], dtype=torch.int32),
        use_cascade=False,
        dcp_context_kv_lens=None,
    )
    attn = SimpleNamespace(
        sliding_window=None,
        impl=SimpleNamespace(
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
            sliding_window=(-1, -1),
            vllm_flash_attn_version=None,
        ),
    )
    output = torch.zeros_like(query_states)
    selection_calls = []
    page_table_calls = []
    flash_calls = []

    def fake_flash_attn(**kwargs):
        flash_calls.append(kwargs)
        kwargs["out"].copy_(torch.ones_like(kwargs["out"]))

    def fake_select_blocks_batched(**kwargs):
        selection_calls.append(kwargs)
        selected_blocks = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        selected_counts = torch.tensor([2, 2], dtype=torch.int32)
        return efficient_components._EfficientBatchedChunkBlockSelections(
            selected_block_indices=selected_blocks,
            selected_block_valid=None,
            selected_block_counts=selected_counts,
            seq_slices={0: (0, 1, 2), 1: (1, 2, 2)},
            chunk_top_k_by_seq={0: 2, 1: 2},
            row_seq_ids=torch.tensor([0, 1], dtype=torch.int32),
            row_current_chunks=torch.tensor([2, 2], dtype=torch.int32),
            row_tail_lens=torch.tensor([3, 1], dtype=torch.int32),
            per_seq={
                0: efficient_components._EfficientChunkBlockSelection(
                    selected_block_indices=selected_blocks[:1],
                    selected_block_counts=selected_counts[:1],
                ),
                1: efficient_components._EfficientChunkBlockSelection(
                    selected_block_indices=selected_blocks[1:],
                    selected_block_counts=selected_counts[1:],
                ),
            },
        )

    def fake_build_page_tables_batched(**kwargs):
        page_table_calls.append(kwargs)
        return (
            torch.empty(2, 3, dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([11, 9], dtype=torch.int32),
            1,
            11,
        )

    monkeypatch.setattr(
        pytorch_components,
        "_get_flash_attn_varlen_func",
        lambda: fake_flash_attn,
    )
    monkeypatch.setattr(
        provider,
        "try_select_blocks_batched",
        fake_select_blocks_batched,
    )
    monkeypatch.setattr(
        provider,
        "try_build_page_tables_batched",
        fake_build_page_tables_batched,
    )

    handled = provider._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
        hidden_states=hidden_states,
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn=attn,
        attn_metadata=attn_metadata,
        positions=torch.arange(4),
        active_seq_infos=active_seq_infos,
        batched_chunk_representatives=representatives,
        output=output,
        local_kv_head_indices=torch.tensor([0], dtype=torch.long),
        indexer_q_proj=lambda _: (_ for _ in ()).throw(AssertionError()),
        precomputed_indexer_q=torch.zeros(4, 1, 2),
    )

    assert handled == {0, 1}
    assert len(selection_calls) == 1
    assert len(page_table_calls) == 1
    assert len(flash_calls) == 1
    assert selection_calls[0]["indexer_q"].shape[0] == 2
    assert page_table_calls[0]["total_rows"] == 2
    assert page_table_calls[0]["num_actual_tokens"] == 2
    assert page_table_calls[0]["active_seq_count"] == 2
    assert page_table_calls[0]["query_start_loc"] is attn_metadata.query_start_loc
    assert page_table_calls[0]["seq_lens"] is attn_metadata.seq_lens
    torch.testing.assert_close(output[:2], torch.ones_like(output[:2]))
    torch.testing.assert_close(output[2:], torch.zeros_like(output[2:]))


def test_efficient_mixed_page_table_hook_reuses_batched_sparse_selection(
    monkeypatch,
):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=4,
    )
    provider.q_indexer_use_full_attention_short_seq = True
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ],
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 99],
        ],
        dtype=torch.int32,
    )
    selected_counts = torch.tensor(
        [0, 0, 0, 0, 1, 2, 0, 0, 0, 1],
        dtype=torch.int32,
    )
    row_current_chunks = torch.tensor([2, 2, 1], dtype=torch.int32)
    row_tail_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
    selection_by_seq = efficient_components._EfficientBatchedChunkBlockSelections(
        selected_block_indices=selected_blocks,
        selected_block_valid=None,
        selected_block_counts=selected_counts,
        seq_slices={
            1: (4, 6, 2),
            3: (9, 10, 2),
        },
        chunk_top_k_by_seq={
            1: 2,
            3: 2,
        },
        row_seq_ids=torch.tensor([1, 1, 3], dtype=torch.int32),
        row_current_chunks=row_current_chunks,
        row_tail_lens=row_tail_lens,
        per_seq={
            1: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[4:6],
                selected_block_counts=selected_counts[4:6],
            ),
            3: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[9:10],
                selected_block_counts=selected_counts[9:10],
            ),
        },
    )
    query_start_loc = torch.tensor([0, 4, 6, 9, 10], dtype=torch.int32)
    seq_lens = torch.tensor([8, 12, 7, 6], dtype=torch.int32)
    calls = []

    def fake_mixed_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(5, 3, dtype=torch.int32),
            torch.tensor([8, 9, 10, 7, 6], dtype=torch.int32),
            torch.tensor([0, 4, 5, 6, 9, 10], dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_unified_page_table_triton",
        fake_mixed_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=block_table,
        active_seq_infos=[
            (0, 0, 4, 8),
            (1, 4, 6, 12),
            (2, 6, 9, 7),
            (3, 9, 10, 6),
        ],
        sparse_infos=[
            (1, 4, 6, 12, 3, 10, torch.tensor([2, 2], dtype=torch.long)),
            (3, 9, 10, 6, 2, 5, torch.tensor([1], dtype=torch.long)),
        ],
        block_selection_by_seq=selection_by_seq,
        total_rows=10,
        device=torch.device("cpu"),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=10,
        active_seq_count=4,
        dense_decode_threshold=8,
        dense_prefill_threshold=8,
    )

    assert result is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["selected_blocks"] is selected_blocks
    assert call["selected_counts"] is selected_counts
    assert call["query_start_loc"] is query_start_loc
    assert call["seq_lens"] is seq_lens
    assert call["num_requests"] == 5
    assert call["table_width"] == 3
    assert call["max_q_len"] == 4
    page_table, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k = result
    assert tuple(page_table.shape) == (5, 3)
    torch.testing.assert_close(
        cu_seqlens_q,
        torch.tensor([0, 4, 5, 6, 9, 10], dtype=torch.int32),
    )
    torch.testing.assert_close(
        seqused_k,
        torch.tensor([8, 9, 10, 7, 6], dtype=torch.int32),
    )
    assert max_seqlen_q == 4
    assert max_seqlen_k == 12


def test_efficient_mixed_page_table_hook_passes_selection_counts(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=2,
        logit_scale=1.0,
        chunk_top_k=4,
    )
    provider.q_indexer_use_full_attention_short_seq = True
    block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
            [40, 41, 42, 43],
        ],
        dtype=torch.int32,
    )
    selected_blocks = torch.tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 99],
        ],
        dtype=torch.int32,
    )
    selected_counts = torch.tensor(
        [0, 0, 0, 0, 1, 2, 0, 0, 0, 1],
        dtype=torch.int32,
    )
    row_current_chunks = torch.tensor([2, 2, 1], dtype=torch.int32)
    row_tail_lens = torch.tensor([1, 2, 3], dtype=torch.int32)
    selection_by_seq = efficient_components._EfficientBatchedChunkBlockSelections(
        selected_block_indices=selected_blocks,
        selected_block_valid=None,
        selected_block_counts=selected_counts,
        seq_slices={
            1: (4, 6, 2),
            3: (9, 10, 2),
        },
        chunk_top_k_by_seq={
            1: 2,
            3: 2,
        },
        row_seq_ids=torch.tensor([1, 1, 3], dtype=torch.int32),
        row_current_chunks=row_current_chunks,
        row_tail_lens=row_tail_lens,
        per_seq={
            1: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[4:6],
                selected_block_counts=selected_counts[4:6],
            ),
            3: efficient_components._EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[9:10],
                selected_block_counts=selected_counts[9:10],
            ),
        },
    )
    query_start_loc = torch.tensor([0, 4, 6, 9, 10], dtype=torch.int32)
    seq_lens = torch.tensor([8, 12, 7, 6], dtype=torch.int32)
    calls = []

    def fake_mixed_page_table(**kwargs):
        calls.append(kwargs)
        return (
            torch.empty(5, 3, dtype=torch.int32),
            torch.tensor([8, 9, 10, 7, 6], dtype=torch.int32),
            torch.tensor([0, 4, 5, 6, 9, 10], dtype=torch.int32),
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_unified_page_table_triton",
        fake_mixed_page_table,
    )

    result = provider.try_build_page_tables_batched(
        block_table=block_table,
        active_seq_infos=[
            (0, 0, 4, 8),
            (1, 4, 6, 12),
            (2, 6, 9, 7),
            (3, 9, 10, 6),
        ],
        sparse_infos=[
            (1, 4, 6, 12, 3, 10, torch.tensor([2, 2], dtype=torch.long)),
            (3, 9, 10, 6, 2, 5, torch.tensor([1], dtype=torch.long)),
        ],
        block_selection_by_seq=selection_by_seq,
        total_rows=10,
        device=torch.device("cpu"),
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        num_actual_tokens=10,
        active_seq_count=4,
        dense_decode_threshold=8,
        dense_prefill_threshold=8,
    )

    assert result is not None
    assert len(calls) == 1
    call = calls[0]
    assert call["selected_blocks"] is selected_blocks
    assert call["selected_counts"] is selected_counts
    assert call["query_start_loc"] is query_start_loc
    assert call["seq_lens"] is seq_lens
    assert call["num_requests"] == 5
    assert call["table_width"] == 3


def test_torch_nonchunked_representatives_are_token_keys():
    provider = TorchNonChunkedDSARepresentativeProvider(
        q_indexer_dim=2,
        num_kv_heads=1,
        head_dim=3,
    )
    key_len = 7
    key_states = torch.arange(21, dtype=torch.float32).view(key_len, 1, 3)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, 3, block_table)

    result = provider(
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
    )

    assert provider.is_available(result)
    representatives = provider.get_for_sequence(result)
    assert representatives is not None
    torch.testing.assert_close(representatives, key_states[..., :2])


def test_torch_nonchunked_scoring_masks_future_tokens():
    provider = TorchNonChunkedDSAScoringProvider(
        q_indexer_dim=2,
        logit_scale=1.5,
    )
    score_query_states = torch.tensor(
        [
            [1.0, 0.5],
            [0.25, -0.75],
        ],
        dtype=torch.float32,
    )
    token_representatives = torch.tensor(
        [
            [0.5, 0.25],
            [1.0, -1.0],
            [-0.5, 0.5],
            [0.75, 0.25],
        ],
        dtype=torch.float32,
    )
    query_positions = torch.tensor([1, 3], dtype=torch.long)

    result = provider(
        score_query_states=score_query_states,
        token_representatives=token_representatives,
        query_positions=query_positions,
    )

    assert provider.is_available(result)
    actual = provider.get_scores(result)
    assert actual is not None
    expected_valid = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    expected_logits = torch.matmul(
        score_query_states,
        token_representatives.transpose(0, 1),
    )
    expected_logits.mul_(1.5 / math.sqrt(2))
    expected_logits = expected_logits.masked_fill(
        ~expected_valid,
        torch.finfo(expected_logits.dtype).min,
    )
    torch.testing.assert_close(actual[0], expected_logits)
    torch.testing.assert_close(actual[1], expected_valid)


def test_torch_nonchunked_topk_selection_returns_token_indices():
    scoring_provider = TorchNonChunkedDSAScoringProvider(
        q_indexer_dim=2,
        logit_scale=1.0,
    )
    selection_provider = TorchTopKTokenDSASelectionProvider()
    score_state = scoring_provider(
        score_query_states=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        token_representatives=torch.tensor(
            [
                [0.25, 0.0],
                [0.75, 0.0],
                [0.5, 0.0],
                [1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        query_positions=torch.tensor([2], dtype=torch.long),
    )

    result = selection_provider(score_state=score_state, token_top_k=2)

    assert selection_provider.is_available(result)
    actual = selection_provider.get_selected_tokens(result)
    assert actual is not None
    torch.testing.assert_close(actual[0], torch.tensor([[1, 2]]))
    torch.testing.assert_close(actual[1], torch.tensor([[True, True]]))


def test_efficient_chunk_representatives_use_original_table_rows(monkeypatch):
    provider = TritonBatchedChunkedDSARepresentativeProvider(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
    )
    key_cache = torch.zeros(16, 4, 1, 2)
    block_table = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9],
        ],
        dtype=torch.int32,
    )
    expected_output = torch.arange(2 * 3 * 1 * 2, dtype=torch.float32).view(
        2,
        3,
        1,
        2,
    )
    calls = []

    def fake_dsa_block_summaries_triton(
        *,
        key_cache,
        block_table,
        seq_lens,
        q_indexer_dim,
        max_chunks,
    ):
        calls.append(
            (tuple(block_table.shape), seq_lens.tolist(), q_indexer_dim, max_chunks)
        )
        return expected_output.to(device=block_table.device)

    monkeypatch.setattr(
        efficient_components,
        "dsa_block_summaries_triton",
        fake_dsa_block_summaries_triton,
    )

    result = provider(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=torch.tensor([5, 9], dtype=torch.int32),
        active_seq_infos=[(0, 0, 1, 5), (1, 1, 2, 9)],
        cache_info=("NHD", 4),
        should_skip_sequence=lambda *_: False,
    )

    assert provider.is_available(result)
    assert calls == [((2, 5), [5, 9], 2, 3)]
    torch.testing.assert_close(
        provider.get_for_sequence(result, seq_idx=0),
        expected_output[0, :2],
    )
    torch.testing.assert_close(
        provider.get_for_sequence(result, seq_idx=1),
        expected_output[1, :3],
    )
    assert provider.get_for_sequence(result, seq_idx=99) is None


def test_efficient_chunk_representatives_return_unavailable_for_cpu_inputs():
    provider = TritonBatchedChunkedDSARepresentativeProvider(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
    )

    result = provider(
        key_cache=torch.zeros(4, 4, 1, 2),
        block_table=torch.zeros(2, 3, dtype=torch.long),
        seq_lens=torch.tensor([5, 9], dtype=torch.int32),
        active_seq_infos=[(0, 0, 1, 5), (1, 1, 2, 9)],
        cache_info=("NHD", 4),
    )

    assert not provider.is_available(result)
    assert provider.get_for_sequence(result, seq_idx=0) is None


def test_efficient_chunk_representatives_match_torch_on_cuda():
    if not torch.cuda.is_available() or not HAS_TRITON:
        pytest.skip("CUDA and Triton are required for efficient representative parity")

    torch.manual_seed(37)
    q_indexer_dim = 3
    chunk_size = 4
    num_kv_heads = 2
    head_dim = 5
    seq_lens = [1, 4, 5, 8, 10]
    max_chunks = 6
    block_table = torch.arange(
        len(seq_lens) * max_chunks,
        device="cuda",
        dtype=torch.long,
    ).view(len(seq_lens), max_chunks)
    key_cache = torch.randn(
        int(block_table.max().item()) + 1,
        chunk_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    torch_provider = TorchChunkedDSARepresentativeProvider(
        q_indexer_dim=q_indexer_dim,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )
    efficient_provider = TritonBatchedChunkedDSARepresentativeProvider(
        q_indexer_dim=q_indexer_dim,
        chunk_size=chunk_size,
        num_kv_heads=num_kv_heads,
    )

    result = efficient_provider(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
        query_start_loc=torch.arange(
            len(seq_lens) + 1,
            device="cuda",
            dtype=torch.int32,
        ),
        active_seq_infos=[
            (seq_idx, seq_idx, seq_idx + 1, key_len)
            for seq_idx, key_len in enumerate(seq_lens)
        ],
        cache_info=("NHD", chunk_size),
    )

    assert efficient_provider.is_available(result)
    for seq_idx, key_len in enumerate(seq_lens):
        expected_result = torch_provider(
            key_cache=key_cache,
            block_table=block_table[seq_idx],
            key_len=key_len,
        )
        actual = efficient_provider.get_for_sequence(result, seq_idx=seq_idx)
        assert actual is not None
        assert actual.dtype == torch.bfloat16
        expected = torch_provider.get_for_sequence(
            expected_result,
            seq_idx=seq_idx,
        )
        assert expected is not None
        torch.testing.assert_close(
            actual,
            expected.to(torch.bfloat16),
            atol=1e-2,
            rtol=1e-2,
        )


def test_efficient_fake_pages_share_nhd_storage_with_expected_strides():
    cache = torch.arange(3 * 4 * 2 * 5).view(3, 4, 2, 5)
    fake = efficient_components._make_nhd_fake_page_view(cache, num_kv_heads=2)
    assert fake is not None
    assert fake.shape == (18, 4, 1, 5)
    assert fake.stride() == (5, 10, 5, 1)
    assert fake.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert fake[17, 3, 0, 4] == cache[2, 3, 1, 4]


def test_efficient_fake_page_mapping_matches_physical_block_and_kv_head():
    cache = torch.arange(4 * 3 * 3 * 2).view(4, 3, 3, 2)
    fake = efficient_components._make_nhd_fake_page_view(cache, num_kv_heads=3)
    assert fake is not None
    for physical_block in range(4):
        for kv_head in range(3):
            fake_page = physical_block * 3 * 3 + kv_head
            torch.testing.assert_close(
                fake[fake_page, :, 0], cache[physical_block, :, kv_head]
            )


def test_efficient_fake_pages_skip_interleaved_kv_storage():
    backing = torch.arange(4 * 2 * 3 * 2 * 5).view(4, 2, 3, 2, 5)
    key_cache, value_cache = backing.unbind(1)
    assert key_cache.stride() == (60, 10, 5, 1)
    for cache in (key_cache, value_cache):
        fake = efficient_components._make_nhd_fake_page_view(cache, num_kv_heads=2)
        assert fake is not None
        assert fake.shape == (38, 3, 1, 5)
        assert fake.stride() == (5, 10, 5, 1)
        assert fake.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        for physical_block in range(4):
            for kv_head in range(2):
                fake_page = physical_block * 12 + kv_head
                torch.testing.assert_close(
                    fake[fake_page, :, 0], cache[physical_block, :, kv_head]
                )


def test_efficient_fake_pages_support_current_packed_kv_storage():
    packed = torch.arange(4 * 3 * 2 * 10).view(4, 3, 2, 10)
    key_cache, value_cache = packed.split(5, dim=-1)
    assert key_cache.stride() == (60, 20, 10, 1)
    for cache in (key_cache, value_cache):
        fake = efficient_components._make_nhd_fake_page_view(
            cache, num_kv_heads=2
        )
        assert fake is not None
        assert fake.shape == (20, 3, 1, 5)
        assert fake.stride() == (10, 20, 10, 1)
        assert fake.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        for physical_block in range(4):
            for kv_head in range(2):
                fake_page = physical_block * 6 + kv_head
                torch.testing.assert_close(
                    fake[fake_page, :, 0], cache[physical_block, :, kv_head]
                )


def test_efficient_fake_page_block_table_remap_is_in_storage_bounds():
    block_table = torch.tensor([[3, 0, 2], [1, 3, 0]], dtype=torch.int32)
    remapped = efficient_components._remap_nhd_block_table_for_kv_head(
        block_table, fake_page_pitch=24, num_kv_heads=3, kv_head_idx=2
    )
    torch.testing.assert_close(remapped, block_table * 24 + 2)
    assert int(remapped.max()) < (4 - 1) * 24 + 3


def test_efficient_multi_kv_metadata_expansion_restores_head_order(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=2,
        head_dim=3,
        logit_scale=1.0,
        num_heads=4,
    )
    query = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    backing = torch.arange(3 * 2 * 4 * 2 * 3, dtype=torch.float32).view(3, 2, 4, 2, 3)
    key_cache, value_cache = backing.unbind(1)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)
    output = torch.empty_like(query)
    planned_tables = []

    def fake_one_head_plan(self, **kwargs):
        del self
        planned_tables.append(kwargs["block_table"])
        kwargs["flash_attn_override"](
            q=kwargs["query_states"].contiguous(),
            k=kwargs["key_cache"],
            v=kwargs["value_cache"],
            out=kwargs["output"],
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            seqused_k=torch.tensor([8], dtype=torch.int32),
            max_seqlen_q=2,
            max_seqlen_k=8,
            block_table=kwargs["block_table"],
            dropout_p=0.0,
            softmax_scale=1.0,
            causal=True,
        )
        return {0}

    real_calls = []

    def fake_flash_attn(**kwargs):
        real_calls.append(kwargs)
        kwargs["out"].copy_(kwargs["q"] + 7)

    monkeypatch.setattr(
        pytorch_components.ChunkedDSAAttentionProviderMixin,
        "_forward_dsa_chunked_one_kv_head_page_table_fa_bucket",
        fake_one_head_plan,
    )
    monkeypatch.setattr(
        efficient_components, "_get_flash_attn_varlen_func", lambda: fake_flash_attn
    )
    handled = provider._forward_dsa_chunked_multi_kv_head_page_table_fa_bucket(
        hidden_states=torch.empty(2, 1),
        query_states=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn=SimpleNamespace(),
        attn_metadata=None,
        positions=torch.arange(2),
        active_seq_infos=[(0, 0, 2, 8)],
        batched_chunk_representatives=None,
        output=output,
        indexer_q_proj=lambda hidden: (hidden, None),
        local_kv_head_indices=torch.tensor([0, 1]),
    )
    assert handled == {0}
    assert len(real_calls) == 2
    torch.testing.assert_close(output, query + 7)
    torch.testing.assert_close(planned_tables[0], block_table * 16)
    torch.testing.assert_close(planned_tables[1], block_table * 16 + 1)
    assert all(
        call["k"].untyped_storage().data_ptr() == key_cache.untyped_storage().data_ptr()
        for call in real_calls
    )
    assert all(
        call["v"].untyped_storage().data_ptr()
        == value_cache.untyped_storage().data_ptr()
        for call in real_calls
    )


def test_efficient_one_kv_head_dispatch_remains_on_single_head_path(monkeypatch):
    provider = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=2,
        chunk_size=4,
        num_kv_heads=1,
        head_dim=3,
        logit_scale=1.0,
        num_heads=2,
    )
    sentinel = {9}

    def fake_single(self, **kwargs):
        del self, kwargs
        return sentinel

    monkeypatch.setattr(
        pytorch_components.ChunkedDSAAttentionProviderMixin,
        "_forward_dsa_chunked_single_kv_head_page_table_fa_bucket",
        fake_single,
    )
    result = provider._forward_dsa_chunked_unified_page_table_fa_bucket(
        hidden_states=torch.empty(1, 1),
        query_states=torch.empty(1, 2, 3),
        key_cache=torch.empty(1, 4, 1, 3),
        value_cache=torch.empty(1, 4, 1, 3),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        attn=SimpleNamespace(),
        attn_metadata=None,
        positions=torch.zeros(1, dtype=torch.long),
        active_seq_infos=[(0, 0, 1, 1)],
        batched_chunk_representatives=None,
        output=torch.empty(1, 2, 3),
        indexer_q_proj=lambda hidden: (hidden, None),
        local_kv_head_indices=torch.tensor([0]),
    )
    assert result is sentinel
