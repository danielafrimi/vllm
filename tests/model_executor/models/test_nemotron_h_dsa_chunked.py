# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import torch

from vllm.model_executor.models import nemotron_h
from vllm.model_executor.models.nemotron_h import NemotronHDSASelectiveAttention


def _make_chunked_dsa_attn() -> NemotronHDSASelectiveAttention:
    attn = NemotronHDSASelectiveAttention.__new__(NemotronHDSASelectiveAttention)
    attn.q_indexer_chunk_size = 4
    attn.q_indexer_chunk_top_k = 1
    attn.q_indexer_chunked_query_chunk_size = 3
    attn.q_indexer_logit_scale = 1.0
    attn.q_indexer_dim = 2
    attn.q_indexer_attn_mode = "chunked_topk_sparse"
    attn.num_kv_heads = 1
    attn.num_heads = 1
    attn.head_dim = 2
    attn.layer_idx = 0
    attn.q_indexer_use_page_table_fa = False
    attn.q_indexer_use_prefill_page_table_fa = False
    attn.q_indexer_use_full_attention_short_seq = False
    attn.q_indexer_share_chunk_topk = False
    attn.q_indexer_use_shared_prefill_page_table_fa = False
    attn.q_indexer_use_union_prefill_kernel = False
    attn.q_indexer_use_union_superset_prefill_page_table_fa = False
    attn.q_indexer_union_chunks_per_iter = 8
    attn._dsa_cache_config_block_size = attn.q_indexer_chunk_size
    attn.q_indexer_use_summary_cache = True
    attn.q_indexer_summary_cache_max_blocks = 1024
    attn._dsa_summary_cache_block_ids = None
    attn._dsa_summary_cache_values = None
    attn._dsa_summary_cache_valid = None
    attn._dsa_summary_cache_block_size = None
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


def _causal_full_attention_reference(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    positions: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    num_heads = query_states.shape[1]
    num_kv_heads = key_states.shape[1]
    group_size = num_heads // num_kv_heads
    rows = []
    for row, position in enumerate(positions.tolist()):
        head_rows = []
        for head_idx in range(num_heads):
            kv_head_idx = head_idx // group_size
            selected_k = key_states[: position + 1, kv_head_idx].float()
            selected_v = value_states[: position + 1, kv_head_idx]
            logits = torch.mv(selected_k, query_states[row, head_idx].float())
            weights = torch.softmax(logits * softmax_scale, dim=-1).to(query_states.dtype)
            head_rows.append(torch.mv(selected_v.transpose(0, 1), weights))
        rows.append(torch.stack(head_rows))
    return torch.stack(rows)


def test_dsa_chunk_representatives_average_partial_chunk():
    attn = _make_chunked_dsa_attn()
    key_states = torch.arange(20, dtype=torch.float32).view(10, 1, 2)

    representatives = attn._build_indexer_chunk_representatives(key_states)

    expected = torch.stack(
        [
            key_states[0:4].mean(dim=0),
            key_states[4:8].mean(dim=0),
            key_states[8:10].mean(dim=0),
        ]
    )
    torch.testing.assert_close(representatives, expected)


def test_dsa_chunk_representatives_use_summary_cache_from_page_blocks():
    attn = _make_chunked_dsa_attn()
    key_len = 10
    block_size = attn.q_indexer_chunk_size
    key_states = torch.arange(20, dtype=torch.float32).view(key_len, 1, 2)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)

    representatives = attn._get_indexer_chunk_representatives(
        key_states=None,
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
    )

    expected = attn._build_indexer_chunk_representatives(key_states)
    torch.testing.assert_close(representatives, expected)
    assert attn._dsa_summary_cache_block_ids.tolist() == [0, 2]
    assert attn._dsa_summary_cache_valid.tolist() == [True, True]


def test_dsa_summary_cache_invalidates_written_slots():
    attn = _make_chunked_dsa_attn()
    key_len = 8
    block_size = attn.q_indexer_chunk_size
    key_states = torch.arange(16, dtype=torch.float32).view(key_len, 1, 2)
    block_table = torch.tensor([2, 0], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)

    attn._get_indexer_chunk_representatives(
        key_states=None,
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
    )

    key_cache[2, :, :, :].add_(100.0)
    attn._invalidate_dsa_summary_cache_for_slots(
        SimpleNamespace(slot_mapping=torch.tensor([2 * block_size + 1])),
        block_size=block_size,
    )

    representatives = attn._get_indexer_chunk_representatives(
        key_states=None,
        key_cache=key_cache,
        block_table=block_table,
        key_len=key_len,
    )

    expected = torch.stack((key_cache[2].mean(dim=0), key_cache[0].mean(dim=0)))
    torch.testing.assert_close(representatives, expected)
    assert attn._dsa_summary_cache_valid.tolist() == [True, True]


def test_dsa_chunked_recall_matches_causal_reference():
    nemotron_h._DSA_DEBUG_FORWARD_PRINT_COUNT = (
        nemotron_h._DSA_DEBUG_FORWARD_PRINT_LIMIT
    )
    attn = _make_chunked_dsa_attn()
    torch.manual_seed(0)

    key_len = 10
    block_size = 4
    key_states = torch.randn(key_len, 1, 2)
    value_states = torch.randn(key_len, 1, 2)
    query_states = torch.randn(key_len, 1, 2)
    indexer_query_states = torch.randn(key_len, 1, 2)
    positions = torch.arange(key_len)
    block_table = torch.arange(math.ceil(key_len / block_size))
    key_cache = _pack_nhd_cache(key_states, block_size)
    value_cache = _pack_nhd_cache(value_states, block_size)

    output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    representatives = attn._build_indexer_chunk_representatives(key_states)
    indexer_scale = 1.0 / math.sqrt(attn.q_indexer_dim)
    main_scale = 1.0 / math.sqrt(attn.head_dim)
    expected_rows = []
    for position in range(key_len):
        current_chunk = position // attn.q_indexer_chunk_size
        recall_indices = []
        if current_chunk > 0:
            logits = torch.mv(
                representatives[:current_chunk, 0],
                indexer_query_states[position, 0].float(),
            )
            top_chunk = int((logits * indexer_scale).topk(k=1).indices[0])
            chunk_start = top_chunk * attn.q_indexer_chunk_size
            chunk_end = min(chunk_start + attn.q_indexer_chunk_size, key_len)
            recall_indices.extend(range(chunk_start, chunk_end))

        tail_start = current_chunk * attn.q_indexer_chunk_size
        recall_indices.extend(range(tail_start, position + 1))
        assert all(index <= position for index in recall_indices)

        selected_k = key_states[recall_indices, 0].float()
        selected_v = value_states[recall_indices, 0]
        logits = torch.mv(selected_k, query_states[position, 0].float())
        weights = torch.softmax(logits * main_scale, dim=-1)
        expected_rows.append(torch.mv(selected_v.transpose(0, 1), weights))

    expected = torch.stack(expected_rows).view(key_len, 1, 2)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_dsa_short_sequence_full_page_table_fa_matches_causal_reference(
    monkeypatch,
):
    nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = (
        nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT
    )
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    attn.q_indexer_chunk_top_k = 3
    attn.q_indexer_use_full_attention_short_seq = True
    torch.manual_seed(4)

    key_len = 9
    query_len = 5
    block_size = attn.q_indexer_chunk_size
    key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    query_states = torch.randn(query_len, attn.num_heads, attn.head_dim)
    positions = torch.arange(key_len - query_len, key_len)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)
    value_cache = _pack_nhd_cache(value_states, block_size, block_table)
    calls = []

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
        assert cu_seqlens_q.tolist() == [0, query_len]
        assert max_seqlen_q == query_len
        assert seqused_k.tolist() == [key_len]
        assert max_seqlen_k == key_len
        assert dropout_p == 0.0
        assert causal is True
        assert "fa_version" in kwargs
        calls.append(block_table.detach().cpu().clone())

        dense_k = []
        dense_v = []
        for token_idx in range(key_len):
            page_idx = token_idx // block_size
            page_offset = token_idx % block_size
            block_id = int(block_table[0, page_idx].item())
            dense_k.append(k[block_id, page_offset])
            dense_v.append(v[block_id, page_offset])
        dense_k = torch.stack(dense_k)
        dense_v = torch.stack(dense_v)
        expected = _causal_full_attention_reference(
            q,
            dense_k,
            dense_v,
            positions,
            softmax_scale,
        )
        out.copy_(expected)
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )

    output = attn._forward_dsa_full_page_table_fa_sequence(
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
        key_len=key_len,
    )

    expected = _causal_full_attention_reference(
        query_states,
        key_states,
        value_states,
        positions,
        1.0 / math.sqrt(attn.head_dim),
    )
    torch.testing.assert_close(output, expected)
    assert [call.tolist()[0] for call in calls] == [[2, 0, 3]]


def test_dsa_short_sequence_full_page_table_fa_requires_tail_positions(
    monkeypatch,
):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_use_full_attention_short_seq = True

    def unexpected_flash_attn_varlen_func(**kwargs):
        raise AssertionError("full page-table FA should reject non-tail positions")

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        unexpected_flash_attn_varlen_func,
    )

    output = attn._forward_dsa_full_page_table_fa_sequence(
        query_states=torch.zeros(2, 1, 2),
        key_cache=torch.zeros(1, 4, 1, 2),
        value_cache=torch.zeros(1, 4, 1, 2),
        block_table=torch.tensor([0], dtype=torch.int32),
        attn_metadata=None,
        positions=torch.tensor([0, 2]),
        key_len=3,
    )

    assert output is None


def test_dsa_chunked_page_table_fa_prefill_matches_gather_path(monkeypatch):
    nemotron_h._DSA_DEBUG_FORWARD_PRINT_COUNT = (
        nemotron_h._DSA_DEBUG_FORWARD_PRINT_LIMIT
    )
    nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = (
        nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT
    )
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    attn.q_indexer_chunk_top_k = 1
    attn.q_indexer_chunked_query_chunk_size = 8
    torch.manual_seed(2)

    key_len = 12
    block_size = attn.q_indexer_chunk_size
    key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    query_states = torch.randn(8, attn.num_heads, attn.head_dim)
    indexer_query_states = torch.randn(8, attn.num_kv_heads, attn.q_indexer_dim)
    positions = torch.arange(4, 12)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)
    value_cache = _pack_nhd_cache(value_states, block_size, block_table)

    gather_output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    calls = []

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
        assert cu_seqlens_q.tolist() == list(range(q.shape[0] + 1))
        assert max_seqlen_q == 1
        assert dropout_p == 0.0
        assert causal is False
        assert "fa_version" in kwargs
        assert max_seqlen_k == int(seqused_k.max().item())
        calls.append(block_table.detach().cpu().clone())

        for row in range(q.shape[0]):
            recalled_tokens = int(seqused_k[row].item())
            selected_k = []
            selected_v = []
            for token_idx in range(recalled_tokens):
                page_idx = token_idx // block_size
                page_offset = token_idx % block_size
                block_id = int(block_table[row, page_idx].item())
                selected_k.append(k[block_id, page_offset, 0])
                selected_v.append(v[block_id, page_offset, 0])
            selected_k = torch.stack(selected_k)
            selected_v = torch.stack(selected_v)
            logits = torch.einsum("hd,kd->hk", q[row].float(), selected_k.float())
            weights = torch.softmax(logits * softmax_scale, dim=-1).to(q.dtype)
            out[row].copy_(torch.einsum("hk,kd->hd", weights, selected_v))
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )
    attn.q_indexer_use_prefill_page_table_fa = True
    page_table_output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=None,
        key_len=key_len,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    torch.testing.assert_close(page_table_output, gather_output)
    assert len(calls) == attn.num_kv_heads


def test_dsa_chunked_page_table_fa_prefill_falls_back_for_invalid_top_chunks(
    monkeypatch,
):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_chunk_top_k = 2
    attn.q_indexer_chunked_query_chunk_size = 8
    attn.q_indexer_use_prefill_page_table_fa = True
    torch.manual_seed(3)

    key_len = 12
    block_size = attn.q_indexer_chunk_size
    key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    query_states = torch.randn(8, attn.num_heads, attn.head_dim)
    indexer_query_states = torch.randn(8, attn.num_kv_heads, attn.q_indexer_dim)
    positions = torch.arange(4, 12)
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)
    value_cache = _pack_nhd_cache(value_states, block_size, block_table)

    def unexpected_flash_attn_varlen_func(**kwargs):
        raise AssertionError("prefill page-table FA should fall back")

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        unexpected_flash_attn_varlen_func,
    )

    output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    assert output.shape == query_states.shape


def test_dsa_chunked_page_table_fa_decode_matches_gather_path(monkeypatch):
    nemotron_h._DSA_DEBUG_FORWARD_PRINT_COUNT = (
        nemotron_h._DSA_DEBUG_FORWARD_PRINT_LIMIT
    )
    nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = (
        nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT
    )
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    attn.q_indexer_chunk_top_k = 1
    torch.manual_seed(1)

    key_len = 10
    block_size = attn.q_indexer_chunk_size
    key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    query_states = torch.randn(1, attn.num_heads, attn.head_dim)
    indexer_query_states = torch.randn(1, attn.num_kv_heads, attn.q_indexer_dim)
    positions = torch.tensor([key_len - 1])
    block_table = torch.tensor([2, 0, 3], dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)
    value_cache = _pack_nhd_cache(value_states, block_size, block_table)

    gather_output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    calls = []

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
        assert cu_seqlens_q.tolist() == [0, 1]
        assert max_seqlen_q == 1
        assert dropout_p == 0.0
        assert causal is False
        assert "fa_version" in kwargs
        recalled_tokens = int(seqused_k[0].item())
        assert max_seqlen_k == recalled_tokens
        calls.append(block_table.detach().cpu().clone())

        selected_k = []
        selected_v = []
        for token_idx in range(recalled_tokens):
            page_idx = token_idx // block_size
            page_offset = token_idx % block_size
            block_id = int(block_table[0, page_idx].item())
            selected_k.append(k[block_id, page_offset, 0])
            selected_v.append(v[block_id, page_offset, 0])
        selected_k = torch.stack(selected_k)
        selected_v = torch.stack(selected_v)
        logits = torch.einsum("qhd,kd->hqk", q.float(), selected_k.float())
        weights = torch.softmax(logits * softmax_scale, dim=-1).to(q.dtype)
        out.copy_(torch.einsum("hqk,kd->qhd", weights, selected_v))
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )
    attn.q_indexer_use_page_table_fa = True
    page_table_output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=None,
        key_len=key_len,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )

    torch.testing.assert_close(page_table_output, gather_output)
    assert len(calls) == attn.num_kv_heads

    representatives = attn._build_indexer_chunk_representatives(key_states)
    current_chunk = int(positions[0].item()) // block_size
    expected_page_tables = []
    for group_idx in range(attn.num_kv_heads):
        logits = torch.mv(
            representatives[:current_chunk, group_idx],
            indexer_query_states[0, group_idx].float(),
        )
        top_chunk = int(logits.topk(k=1).indices[0].item())
        expected_page_tables.append(
            [
                int(block_table[top_chunk].item()),
                int(block_table[current_chunk].item()),
            ]
        )

    assert [call.tolist()[0] for call in calls] == expected_page_tables


def test_dsa_chunked_page_table_fa_rejects_hnd_layout(monkeypatch):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_use_page_table_fa = True
    key_cache = torch.zeros(1, 4, 1, 2)
    block_table = torch.tensor([0], dtype=torch.int32)

    def unexpected_flash_attn_varlen_func(**kwargs):
        raise AssertionError("page-table FA should reject HND layout")

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        unexpected_flash_attn_varlen_func,
    )
    monkeypatch.setattr(nemotron_h, "_get_dsa_kv_cache_layout", lambda: "HND")

    output = attn._forward_dsa_chunked_page_table_fa_decode(
        query_states=torch.zeros(1, 1, 2),
        key_cache=key_cache,
        value_cache=key_cache,
        block_table=block_table,
        attn_metadata=None,
        top_chunk_indices=torch.empty(1, 0, dtype=torch.long),
        top_chunk_valid=torch.empty(1, 0, dtype=torch.bool),
        current_chunks=torch.tensor([0]),
        query_positions=torch.tensor([0]),
        key_len=1,
        group_idx=0,
        softmax_scale=1.0,
    )

    assert output is None
