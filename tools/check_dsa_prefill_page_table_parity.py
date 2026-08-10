#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("VLLM_DSA_PARITY_USE_SITE_PACKAGE", "0") != "1":
    sys.path.insert(0, str(REPO_ROOT))

from vllm.model_executor.models import nemotron_h
from vllm.model_executor.models.nemotron_h import NemotronHDSASelectiveAttention


def _make_chunked_dsa_attn() -> NemotronHDSASelectiveAttention:
    attn = NemotronHDSASelectiveAttention.__new__(NemotronHDSASelectiveAttention)
    attn.q_indexer_chunk_size = 4
    attn.q_indexer_chunk_top_k = 1
    attn.q_indexer_chunked_query_chunk_size = 8
    attn.q_indexer_logit_scale = 1.0
    attn.q_indexer_dim = 2
    attn.q_indexer_attn_mode = "chunked_topk_sparse"
    attn.num_kv_heads = 2
    attn.num_heads = 4
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
    block_table: torch.Tensor,
) -> torch.Tensor:
    key_len, num_kv_heads, head_dim = value_states.shape
    num_blocks = int(block_table.max().item()) + 1
    cache = value_states.new_zeros(num_blocks, block_size, num_kv_heads, head_dim)
    for token in range(key_len):
        block_id = int(block_table[token // block_size].item())
        cache[block_id, token % block_size] = value_states[token]
    return cache


def _fake_flash_attn_varlen_func(block_size: int):
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
        assert max_seqlen_k == int(seqused_k.max().item())
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

    return fake_flash_attn_varlen_func


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
            weights = torch.softmax(logits * softmax_scale, dim=-1).to(
                query_states.dtype
            )
            head_rows.append(torch.mv(selected_v.transpose(0, 1), weights))
        rows.append(torch.stack(head_rows))
    return torch.stack(rows)


def _fake_full_flash_attn_varlen_func(
    block_size: int,
    positions: torch.Tensor,
):
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
        key_len = int(seqused_k[0].item())
        assert cu_seqlens_q.tolist() == [0, q.shape[0]]
        assert max_seqlen_q == q.shape[0]
        assert max_seqlen_k == key_len
        assert dropout_p == 0.0
        assert causal is True

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
        out.copy_(
            _causal_full_attention_reference(
                q,
                dense_k,
                dense_v,
                positions,
                softmax_scale,
            )
        )
        return out

    return fake_flash_attn_varlen_func


def main() -> None:
    nemotron_h._DSA_DEBUG_FORWARD_PRINT_COUNT = (
        nemotron_h._DSA_DEBUG_FORWARD_PRINT_LIMIT
    )
    nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = (
        nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT
    )
    attn = _make_chunked_dsa_attn()
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

    original_flash_attn = nemotron_h.flash_attn_varlen_func
    nemotron_h.flash_attn_varlen_func = _fake_flash_attn_varlen_func(block_size)
    try:
        attn.q_indexer_use_prefill_page_table_fa = True
        page_table_output = attn._forward_dsa_chunked_sequence(
            query_states=query_states,
            indexer_query_states=indexer_query_states,
            key_states=key_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=None,
            positions=positions,
        )
    finally:
        nemotron_h.flash_attn_varlen_func = original_flash_attn

    torch.testing.assert_close(page_table_output, gather_output)
    assert all(int(pos.item()) >= 0 for pos in positions)
    assert math.ceil(key_len / block_size) == len(block_table)
    print("prefill_page_table_parity=passed")

    attn.q_indexer_chunk_top_k = math.ceil(key_len / block_size)
    attn.q_indexer_use_full_attention_short_seq = True
    full_query_len = 5
    full_positions = torch.arange(key_len - full_query_len, key_len)
    full_query_states = torch.randn(full_query_len, attn.num_heads, attn.head_dim)
    original_flash_attn = nemotron_h.flash_attn_varlen_func
    nemotron_h.flash_attn_varlen_func = _fake_full_flash_attn_varlen_func(
        block_size,
        full_positions,
    )
    try:
        full_output = attn._forward_dsa_full_page_table_fa_sequence(
            query_states=full_query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=None,
            positions=full_positions,
            key_len=key_len,
        )
    finally:
        nemotron_h.flash_attn_varlen_func = original_flash_attn

    expected_full = _causal_full_attention_reference(
        full_query_states,
        key_states,
        value_states,
        full_positions,
        1.0 / math.sqrt(attn.head_dim),
    )
    torch.testing.assert_close(full_output, expected_full)
    print("short_sequence_full_page_table_parity=passed")


if __name__ == "__main__":
    main()
