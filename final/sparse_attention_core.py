"""Core flattened sparse attention paths shared by the final benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass(frozen=True)
class FlashAttentionConfig:
    block_size: int
    heads: int
    kv_heads: int
    head_dim: int
    fa_version: int = 2
    num_splits: int = 0


def make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return cu


def lse_as_rows(softmax_lse: torch.Tensor, total_q: int, num_heads: int) -> torch.Tensor:
    if softmax_lse.shape == (num_heads, total_q):
        return softmax_lse.transpose(0, 1).contiguous()
    if softmax_lse.shape == (total_q, num_heads):
        return softmax_lse.contiguous()
    raise ValueError(f"unexpected LSE shape: {tuple(softmax_lse.shape)}")


def make_flattened_grouped_kv(
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    *,
    batch_size: int,
    context_len: int,
    config: FlashAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = context_len // config.block_size
    flat_key_cache = grouped_key_cache.view(
        config.kv_heads,
        batch_size,
        num_blocks,
        config.block_size,
        1,
        config.head_dim,
    ).permute(1, 0, 2, 3, 4, 5).reshape(
        batch_size * config.kv_heads * num_blocks,
        config.block_size,
        1,
        config.head_dim,
    )
    flat_value_cache = grouped_value_cache.view(
        config.kv_heads,
        batch_size,
        num_blocks,
        config.block_size,
        1,
        config.head_dim,
    ).permute(1, 0, 2, 3, 4, 5).reshape(
        batch_size * config.kv_heads * num_blocks,
        config.block_size,
        1,
        config.head_dim,
    )
    return flat_key_cache, flat_value_cache


def make_flattened_dense_block_table(
    *,
    batch_size: int,
    context_len: int,
    device: torch.device,
    config: FlashAttentionConfig,
) -> torch.Tensor:
    num_blocks = context_len // config.block_size
    return torch.arange(
        batch_size * config.kv_heads * num_blocks,
        device=device,
        dtype=torch.int32,
    ).view(batch_size * config.kv_heads, num_blocks)


def build_decode_sparse_block_table(
    full_block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
) -> torch.Tensor:
    """Build [top-k full remote blocks..., current/latest block].

    ``selected_blocks`` is shaped [batch, kv_heads, active_blocks] and contains
    logical block indices over the full per-row block table. The final column is
    the current/latest logical block.
    """
    flat_rows = full_block_table.shape[0]
    active_blocks = selected_blocks.shape[-1]
    flat_selected = selected_blocks.view(flat_rows, active_blocks)
    return torch.gather(full_block_table, dim=-1, index=flat_selected)


def make_decode_sparse_seqused_k(
    local_prefixes: torch.Tensor,
    *,
    remote_blocks: int,
    config: FlashAttentionConfig,
) -> torch.Tensor:
    return (
        remote_blocks * config.block_size
        + local_prefixes.view(-1, 1).expand(-1, config.kv_heads).reshape(-1)
    ).to(torch.int32)


def flash_sparse_requests(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    config: FlashAttentionConfig,
    request_chunk_size: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    out_chunks = []
    lse_chunks = []
    chunk_size = request_chunk_size or q.shape[0]
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        cu_seqlens_q = torch.arange(end - start + 1, device=q.device, dtype=torch.int32)
        result = flash_attn_varlen_func(
            q=q[start:end],
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            seqused_k=seqused_k[start:end],
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            causal=False,
            dropout_p=0.0,
            block_table=block_table[start:end],
            return_softmax_lse=return_lse,
            fa_version=config.fa_version,
            num_splits=config.num_splits,
        )
        if return_lse:
            chunk_out, chunk_lse = result
            out_chunks.append(chunk_out)
            lse_chunks.append(lse_as_rows(chunk_lse, chunk_out.shape[0], chunk_out.shape[1]))
        else:
            out_chunks.append(result)

    out = torch.cat(out_chunks, dim=0) if len(out_chunks) > 1 else out_chunks[0]
    if not return_lse:
        return out
    lse = torch.cat(lse_chunks, dim=0) if len(lse_chunks) > 1 else lse_chunks[0]
    return out, lse


def flattened_sparse_decode(
    q: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    selected_blocks: torch.Tensor,
    full_block_table: torch.Tensor,
    local_prefixes: torch.Tensor,
    flat_cu_seqlens_q: torch.Tensor,
    *,
    config: FlashAttentionConfig,
) -> torch.Tensor:
    batch_size = q.shape[0]
    active_blocks = selected_blocks.shape[-1]
    remote_blocks = active_blocks - 1
    q_heads_per_kv = config.heads // config.kv_heads
    q_flat = q.view(batch_size, config.kv_heads, q_heads_per_kv, config.head_dim).reshape(
        batch_size * config.kv_heads,
        q_heads_per_kv,
        config.head_dim,
    )
    block_table = build_decode_sparse_block_table(
        full_block_table,
        selected_blocks,
    )
    seqused_k = make_decode_sparse_seqused_k(
        local_prefixes,
        remote_blocks=remote_blocks,
        config=config,
    )
    out_flat = flash_attn_varlen_func(
        q=q_flat,
        k=flat_key_cache,
        v=flat_value_cache,
        cu_seqlens_q=flat_cu_seqlens_q,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=active_blocks * config.block_size,
        causal=False,
        dropout_p=0.0,
        block_table=block_table,
        return_softmax_lse=False,
        fa_version=config.fa_version,
        num_splits=config.num_splits,
    )
    return out_flat.view(
        batch_size,
        config.heads,
        config.head_dim,
    )


def flattened_sparse_prefill(
    q: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    remote_blocks: torch.Tensor,
    remote_lens: torch.Tensor,
    *,
    context_len: int,
    active_blocks_per_token: int,
    request_chunk_size: int,
    config: FlashAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_len = q.shape[:2]
    num_blocks = context_len // config.block_size
    q_heads_per_kv = config.heads // config.kv_heads
    remote_slots = active_blocks_per_token - 1
    flat_requests = batch_size * query_len * config.kv_heads
    device = q.device

    q_flat = q.view(
        batch_size,
        query_len,
        config.kv_heads,
        q_heads_per_kv,
        config.head_dim,
    ).reshape(flat_requests, q_heads_per_kv, config.head_dim)

    query_start = context_len - query_len
    q_pos = torch.arange(query_start, context_len, device=device, dtype=torch.int32)
    q_blocks = q_pos // config.block_size
    local_prefix = q_pos % config.block_size + 1
    all_remote_slots_filled = query_start // config.block_size >= remote_slots
    flat_batch_group_offsets = (
        torch.arange(batch_size * config.kv_heads, device=device, dtype=torch.int32)
        .view(batch_size, 1, config.kv_heads)
        .expand(batch_size, query_len, config.kv_heads)
        .reshape(flat_requests)
        * num_blocks
    )
    flat_current_blocks = (
        flat_batch_group_offsets
        + q_blocks.view(1, query_len, 1).expand(batch_size, query_len, config.kv_heads).reshape(flat_requests)
    )
    flat_local_prefix = (
        local_prefix.view(1, query_len, 1)
        .expand(batch_size, query_len, config.kv_heads)
        .reshape(flat_requests)
    )

    block_table = torch.empty(flat_requests, active_blocks_per_token, device=device, dtype=torch.int32)
    if all_remote_slots_filled:
        seqused_k = (remote_slots * config.block_size + flat_local_prefix).to(torch.int32)
        remote_flat = remote_blocks.reshape(flat_requests, remote_slots)
        if remote_slots > 0:
            block_table[:, :remote_slots] = remote_flat + flat_batch_group_offsets.view(flat_requests, 1)
        block_table[:, remote_slots] = flat_current_blocks
    else:
        flat_remote_lens = (
            remote_lens.view(1, query_len, 1)
            .expand(batch_size, query_len, config.kv_heads)
            .reshape(flat_requests)
        )
        seqused_k = (flat_remote_lens * config.block_size + flat_local_prefix).to(torch.int32)
        block_table.zero_()
        if remote_slots > 0:
            remote_flat = remote_blocks.reshape(flat_requests, remote_slots)
            block_table[:, :remote_slots] = torch.where(
                remote_flat >= 0,
                remote_flat + flat_batch_group_offsets.view(flat_requests, 1),
                torch.zeros_like(remote_flat),
            )
        block_table.scatter_(1, flat_remote_lens.long().unsqueeze(1), flat_current_blocks.unsqueeze(1))

    flat_out, flat_lse = flash_sparse_requests(
        q=q_flat,
        k=flat_key_cache,
        v=flat_value_cache,
        block_table=block_table,
        seqused_k=seqused_k,
        max_seqlen_k=active_blocks_per_token * config.block_size,
        config=config,
        request_chunk_size=request_chunk_size,
        return_lse=True,
    )
    out = flat_out.view(
        batch_size,
        query_len,
        config.heads,
        config.head_dim,
    )
    lse = flat_lse.view(
        batch_size,
        query_len,
        config.heads,
    )
    return out, lse
