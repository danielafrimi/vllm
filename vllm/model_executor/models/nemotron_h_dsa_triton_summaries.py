# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton KV-page summary builder for Nemotron-H chunked DSA."""

from __future__ import annotations

import torch

try:
    from vllm.triton_utils import HAS_TRITON, tl, triton
except ImportError:
    HAS_TRITON = False
    tl = None
    triton = None


if HAS_TRITON and triton is not None and tl is not None:

    @triton.jit(
        do_not_specialize=[
            "key_stride_block",
            "key_stride_token",
            "key_stride_kv_head",
            "key_stride_dim",
            "seq_lens_stride",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "max_chunks",
            "kv_heads",
            "q_indexer_dim",
        ]
    )
    def _dsa_block_summaries_kernel(
        key_cache,
        block_table,
        seq_lens,
        output,
        key_stride_block,
        key_stride_token,
        key_stride_kv_head,
        key_stride_dim,
        seq_lens_stride,
        block_table_stride_seq,
        block_table_stride_chunk,
        max_chunks,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        BLOCK_D: tl.constexpr,
    ):
        seq_chunk = tl.program_id(0)
        kv_head = tl.program_id(1)
        seq = seq_chunk // max_chunks
        chunk = seq_chunk - seq * max_chunks
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < q_indexer_dim

        seq_len = tl.load(seq_lens + seq * seq_lens_stride)
        num_chunks = tl.cdiv(seq_len, block_size)
        active = chunk < num_chunks
        remaining = seq_len - chunk * block_size
        valid_len = tl.minimum(remaining, block_size)
        valid_len = tl.maximum(valid_len, 0)
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=0,
        )

        acc = tl.zeros((BLOCK_D,), tl.float32)
        for offset in tl.static_range(0, block_size):
            token_valid = active & (offset < valid_len)
            key_offsets = (
                physical_block * key_stride_block
                + offset * key_stride_token
                + kv_head * key_stride_kv_head
                + dims * key_stride_dim
            )
            values = tl.load(
                key_cache + key_offsets,
                mask=token_valid & dim_mask,
                other=0.0,
            )
            acc += values.to(tl.float32)

        denom = tl.maximum(valid_len, 1).to(tl.float32)
        acc = acc / denom
        out_offsets = (
            (((seq * max_chunks + chunk) * kv_heads + kv_head) * q_indexer_dim)
            + dims
        )
        tl.store(output + out_offsets, acc, mask=dim_mask)


    @triton.jit(
        do_not_specialize=[
            "representative_stride_seq",
            "representative_stride_chunk",
            "representative_stride_kv_head",
            "representative_stride_dim",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "query_lens_stride",
            "cache_stride_block",
            "cache_stride_kv_head",
            "cache_stride_dim",
            "max_chunks",
            "max_seed_chunks",
            "kv_heads",
            "q_indexer_dim",
            "num_cache_blocks",
        ]
    )
    def _dsa_seed_block_summary_cache_kernel(
        representatives,
        block_table,
        seq_lens,
        query_lens,
        cache_values,
        cache_valid,
        representative_stride_seq,
        representative_stride_chunk,
        representative_stride_kv_head,
        representative_stride_dim,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        query_lens_stride,
        cache_stride_block,
        cache_stride_kv_head,
        cache_stride_dim,
        max_chunks,
        max_seed_chunks,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        num_cache_blocks,
        BLOCK_D: tl.constexpr,
    ):
        seq_seed_chunk = tl.program_id(0)
        kv_head = tl.program_id(1)
        seq = seq_seed_chunk // max_seed_chunks
        relative_chunk = seq_seed_chunk - seq * max_seed_chunks
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < q_indexer_dim

        seq_len = tl.load(seq_lens + seq * seq_lens_stride)
        query_len = tl.load(query_lens + seq * query_lens_stride)
        num_chunks = tl.cdiv(seq_len, block_size)
        first_token = tl.maximum(seq_len - query_len, 0)
        first_chunk = first_token // block_size
        chunk = first_chunk + relative_chunk
        active = chunk < num_chunks
        full = active & ((chunk + 1) * block_size <= seq_len)
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=0,
        ).to(tl.int64)
        physical_valid = (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )

        representative_offsets = (
            seq * representative_stride_seq
            + chunk * representative_stride_chunk
            + kv_head * representative_stride_kv_head
            + dims * representative_stride_dim
        )
        values = tl.load(
            representatives + representative_offsets,
            mask=full & physical_valid & dim_mask,
            other=0.0,
        )
        cache_offsets = (
            physical_block * cache_stride_block
            + kv_head * cache_stride_kv_head
            + dims * cache_stride_dim
        )
        tl.store(
            cache_values + cache_offsets,
            values,
            mask=full & physical_valid & dim_mask,
        )

        # This kernel finishes before any cache consumer launches. Publishing
        # validity here is therefore safe even though KV heads are separate
        # programs. Partial current pages are explicitly invalidated.
        tl.store(
            cache_valid + physical_block,
            tl.where(full, 1, 0),
            mask=active & physical_valid & (kv_head == 0),
        )


    @triton.jit(
        do_not_specialize=[
            "key_stride_block",
            "key_stride_token",
            "key_stride_kv_head",
            "key_stride_dim",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "cache_stride_block",
            "cache_stride_kv_head",
            "cache_stride_dim",
            "max_chunks",
            "kv_heads",
            "q_indexer_dim",
            "num_cache_blocks",
            "total_entries",
        ]
    )
    def _dsa_cached_block_summaries_kernel(
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        output,
        key_stride_block,
        key_stride_token,
        key_stride_kv_head,
        key_stride_dim,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        cache_stride_block,
        cache_stride_kv_head,
        cache_stride_dim,
        max_chunks,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        num_cache_blocks,
        total_entries,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        seq_chunks = tl.program_id(0) * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        kv_head = tl.program_id(1)
        entry_valid = seq_chunks < total_entries
        seq = seq_chunks // max_chunks
        chunk = seq_chunks - seq * max_chunks
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < q_indexer_dim

        seq_len = tl.load(
            seq_lens + seq * seq_lens_stride,
            mask=entry_valid,
            other=0,
        )
        num_chunks = tl.cdiv(seq_len, block_size)
        active = entry_valid & (chunk < num_chunks)
        remaining = seq_len - chunk * block_size
        valid_len = tl.minimum(remaining, block_size)
        valid_len = tl.maximum(valid_len, 0)
        current = active & (chunk == num_chunks - 1)
        full = active & (valid_len == block_size)
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=0,
        ).to(tl.int64)
        physical_valid = (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )
        valid = tl.load(
            cache_valid + physical_block,
            mask=active & physical_valid,
            other=0,
        )
        cache_hit = active & physical_valid & ~current & (valid != 0)
        compute = active & physical_valid & ~cache_hit

        acc = tl.zeros((BLOCK_CHUNKS, BLOCK_D), tl.float32)
        for offset in tl.static_range(0, block_size):
            token_valid = compute & (offset < valid_len)
            key_offsets = (
                physical_block[:, None] * key_stride_block
                + offset * key_stride_token
                + kv_head * key_stride_kv_head
                + dims[None, :] * key_stride_dim
            )
            values = tl.load(
                key_cache + key_offsets,
                mask=token_valid[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += values.to(tl.float32)

        cache_offsets = (
            physical_block[:, None] * cache_stride_block
            + kv_head * cache_stride_kv_head
            + dims[None, :] * cache_stride_dim
        )
        cached = tl.load(
            cache_values + cache_offsets,
            mask=cache_hit[:, None] & dim_mask[None, :],
            other=0.0,
        )
        denom = tl.maximum(valid_len, 1).to(tl.float32)
        computed = acc / denom[:, None]
        result = tl.where(cache_hit[:, None], cached, computed)
        out_offsets = (
            (
                ((seq[:, None] * max_chunks + chunk[:, None]) * kv_heads + kv_head)
                * q_indexer_dim
            )
            + dims[None, :]
        )
        tl.store(
            output + out_offsets,
            result,
            mask=active[:, None] & dim_mask[None, :],
        )

        # Misses and the current page are written now, but validity is only
        # published by a separate kernel after every KV-head program finishes.
        tl.store(
            cache_values + cache_offsets,
            computed,
            mask=(compute & full)[:, None] & dim_mask[None, :],
        )


    @triton.jit(
        do_not_specialize=[
            "key_stride_block",
            "key_stride_token",
            "key_stride_kv_head",
            "key_stride_dim",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "cache_stride_block",
            "cache_stride_kv_head",
            "cache_stride_dim",
            "max_chunks",
            "kv_heads",
            "q_indexer_dim",
            "num_cache_blocks",
            "total_entries",
        ]
    )
    def _dsa_update_block_summary_cache_kernel(
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        key_stride_block,
        key_stride_token,
        key_stride_kv_head,
        key_stride_dim,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        cache_stride_block,
        cache_stride_kv_head,
        cache_stride_dim,
        max_chunks,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        num_cache_blocks,
        total_entries,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Populate sidecar misses without materializing logical summaries."""
        seq_chunks = tl.program_id(0) * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        kv_head = tl.program_id(1)
        entry_valid = seq_chunks < total_entries
        seq = seq_chunks // max_chunks
        chunk = seq_chunks - seq * max_chunks
        dims = tl.arange(0, BLOCK_D)
        dim_mask = dims < q_indexer_dim

        seq_len = tl.load(
            seq_lens + seq * seq_lens_stride,
            mask=entry_valid,
            other=0,
        )
        num_chunks = tl.cdiv(seq_len, block_size)
        active = entry_valid & (chunk < num_chunks)
        full = active & ((chunk + 1) * block_size <= seq_len)
        current = active & (chunk == num_chunks - 1)
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=0,
        ).to(tl.int64)
        physical_valid = (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )
        valid = tl.load(
            cache_valid + physical_block,
            mask=active & physical_valid,
            other=0,
        )

        # Historical misses are repaired once. A just-completed current page is
        # recomputed even if its physical ID was previously valid, which makes
        # page recycling safe. Partial current pages are neither read nor
        # summarized because selection only scores strictly prior chunks.
        compute = full & physical_valid & (current | (valid == 0))
        if tl.sum(compute.to(tl.int32), axis=0) > 0:
            acc = tl.zeros((BLOCK_CHUNKS, BLOCK_D), tl.float32)
            for offset in tl.static_range(0, block_size):
                key_offsets = (
                    physical_block[:, None] * key_stride_block
                    + offset * key_stride_token
                    + kv_head * key_stride_kv_head
                    + dims[None, :] * key_stride_dim
                )
                values = tl.load(
                    key_cache + key_offsets,
                    mask=compute[:, None] & dim_mask[None, :],
                    other=0.0,
                )
                acc += values.to(tl.float32)

            cache_offsets = (
                physical_block[:, None] * cache_stride_block
                + kv_head * cache_stride_kv_head
                + dims[None, :] * cache_stride_dim
            )
            computed = acc / block_size
            tl.store(
                cache_values + cache_offsets,
                computed,
                mask=compute[:, None] & dim_mask[None, :],
            )


    @triton.jit(
        do_not_specialize=[
            "key_stride_block",
            "key_stride_token",
            "key_stride_kv_head",
            "key_stride_dim",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "cache_stride_block",
            "cache_stride_kv_head",
            "cache_stride_dim",
            "kv_heads",
            "q_indexer_dim",
            "num_cache_blocks",
        ]
    )
    def _dsa_update_current_block_summary_cache_kernel(
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        key_stride_block,
        key_stride_token,
        key_stride_kv_head,
        key_stride_dim,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        cache_stride_block,
        cache_stride_kv_head,
        cache_stride_dim,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        num_cache_blocks,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Update only the current physical page for one decode sequence."""
        seq = tl.program_id(0)
        seq_len = tl.load(seq_lens + seq * seq_lens_stride)
        active = seq_len > 0
        current_chunk = (tl.maximum(seq_len, 1) - 1) // block_size
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + current_chunk * block_table_stride_chunk,
            mask=active,
            other=-1,
        ).to(tl.int64)
        physical_valid = active & (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )
        full = physical_valid & ((seq_len % block_size) == 0)

        heads = tl.arange(0, BLOCK_H)
        dims = tl.arange(0, BLOCK_D)
        head_mask = heads < kv_heads
        dim_mask = dims < q_indexer_dim
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
        for offset in tl.static_range(0, block_size):
            key_offsets = (
                physical_block * key_stride_block
                + offset * key_stride_token
                + heads[:, None] * key_stride_kv_head
                + dims[None, :] * key_stride_dim
            )
            values = tl.load(
                key_cache + key_offsets,
                mask=full & head_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += values.to(tl.float32)

        cache_offsets = (
            physical_block * cache_stride_block
            + heads[:, None] * cache_stride_kv_head
            + dims[None, :] * cache_stride_dim
        )
        tl.store(
            cache_values + cache_offsets,
            acc / block_size,
            mask=full & head_mask[:, None] & dim_mask[None, :],
        )
        # A recycled page is invalid while it is partial. One program writes all
        # local KV heads, so publishing a completed page here is race-free.
        tl.store(
            cache_valid + physical_block,
            tl.where(full, 1, 0),
            mask=physical_valid,
        )


    @triton.jit(
        do_not_specialize=[
            "key_stride_block",
            "key_stride_token",
            "key_stride_kv_head",
            "key_stride_dim",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "query_lens_stride",
            "cache_stride_block",
            "cache_stride_kv_head",
            "cache_stride_dim",
            "max_written_chunks",
            "kv_heads",
            "q_indexer_dim",
            "num_cache_blocks",
        ]
    )
    def _dsa_update_written_block_summary_cache_kernel(
        key_cache,
        block_table,
        seq_lens,
        query_lens,
        cache_values,
        cache_valid,
        key_stride_block,
        key_stride_token,
        key_stride_kv_head,
        key_stride_dim,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        query_lens_stride,
        cache_stride_block,
        cache_stride_kv_head,
        cache_stride_dim,
        max_written_chunks,
        block_size: tl.constexpr,
        kv_heads,
        q_indexer_dim,
        num_cache_blocks,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Refresh only physical pages touched by the current scheduler slice."""
        entry = tl.program_id(0)
        seq = entry // max_written_chunks
        relative_chunk = entry - seq * max_written_chunks

        seq_len = tl.load(seq_lens + seq * seq_lens_stride)
        query_len = tl.load(query_lens + seq * query_lens_stride)
        first_written_token = tl.maximum(seq_len - query_len, 0)
        first_written_chunk = first_written_token // block_size
        num_chunks = tl.cdiv(seq_len, block_size)
        written_chunks = tl.maximum(num_chunks - first_written_chunk, 0)
        active = (query_len > 0) & (relative_chunk < written_chunks)
        chunk = first_written_chunk + relative_chunk

        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=-1,
        ).to(tl.int64)
        physical_valid = active & (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )
        full = physical_valid & ((chunk + 1) * block_size <= seq_len)

        heads = tl.arange(0, BLOCK_H)
        dims = tl.arange(0, BLOCK_D)
        head_mask = heads < kv_heads
        dim_mask = dims < q_indexer_dim
        acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
        for offset in tl.static_range(0, block_size):
            key_offsets = (
                physical_block * key_stride_block
                + offset * key_stride_token
                + heads[:, None] * key_stride_kv_head
                + dims[None, :] * key_stride_dim
            )
            values = tl.load(
                key_cache + key_offsets,
                mask=full & head_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            acc += values.to(tl.float32)

        cache_offsets = (
            physical_block * cache_stride_block
            + heads[:, None] * cache_stride_kv_head
            + dims[None, :] * cache_stride_dim
        )
        tl.store(
            cache_values + cache_offsets,
            acc / block_size,
            mask=full & head_mask[:, None] & dim_mask[None, :],
        )
        # This program writes every local KV head, so validity can be
        # published without a second kernel or an inter-program race.
        tl.store(
            cache_valid + physical_block,
            tl.where(full, 1, 0),
            mask=physical_valid,
        )


    @triton.jit(
        do_not_specialize=[
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "seq_lens_stride",
            "max_chunks",
            "num_cache_blocks",
            "total_entries",
        ]
    )
    def _dsa_publish_block_summary_cache_kernel(
        block_table,
        seq_lens,
        cache_valid,
        block_table_stride_seq,
        block_table_stride_chunk,
        seq_lens_stride,
        max_chunks,
        block_size: tl.constexpr,
        num_cache_blocks,
        total_entries,
        BLOCK_ENTRIES: tl.constexpr,
    ):
        seq_chunks = tl.program_id(0) * BLOCK_ENTRIES + tl.arange(
            0, BLOCK_ENTRIES
        )
        entry_valid = seq_chunks < total_entries
        seq = seq_chunks // max_chunks
        chunk = seq_chunks - seq * max_chunks

        seq_len = tl.load(
            seq_lens + seq * seq_lens_stride,
            mask=entry_valid,
            other=0,
        )
        num_chunks = tl.cdiv(seq_len, block_size)
        active = entry_valid & (chunk < num_chunks)
        full = active & ((chunk + 1) * block_size <= seq_len)
        physical_block = tl.load(
            block_table
            + seq * block_table_stride_seq
            + chunk * block_table_stride_chunk,
            mask=active,
            other=0,
        ).to(tl.int64)
        physical_valid = (physical_block >= 0) & (
            physical_block < num_cache_blocks
        )
        old_valid = tl.load(
            cache_valid + physical_block,
            mask=active & physical_valid,
            other=0,
        )
        tl.store(
            cache_valid + physical_block,
            1,
            mask=full & physical_valid & (old_valid == 0),
        )
        tl.store(
            cache_valid + physical_block,
            0,
            mask=active & ~full & physical_valid,
        )


def dsa_block_summaries_triton(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indexer_dim: int,
    max_chunks: int | None = None,
) -> torch.Tensor | None:
    """Build DSA chunk representatives from NHD KV-cache pages.

    Returns ``None`` when the Triton path is unavailable or the inputs do not
    match the NHD/block-aligned assumptions, so callers can fall back cleanly.
    """
    if not HAS_TRITON or triton is None or tl is None:
        return None
    if not key_cache.is_cuda or not block_table.is_cuda or not seq_lens.is_cuda:
        return None
    if key_cache.dim() != 4 or block_table.dim() != 2 or seq_lens.dim() != 1:
        return None
    if (
        block_table.device != key_cache.device
        or seq_lens.device != key_cache.device
    ):
        return None
    batch, table_width = block_table.shape
    if seq_lens.shape[0] != batch:
        return None
    if max_chunks is None:
        max_chunks = int(table_width)
    else:
        max_chunks = int(max_chunks)

    _, block_size, kv_heads, head_dim = key_cache.shape
    if q_indexer_dim <= 0 or q_indexer_dim > head_dim:
        return None
    if block_size <= 0 or max_chunks < 0 or max_chunks > int(table_width):
        return None

    output = torch.empty(
        batch,
        max_chunks,
        kv_heads,
        q_indexer_dim,
        device=key_cache.device,
        dtype=torch.bfloat16,
    )
    if batch == 0 or max_chunks == 0:
        return output

    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_block_summaries_kernel[(batch * max_chunks, kv_heads)](
        key_cache,
        block_table,
        seq_lens,
        output,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(seq_lens.stride(0)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        max_chunks,
        block_size,
        kv_heads,
        q_indexer_dim,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output


def _validate_summary_cache(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    q_indexer_dim: int,
    max_chunks: int | None,
) -> tuple[int, int, int, int] | None:
    if not HAS_TRITON or triton is None or tl is None:
        return None
    if (
        not key_cache.is_cuda
        or not block_table.is_cuda
        or not seq_lens.is_cuda
        or not cache_values.is_cuda
        or not cache_valid.is_cuda
    ):
        return None
    if (
        key_cache.dim() != 4
        or block_table.dim() != 2
        or seq_lens.dim() != 1
        or cache_values.dim() != 3
        or cache_valid.dim() != 1
    ):
        return None
    if any(
        tensor.device != key_cache.device
        for tensor in (block_table, seq_lens, cache_values, cache_valid)
    ):
        return None
    if key_cache.dtype not in (torch.bfloat16, torch.float16):
        return None
    if cache_values.dtype != torch.bfloat16 or cache_valid.dtype != torch.uint8:
        return None
    if not cache_values.is_contiguous() or not cache_valid.is_contiguous():
        return None

    batch, table_width = block_table.shape
    if seq_lens.shape[0] != batch:
        return None
    if max_chunks is None:
        max_chunks = int(table_width)
    else:
        max_chunks = int(max_chunks)

    num_cache_blocks, block_size, kv_heads, head_dim = key_cache.shape
    expected_cache_shape = (num_cache_blocks, kv_heads, q_indexer_dim)
    if tuple(cache_values.shape) != expected_cache_shape:
        return None
    if tuple(cache_valid.shape) != (num_cache_blocks,):
        return None
    if q_indexer_dim <= 0 or q_indexer_dim > head_dim:
        return None
    if block_size <= 0 or max_chunks < 0 or max_chunks > int(table_width):
        return None
    return batch, max_chunks, block_size, kv_heads


def _dsa_seed_block_summary_cache_triton(
    *,
    representatives: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    block_size: int,
    max_seed_chunks: int,
) -> bool:
    """Implementation helper with the physical page size supplied explicitly."""
    batch, max_chunks, kv_heads, q_indexer_dim = representatives.shape
    if block_size <= 0 or max_seed_chunks <= 0:
        return False
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_seed_block_summary_cache_kernel[(batch * max_seed_chunks, kv_heads)](
        representatives,
        block_table,
        seq_lens,
        query_lens,
        cache_values,
        cache_valid,
        int(representatives.stride(0)),
        int(representatives.stride(1)),
        int(representatives.stride(2)),
        int(representatives.stride(3)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        int(query_lens.stride(0)),
        int(cache_values.stride(0)),
        int(cache_values.stride(1)),
        int(cache_values.stride(2)),
        max_chunks,
        max_seed_chunks,
        block_size,
        kv_heads,
        q_indexer_dim,
        int(cache_values.shape[0]),
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return True


def dsa_seed_block_summary_cache_triton(
    *,
    representatives: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    block_size: int,
    max_seed_chunks: int,
) -> bool:
    """Seed pages written by the current prefill or mixed-batch slice."""
    if representatives.dim() != 4:
        return False
    batch, max_chunks, kv_heads, q_indexer_dim = representatives.shape
    if (
        not HAS_TRITON
        or triton is None
        or tl is None
        or not representatives.is_cuda
        or not block_table.is_cuda
        or not seq_lens.is_cuda
        or not query_lens.is_cuda
        or not cache_values.is_cuda
        or not cache_valid.is_cuda
        or representatives.dtype != torch.bfloat16
        or cache_values.dtype != torch.bfloat16
        or cache_valid.dtype != torch.uint8
        or representatives.device != cache_values.device
        or block_table.device != cache_values.device
        or seq_lens.device != cache_values.device
        or query_lens.device != cache_values.device
        or cache_valid.device != cache_values.device
        or block_table.dim() != 2
        or seq_lens.dim() != 1
        or query_lens.dim() != 1
        or cache_values.dim() != 3
        or cache_valid.dim() != 1
        or int(block_table.shape[0]) != batch
        or int(seq_lens.shape[0]) != batch
        or int(query_lens.shape[0]) != batch
        or max_chunks > int(block_table.shape[1])
        or tuple(cache_values.shape[1:]) != (kv_heads, q_indexer_dim)
        or int(cache_valid.shape[0]) != int(cache_values.shape[0])
        or not cache_values.is_contiguous()
        or not cache_valid.is_contiguous()
        or block_size <= 0
        or max_seed_chunks <= 0
        or max_seed_chunks > max_chunks
    ):
        return False
    if batch == 0 or max_chunks == 0:
        return True
    return _dsa_seed_block_summary_cache_triton(
        representatives=representatives,
        block_table=block_table,
        seq_lens=seq_lens,
        query_lens=query_lens,
        cache_values=cache_values,
        cache_valid=cache_valid,
        block_size=int(block_size),
        max_seed_chunks=int(max_seed_chunks),
    )


def dsa_cached_block_summaries_triton(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    q_indexer_dim: int,
    max_chunks: int | None = None,
) -> torch.Tensor | None:
    """Build representatives from a physical-page sidecar during decode.

    Historical full pages use cached BF16 representatives. The current page
    always reads K directly, even when it is full, so recycled physical pages
    cannot expose stale summaries. A second kernel publishes validity after all
    KV-head values have been written.
    """
    validated = _validate_summary_cache(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cache_values=cache_values,
        cache_valid=cache_valid,
        q_indexer_dim=q_indexer_dim,
        max_chunks=max_chunks,
    )
    if validated is None:
        return None
    batch, max_chunks, block_size, kv_heads = validated
    output = torch.empty(
        batch,
        max_chunks,
        kv_heads,
        q_indexer_dim,
        device=key_cache.device,
        dtype=torch.bfloat16,
    )
    if batch == 0 or max_chunks == 0:
        return output

    block_d = triton.next_power_of_2(q_indexer_dim)
    cache_block_chunks = 8
    _dsa_cached_block_summaries_kernel[
        (triton.cdiv(batch * max_chunks, cache_block_chunks), kv_heads)
    ](
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        output,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        int(cache_values.stride(0)),
        int(cache_values.stride(1)),
        int(cache_values.stride(2)),
        max_chunks,
        block_size,
        kv_heads,
        q_indexer_dim,
        int(cache_values.shape[0]),
        batch * max_chunks,
        BLOCK_CHUNKS=cache_block_chunks,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    publish_block_entries = 256
    _dsa_publish_block_summary_cache_kernel[
        (triton.cdiv(batch * max_chunks, publish_block_entries),)
    ](
        block_table,
        seq_lens,
        cache_valid,
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        max_chunks,
        block_size,
        int(cache_values.shape[0]),
        batch * max_chunks,
        BLOCK_ENTRIES=publish_block_entries,
        num_warps=4,
        num_stages=1,
    )
    return output


def dsa_update_block_summary_cache_triton(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    q_indexer_dim: int,
    max_chunks: int | None = None,
) -> bool:
    """Make all complete logical pages available in the physical sidecar.

    Unlike :func:`dsa_cached_block_summaries_triton`, this path never allocates
    or writes ``[batch, logical_chunks, heads, dim]``. Steady decode scans only
    block IDs and validity bytes; source K is read only for a cache miss or a
    current page on the token that completes it.
    """
    validated = _validate_summary_cache(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cache_values=cache_values,
        cache_valid=cache_valid,
        q_indexer_dim=q_indexer_dim,
        max_chunks=max_chunks,
    )
    if validated is None:
        return False
    batch, max_chunks, block_size, kv_heads = validated
    if batch == 0 or max_chunks == 0:
        return True

    block_d = triton.next_power_of_2(q_indexer_dim)
    cache_block_chunks = 8
    _dsa_update_block_summary_cache_kernel[
        (triton.cdiv(batch * max_chunks, cache_block_chunks), kv_heads)
    ](
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        int(cache_values.stride(0)),
        int(cache_values.stride(1)),
        int(cache_values.stride(2)),
        max_chunks,
        block_size,
        kv_heads,
        q_indexer_dim,
        int(cache_values.shape[0]),
        batch * max_chunks,
        BLOCK_CHUNKS=cache_block_chunks,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    publish_block_entries = 256
    _dsa_publish_block_summary_cache_kernel[
        (triton.cdiv(batch * max_chunks, publish_block_entries),)
    ](
        block_table,
        seq_lens,
        cache_valid,
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        max_chunks,
        block_size,
        int(cache_values.shape[0]),
        batch * max_chunks,
        BLOCK_ENTRIES=publish_block_entries,
        num_warps=4,
        num_stages=1,
    )
    return True


def dsa_update_current_block_summary_cache_triton(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    q_indexer_dim: int,
    max_chunks: int | None = None,
) -> bool:
    """Maintain only each sequence's current page during steady decode.

    All historical pages must already be valid. Partial current pages are
    invalidated to make physical-page recycling safe. A page is summarized and
    published only on the decode step that fills its final token.
    """
    validated = _validate_summary_cache(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cache_values=cache_values,
        cache_valid=cache_valid,
        q_indexer_dim=q_indexer_dim,
        max_chunks=max_chunks,
    )
    if validated is None:
        return False
    batch, _, block_size, kv_heads = validated
    if batch == 0:
        return True

    block_h = triton.next_power_of_2(kv_heads)
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_update_current_block_summary_cache_kernel[(batch,)](
        key_cache,
        block_table,
        seq_lens,
        cache_values,
        cache_valid,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        int(cache_values.stride(0)),
        int(cache_values.stride(1)),
        int(cache_values.stride(2)),
        block_size,
        kv_heads,
        q_indexer_dim,
        int(cache_values.shape[0]),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return True


def dsa_update_written_block_summary_cache_triton(
    *,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    cache_values: torch.Tensor,
    cache_valid: torch.Tensor,
    q_indexer_dim: int,
    max_written_chunks: int,
    max_chunks: int | None = None,
) -> bool:
    """Refresh pages written by a prefill or mixed scheduler slice.

    Historical pages remain untouched. Every complete page intersecting the
    current query slice is recomputed from K and published, while a final
    partial page is invalidated. This makes recycled physical pages safe
    without rescanning or rematerializing the full logical history.
    """
    validated = _validate_summary_cache(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        cache_values=cache_values,
        cache_valid=cache_valid,
        q_indexer_dim=q_indexer_dim,
        max_chunks=max_chunks,
    )
    if validated is None:
        return False
    batch, max_chunks, block_size, kv_heads = validated
    if (
        query_lens.dim() != 1
        or int(query_lens.shape[0]) != batch
        or query_lens.device != key_cache.device
        or query_lens.dtype not in (torch.int32, torch.int64)
        or max_written_chunks < 0
        or max_written_chunks > max_chunks
    ):
        return False
    if batch == 0 or max_written_chunks == 0:
        return True

    block_h = triton.next_power_of_2(kv_heads)
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_update_written_block_summary_cache_kernel[
        (batch * max_written_chunks,)
    ](
        key_cache,
        block_table,
        seq_lens,
        query_lens,
        cache_values,
        cache_valid,
        int(key_cache.stride(0)),
        int(key_cache.stride(1)),
        int(key_cache.stride(2)),
        int(key_cache.stride(3)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(seq_lens.stride(0)),
        int(query_lens.stride(0)),
        int(cache_values.stride(0)),
        int(cache_values.stride(1)),
        int(cache_values.stride(2)),
        int(max_written_chunks),
        block_size,
        kv_heads,
        q_indexer_dim,
        int(cache_values.shape[0]),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )
    return True
