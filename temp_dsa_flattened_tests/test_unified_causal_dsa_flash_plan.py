"""Prototype a unified causal FlashAttention plan for mixed DSA rows.

This standalone test intentionally avoids the Nemotron-H module.  It starts at
the DSA attention boundary: representatives, scoring, physical page-table
materialization, a single causal FlashAttention call, and output layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import os
import time

import torch

try:
    import pytest
except ModuleNotFoundError:

    class _PytestMarkFallback:

        @staticmethod
        def parametrize(*_args, **_kwargs):
            return lambda fn: fn

        @staticmethod
        def skipif(*_args, **_kwargs):
            return lambda fn: fn

    class _PytestFallback:
        mark = _PytestMarkFallback()

        @staticmethod
        def param(*values, **_kwargs):
            return values

        @staticmethod
        def importorskip(module_name: str):
            return importlib.import_module(module_name)

        @staticmethod
        def skip(reason: str):
            raise RuntimeError(reason)

    pytest = _PytestFallback()


@dataclass(frozen=True)
class SeqSpec:
    name: str
    q_len: int
    key_len: int
    sparse: bool
    shared_indexer: bool = False


@dataclass
class MixedBatch:
    specs: list[SeqSpec]
    query_states: torch.Tensor
    indexer_query_states: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    flat_key_cache: torch.Tensor
    flat_value_cache: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    key_lens: torch.Tensor
    positions: torch.Tensor
    current_chunks: torch.Tensor
    representatives: list[torch.Tensor]
    num_physical_blocks: int


@dataclass
class UnifiedPlan:
    q: torch.Tensor
    block_table: torch.Tensor
    seqused_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    out_rows: torch.Tensor
    out_kv_heads: torch.Tensor
    request_lens: torch.Tensor
    request_kinds: list[str]


@dataclass
class DirectOutputPlan:
    path: str
    q: torch.Tensor
    block_table: torch.Tensor
    seqused_k: torch.Tensor
    cu_seqlens_q: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    request_lens: torch.Tensor
    request_kinds: list[str]


def _make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return cu


def _test_fa_version(default: int = 2) -> int:
    return int(os.environ.get("DSA_UNIFIED_FA_VERSION", str(default)))


def _check_fa_version_support(flash_attn, fa_version: int) -> None:
    if os.environ.get("DSA_UNIFIED_BYPASS_FA_VERSION_CHECK"):
        return
    if not flash_attn.is_fa_version_supported(fa_version):
        reason = flash_attn.fa_version_unsupported_reason(fa_version)
        pytest.skip(reason or f"FA version {fa_version} is unavailable")


def _chunk_representatives(
    key_states: torch.Tensor,
    *,
    block_size: int,
    q_indexer_dim: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, key_states.shape[0], block_size):
        end = min(start + block_size, key_states.shape[0])
        chunks.append(key_states[start:end, :, :q_indexer_dim].float().mean(dim=0))
    return torch.stack(chunks, dim=0)


def _pack_nhd_cache(
    key_states_by_seq: list[torch.Tensor],
    value_states_by_seq: list[torch.Tensor],
    block_tables: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_physical_blocks = int(block_tables.max().item()) + 1
    sample = key_states_by_seq[0]
    kv_heads = sample.shape[1]
    head_dim = sample.shape[2]
    key_cache = sample.new_zeros(
        num_physical_blocks,
        block_size,
        kv_heads,
        head_dim,
    )
    value_cache = key_cache.new_zeros(key_cache.shape)
    flat_key_cache = sample.new_zeros(
        num_physical_blocks * kv_heads,
        block_size,
        1,
        head_dim,
    )
    flat_value_cache = flat_key_cache.new_zeros(flat_key_cache.shape)
    kv_offsets = torch.arange(
        kv_heads,
        device=sample.device,
        dtype=torch.long,
    ) * num_physical_blocks
    for seq_idx, key_states in enumerate(key_states_by_seq):
        value_states = value_states_by_seq[seq_idx]
        for token_idx in range(key_states.shape[0]):
            block_id = int(block_tables[seq_idx, token_idx // block_size].item())
            offset = token_idx % block_size
            key_cache[block_id, offset] = key_states[token_idx]
            value_cache[block_id, offset] = value_states[token_idx]
            flat_block_ids = kv_offsets + block_id
            flat_key_cache[flat_block_ids, offset, 0] = key_states[token_idx]
            flat_value_cache[flat_block_ids, offset, 0] = value_states[token_idx]
    return key_cache, value_cache, flat_key_cache, flat_value_cache


def _make_mixed_batch(
    *,
    device: torch.device,
    dtype: torch.dtype,
    block_size: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    q_indexer_dim: int,
    seed: int,
) -> MixedBatch:
    specs = [
        SeqSpec("dense_short_prefill", q_len=6, key_len=11, sparse=False),
        SeqSpec("sparse_decode", q_len=1, key_len=137, sparse=True),
        SeqSpec("sparse_mtp_shared", q_len=4, key_len=141, sparse=True,
                shared_indexer=True),
        SeqSpec("sparse_chunked_prefill", q_len=65, key_len=513, sparse=True),
        SeqSpec("dense_decode", q_len=1, key_len=7, sparse=False),
        SeqSpec("sparse_short_prefill", q_len=6, key_len=102, sparse=True),
    ]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    query_lens = torch.tensor([s.q_len for s in specs], device=device, dtype=torch.int32)
    query_start_loc = _make_cu_seqlens(query_lens)
    total_q = int(query_start_loc[-1].item())
    key_lens = torch.tensor([s.key_len for s in specs], device=device, dtype=torch.int32)
    max_blocks = max(math.ceil(s.key_len / block_size) for s in specs)

    block_tables = torch.empty(
        len(specs),
        max_blocks,
        device=device,
        dtype=torch.int32,
    )
    next_block = 0
    for seq_idx, spec in enumerate(specs):
        num_blocks = math.ceil(spec.key_len / block_size)
        permuted = torch.randperm(
            num_blocks,
            device=device,
            generator=generator,
            dtype=torch.int32,
        ) + next_block
        block_tables[seq_idx, :num_blocks] = permuted
        if num_blocks < max_blocks:
            block_tables[seq_idx, num_blocks:] = 0
        next_block += num_blocks

    key_states_by_seq = []
    value_states_by_seq = []
    representatives = []
    for spec in specs:
        key_states = torch.randn(
            spec.key_len,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        ) * 0.25
        value_states = torch.randn(
            spec.key_len,
            kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        key_states_by_seq.append(key_states)
        value_states_by_seq.append(value_states)
        representatives.append(
            _chunk_representatives(
                key_states,
                block_size=block_size,
                q_indexer_dim=q_indexer_dim,
            )
        )

    key_cache, value_cache, flat_key_cache, flat_value_cache = _pack_nhd_cache(
        key_states_by_seq,
        value_states_by_seq,
        block_tables,
        block_size=block_size,
    )

    query_states = torch.randn(
        total_q,
        heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 0.25
    indexer_query_states = torch.randn(
        total_q,
        kv_heads,
        q_indexer_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 0.25
    positions = torch.empty(total_q, device=device, dtype=torch.long)
    current_chunks = torch.empty(total_q, device=device, dtype=torch.long)
    for seq_idx, spec in enumerate(specs):
        start = int(query_start_loc[seq_idx].item())
        end = int(query_start_loc[seq_idx + 1].item())
        seq_positions = torch.arange(
            spec.key_len - spec.q_len,
            spec.key_len,
            device=device,
            dtype=torch.long,
        )
        positions[start:end] = seq_positions
        current_chunks[start:end] = torch.div(
            seq_positions,
            block_size,
            rounding_mode="floor",
        )
        if spec.shared_indexer:
            indexer_query_states[start:end] = indexer_query_states[start:start + 1]

    return MixedBatch(
        specs=specs,
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        flat_key_cache=flat_key_cache,
        flat_value_cache=flat_value_cache,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        key_lens=key_lens,
        positions=positions,
        current_chunks=current_chunks,
        representatives=representatives,
        num_physical_blocks=key_cache.shape[0],
    )


def _score_sparse_rows(
    batch: MixedBatch,
    *,
    block_size: int,
    top_k: int,
    q_indexer_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q = batch.query_states.shape[0]
    kv_heads = batch.key_cache.shape[2]
    top_indices = torch.zeros(
        total_q,
        kv_heads,
        top_k,
        device=batch.query_states.device,
        dtype=torch.long,
    )
    top_valid = torch.zeros_like(top_indices, dtype=torch.bool)
    score_scale = 1.0 / math.sqrt(q_indexer_dim)
    for seq_idx, spec in enumerate(batch.specs):
        if not spec.sparse:
            continue
        row_start = int(batch.query_start_loc[seq_idx].item())
        row_end = int(batch.query_start_loc[seq_idx + 1].item())
        reps = batch.representatives[seq_idx]
        for row in range(row_start, row_end):
            current_chunk = int(batch.current_chunks[row].item())
            if current_chunk <= 0:
                continue
            k = min(top_k, current_chunk)
            for kv_head in range(kv_heads):
                logits = torch.mv(
                    reps[:current_chunk, kv_head],
                    batch.indexer_query_states[row, kv_head].float(),
                ) * score_scale
                selected = logits.topk(k=k).indices
                top_indices[row, kv_head, :k] = selected
                top_valid[row, kv_head, :k] = True
    return top_indices, top_valid


def _logical_sparse_tokens(
    top_chunks: torch.Tensor,
    *,
    current_chunk: int,
    position: int,
    key_len: int,
    block_size: int,
) -> list[int]:
    tokens: list[int] = []
    for chunk in top_chunks.tolist():
        start = int(chunk) * block_size
        tokens.extend(range(start, min(start + block_size, key_len)))
    tokens.extend(range(current_chunk * block_size, position + 1))
    return tokens


def _gather_sequence_from_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    key_len: int,
    *,
    block_size: int,
) -> torch.Tensor:
    blocks = []
    for token in range(key_len):
        block_id = int(block_table[token // block_size].item())
        blocks.append(cache[block_id, token % block_size])
    return torch.stack(blocks, dim=0)


def _reference_mixed_attention(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    total_q, heads, head_dim = batch.query_states.shape
    kv_heads = batch.key_cache.shape[2]
    group_size = heads // kv_heads
    output = batch.query_states.new_empty(total_q, heads, head_dim)

    for seq_idx, spec in enumerate(batch.specs):
        row_start = int(batch.query_start_loc[seq_idx].item())
        row_end = int(batch.query_start_loc[seq_idx + 1].item())
        key_len = int(batch.key_lens[seq_idx].item())
        key_states = _gather_sequence_from_cache(
            batch.key_cache,
            batch.block_tables[seq_idx],
            key_len,
            block_size=block_size,
        )
        value_states = _gather_sequence_from_cache(
            batch.value_cache,
            batch.block_tables[seq_idx],
            key_len,
            block_size=block_size,
        )

        for row in range(row_start, row_end):
            position = int(batch.positions[row].item())
            current_chunk = int(batch.current_chunks[row].item())
            for head in range(heads):
                kv_head = head // group_size
                if spec.sparse:
                    valid = top_valid[row, kv_head]
                    recall = _logical_sparse_tokens(
                        top_indices[row, kv_head].masked_select(valid),
                        current_chunk=current_chunk,
                        position=position,
                        key_len=key_len,
                        block_size=block_size,
                    )
                else:
                    recall = list(range(position + 1))
                selected_k = key_states[recall, kv_head].float()
                selected_v = value_states[recall, kv_head]
                logits = torch.mv(selected_k, batch.query_states[row, head].float())
                weights = torch.softmax(logits * softmax_scale, dim=-1)
                output[row, head] = torch.mv(
                    selected_v.float().transpose(0, 1),
                    weights,
                ).to(output.dtype)
    return output


def _same_sparse_request(
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    current_chunks: torch.Tensor,
    prev_row: int,
    row: int,
    kv_head: int,
) -> bool:
    return (
        int(current_chunks[row].item()) == int(current_chunks[prev_row].item())
        and bool(torch.equal(top_valid[row, kv_head], top_valid[prev_row, kv_head]))
        and bool(torch.equal(top_indices[row, kv_head], top_indices[prev_row, kv_head]))
    )


def _build_unified_causal_plan(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
) -> UnifiedPlan:
    device = batch.query_states.device
    heads = batch.query_states.shape[1]
    kv_heads = batch.key_cache.shape[2]
    group_size = heads // kv_heads

    q_parts = []
    block_rows = []
    seqused = []
    request_lens = []
    request_kinds = []
    out_rows = []
    out_kv_heads = []

    for seq_idx, spec in enumerate(batch.specs):
        seq_row_start = int(batch.query_start_loc[seq_idx].item())
        seq_row_end = int(batch.query_start_loc[seq_idx + 1].item())
        key_len = int(batch.key_lens[seq_idx].item())
        num_chunks = math.ceil(key_len / block_size)
        for kv_head in range(kv_heads):
            head_start = kv_head * group_size
            head_end = head_start + group_size
            kv_offset = kv_head * batch.num_physical_blocks
            if not spec.sparse:
                logical_pages = torch.arange(num_chunks, device=device, dtype=torch.long)
                physical_pages = batch.block_tables[seq_idx].index_select(
                    0,
                    logical_pages,
                ).to(torch.int32) + kv_offset
                rows = torch.arange(seq_row_start, seq_row_end, device=device)
                q_parts.append(batch.query_states[rows, head_start:head_end])
                block_rows.append(physical_pages)
                seqused.append(key_len)
                request_lens.append(rows.numel())
                request_kinds.append(f"dense:{spec.name}")
                out_rows.append(rows)
                out_kv_heads.append(torch.full_like(rows, kv_head))
                continue

            row = seq_row_start
            while row < seq_row_end:
                run_end = row + 1
                while run_end < seq_row_end:
                    if int(batch.positions[run_end].item()) != (
                        int(batch.positions[run_end - 1].item()) + 1
                    ):
                        break
                    if not _same_sparse_request(
                        top_indices,
                        top_valid,
                        batch.current_chunks,
                        run_end - 1,
                        run_end,
                        kv_head,
                    ):
                        break
                    run_end += 1

                valid_count = int(top_valid[row, kv_head].sum().item())
                current_chunk = int(batch.current_chunks[row].item())
                logical_pages = top_indices[row, kv_head, :valid_count].to(torch.long)
                logical_pages = torch.cat(
                    (
                        logical_pages,
                        torch.tensor([current_chunk], device=device, dtype=torch.long),
                    )
                )
                physical_pages = batch.block_tables[seq_idx].index_select(
                    0,
                    logical_pages,
                ).to(torch.int32) + kv_offset
                rows = torch.arange(row, run_end, device=device)
                last_position = int(batch.positions[run_end - 1].item())
                tail_len = last_position - current_chunk * block_size + 1

                q_parts.append(batch.query_states[rows, head_start:head_end])
                block_rows.append(physical_pages)
                seqused.append(valid_count * block_size + tail_len)
                request_lens.append(rows.numel())
                request_kinds.append(f"sparse:{spec.name}")
                out_rows.append(rows)
                out_kv_heads.append(torch.full_like(rows, kv_head))
                row = run_end

    request_lens_t = torch.tensor(request_lens, device=device, dtype=torch.int32)
    max_pages = max(int(row.numel()) for row in block_rows)
    plan_block_table = torch.zeros(
        len(block_rows),
        max_pages,
        device=device,
        dtype=torch.int32,
    )
    for request_idx, pages in enumerate(block_rows):
        plan_block_table[request_idx, : pages.numel()] = pages

    return UnifiedPlan(
        q=torch.cat(q_parts, dim=0).contiguous(),
        block_table=plan_block_table,
        seqused_k=torch.tensor(seqused, device=device, dtype=torch.int32),
        cu_seqlens_q=_make_cu_seqlens(request_lens_t),
        max_seqlen_q=int(request_lens_t.max().item()),
        max_seqlen_k=int(max(seqused)),
        out_rows=torch.cat(out_rows, dim=0).to(torch.long),
        out_kv_heads=torch.cat(out_kv_heads, dim=0).to(torch.long),
        request_lens=request_lens_t,
        request_kinds=request_kinds,
    )


def _finish_direct_output_plan(
    *,
    path: str,
    q: torch.Tensor,
    block_rows: list[torch.Tensor],
    seqused: list[int],
    request_lens: list[int],
    request_kinds: list[str],
) -> DirectOutputPlan:
    device = q.device
    max_pages = max(int(row.numel()) for row in block_rows)
    plan_block_table = torch.zeros(
        len(block_rows),
        max_pages,
        device=device,
        dtype=torch.int32,
    )
    for request_idx, pages in enumerate(block_rows):
        plan_block_table[request_idx, : pages.numel()] = pages

    request_lens_t = torch.tensor(request_lens, device=device, dtype=torch.int32)
    return DirectOutputPlan(
        path=path,
        q=q,
        block_table=plan_block_table,
        seqused_k=torch.tensor(seqused, device=device, dtype=torch.int32),
        cu_seqlens_q=_make_cu_seqlens(request_lens_t),
        max_seqlen_q=int(request_lens_t.max().item()),
        max_seqlen_k=int(max(seqused)),
        request_lens=request_lens_t,
        request_kinds=request_kinds,
    )


def _build_single_kv_head_fast_causal_plan(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
) -> DirectOutputPlan:
    device = batch.query_states.device
    total_q, heads, head_dim = batch.query_states.shape
    kv_heads = batch.key_cache.shape[2]
    if kv_heads != 1:
        raise ValueError(
            "single-KV-head fast path requires exactly one local KV head, "
            f"got kv_heads={kv_heads}"
        )

    q = batch.query_states.view(total_q, heads, head_dim)

    block_rows = []
    seqused = []
    request_lens = []
    request_kinds = []
    for seq_idx, spec in enumerate(batch.specs):
        seq_row_start = int(batch.query_start_loc[seq_idx].item())
        seq_row_end = int(batch.query_start_loc[seq_idx + 1].item())
        if not spec.sparse:
            last_row = seq_row_end - 1
            position = int(batch.positions[last_row].item())
            current_chunk = int(batch.current_chunks[last_row].item())
            logical_pages = torch.arange(
                current_chunk + 1,
                device=device,
                dtype=torch.long,
            )
            physical_pages = batch.block_tables[seq_idx].index_select(
                0,
                logical_pages,
            ).to(torch.int32)
            block_rows.append(physical_pages)
            seqused.append(position + 1)
            request_lens.append(seq_row_end - seq_row_start)
            if seq_row_end > seq_row_start + 1:
                request_kinds.append(f"fast_dense_multi_q:{spec.name}")
            else:
                request_kinds.append(f"fast_dense_q1:{spec.name}")
            continue

        for row in range(seq_row_start, seq_row_end):
            position = int(batch.positions[row].item())
            current_chunk = int(batch.current_chunks[row].item())
            local_prefix = position - current_chunk * block_size + 1
            valid_count = int(top_valid[row, 0].sum().item())
            logical_pages = top_indices[row, 0, :valid_count].to(torch.long)
            logical_pages = torch.cat(
                (
                    logical_pages,
                    torch.tensor([current_chunk], device=device, dtype=torch.long),
                )
            )
            physical_pages = batch.block_tables[seq_idx].index_select(
                0,
                logical_pages,
            ).to(torch.int32)
            block_rows.append(physical_pages)
            seqused.append(valid_count * block_size + local_prefix)
            request_lens.append(1)
            request_kinds.append(f"fast_sparse_q1:{spec.name}")

    return _finish_direct_output_plan(
        path="single_kv_head_fast",
        q=q,
        block_rows=block_rows,
        seqused=seqused,
        request_lens=request_lens,
        request_kinds=request_kinds,
    )


def _max_sparse_suffix_seqused_k(
    *,
    query_position_start: int,
    key_len: int,
    block_size: int,
    top_width: int,
) -> int:
    start_chunk = query_position_start // block_size
    end_chunk = (key_len - 1) // block_size
    max_used = 0
    for chunk_idx in range(start_chunk, end_chunk + 1):
        chunk_query_start = max(query_position_start, chunk_idx * block_size)
        chunk_query_end = min(key_len, (chunk_idx + 1) * block_size)
        if chunk_query_end <= chunk_query_start:
            continue
        local_prefix = chunk_query_end - chunk_idx * block_size
        valid_count = min(top_width, chunk_idx)
        max_used = max(max_used, valid_count * block_size + local_prefix)
    return max_used


def _build_single_kv_head_vectorized_causal_plan(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
) -> DirectOutputPlan:
    device = batch.query_states.device
    total_q, heads, head_dim = batch.query_states.shape
    kv_heads = batch.key_cache.shape[2]
    if kv_heads != 1:
        raise ValueError(
            "single-KV-head vectorized path requires exactly one local KV head, "
            f"got kv_heads={kv_heads}"
        )

    q = batch.query_states.view(total_q, heads, head_dim)
    table_parts: list[torch.Tensor] = []
    seqused_parts: list[torch.Tensor] = []
    request_lens_parts: list[torch.Tensor] = []
    request_kinds: list[str] = []
    top_width = top_indices.shape[-1]
    max_seqlen_q = 0
    max_seqlen_k = 0

    seq_row_start = 0
    for seq_idx, spec in enumerate(batch.specs):
        q_len = spec.q_len
        key_len = spec.key_len
        seq_row_end = seq_row_start + q_len
        if not spec.sparse:
            num_pages = math.ceil(key_len / block_size)
            logical_pages = torch.arange(
                num_pages,
                device=device,
                dtype=torch.long,
            )
            physical_pages = batch.block_tables[seq_idx].index_select(
                0,
                logical_pages,
            ).to(torch.int32).view(1, -1)
            table_parts.append(physical_pages)
            seqused_parts.append(
                torch.full((1,), key_len, device=device, dtype=torch.int32)
            )
            request_lens_parts.append(
                torch.full((1,), q_len, device=device, dtype=torch.int32)
            )
            max_seqlen_q = max(max_seqlen_q, q_len)
            max_seqlen_k = max(max_seqlen_k, key_len)
            if q_len > 1:
                request_kinds.append(f"fast_dense_multi_q:{spec.name}")
            else:
                request_kinds.append(f"fast_dense_q1:{spec.name}")
            seq_row_start = seq_row_end
            continue

        rows = torch.arange(seq_row_start, seq_row_end, device=device)
        valid_counts = top_valid[rows, 0].sum(dim=-1).to(torch.long)
        current_chunks = batch.current_chunks[rows].to(torch.long)
        positions = batch.positions[rows].to(torch.long)
        local_prefixes = positions - current_chunks * block_size + 1

        logical_pages = current_chunks[:, None].expand(
            q_len,
            top_width + 1,
        ).clone()
        if top_width > 0:
            logical_pages[:, :top_width] = top_indices[rows, 0].to(torch.long)
        logical_pages.scatter_(
            dim=1,
            index=valid_counts[:, None],
            src=current_chunks[:, None],
        )
        seq_block_table = batch.block_tables[seq_idx].to(torch.long)
        physical_pages = seq_block_table.expand(q_len, -1).gather(
            dim=1,
            index=logical_pages,
        ).to(torch.int32)
        used_page_mask = (
            torch.arange(top_width + 1, device=device, dtype=torch.long)[None, :]
            <= valid_counts[:, None]
        )
        physical_pages.masked_fill_(~used_page_mask, 0)
        table_parts.append(physical_pages)
        seqused_parts.append(
            (valid_counts * block_size + local_prefixes).to(torch.int32)
        )
        request_lens_parts.append(torch.ones(q_len, device=device, dtype=torch.int32))
        request_kinds.extend([f"fast_sparse_q1:{spec.name}"] * q_len)
        max_seqlen_q = max(max_seqlen_q, 1)
        max_seqlen_k = max(
            max_seqlen_k,
            _max_sparse_suffix_seqused_k(
                query_position_start=key_len - q_len,
                key_len=key_len,
                block_size=block_size,
                top_width=top_width,
            ),
        )
        seq_row_start = seq_row_end

    request_lens = torch.cat(request_lens_parts, dim=0)
    seqused_k = torch.cat(seqused_parts, dim=0)
    max_pages = max(int(part.shape[1]) for part in table_parts)
    total_requests = sum(int(part.shape[0]) for part in table_parts)
    plan_block_table = torch.zeros(
        total_requests,
        max_pages,
        device=device,
        dtype=torch.int32,
    )
    request_start = 0
    for pages in table_parts:
        request_end = request_start + pages.shape[0]
        plan_block_table[request_start:request_end, : pages.shape[1]] = pages
        request_start = request_end

    return DirectOutputPlan(
        path="single_kv_head_vectorized",
        q=q,
        block_table=plan_block_table,
        seqused_k=seqused_k,
        cu_seqlens_q=_make_cu_seqlens(request_lens),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        request_lens=request_lens,
        request_kinds=request_kinds,
    )


def _build_multi_kv_head_fallback_causal_plan(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
) -> DirectOutputPlan:
    device = batch.query_states.device
    total_q, heads, head_dim = batch.query_states.shape
    kv_heads = batch.key_cache.shape[2]
    if kv_heads <= 1:
        raise ValueError(
            "multi-KV-head fallback is only for ranks with more than one "
            f"local KV head, got kv_heads={kv_heads}"
        )
    if heads % kv_heads != 0:
        raise ValueError(f"heads={heads} must be divisible by kv_heads={kv_heads}")
    group_size = heads // kv_heads

    q = batch.query_states.view(
        total_q,
        kv_heads,
        group_size,
        head_dim,
    ).view(total_q * kv_heads, group_size, head_dim)

    block_rows = []
    seqused = []
    request_lens = []
    request_kinds = []
    for seq_idx, spec in enumerate(batch.specs):
        seq_row_start = int(batch.query_start_loc[seq_idx].item())
        seq_row_end = int(batch.query_start_loc[seq_idx + 1].item())
        for row in range(seq_row_start, seq_row_end):
            position = int(batch.positions[row].item())
            current_chunk = int(batch.current_chunks[row].item())
            local_prefix = position - current_chunk * block_size + 1
            for kv_head in range(kv_heads):
                kv_offset = kv_head * batch.num_physical_blocks
                if spec.sparse:
                    valid_count = int(top_valid[row, kv_head].sum().item())
                    logical_pages = top_indices[
                        row,
                        kv_head,
                        :valid_count,
                    ].to(torch.long)
                    logical_pages = torch.cat(
                        (
                            logical_pages,
                            torch.tensor(
                                [current_chunk],
                                device=device,
                                dtype=torch.long,
                            ),
                        )
                    )
                    seqused.append(valid_count * block_size + local_prefix)
                    request_kinds.append(f"fallback_sparse_q1:{spec.name}")
                else:
                    logical_pages = torch.arange(
                        current_chunk + 1,
                        device=device,
                        dtype=torch.long,
                    )
                    seqused.append(position + 1)
                    request_kinds.append(f"fallback_dense_q1:{spec.name}")

                physical_pages = batch.block_tables[seq_idx].index_select(
                    0,
                    logical_pages,
                ).to(torch.int32) + kv_offset
                block_rows.append(physical_pages)
                request_lens.append(1)

    return _finish_direct_output_plan(
        path="multi_kv_head_fallback",
        q=q,
        block_rows=block_rows,
        seqused=seqused,
        request_lens=request_lens,
        request_kinds=request_kinds,
    )


def _build_direct_output_causal_plan(
    batch: MixedBatch,
    top_indices: torch.Tensor,
    top_valid: torch.Tensor,
    *,
    block_size: int,
) -> DirectOutputPlan:
    if batch.key_cache.shape[2] == 1:
        return _build_single_kv_head_vectorized_causal_plan(
            batch,
            top_indices,
            top_valid,
            block_size=block_size,
        )
    return _build_multi_kv_head_fallback_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )


def _execute_unified_plan(
    batch: MixedBatch,
    plan: UnifiedPlan,
    *,
    heads: int,
    fa_version: int,
    softmax_scale: float,
) -> torch.Tensor:
    flash_attn = pytest.importorskip("vllm.vllm_flash_attn")
    _check_fa_version_support(flash_attn, fa_version)

    flat_key, flat_value = batch.flat_key_cache, batch.flat_value_cache
    group_size = heads // batch.key_cache.shape[2]
    if os.environ.get("DSA_UNIFIED_VERBOSE"):
        print(
            "fa_call",
            {
                "q": tuple(plan.q.shape),
                "k": tuple(flat_key.shape),
                "block_table": tuple(plan.block_table.shape),
                "seqused_k": plan.seqused_k.detach().cpu().tolist(),
                "cu_seqlens_q": plan.cu_seqlens_q.detach().cpu().tolist(),
                "request_lens": plan.request_lens.detach().cpu().tolist(),
                "request_kinds": plan.request_kinds,
                "max_seqlen_q": plan.max_seqlen_q,
                "max_seqlen_k": plan.max_seqlen_k,
                "causal": True,
                "fa_version": fa_version,
            },
        )
    fa_out = flash_attn.flash_attn_varlen_func(
        q=plan.q,
        k=flat_key,
        v=flat_value,
        cu_seqlens_q=plan.cu_seqlens_q,
        seqused_k=plan.seqused_k,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=plan.block_table,
        fa_version=fa_version,
    )
    output_by_kv = batch.query_states.new_empty(
        batch.query_states.shape[0],
        batch.key_cache.shape[2],
        group_size,
        batch.query_states.shape[-1],
    )
    output_by_kv[plan.out_rows, plan.out_kv_heads] = fa_out
    return output_by_kv.view(batch.query_states.shape[0], heads,
                             batch.query_states.shape[-1])


def _execute_plan_with_torch_reference(
    batch: MixedBatch,
    plan: UnifiedPlan,
    *,
    heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    flat_key, flat_value = batch.flat_key_cache, batch.flat_value_cache
    group_size = heads // batch.key_cache.shape[2]
    fa_like_out = plan.q.new_empty(plan.q.shape)
    for request_idx in range(plan.seqused_k.numel()):
        q_start = int(plan.cu_seqlens_q[request_idx].item())
        q_end = int(plan.cu_seqlens_q[request_idx + 1].item())
        q_len = q_end - q_start
        k_len = int(plan.seqused_k[request_idx].item())
        page_ids = plan.block_table[request_idx]
        selected_k = []
        selected_v = []
        for token_idx in range(k_len):
            page_id = int(page_ids[token_idx // batch.key_cache.shape[1]].item())
            page_offset = token_idx % batch.key_cache.shape[1]
            selected_k.append(flat_key[page_id, page_offset, 0])
            selected_v.append(flat_value[page_id, page_offset, 0])
        request_k = torch.stack(selected_k, dim=0)
        request_v = torch.stack(selected_v, dim=0)
        for local_row, q_row in enumerate(range(q_start, q_end)):
            causal_k_len = k_len - q_len + local_row + 1
            logits = torch.einsum(
                "hd,kd->hk",
                plan.q[q_row].float(),
                request_k[:causal_k_len].float(),
            )
            weights = torch.softmax(logits * softmax_scale, dim=-1)
            fa_like_out[q_row] = torch.einsum(
                "hk,kd->hd",
                weights,
                request_v[:causal_k_len].float(),
            ).to(plan.q.dtype)

    output_by_kv = batch.query_states.new_empty(
        batch.query_states.shape[0],
        batch.key_cache.shape[2],
        group_size,
        batch.query_states.shape[-1],
    )
    output_by_kv[plan.out_rows, plan.out_kv_heads] = fa_like_out
    return output_by_kv.view(batch.query_states.shape[0], heads,
                             batch.query_states.shape[-1])


def _direct_output_view(
    batch: MixedBatch,
    flat_output: torch.Tensor,
    *,
    heads: int,
) -> torch.Tensor:
    return flat_output.view(
        batch.query_states.shape[0],
        heads,
        batch.query_states.shape[-1],
    )


def _execute_direct_plan_with_torch_reference(
    batch: MixedBatch,
    plan: DirectOutputPlan,
    *,
    heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    flat_key, flat_value = batch.flat_key_cache, batch.flat_value_cache
    direct_out = plan.q.new_empty(plan.q.shape)
    for request_idx in range(plan.seqused_k.numel()):
        q_start = int(plan.cu_seqlens_q[request_idx].item())
        q_end = int(plan.cu_seqlens_q[request_idx + 1].item())
        q_len = q_end - q_start
        k_len = int(plan.seqused_k[request_idx].item())
        page_ids = plan.block_table[request_idx]
        selected_k = []
        selected_v = []
        for token_idx in range(k_len):
            page_id = int(page_ids[token_idx // batch.key_cache.shape[1]].item())
            page_offset = token_idx % batch.key_cache.shape[1]
            selected_k.append(flat_key[page_id, page_offset, 0])
            selected_v.append(flat_value[page_id, page_offset, 0])
        request_k = torch.stack(selected_k, dim=0)
        request_v = torch.stack(selected_v, dim=0)
        for local_row, q_row in enumerate(range(q_start, q_end)):
            causal_k_len = k_len - q_len + local_row + 1
            logits = torch.einsum(
                "hd,kd->hk",
                plan.q[q_row].float(),
                request_k[:causal_k_len].float(),
            )
            weights = torch.softmax(logits * softmax_scale, dim=-1)
            direct_out[q_row] = torch.einsum(
                "hk,kd->hd",
                weights,
                request_v[:causal_k_len].float(),
            ).to(plan.q.dtype)

    return _direct_output_view(batch, direct_out, heads=heads)


def _execute_direct_output_plan(
    batch: MixedBatch,
    plan: DirectOutputPlan,
    *,
    heads: int,
    fa_version: int,
    softmax_scale: float,
) -> torch.Tensor:
    flash_attn = pytest.importorskip("vllm.vllm_flash_attn")
    _check_fa_version_support(flash_attn, fa_version)

    flat_key, flat_value = batch.flat_key_cache, batch.flat_value_cache
    direct_out = torch.empty_like(plan.q)
    if os.environ.get("DSA_UNIFIED_VERBOSE"):
        print(
            "direct_fa_call",
            {
                "path": plan.path,
                "q": tuple(plan.q.shape),
                "k": tuple(flat_key.shape),
                "out": tuple(direct_out.shape),
                "block_table": tuple(plan.block_table.shape),
                "seqused_k": plan.seqused_k.detach().cpu().tolist(),
                "cu_seqlens_q": plan.cu_seqlens_q.detach().cpu().tolist(),
                "request_lens": plan.request_lens.detach().cpu().tolist(),
                "request_kinds": plan.request_kinds,
                "max_seqlen_q": plan.max_seqlen_q,
                "max_seqlen_k": plan.max_seqlen_k,
                "causal": True,
                "fa_version": fa_version,
            },
        )
    flash_attn.flash_attn_varlen_func(
        q=plan.q,
        k=flat_key,
        v=flat_value,
        cu_seqlens_q=plan.cu_seqlens_q,
        seqused_k=plan.seqused_k,
        max_seqlen_q=plan.max_seqlen_q,
        max_seqlen_k=plan.max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        block_table=plan.block_table,
        fa_version=fa_version,
        out=direct_out,
    )
    return _direct_output_view(batch, direct_out, heads=heads)


HEAD_CASES = [(8, 2), (6, 2), (1, 1), (4, 1)]
FAST_HEAD_CASES = [
    pytest.param(1, 1, id="tp_single_head"),
    pytest.param(4, 1, id="single_kv_group4"),
]
FALLBACK_HEAD_CASES = [
    pytest.param(8, 2, id="gqa_group4"),
    pytest.param(6, 2, id="gqa_group3"),
]
DIRECT_HEAD_CASES = [
    *FALLBACK_HEAD_CASES,
    *FAST_HEAD_CASES,
]


def _make_expected_and_direct_plan(
    *,
    seed: int,
    heads: int,
    kv_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    builder=None,
) -> tuple[MixedBatch, torch.Tensor, DirectOutputPlan, float]:
    block_size = 16
    head_dim = 32
    q_indexer_dim = 8
    top_k = 4
    batch = _make_mixed_batch(
        device=device,
        dtype=dtype,
        block_size=block_size,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        seed=seed,
    )
    top_indices, top_valid = _score_sparse_rows(
        batch,
        block_size=block_size,
        top_k=top_k,
        q_indexer_dim=q_indexer_dim,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    expected = _reference_mixed_attention(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )
    plan_builder = builder or _build_direct_output_causal_plan
    plan = plan_builder(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )
    return batch, expected, plan, softmax_scale


def _make_single_kv_metadata_stress_batch(
    *,
    q_rows: int,
    key_len: int,
    top_k: int,
    block_size: int,
    heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MixedBatch, torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(key_len / block_size)
    spec = SeqSpec(
        "stress_sparse_chunked_prefill",
        q_len=q_rows,
        key_len=key_len,
        sparse=True,
    )
    query_states = torch.empty(q_rows, heads, head_dim, device=device, dtype=dtype)
    indexer_query_states = torch.empty(q_rows, 1, 8, device=device, dtype=dtype)
    key_cache = torch.empty(num_blocks, block_size, 1, head_dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    flat_key_cache = torch.empty(
        num_blocks,
        block_size,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )
    flat_value_cache = torch.empty_like(flat_key_cache)
    block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    query_start_loc = torch.tensor([0, q_rows], device=device, dtype=torch.int32)
    key_lens = torch.tensor([key_len], device=device, dtype=torch.int32)
    positions = torch.arange(key_len - q_rows, key_len, device=device, dtype=torch.long)
    current_chunks = torch.div(positions, block_size, rounding_mode="floor")
    top_indices = torch.arange(top_k, device=device, dtype=torch.long).view(1, 1, top_k)
    top_indices = top_indices.expand(q_rows, 1, top_k).clone()
    top_valid = torch.ones_like(top_indices, dtype=torch.bool)
    batch = MixedBatch(
        specs=[spec],
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        flat_key_cache=flat_key_cache,
        flat_value_cache=flat_value_cache,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        key_lens=key_lens,
        positions=positions,
        current_chunks=current_chunks,
        representatives=[],
        num_physical_blocks=num_blocks,
    )
    return batch, top_indices, top_valid


def _make_single_kv_real_mixed_stress_batch(
    *,
    block_size: int,
    heads: int,
    head_dim: int,
    top_k: int,
    long_q_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[MixedBatch, torch.Tensor, torch.Tensor]:
    specs = [
        SeqSpec("real_sparse_decode", q_len=1, key_len=32768, sparse=True),
        SeqSpec("real_sparse_mtp4", q_len=4, key_len=256, sparse=True),
        SeqSpec("real_dense_short_prefill", q_len=6, key_len=11, sparse=False),
        SeqSpec(
            "real_sparse_chunked_prefill_8k",
            q_len=long_q_len,
            key_len=1048576,
            sparse=True,
        ),
        SeqSpec("real_dense_decode", q_len=1, key_len=7, sparse=False),
    ]
    q_lens = torch.tensor([spec.q_len for spec in specs], device=device, dtype=torch.int32)
    query_start_loc = _make_cu_seqlens(q_lens)
    total_q = sum(spec.q_len for spec in specs)
    key_lens = torch.tensor([spec.key_len for spec in specs], device=device, dtype=torch.int32)
    max_blocks = max(math.ceil(spec.key_len / block_size) for spec in specs)
    block_tables = torch.zeros(len(specs), max_blocks, device=device, dtype=torch.int32)
    next_block = 0
    for seq_idx, spec in enumerate(specs):
        num_blocks = math.ceil(spec.key_len / block_size)
        block_tables[seq_idx, :num_blocks] = torch.arange(
            next_block,
            next_block + num_blocks,
            device=device,
            dtype=torch.int32,
        )
        next_block += num_blocks

    positions = torch.empty(total_q, device=device, dtype=torch.long)
    current_chunks = torch.empty(total_q, device=device, dtype=torch.long)
    row_start = 0
    for spec in specs:
        start = row_start
        end = start + spec.q_len
        row_start = end
        seq_positions = torch.arange(
            spec.key_len - spec.q_len,
            spec.key_len,
            device=device,
            dtype=torch.long,
        )
        positions[start:end] = seq_positions
        current_chunks[start:end] = torch.div(
            seq_positions,
            block_size,
            rounding_mode="floor",
        )

    query_states = torch.empty(total_q, heads, head_dim, device=device, dtype=dtype)
    indexer_query_states = torch.empty(total_q, 1, 8, device=device, dtype=dtype)
    key_cache = torch.empty(next_block, block_size, 1, head_dim, device=device, dtype=dtype)
    value_cache = torch.empty_like(key_cache)
    flat_key_cache = torch.empty(next_block, block_size, 1, head_dim, device=device, dtype=dtype)
    flat_value_cache = torch.empty_like(flat_key_cache)

    top_indices = torch.arange(top_k, device=device, dtype=torch.long).view(1, 1, top_k)
    top_indices = top_indices.expand(total_q, 1, top_k).clone()
    sparse_mask = torch.zeros(total_q, device=device, dtype=torch.bool)
    row_start = 0
    for spec in specs:
        start = row_start
        end = start + spec.q_len
        row_start = end
        if spec.sparse:
            sparse_mask[start:end] = True
    valid_counts = torch.minimum(
        current_chunks,
        torch.full_like(current_chunks, top_k),
    )
    top_slots = torch.arange(top_k, device=device, dtype=torch.long)
    top_valid = (top_slots.view(1, 1, top_k) < valid_counts.view(total_q, 1, 1))
    top_valid &= sparse_mask.view(total_q, 1, 1)

    batch = MixedBatch(
        specs=specs,
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        flat_key_cache=flat_key_cache,
        flat_value_cache=flat_value_cache,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        key_lens=key_lens,
        positions=positions,
        current_chunks=current_chunks,
        representatives=[],
        num_physical_blocks=next_block,
    )
    return batch, top_indices, top_valid


def _time_cuda_builder(builder, batch, top_indices, top_valid, *, block_size: int) -> tuple[DirectOutputPlan, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    plan = builder(batch, top_indices, top_valid, block_size=block_size)
    torch.cuda.synchronize()
    return plan, time.perf_counter() - start


def _assert_single_kv_head_fast_plan(batch: MixedBatch, plan: DirectOutputPlan) -> None:
    assert plan.path in {"single_kv_head_fast", "single_kv_head_vectorized"}
    assert plan.q.data_ptr() == batch.query_states.data_ptr()
    assert plan.q.shape == batch.query_states.shape
    assert plan.cu_seqlens_q[-1].item() == batch.query_states.shape[0]
    assert plan.max_seqlen_q > 1
    assert plan.block_table.shape[0] == plan.request_lens.numel()
    assert plan.seqused_k.shape[0] == plan.request_lens.numel()
    assert any(kind.startswith("fast_dense_multi_q:") for kind in plan.request_kinds)
    assert any(kind.startswith("fast_dense_q1:") for kind in plan.request_kinds)
    assert any(kind.startswith("fast_sparse_q1:") for kind in plan.request_kinds)
    for request_len, kind in zip(plan.request_lens.tolist(), plan.request_kinds):
        if kind.startswith("fast_sparse_q1:"):
            assert request_len == 1
    assert not any(kind.startswith("fallback_") for kind in plan.request_kinds)


def _assert_multi_kv_head_fallback_plan(
    batch: MixedBatch,
    plan: DirectOutputPlan,
) -> None:
    total_q, heads, _ = batch.query_states.shape
    kv_heads = batch.key_cache.shape[2]
    assert plan.path == "multi_kv_head_fallback"
    assert plan.q.shape[0] == total_q * kv_heads
    assert plan.q.shape[1] == heads // kv_heads
    assert plan.cu_seqlens_q[-1].item() == plan.q.shape[0]
    assert plan.block_table.shape[0] == plan.request_lens.numel()
    assert plan.seqused_k.shape[0] == plan.request_lens.numel()
    assert plan.request_lens.min().item() == 1
    assert plan.request_lens.max().item() == 1
    assert plan.max_seqlen_q == 1
    assert any(kind.startswith("fallback_dense_q1:") for kind in plan.request_kinds)
    assert any(kind.startswith("fallback_sparse_q1:") for kind in plan.request_kinds)
    assert not any(kind.startswith("fast_") for kind in plan.request_kinds)


@pytest.mark.parametrize("seed", [17, 23])
def test_unified_causal_plan_torch_executor_matches_reference(seed: int):
    block_size = 16
    heads = 8
    kv_heads = 2
    head_dim = 32
    q_indexer_dim = 8
    top_k = 4
    device = torch.device("cpu")
    dtype = torch.float32

    batch = _make_mixed_batch(
        device=device,
        dtype=dtype,
        block_size=block_size,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        seed=seed,
    )
    top_indices, top_valid = _score_sparse_rows(
        batch,
        block_size=block_size,
        top_k=top_k,
        q_indexer_dim=q_indexer_dim,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)
    expected = _reference_mixed_attention(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )
    plan = _build_unified_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    assert plan.request_lens.min().item() == 1
    assert plan.request_lens.max().item() > 1
    assert any(kind == "sparse:sparse_mtp_shared" for kind in plan.request_kinds)

    actual = _execute_plan_with_torch_reference(
        batch,
        plan,
        heads=heads,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), FAST_HEAD_CASES)
def test_single_kv_head_fast_plan_torch_executor_matches_reference(
    seed: int,
    heads: int,
    kv_heads: int,
):
    batch, expected, plan, softmax_scale = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
        builder=_build_single_kv_head_fast_causal_plan,
    )

    _assert_single_kv_head_fast_plan(batch, plan)
    actual = _execute_direct_plan_with_torch_reference(
        batch,
        plan,
        heads=heads,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), FAST_HEAD_CASES)
def test_single_kv_head_vectorized_plan_torch_executor_matches_reference(
    seed: int,
    heads: int,
    kv_heads: int,
):
    batch, expected, plan, softmax_scale = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
        builder=_build_single_kv_head_vectorized_causal_plan,
    )

    assert plan.path == "single_kv_head_vectorized"
    _assert_single_kv_head_fast_plan(batch, plan)
    actual = _execute_direct_plan_with_torch_reference(
        batch,
        plan,
        heads=heads,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), FAST_HEAD_CASES)
def test_single_kv_head_vectorized_metadata_matches_slow_builder(
    seed: int,
    heads: int,
    kv_heads: int,
):
    block_size = 16
    batch, _, slow_plan, _ = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
        builder=_build_single_kv_head_fast_causal_plan,
    )
    top_indices, top_valid = _score_sparse_rows(
        batch,
        block_size=block_size,
        top_k=4,
        q_indexer_dim=8,
    )
    vectorized_plan = _build_single_kv_head_vectorized_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    assert vectorized_plan.q.data_ptr() == slow_plan.q.data_ptr()
    torch.testing.assert_close(vectorized_plan.request_lens, slow_plan.request_lens)
    torch.testing.assert_close(vectorized_plan.cu_seqlens_q, slow_plan.cu_seqlens_q)
    torch.testing.assert_close(vectorized_plan.seqused_k, slow_plan.seqused_k)
    torch.testing.assert_close(vectorized_plan.block_table, slow_plan.block_table)
    assert vectorized_plan.max_seqlen_q == slow_plan.max_seqlen_q
    assert vectorized_plan.max_seqlen_k == slow_plan.max_seqlen_k
    assert vectorized_plan.request_kinds == slow_plan.request_kinds


def test_single_kv_head_vectorized_metadata_matches_slow_builder_real_mixed_8k():
    block_size = 16
    batch, top_indices, top_valid = _make_single_kv_real_mixed_stress_batch(
        block_size=block_size,
        heads=4,
        head_dim=32,
        top_k=2048,
        long_q_len=8192,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    slow_plan = _build_single_kv_head_fast_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )
    vectorized_plan = _build_single_kv_head_vectorized_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    assert vectorized_plan.q.data_ptr() == batch.query_states.data_ptr()
    assert vectorized_plan.q.data_ptr() == slow_plan.q.data_ptr()
    assert vectorized_plan.request_lens.numel() == 8199
    assert int(vectorized_plan.request_lens.sum().item()) == 8204
    assert vectorized_plan.max_seqlen_q == 6
    assert vectorized_plan.block_table.shape == (8199, 2049)
    assert vectorized_plan.request_kinds[:6] == [
        "fast_sparse_q1:real_sparse_decode",
        "fast_sparse_q1:real_sparse_mtp4",
        "fast_sparse_q1:real_sparse_mtp4",
        "fast_sparse_q1:real_sparse_mtp4",
        "fast_sparse_q1:real_sparse_mtp4",
        "fast_dense_multi_q:real_dense_short_prefill",
    ]
    assert (
        vectorized_plan.request_kinds[-1]
        == "fast_dense_q1:real_dense_decode"
    )
    assert (
        vectorized_plan.request_kinds.count(
            "fast_sparse_q1:real_sparse_chunked_prefill_8k"
        )
        == 8192
    )

    torch.testing.assert_close(vectorized_plan.request_lens, slow_plan.request_lens)
    torch.testing.assert_close(vectorized_plan.cu_seqlens_q, slow_plan.cu_seqlens_q)
    torch.testing.assert_close(vectorized_plan.seqused_k, slow_plan.seqused_k)
    torch.testing.assert_close(vectorized_plan.block_table, slow_plan.block_table)
    assert vectorized_plan.max_seqlen_k == slow_plan.max_seqlen_k
    assert vectorized_plan.request_kinds == slow_plan.request_kinds


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), FALLBACK_HEAD_CASES)
def test_multi_kv_head_fallback_plan_torch_executor_matches_reference(
    seed: int,
    heads: int,
    kv_heads: int,
):
    batch, expected, plan, softmax_scale = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
        builder=_build_multi_kv_head_fallback_causal_plan,
    )

    _assert_multi_kv_head_fallback_plan(batch, plan)
    actual = _execute_direct_plan_with_torch_reference(
        batch,
        plan,
        heads=heads,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_fast_and_fallback_builders_reject_wrong_local_kv_head_counts():
    block_size = 16
    head_dim = 32
    q_indexer_dim = 8

    multi_kv_batch = _make_mixed_batch(
        device=torch.device("cpu"),
        dtype=torch.float32,
        block_size=block_size,
        heads=8,
        kv_heads=2,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        seed=17,
    )
    multi_top_indices, multi_top_valid = _score_sparse_rows(
        multi_kv_batch,
        block_size=block_size,
        top_k=4,
        q_indexer_dim=q_indexer_dim,
    )
    try:
        _build_single_kv_head_fast_causal_plan(
            multi_kv_batch,
            multi_top_indices,
            multi_top_valid,
            block_size=block_size,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("single-KV-head fast path accepted kv_heads=2")

    single_kv_batch = _make_mixed_batch(
        device=torch.device("cpu"),
        dtype=torch.float32,
        block_size=block_size,
        heads=4,
        kv_heads=1,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        seed=17,
    )
    single_top_indices, single_top_valid = _score_sparse_rows(
        single_kv_batch,
        block_size=block_size,
        top_k=4,
        q_indexer_dim=q_indexer_dim,
    )
    try:
        _build_multi_kv_head_fallback_causal_plan(
            single_kv_batch,
            single_top_indices,
            single_top_valid,
            block_size=block_size,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("multi-KV-head fallback accepted kv_heads=1")


@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), DIRECT_HEAD_CASES)
def test_direct_output_causal_plan_torch_executor_matches_reference(
    seed: int,
    heads: int,
    kv_heads: int,
):
    batch, expected, plan, softmax_scale = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    if kv_heads == 1:
        _assert_single_kv_head_fast_plan(batch, plan)
    else:
        _assert_multi_kv_head_fallback_plan(batch, plan)

    actual = _execute_direct_plan_with_torch_reference(
        batch,
        plan,
        heads=heads,
        softmax_scale=softmax_scale,
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not os.environ.get("DSA_UNIFIED_RUN_EFFICIENCY"),
    reason="set DSA_UNIFIED_RUN_EFFICIENCY=1 to run metadata builder benchmark",
)
def test_single_kv_vectorized_metadata_builder_efficiency():
    block_size = 16
    batch, top_indices, top_valid = _make_single_kv_metadata_stress_batch(
        q_rows=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_Q_ROWS", "4096")),
        key_len=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_KEY_LEN", "1048576")),
        top_k=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_TOP_K", "2048")),
        block_size=block_size,
        heads=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_HEADS", "4")),
        head_dim=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_HEAD_DIM", "32")),
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    slow_plan, slow_s = _time_cuda_builder(
        _build_single_kv_head_fast_causal_plan,
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )
    vectorized_plan, vectorized_s = _time_cuda_builder(
        _build_single_kv_head_vectorized_causal_plan,
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    torch.testing.assert_close(vectorized_plan.request_lens, slow_plan.request_lens)
    torch.testing.assert_close(vectorized_plan.cu_seqlens_q, slow_plan.cu_seqlens_q)
    torch.testing.assert_close(vectorized_plan.seqused_k, slow_plan.seqused_k)
    torch.testing.assert_close(vectorized_plan.block_table, slow_plan.block_table)
    print(
        "single_kv_metadata_builder_efficiency "
        f"requests={vectorized_plan.request_lens.numel()} "
        f"pages={vectorized_plan.block_table.shape[1]} "
        f"slow_s={slow_s:.6f} "
        f"vectorized_s={vectorized_s:.6f} "
        f"speedup={slow_s / max(vectorized_s, 1e-9):.2f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.skipif(
    not os.environ.get("DSA_UNIFIED_RUN_EFFICIENCY"),
    reason="set DSA_UNIFIED_RUN_EFFICIENCY=1 to run metadata builder benchmark",
)
def test_single_kv_vectorized_metadata_builder_real_mixed_8k_efficiency():
    block_size = 16
    batch, top_indices, top_valid = _make_single_kv_real_mixed_stress_batch(
        block_size=block_size,
        heads=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_HEADS", "4")),
        head_dim=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_HEAD_DIM", "32")),
        top_k=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_TOP_K", "2048")),
        long_q_len=int(os.environ.get("DSA_UNIFIED_EFFICIENCY_LONG_Q", "8192")),
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    slow_plan, slow_s = _time_cuda_builder(
        _build_single_kv_head_fast_causal_plan,
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )
    vectorized_plan, vectorized_s = _time_cuda_builder(
        _build_single_kv_head_vectorized_causal_plan,
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    torch.testing.assert_close(vectorized_plan.request_lens, slow_plan.request_lens)
    torch.testing.assert_close(vectorized_plan.cu_seqlens_q, slow_plan.cu_seqlens_q)
    torch.testing.assert_close(vectorized_plan.seqused_k, slow_plan.seqused_k)
    torch.testing.assert_close(vectorized_plan.block_table, slow_plan.block_table)
    print(
        "single_kv_real_mixed_8k_metadata_builder_efficiency "
        f"requests={vectorized_plan.request_lens.numel()} "
        f"q_rows={int(vectorized_plan.request_lens.sum().item())} "
        f"pages={vectorized_plan.block_table.shape[1]} "
        f"slow_s={slow_s:.6f} "
        f"vectorized_s={vectorized_s:.6f} "
        f"speedup={slow_s / max(vectorized_s, 1e-9):.2f}x"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [17, 23])
def test_unified_causal_fa_plan_matches_per_sequence_reference(seed: int):
    block_size = 16
    heads = 8
    kv_heads = 2
    head_dim = 32
    q_indexer_dim = 8
    top_k = 4
    fa_version = _test_fa_version()
    device = torch.device("cuda")
    dtype = torch.float16

    batch = _make_mixed_batch(
        device=device,
        dtype=dtype,
        block_size=block_size,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        seed=seed,
    )
    top_indices, top_valid = _score_sparse_rows(
        batch,
        block_size=block_size,
        top_k=top_k,
        q_indexer_dim=q_indexer_dim,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    expected = _reference_mixed_attention(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )
    plan = _build_unified_causal_plan(
        batch,
        top_indices,
        top_valid,
        block_size=block_size,
    )

    assert plan.block_table.dim() == 2
    assert plan.seqused_k.shape[0] == plan.block_table.shape[0]
    assert plan.request_lens.min().item() == 1
    assert plan.request_lens.max().item() > 1
    assert any(kind.startswith("dense:") for kind in plan.request_kinds)
    assert any(kind.startswith("sparse:") for kind in plan.request_kinds)

    actual = _execute_unified_plan(
        batch,
        plan,
        heads=heads,
        fa_version=fa_version,
        softmax_scale=softmax_scale,
    )
    if os.environ.get("DSA_UNIFIED_VERBOSE"):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        print(f"fa_assert seed={seed} max_diff={max_diff:.6g}")
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=3e-2,
        rtol=3e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("seed", [17, 23])
@pytest.mark.parametrize(("heads", "kv_heads"), DIRECT_HEAD_CASES)
def test_direct_output_causal_fa_plan_matches_per_sequence_reference(
    seed: int,
    heads: int,
    kv_heads: int,
):
    fa_version = _test_fa_version()
    batch, expected, plan, softmax_scale = _make_expected_and_direct_plan(
        seed=seed,
        heads=heads,
        kv_heads=kv_heads,
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    if kv_heads == 1:
        _assert_single_kv_head_fast_plan(batch, plan)
    else:
        _assert_multi_kv_head_fallback_plan(batch, plan)

    actual = _execute_direct_output_plan(
        batch,
        plan,
        heads=heads,
        fa_version=fa_version,
        softmax_scale=softmax_scale,
    )
    if os.environ.get("DSA_UNIFIED_VERBOSE"):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        print(
            "direct_fa_assert "
            f"path={plan.path} "
            f"seed={seed} heads={heads} kv_heads={kv_heads} "
            f"max_diff={max_diff:.6g}"
        )
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=3e-2,
        rtol=3e-2,
    )


def _run_as_script() -> None:
    for seed in (17, 23):
        test_unified_causal_plan_torch_executor_matches_reference(seed)
    for heads, kv_heads in HEAD_CASES:
        for seed in (17, 23):
            test_direct_output_causal_plan_torch_executor_matches_reference(
                seed,
                heads,
                kv_heads,
            )
            test_direct_output_causal_fa_plan_matches_per_sequence_reference(
                seed,
                heads,
                kv_heads,
            )
    print("standalone unified causal DSA checks passed")


if __name__ == "__main__":
    _run_as_script()
