#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Synthetic prefill benchmark for Nemotron-H chunked DSA kernels.

This is intentionally narrower than an end-to-end model run. It builds one
local tensor-parallel attention slice with fake data and compares:

* dense full FlashAttention over the full KV history, for a speed target;
* the current chunked DSA Python + page-table FlashAttention path;
* the current chunked DSA page-table apply path with top chunks precomputed.

Correctness is checked on a smaller shape by comparing the page-table path to
the exact manual gather path for the same DSA chunk choices. Dense FA is a speed
baseline only because it has different semantics from sparse DSA.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get("VLLM_DSA_BENCH_USE_SITE_PACKAGE", "0") != "1":
    sys.path.insert(0, str(REPO_ROOT))

from vllm.model_executor.models import nemotron_h
from vllm.model_executor.layers.dsa_moonshot_attention import (
    dsa_prefill_gqa_attention,
    dsa_prefill_gqa_splitk_attention,
    dsa_prefill_gqa_union_attention,
    dsa_prefill_gqa_union_qh_attention,
    dsa_prefill_gqa_wide_union_attention,
)
from vllm.model_executor.models.nemotron_h import NemotronHDSASelectiveAttention


@dataclass
class BenchTensors:
    query_states: torch.Tensor
    indexer_query_states: torch.Tensor
    key_states: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    positions: torch.Tensor


@dataclass
class PrefillWorkItem:
    query_start: int
    query_end: int
    group_idx: int
    top_chunk_indices: torch.Tensor
    top_chunk_valid: torch.Tensor
    current_chunks: torch.Tensor
    query_positions: torch.Tensor


@dataclass
class SharedRunWorkItem:
    query_start: int
    query_end: int
    group_idx: int
    run_block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seqused_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass
class UnionWorkItem:
    query_start: int
    query_end: int
    group_idx: int
    union_chunks: torch.Tensor
    union_masks: torch.Tensor
    union_counts: torch.Tensor
    row_starts: torch.Tensor
    row_counts: torch.Tensor
    current_chunks: torch.Tensor
    tail_lens: torch.Tensor


@dataclass
class WideUnionWorkItem:
    query_start: int
    query_end: int
    group_idx: int
    union_chunks: torch.Tensor
    full_masks: torch.Tensor
    current_masks: torch.Tensor
    union_counts: torch.Tensor
    row_starts: torch.Tensor
    row_counts: torch.Tensor
    tail_lens: torch.Tensor


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def _summarize_times(times_ms: list[float]) -> dict[str, float]:
    return {
        "avg_ms": statistics.mean(times_ms),
        "min_ms": min(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p90_ms": _percentile(times_ms, 0.90),
        "max_ms": max(times_ms),
    }


def _measure_cuda(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    assert output is not None
    stats = _summarize_times(times_ms)
    stats["iters"] = float(iters)
    print(f"{name}: {json.dumps(stats, sort_keys=True)}", flush=True)
    return output, stats


def _make_attn(args: argparse.Namespace, *, use_page_table: bool) -> NemotronHDSASelectiveAttention:
    attn = NemotronHDSASelectiveAttention.__new__(NemotronHDSASelectiveAttention)
    attn.q_indexer_chunk_size = args.chunk_size
    attn.q_indexer_chunk_top_k = args.top_chunks
    attn.q_indexer_chunked_query_chunk_size = args.query_chunk_size
    attn.q_indexer_logit_scale = 1.0
    attn.q_indexer_dim = args.q_indexer_dim
    attn.q_indexer_attn_mode = "chunked_topk_sparse"
    attn.num_kv_heads = args.num_kv_heads
    attn.num_heads = args.num_heads
    attn.head_dim = args.head_dim
    attn.layer_idx = 0
    attn.q_indexer_use_flash_topk = False
    attn.q_indexer_use_page_table_fa = False
    attn.q_indexer_use_prefill_page_table_fa = use_page_table
    attn.q_indexer_use_full_attention_short_seq = False
    attn.q_indexer_share_chunk_topk = False
    attn.q_indexer_use_shared_prefill_page_table_fa = False
    attn.q_indexer_use_union_prefill_kernel = False
    attn.q_indexer_use_union_superset_prefill_page_table_fa = False
    attn.q_indexer_union_chunks_per_iter = args.union_chunks_per_iter
    attn._dsa_cache_config_block_size = args.chunk_size
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


def _pack_identity_nhd_cache(states: torch.Tensor, block_size: int) -> torch.Tensor:
    key_len, num_kv_heads, head_dim = states.shape
    num_blocks = math.ceil(key_len / block_size)
    padded_len = num_blocks * block_size
    if padded_len != key_len:
        padding = states.new_zeros(padded_len - key_len, num_kv_heads, head_dim)
        states = torch.cat((states, padding), dim=0)
    return states.view(num_blocks, block_size, num_kv_heads, head_dim).contiguous()


def _make_tensors(
    args: argparse.Namespace,
    *,
    key_len: int,
    q_len: int,
    top_chunks: int | None = None,
) -> BenchTensors:
    if key_len < q_len:
        raise ValueError(f"key_len must be >= q_len, got {key_len=} {q_len=}")
    dtype = _dtype_from_name(args.dtype)
    device = torch.device(args.device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    key_states = torch.randn(
        key_len,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    value_states = torch.randn(
        key_len,
        args.num_kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    query_states = torch.randn(
        q_len,
        args.num_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    indexer_query_states = torch.randn(
        q_len,
        args.num_kv_heads,
        args.q_indexer_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    num_blocks = math.ceil(key_len / args.chunk_size)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)
    positions = torch.arange(key_len - q_len, key_len, device=device, dtype=torch.long)
    if top_chunks is not None:
        args.top_chunks = top_chunks
    return BenchTensors(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=_pack_identity_nhd_cache(key_states, args.chunk_size),
        value_cache=_pack_identity_nhd_cache(value_states, args.chunk_size),
        block_table=block_table,
        positions=positions,
    )


def _silence_dsa_debug() -> None:
    nemotron_h._DSA_DEBUG_FORWARD_PRINT_COUNT = (
        nemotron_h._DSA_DEBUG_FORWARD_PRINT_LIMIT
    )
    nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = (
        nemotron_h._DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT
    )


def _build_prefill_work(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    *,
    top_chunk_pattern: str = "indexer",
    shared_noise_chunks: int = 32,
) -> list[PrefillWorkItem]:
    q_len = tensors.query_states.shape[0]
    key_len = tensors.key_states.shape[0]
    chunk_size = attn.q_indexer_chunk_size
    num_chunks = math.ceil(key_len / chunk_size)
    query_chunk_size = min(attn.q_indexer_chunked_query_chunk_size, q_len)
    indexer_scale = attn.q_indexer_logit_scale / math.sqrt(attn.q_indexer_dim)
    indexer_key_states = tensors.key_states[..., : attn.q_indexer_dim]
    chunk_representatives = attn._build_indexer_chunk_representatives(
        indexer_key_states
    )
    work: list[PrefillWorkItem] = []

    for query_start in range(0, q_len, query_chunk_size):
        query_end = min(query_start + query_chunk_size, q_len)
        query_positions = tensors.positions[query_start:query_end].to(torch.long)
        current_chunks = torch.div(
            query_positions,
            chunk_size,
            rounding_mode="floor",
        ).clamp_(min=0, max=num_chunks - 1)
        max_prior_chunks = int(current_chunks.max().item())
        chunk_ids = torch.arange(
            max_prior_chunks,
            device=tensors.query_states.device,
            dtype=torch.long,
        )
        chunk_len = query_end - query_start

        for group_idx in range(attn.num_kv_heads):
            if max_prior_chunks > 0:
                chunk_top_k = min(attn.q_indexer_chunk_top_k, max_prior_chunks)
                if top_chunk_pattern == "shared-noise-current-chunk":
                    top_chunk_indices = torch.empty(
                        chunk_len,
                        chunk_top_k,
                        device=tensors.query_states.device,
                        dtype=torch.long,
                    )
                    top_chunk_valid = torch.ones(
                        chunk_len,
                        chunk_top_k,
                        device=tensors.query_states.device,
                        dtype=torch.bool,
                    )
                    for chunk in current_chunks.unique(sorted=True):
                        rows = torch.nonzero(current_chunks == chunk, as_tuple=False)
                        if rows.numel() == 0:
                            continue
                        row_indices = rows.reshape(-1)
                        prior_count = int(chunk.item())
                        if prior_count <= chunk_top_k:
                            top_chunk_indices[row_indices] = torch.arange(
                                chunk_top_k,
                                device=tensors.query_states.device,
                                dtype=torch.long,
                            )
                            continue
                        noise_count = min(shared_noise_chunks, chunk_top_k)
                        shared_count = chunk_top_k - noise_count
                        needed = min(
                            prior_count,
                            shared_count + noise_count * int(row_indices.numel()),
                        )
                        generator = torch.Generator(device="cpu")
                        generator.manual_seed(
                            args_seed_from_chunk(prior_count, int(group_idx))
                        )
                        pool = torch.randperm(
                            prior_count,
                            generator=generator,
                            dtype=torch.long,
                        )
                        shared = pool[:shared_count]
                        noise_pool = pool[shared_count:needed]
                        if noise_pool.numel() < noise_count:
                            noise_pool = pool[shared_count:]
                        for local_row, row_idx in enumerate(row_indices.tolist()):
                            noise_start = local_row * noise_count
                            noise_end = noise_start + noise_count
                            if noise_end <= noise_pool.numel():
                                noise = noise_pool[noise_start:noise_end]
                            else:
                                noise = noise_pool[:noise_count]
                            top_chunks = torch.cat((shared, noise)).to(
                                device=tensors.query_states.device
                            )
                            top_chunk_indices[row_idx] = top_chunks
                else:
                    chunk_logits = torch.matmul(
                        tensors.indexer_query_states[
                            query_start:query_end, group_idx
                        ].float(),
                        chunk_representatives[:max_prior_chunks, group_idx].transpose(
                            0, 1
                        ),
                    )
                    chunk_logits.mul_(indexer_scale)
                    chunk_valid = chunk_ids[None, :] < current_chunks[:, None]
                    chunk_logits = chunk_logits.masked_fill(
                        ~chunk_valid,
                        torch.finfo(chunk_logits.dtype).min,
                    )
                    top_chunk_indices = chunk_logits.topk(
                        k=chunk_top_k,
                        dim=-1,
                    ).indices
                    top_chunk_valid = chunk_valid.gather(
                        dim=-1,
                        index=top_chunk_indices,
                    )
                if top_chunk_pattern == "shared-current-chunk":
                    for chunk in current_chunks.unique(sorted=True):
                        rows = torch.nonzero(current_chunks == chunk, as_tuple=False)
                        if rows.numel() == 0:
                            continue
                        src = int(rows[-1].item())
                        row_indices = rows.reshape(-1)
                        top_chunk_indices[row_indices] = top_chunk_indices[src].clone()
                        top_chunk_valid[row_indices] = top_chunk_valid[src].clone()
                elif top_chunk_pattern not in {
                    "indexer",
                    "shared-noise-current-chunk",
                }:
                    raise ValueError(f"unknown top_chunk_pattern={top_chunk_pattern!r}")
            else:
                top_chunk_indices = chunk_ids.new_empty(chunk_len, 0)
                top_chunk_valid = torch.empty(
                    chunk_len,
                    0,
                    device=tensors.query_states.device,
                    dtype=torch.bool,
                )
            work.append(
                PrefillWorkItem(
                    query_start=query_start,
                    query_end=query_end,
                    group_idx=group_idx,
                    top_chunk_indices=top_chunk_indices,
                    top_chunk_valid=top_chunk_valid,
                    current_chunks=current_chunks,
                    query_positions=query_positions,
                )
            )
    return work


def args_seed_from_chunk(current_chunk: int, group_idx: int) -> int:
    return (current_chunk * 1103515245 + group_idx * 12345 + 0x5EED) & 0x7FFFFFFF


def _build_shared_run_work(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
) -> list[SharedRunWorkItem]:
    chunk_size = attn.q_indexer_chunk_size
    shared_work: list[SharedRunWorkItem] = []
    for item in work:
        top_chunk_indices = item.top_chunk_indices.detach().cpu()
        top_chunk_valid = item.top_chunk_valid.detach().cpu()
        current_chunks = item.current_chunks.detach().cpu()
        query_positions = item.query_positions.detach().cpu()
        rows = top_chunk_indices.shape[0]
        top_chunks = top_chunk_indices.shape[1]
        run_block_tables: list[torch.Tensor] = []
        cu_seqlens = [0]
        seqused = []
        row = 0
        while row < rows:
            end = row + 1
            while (
                end < rows
                and int(current_chunks[end].item()) == int(current_chunks[row].item())
                and bool(torch.equal(top_chunk_indices[end], top_chunk_indices[row]))
                and bool(torch.equal(top_chunk_valid[end], top_chunk_valid[row]))
            ):
                end += 1
            if not bool(top_chunk_valid[row].all().item()):
                raise RuntimeError("shared-run FA prototype expects valid top chunks")
            logical_chunks = torch.cat(
                (
                    top_chunk_indices[row].to(torch.long),
                    current_chunks[row : row + 1].to(torch.long),
                )
            ).to(device=tensors.block_table.device)
            run_block_tables.append(
                tensors.block_table.index_select(0, logical_chunks).to(torch.int32)
            )
            cu_seqlens.append(cu_seqlens[-1] + (end - row))
            current_start = int(current_chunks[row].item()) * chunk_size
            tail_len = int(query_positions[end - 1].item()) - current_start + 1
            seqused.append(top_chunks * chunk_size + tail_len)
            row = end

        run_block_table = torch.stack(run_block_tables).to(
            device=tensors.query_states.device
        )
        cu_seqlens_q = torch.tensor(
            cu_seqlens,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        seqused_k = torch.tensor(
            seqused,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        shared_work.append(
            SharedRunWorkItem(
                query_start=item.query_start,
                query_end=item.query_end,
                group_idx=item.group_idx,
                run_block_table=run_block_table,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                max_seqlen_q=max(
                    cu_seqlens[i + 1] - cu_seqlens[i]
                    for i in range(len(cu_seqlens) - 1)
                ),
                max_seqlen_k=int(max(seqused)),
            )
        )
    return shared_work


def _build_union_work(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
    *,
    union_rows: int,
) -> list[UnionWorkItem]:
    if union_rows <= 0 or union_rows > 16:
        raise ValueError(f"union_rows must be in [1, 16], got {union_rows}")

    chunk_size = attn.q_indexer_chunk_size
    union_work: list[UnionWorkItem] = []
    for item in work:
        top_chunk_indices = item.top_chunk_indices.detach().cpu()
        top_chunk_valid = item.top_chunk_valid.detach().cpu()
        current_chunks = item.current_chunks.detach().cpu()
        query_positions = item.query_positions.detach().cpu()
        rows = top_chunk_indices.shape[0]

        chunks_by_block: list[torch.Tensor] = []
        masks_by_block: list[torch.Tensor] = []
        counts: list[int] = []
        row_starts: list[int] = []
        row_counts: list[int] = []
        current_by_block: list[int] = []
        tails_by_block: list[list[int]] = []

        row = 0
        while row < rows:
            run_end = row + 1
            while (
                run_end < rows
                and int(current_chunks[run_end].item()) == int(current_chunks[row].item())
            ):
                run_end += 1

            block_start = row
            while block_start < run_end:
                block_end = min(block_start + union_rows, run_end)
                membership: dict[int, int] = {}
                for local_row, src_row in enumerate(range(block_start, block_end)):
                    valid_chunks = top_chunk_indices[src_row][top_chunk_valid[src_row]]
                    for chunk in valid_chunks.tolist():
                        membership[int(chunk)] = membership.get(int(chunk), 0) | (
                            1 << local_row
                        )
                logical_chunks = sorted(membership)
                chunks_by_block.append(torch.tensor(logical_chunks, dtype=torch.int32))
                masks_by_block.append(
                    torch.tensor(
                        [membership[chunk] for chunk in logical_chunks],
                        dtype=torch.int32,
                    )
                )
                counts.append(len(logical_chunks))
                row_starts.append(block_start)
                row_counts.append(block_end - block_start)
                current_chunk = int(current_chunks[block_start].item())
                current_by_block.append(current_chunk)
                current_start = current_chunk * chunk_size
                tails = []
                for src_row in range(block_start, block_end):
                    tails.append(int(query_positions[src_row].item()) - current_start + 1)
                tails.extend([0] * (union_rows - len(tails)))
                tails_by_block.append(tails)
                block_start = block_end
            row = run_end

        max_union = max(counts, default=0)
        if max_union == 0:
            union_chunks = torch.empty(
                len(counts),
                0,
                device=tensors.query_states.device,
                dtype=torch.int32,
            )
            union_masks = torch.empty_like(union_chunks)
        else:
            union_chunks_cpu = torch.zeros(len(counts), max_union, dtype=torch.int32)
            union_masks_cpu = torch.zeros(len(counts), max_union, dtype=torch.int32)
            for block_idx, (chunks, masks) in enumerate(
                zip(chunks_by_block, masks_by_block)
            ):
                union_chunks_cpu[block_idx, : chunks.numel()] = chunks
                union_masks_cpu[block_idx, : masks.numel()] = masks
            union_chunks = union_chunks_cpu.to(device=tensors.query_states.device)
            union_masks = union_masks_cpu.to(device=tensors.query_states.device)

        union_work.append(
            UnionWorkItem(
                query_start=item.query_start,
                query_end=item.query_end,
                group_idx=item.group_idx,
                union_chunks=union_chunks,
                union_masks=union_masks,
                union_counts=torch.tensor(
                    counts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                row_starts=torch.tensor(
                    row_starts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                row_counts=torch.tensor(
                    row_counts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                current_chunks=torch.tensor(
                    current_by_block,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                tail_lens=torch.tensor(
                    tails_by_block,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
            )
        )
    return union_work


def _build_union_work_sort_gpu(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
    *,
    union_rows: int,
) -> list[UnionWorkItem]:
    if union_rows <= 0 or union_rows > 16:
        raise ValueError(f"union_rows must be in [1, 16], got {union_rows}")

    chunk_size = attn.q_indexer_chunk_size
    union_work: list[UnionWorkItem] = []
    for item in work:
        rows, top_k = item.top_chunk_indices.shape
        if rows % union_rows != 0 or not bool(item.top_chunk_valid.all().item()):
            return _build_union_work(
                attn,
                tensors,
                work,
                union_rows=union_rows,
            )

        groups = rows // union_rows
        flat = item.top_chunk_indices.to(torch.int32).view(
            groups,
            union_rows * top_k,
        )
        sorted_chunks, sorted_pos = flat.sort(dim=1)
        unique = torch.ones_like(sorted_chunks, dtype=torch.bool)
        unique[:, 1:] = sorted_chunks[:, 1:] != sorted_chunks[:, :-1]
        unique_rank_sorted = unique.cumsum(dim=1, dtype=torch.int32) - 1
        union_counts = unique.sum(dim=1, dtype=torch.int32)
        max_union = int(union_counts.max().item())

        union_chunks = torch.empty(
            groups,
            max_union,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        rank_clamped = unique_rank_sorted.clamp_max(max_union - 1).to(torch.long)
        union_chunks.scatter_(1, rank_clamped, sorted_chunks)

        rank_for_flat = torch.empty_like(unique_rank_sorted)
        rank_for_flat.scatter_(1, sorted_pos, unique_rank_sorted)
        row_ids = torch.arange(
            union_rows,
            device=tensors.query_states.device,
            dtype=torch.int32,
        ).repeat_interleave(top_k)
        row_bits = (1 << row_ids).expand(groups, -1)
        union_masks = torch.zeros(
            groups,
            max_union,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        union_masks.scatter_reduce_(
            1,
            rank_for_flat.to(torch.long),
            row_bits,
            reduce="sum",
            include_self=False,
        )

        row_starts = (
            torch.arange(groups, device=tensors.query_states.device, dtype=torch.int32)
            * union_rows
        )
        row_counts = torch.full(
            (groups,),
            union_rows,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        grouped_current = item.current_chunks.to(torch.int32).view(groups, union_rows)
        current_chunks = grouped_current[:, 0].contiguous()
        grouped_positions = item.query_positions.to(torch.int32).view(groups, union_rows)
        tail_lens = (
            grouped_positions - current_chunks[:, None] * chunk_size + 1
        ).contiguous()

        union_work.append(
            UnionWorkItem(
                query_start=item.query_start,
                query_end=item.query_end,
                group_idx=item.group_idx,
                union_chunks=union_chunks,
                union_masks=union_masks,
                union_counts=union_counts.contiguous(),
                row_starts=row_starts,
                row_counts=row_counts,
                current_chunks=current_chunks,
                tail_lens=tail_lens,
            )
        )
    return union_work


def _build_wide_union_work(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
    *,
    union_rows: int,
) -> list[WideUnionWorkItem]:
    if union_rows <= 0 or union_rows > 63:
        raise ValueError(f"wide union_rows must be in [1, 63], got {union_rows}")

    chunk_size = attn.q_indexer_chunk_size
    wide_work: list[WideUnionWorkItem] = []
    for item in work:
        top_chunk_indices = item.top_chunk_indices.detach().cpu()
        top_chunk_valid = item.top_chunk_valid.detach().cpu()
        current_chunks = item.current_chunks.detach().cpu()
        query_positions = item.query_positions.detach().cpu()
        rows = top_chunk_indices.shape[0]

        chunks_by_block: list[torch.Tensor] = []
        full_masks_by_block: list[torch.Tensor] = []
        current_masks_by_block: list[torch.Tensor] = []
        counts: list[int] = []
        row_starts: list[int] = []
        row_counts: list[int] = []
        tails_by_block: list[list[int]] = []

        block_start = 0
        while block_start < rows:
            block_end = min(block_start + union_rows, rows)
            full_membership: dict[int, int] = {}
            current_membership: dict[int, int] = {}
            tails: list[int] = []
            for local_row, src_row in enumerate(range(block_start, block_end)):
                row_bit = 1 << local_row
                valid_chunks = top_chunk_indices[src_row][top_chunk_valid[src_row]]
                for chunk in valid_chunks.tolist():
                    full_membership[int(chunk)] = full_membership.get(int(chunk), 0) | row_bit
                current_chunk = int(current_chunks[src_row].item())
                current_membership[current_chunk] = (
                    current_membership.get(current_chunk, 0) | row_bit
                )
                current_start = current_chunk * chunk_size
                tails.append(int(query_positions[src_row].item()) - current_start + 1)
            tails.extend([0] * (union_rows - len(tails)))

            logical_chunks = sorted(set(full_membership) | set(current_membership))
            chunks_by_block.append(torch.tensor(logical_chunks, dtype=torch.int32))
            full_masks_by_block.append(
                torch.tensor(
                    [full_membership.get(chunk, 0) for chunk in logical_chunks],
                    dtype=torch.int64,
                )
            )
            current_masks_by_block.append(
                torch.tensor(
                    [current_membership.get(chunk, 0) for chunk in logical_chunks],
                    dtype=torch.int64,
                )
            )
            counts.append(len(logical_chunks))
            row_starts.append(block_start)
            row_counts.append(block_end - block_start)
            tails_by_block.append(tails)
            block_start = block_end

        max_union = max(counts, default=0)
        union_chunks_cpu = torch.zeros(len(counts), max_union, dtype=torch.int32)
        full_masks_cpu = torch.zeros(len(counts), max_union, dtype=torch.int64)
        current_masks_cpu = torch.zeros(len(counts), max_union, dtype=torch.int64)
        for block_idx, (chunks, full_masks, current_masks) in enumerate(
            zip(chunks_by_block, full_masks_by_block, current_masks_by_block)
        ):
            union_chunks_cpu[block_idx, : chunks.numel()] = chunks
            full_masks_cpu[block_idx, : full_masks.numel()] = full_masks
            current_masks_cpu[block_idx, : current_masks.numel()] = current_masks

        wide_work.append(
            WideUnionWorkItem(
                query_start=item.query_start,
                query_end=item.query_end,
                group_idx=item.group_idx,
                union_chunks=union_chunks_cpu.to(device=tensors.query_states.device),
                full_masks=full_masks_cpu.to(device=tensors.query_states.device),
                current_masks=current_masks_cpu.to(device=tensors.query_states.device),
                union_counts=torch.tensor(
                    counts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                row_starts=torch.tensor(
                    row_starts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                row_counts=torch.tensor(
                    row_counts,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
                tail_lens=torch.tensor(
                    tails_by_block,
                    device=tensors.query_states.device,
                    dtype=torch.int32,
                ),
            )
        )
    return wide_work


def _build_wide_union_work_sort_gpu(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
    *,
    union_rows: int,
) -> list[WideUnionWorkItem]:
    if union_rows <= 0 or union_rows > 63:
        raise ValueError(f"wide union_rows must be in [1, 63], got {union_rows}")

    chunk_size = attn.q_indexer_chunk_size
    wide_work: list[WideUnionWorkItem] = []
    for item in work:
        rows, top_k = item.top_chunk_indices.shape
        if rows % union_rows != 0 or not bool(item.top_chunk_valid.all().item()):
            return _build_wide_union_work(
                attn,
                tensors,
                work,
                union_rows=union_rows,
            )

        groups = rows // union_rows
        top_flat = item.top_chunk_indices.to(torch.int32).view(
            groups,
            union_rows * top_k,
        )
        current_flat = item.current_chunks.to(torch.int32).view(groups, union_rows)
        flat = torch.cat((top_flat, current_flat), dim=1)
        sorted_chunks, sorted_pos = flat.sort(dim=1)
        unique = torch.ones_like(sorted_chunks, dtype=torch.bool)
        unique[:, 1:] = sorted_chunks[:, 1:] != sorted_chunks[:, :-1]
        unique_rank_sorted = unique.cumsum(dim=1, dtype=torch.int32) - 1
        union_counts = unique.sum(dim=1, dtype=torch.int32)
        max_union = int(union_counts.max().item())

        union_chunks = torch.empty(
            groups,
            max_union,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        rank_clamped = unique_rank_sorted.clamp_max(max_union - 1).to(torch.long)
        union_chunks.scatter_(1, rank_clamped, sorted_chunks)

        rank_for_flat = torch.empty_like(unique_rank_sorted)
        rank_for_flat.scatter_(1, sorted_pos, unique_rank_sorted)

        top_row_ids = torch.arange(
            union_rows,
            device=tensors.query_states.device,
            dtype=torch.int64,
        ).repeat_interleave(top_k)
        current_row_ids = torch.arange(
            union_rows,
            device=tensors.query_states.device,
            dtype=torch.int64,
        )
        top_bits = (torch.ones_like(top_row_ids, dtype=torch.int64) << top_row_ids)
        current_bits = (
            torch.ones_like(current_row_ids, dtype=torch.int64) << current_row_ids
        )
        full_bits = torch.cat(
            (
                top_bits.expand(groups, -1),
                torch.zeros(groups, union_rows, device=tensors.query_states.device, dtype=torch.int64),
            ),
            dim=1,
        )
        current_bits_flat = torch.cat(
            (
                torch.zeros(
                    groups,
                    union_rows * top_k,
                    device=tensors.query_states.device,
                    dtype=torch.int64,
                ),
                current_bits.expand(groups, -1),
            ),
            dim=1,
        )

        full_masks = torch.zeros(
            groups,
            max_union,
            device=tensors.query_states.device,
            dtype=torch.int64,
        )
        current_masks = torch.zeros_like(full_masks)
        full_masks.scatter_reduce_(
            1,
            rank_for_flat.to(torch.long),
            full_bits,
            reduce="sum",
            include_self=False,
        )
        current_masks.scatter_reduce_(
            1,
            rank_for_flat.to(torch.long),
            current_bits_flat,
            reduce="sum",
            include_self=False,
        )

        row_starts = (
            torch.arange(groups, device=tensors.query_states.device, dtype=torch.int32)
            * union_rows
        )
        row_counts = torch.full(
            (groups,),
            union_rows,
            device=tensors.query_states.device,
            dtype=torch.int32,
        )
        grouped_positions = item.query_positions.to(torch.int32).view(groups, union_rows)
        tail_lens = (
            grouped_positions - current_flat * chunk_size + 1
        ).contiguous()

        wide_work.append(
            WideUnionWorkItem(
                query_start=item.query_start,
                query_end=item.query_end,
                group_idx=item.group_idx,
                union_chunks=union_chunks,
                full_masks=full_masks,
                current_masks=current_masks,
                union_counts=union_counts.contiguous(),
                row_starts=row_starts,
                row_counts=row_counts,
                tail_lens=tail_lens,
            )
        )
    return wide_work


def _kernel_enabled(args: argparse.Namespace, name: str) -> bool:
    kernels = set(args.kernels.split(","))
    return "all" in kernels or name in kernels


def _same_top_chunks(
    top_chunk_indices: torch.Tensor,
    lhs: int,
    rhs: int,
) -> bool:
    return bool(torch.equal(top_chunk_indices[lhs], top_chunk_indices[rhs]))


def _summarize_work_items(work: list[PrefillWorkItem]) -> dict[str, float]:
    run_lengths: list[int] = []
    for item in work:
        top_chunk_indices = item.top_chunk_indices.detach().cpu()
        current_chunks = item.current_chunks.detach().cpu()
        row = 0
        while row < top_chunk_indices.shape[0]:
            end = row + 1
            while (
                end < top_chunk_indices.shape[0]
                and int(current_chunks[end].item()) == int(current_chunks[row].item())
                and _same_top_chunks(top_chunk_indices, row, end)
            ):
                end += 1
            run_lengths.append(end - row)
            row = end

    if not run_lengths:
        return {
            "work_items": 0.0,
            "shared_run_avg": 0.0,
            "shared_run_max": 0.0,
            "shared_run_ge2_frac": 0.0,
        }
    return {
        "work_items": float(len(work)),
        "shared_run_avg": statistics.mean(run_lengths),
        "shared_run_max": float(max(run_lengths)),
        "shared_run_ge2_frac": sum(length >= 2 for length in run_lengths)
        / len(run_lengths),
    }


def _summarize_union_work(union_work: list[UnionWorkItem]) -> dict[str, float]:
    counts: list[int] = []
    row_counts: list[int] = []
    for item in union_work:
        counts.extend(int(v) for v in item.union_counts.detach().cpu().tolist())
        row_counts.extend(int(v) for v in item.row_counts.detach().cpu().tolist())
    if not counts:
        return {
            "union_blocks": 0.0,
            "union_avg_chunks": 0.0,
            "union_max_chunks": 0.0,
            "union_avg_rows": 0.0,
        }
    return {
        "union_blocks": float(len(counts)),
        "union_avg_chunks": statistics.mean(counts),
        "union_max_chunks": float(max(counts)),
        "union_avg_rows": statistics.mean(row_counts),
    }


def _summarize_wide_union_work(
    wide_union_work: list[WideUnionWorkItem],
) -> dict[str, float]:
    counts: list[int] = []
    row_counts: list[int] = []
    for item in wide_union_work:
        counts.extend(int(v) for v in item.union_counts.detach().cpu().tolist())
        row_counts.extend(int(v) for v in item.row_counts.detach().cpu().tolist())
    if not counts:
        return {
            "wide_union_blocks": 0.0,
            "wide_union_avg_chunks": 0.0,
            "wide_union_max_chunks": 0.0,
            "wide_union_avg_rows": 0.0,
        }
    return {
        "wide_union_blocks": float(len(counts)),
        "wide_union_avg_chunks": statistics.mean(counts),
        "wide_union_max_chunks": float(max(counts)),
        "wide_union_avg_rows": statistics.mean(row_counts),
    }


def _gib(num_bytes: float) -> float:
    return num_bytes / float(1024**3)


def _summarize_reverse_mapping(
    work: list[PrefillWorkItem],
    *,
    chunk_size: int,
    group_size: int,
    head_dim: int,
    elem_bytes: int,
) -> dict[str, float]:
    chunks_by_group: dict[int, list[torch.Tensor]] = {}
    q_rows_by_group: dict[int, int] = {}
    top_pairs = 0
    for item in work:
        valid_top_chunks = item.top_chunk_indices[item.top_chunk_valid].to(
            torch.int64
        )
        top_pairs += int(valid_top_chunks.numel())
        current_chunks = item.current_chunks.to(torch.int64)
        chunks = torch.cat((valid_top_chunks.reshape(-1), current_chunks.reshape(-1)))
        chunks_by_group.setdefault(item.group_idx, []).append(chunks)
        q_rows_by_group[item.group_idx] = (
            q_rows_by_group.get(item.group_idx, 0) + item.top_chunk_indices.shape[0]
        )

    total_pairs = 0
    total_dense_cells = 0
    active_blocks: list[int] = []
    count_values: list[int] = []
    for group_idx, chunks_parts in chunks_by_group.items():
        chunks = torch.cat(chunks_parts)
        if chunks.numel() == 0:
            active = 0
            counts = torch.empty(0, device=chunks.device, dtype=torch.int64)
        else:
            _, counts = torch.unique(chunks, return_counts=True)
            active = int(counts.numel())
        pairs = int(chunks.numel())
        q_rows = q_rows_by_group[group_idx]
        total_pairs += pairs
        total_dense_cells += q_rows * active
        active_blocks.append(active)
        count_values.extend(int(v) for v in counts.detach().cpu().tolist())

    if count_values:
        counts_sorted = sorted(count_values)
        p50 = _percentile([float(v) for v in counts_sorted], 0.50)
        p90 = _percentile([float(v) for v in counts_sorted], 0.90)
        p99 = _percentile([float(v) for v in counts_sorted], 0.99)
        max_queries = float(max(count_values))
    else:
        p50 = p90 = p99 = max_queries = 0.0

    sparse_logit_bytes = total_pairs * chunk_size * group_size * elem_bytes
    dense_logit_bytes = total_dense_cells * chunk_size * group_size * elem_bytes
    sparse_chunk_stats_bytes = total_pairs * group_size * 2 * 4
    dense_chunk_stats_bytes = total_dense_cells * group_size * 2 * 4
    sparse_partial_out_bytes = total_pairs * group_size * head_dim * 4
    dense_partial_out_bytes = total_dense_cells * group_size * head_dim * 4
    sparse_pair_index_bytes = total_pairs * 8

    density = total_pairs / total_dense_cells if total_dense_cells else 0.0
    avg_active = statistics.mean(active_blocks) if active_blocks else 0.0
    max_active = float(max(active_blocks)) if active_blocks else 0.0
    return {
        "reverse_groups": float(len(chunks_by_group)),
        "reverse_top_pairs": float(top_pairs),
        "reverse_pairs_with_current": float(total_pairs),
        "reverse_active_blocks_avg": float(avg_active),
        "reverse_active_blocks_max": max_active,
        "reverse_dense_q_block_cells": float(total_dense_cells),
        "reverse_density": float(density),
        "reverse_queries_per_active_block_p50": float(p50),
        "reverse_queries_per_active_block_p90": float(p90),
        "reverse_queries_per_active_block_p99": float(p99),
        "reverse_queries_per_active_block_max": max_queries,
        "reverse_sparse_pair_indices_gib": _gib(sparse_pair_index_bytes),
        "reverse_sparse_logits_gib": _gib(sparse_logit_bytes),
        "reverse_dense_logits_gib": _gib(dense_logit_bytes),
        "reverse_sparse_chunk_max_lse_gib": _gib(sparse_chunk_stats_bytes),
        "reverse_dense_chunk_max_lse_gib": _gib(dense_chunk_stats_bytes),
        "reverse_sparse_partial_output_gib": _gib(sparse_partial_out_bytes),
        "reverse_dense_partial_output_gib": _gib(dense_partial_out_bytes),
    }


def _run_dsa_full_path(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
) -> torch.Tensor:
    return attn._forward_dsa_chunked_sequence(
        query_states=tensors.query_states,
        indexer_query_states=tensors.indexer_query_states,
        key_states=tensors.key_states,
        key_cache=tensors.key_cache,
        value_cache=tensors.value_cache,
        block_table=tensors.block_table,
        attn_metadata=None,
        positions=tensors.positions,
    )


def _run_dsa_apply_only(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        group_output = attn._forward_dsa_chunked_page_table_fa_prefill(
            query_states=group_query_states,
            key_cache=tensors.key_cache,
            value_cache=tensors.value_cache,
            block_table=tensors.block_table,
            attn_metadata=None,
            top_chunk_indices=item.top_chunk_indices,
            top_chunk_valid=item.top_chunk_valid,
            current_chunks=item.current_chunks,
            query_positions=item.query_positions,
            key_len=tensors.key_states.shape[0],
            group_idx=item.group_idx,
            softmax_scale=softmax_scale,
        )
        if group_output is None:
            raise RuntimeError(
                "current page-table apply path rejected generated metadata; "
                "try a longer key_len or smaller top_chunks"
            )
        output[item.query_start : item.query_end, head_start:head_end] = group_output
    return output


def _run_dsa_moonshot_gqa(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        output[item.query_start : item.query_end, head_start:head_end] = (
            dsa_prefill_gqa_attention(
                query_states=group_query_states,
                key_cache=tensors.key_cache,
                value_cache=tensors.value_cache,
                block_table=tensors.block_table,
                top_chunk_indices=item.top_chunk_indices,
                top_chunk_valid=item.top_chunk_valid,
                current_chunks=item.current_chunks,
                query_positions=item.query_positions,
                group_idx=item.group_idx,
                softmax_scale=softmax_scale,
            )
        )
    return output


def _run_dsa_moonshot_splitk(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    work: list[PrefillWorkItem],
    split_top_chunks: int,
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        output[item.query_start : item.query_end, head_start:head_end] = (
            dsa_prefill_gqa_splitk_attention(
                query_states=group_query_states,
                key_cache=tensors.key_cache,
                value_cache=tensors.value_cache,
                block_table=tensors.block_table,
                top_chunk_indices=item.top_chunk_indices,
                top_chunk_valid=item.top_chunk_valid,
                current_chunks=item.current_chunks,
                query_positions=item.query_positions,
                group_idx=item.group_idx,
                softmax_scale=softmax_scale,
                split_top_chunks=split_top_chunks,
            )
        )
    return output


def _run_dsa_moonshot_union(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    union_work: list[UnionWorkItem],
    chunks_per_iter: int,
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in union_work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        output[item.query_start : item.query_end, head_start:head_end] = (
            dsa_prefill_gqa_union_attention(
                query_states=group_query_states,
                key_cache=tensors.key_cache,
                value_cache=tensors.value_cache,
                block_table=tensors.block_table,
                union_chunks=item.union_chunks,
                union_masks=item.union_masks,
                union_counts=item.union_counts,
                row_starts=item.row_starts,
                row_counts=item.row_counts,
                current_chunks=item.current_chunks,
                tail_lens=item.tail_lens,
                group_idx=item.group_idx,
                softmax_scale=softmax_scale,
                chunks_per_iter=chunks_per_iter,
            )
        )
    return output


def _run_dsa_moonshot_union_qh(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    union_work: list[UnionWorkItem],
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in union_work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        output[item.query_start : item.query_end, head_start:head_end] = (
            dsa_prefill_gqa_union_qh_attention(
                query_states=group_query_states,
                key_cache=tensors.key_cache,
                value_cache=tensors.value_cache,
                block_table=tensors.block_table,
                union_chunks=item.union_chunks,
                union_masks=item.union_masks,
                union_counts=item.union_counts,
                row_starts=item.row_starts,
                row_counts=item.row_counts,
                current_chunks=item.current_chunks,
                tail_lens=item.tail_lens,
                group_idx=item.group_idx,
                softmax_scale=softmax_scale,
            )
        )
    return output


def _run_dsa_moonshot_wide_union(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    wide_union_work: list[WideUnionWorkItem],
    chunks_per_iter: int,
) -> torch.Tensor:
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    for item in wide_union_work:
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        output[item.query_start : item.query_end, head_start:head_end] = (
            dsa_prefill_gqa_wide_union_attention(
                query_states=group_query_states,
                key_cache=tensors.key_cache,
                value_cache=tensors.value_cache,
                block_table=tensors.block_table,
                union_chunks=item.union_chunks,
                full_masks=item.full_masks,
                current_masks=item.current_masks,
                union_counts=item.union_counts,
                row_starts=item.row_starts,
                row_counts=item.row_counts,
                tail_lens=item.tail_lens,
                group_idx=item.group_idx,
                softmax_scale=softmax_scale,
                chunks_per_iter=chunks_per_iter,
            )
        )
    return output


def _run_dsa_shared_run_page_table_fa(
    attn: NemotronHDSASelectiveAttention,
    tensors: BenchTensors,
    shared_work: list[SharedRunWorkItem],
) -> torch.Tensor:
    if nemotron_h.flash_attn_varlen_func is None:
        raise RuntimeError("flash_attn_varlen_func is unavailable")
    group_size = attn.num_heads // attn.num_kv_heads
    softmax_scale = 1.0 / math.sqrt(attn.head_dim)
    output = torch.empty_like(tensors.query_states)
    group_key_cache = tensors.key_cache[:, :, 0:1, :]
    group_value_cache = tensors.value_cache[:, :, 0:1, :]
    for item in shared_work:
        if item.group_idx != 0:
            group_key_cache = tensors.key_cache[
                :,
                :,
                item.group_idx : item.group_idx + 1,
                :,
            ]
            group_value_cache = tensors.value_cache[
                :,
                :,
                item.group_idx : item.group_idx + 1,
                :,
            ]
        head_start = item.group_idx * group_size
        head_end = head_start + group_size
        group_query_states = tensors.query_states[
            item.query_start : item.query_end,
            head_start:head_end,
        ]
        group_output = torch.empty_like(group_query_states)
        nemotron_h.flash_attn_varlen_func(
            q=group_query_states.contiguous(),
            k=group_key_cache,
            v=group_value_cache,
            out=group_output,
            cu_seqlens_q=item.cu_seqlens_q,
            max_seqlen_q=item.max_seqlen_q,
            seqused_k=item.seqused_k,
            max_seqlen_k=item.max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=item.run_block_table,
            fa_version=2,
        )
        output[item.query_start : item.query_end, head_start:head_end] = group_output
    return output


def _run_dense_flashattention(
    args: argparse.Namespace,
    tensors: BenchTensors,
) -> torch.Tensor:
    if nemotron_h.flash_attn_varlen_func is None:
        raise RuntimeError("flash_attn_varlen_func is unavailable")
    q_len = tensors.query_states.shape[0]
    key_len = tensors.key_states.shape[0]
    device = tensors.query_states.device
    cu_seqlens_q = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([key_len], device=device, dtype=torch.int32)
    output = torch.empty_like(tensors.query_states)
    nemotron_h.flash_attn_varlen_func(
        q=tensors.query_states.contiguous(),
        k=tensors.key_cache,
        v=tensors.value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=q_len,
        seqused_k=seqused_k,
        max_seqlen_k=key_len,
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(args.head_dim),
        causal=True,
        block_table=tensors.block_table.reshape(1, -1),
        fa_version=2,
    )
    return output


def _run_correctness(args: argparse.Namespace) -> dict[str, float]:
    check_args = argparse.Namespace(**vars(args))
    check_args.top_chunks = args.check_top_chunks
    check_args.query_chunk_size = args.check_query_chunk_size
    tensors = _make_tensors(
        check_args,
        key_len=args.check_key_len,
        q_len=args.check_q_len,
    )
    manual_attn = _make_attn(check_args, use_page_table=False)
    page_table_attn = _make_attn(check_args, use_page_table=True)
    _silence_dsa_debug()
    expected = _run_dsa_full_path(manual_attn, tensors)
    actual = _run_dsa_full_path(page_table_attn, tensors)
    work = _build_prefill_work(page_table_attn, tensors)
    moonshot = _run_dsa_moonshot_gqa(page_table_attn, tensors, work)
    splitk = _run_dsa_moonshot_splitk(
        page_table_attn,
        tensors,
        work,
        args.split_top_chunks,
    )
    old_union_rows = min(args.union_rows, 16)
    union_work = _build_union_work(
        page_table_attn,
        tensors,
        work,
        union_rows=old_union_rows,
    )
    union = _run_dsa_moonshot_union(
        page_table_attn,
        tensors,
        union_work,
        args.union_chunks_per_iter,
    )
    union_qh = _run_dsa_moonshot_union_qh(page_table_attn, tensors, union_work)
    wide_union_work = _build_wide_union_work_sort_gpu(
        page_table_attn,
        tensors,
        work,
        union_rows=args.union_rows,
    )
    wide_union = _run_dsa_moonshot_wide_union(
        page_table_attn,
        tensors,
        wide_union_work,
        args.union_chunks_per_iter,
    )
    torch.cuda.synchronize()
    diff = (actual.float() - expected.float()).abs()
    moonshot_diff = (moonshot.float() - expected.float()).abs()
    splitk_diff = (splitk.float() - expected.float()).abs()
    union_diff = (union.float() - expected.float()).abs()
    union_qh_diff = (union_qh.float() - expected.float()).abs()
    wide_union_diff = (wide_union.float() - expected.float()).abs()
    max_abs = float(diff.max().item())
    moonshot_max_abs = float(moonshot_diff.max().item())
    splitk_max_abs = float(splitk_diff.max().item())
    union_max_abs = float(union_diff.max().item())
    union_qh_max_abs = float(union_qh_diff.max().item())
    wide_union_max_abs = float(wide_union_diff.max().item())
    denom = expected.float().abs().clamp_min(1e-6)
    max_rel = float((diff / denom).max().item())
    moonshot_max_rel = float((moonshot_diff / denom).max().item())
    splitk_max_rel = float((splitk_diff / denom).max().item())
    union_max_rel = float((union_diff / denom).max().item())
    union_qh_max_rel = float((union_qh_diff / denom).max().item())
    wide_union_max_rel = float((wide_union_diff / denom).max().item())
    print(
        "correctness: "
        f"max_abs={max_abs:.6g} max_rel={max_rel:.6g} "
        f"moonshot_max_abs={moonshot_max_abs:.6g} "
        f"moonshot_max_rel={moonshot_max_rel:.6g} "
        f"splitk_max_abs={splitk_max_abs:.6g} "
        f"splitk_max_rel={splitk_max_rel:.6g} "
        f"union_max_abs={union_max_abs:.6g} "
        f"union_max_rel={union_max_rel:.6g} "
        f"union_qh_max_abs={union_qh_max_abs:.6g} "
        f"union_qh_max_rel={union_qh_max_rel:.6g} "
        f"wide_union_max_abs={wide_union_max_abs:.6g} "
        f"wide_union_max_rel={wide_union_max_rel:.6g} "
        f"atol={args.atol} rtol={args.rtol}",
        flush=True,
    )
    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(moonshot, expected, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(splitk, expected, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(union, expected, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(union_qh, expected, atol=args.atol, rtol=args.rtol)
    torch.testing.assert_close(wide_union, expected, atol=args.atol, rtol=args.rtol)
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "moonshot_max_abs": moonshot_max_abs,
        "moonshot_max_rel": moonshot_max_rel,
        "splitk_max_abs": splitk_max_abs,
        "splitk_max_rel": splitk_max_rel,
        "union_max_abs": union_max_abs,
        "union_max_rel": union_max_rel,
        "union_qh_max_abs": union_qh_max_abs,
        "union_qh_max_rel": union_qh_max_rel,
        "wide_union_max_abs": wide_union_max_abs,
        "wide_union_max_rel": wide_union_max_rel,
    }


def _run_bench(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    tensors = _make_tensors(args, key_len=args.key_len, q_len=args.q_len)
    page_table_attn = _make_attn(args, use_page_table=True)
    _silence_dsa_debug()
    work = _build_prefill_work(
        page_table_attn,
        tensors,
        top_chunk_pattern=args.top_chunk_pattern,
        shared_noise_chunks=args.shared_noise_chunks,
    )
    shared_work = (
        _build_shared_run_work(page_table_attn, tensors, work)
        if _kernel_enabled(args, "sharedfa")
        else []
    )
    union_builder = (
        _build_union_work_sort_gpu
        if args.union_build == "gpu-sort"
        else _build_union_work
    )
    needs_union = _kernel_enabled(args, "union") or _kernel_enabled(args, "unionqh")
    needs_wide_union = _kernel_enabled(args, "wideunion")
    old_union_rows = min(args.union_rows, 16)
    if args.time_union_build and args.union_build == "gpu-sort":
        _, results_build = _measure_cuda(
            "union_build_gpu_sort",
            lambda: union_builder(
                page_table_attn,
                tensors,
                work,
                union_rows=old_union_rows,
            )[0].union_chunks,
            warmup=args.warmup,
            iters=args.iters,
        )
    else:
        results_build = None
    union_work = (
        union_builder(
            page_table_attn,
            tensors,
            work,
            union_rows=old_union_rows,
        )
        if needs_union
        else []
    )
    wide_union_work = (
        _build_wide_union_work_sort_gpu(
            page_table_attn,
            tensors,
            work,
            union_rows=args.union_rows,
        )
        if needs_wide_union
        else []
    )
    work_stats = _summarize_work_items(work)
    if union_work:
        work_stats.update(_summarize_union_work(union_work))
    if wide_union_work:
        work_stats.update(_summarize_wide_union_work(wide_union_work))
    print(f"work_stats: {json.dumps(work_stats, sort_keys=True)}", flush=True)
    reverse_stats = None
    if args.reverse_stats:
        elem_bytes = torch.finfo(_dtype_from_name(args.dtype)).bits // 8
        reverse_stats = _summarize_reverse_mapping(
            work,
            chunk_size=args.chunk_size,
            group_size=page_table_attn.num_heads // page_table_attn.num_kv_heads,
            head_dim=args.head_dim,
            elem_bytes=elem_bytes,
        )
        print(
            f"reverse_mapping_stats: {json.dumps(reverse_stats, sort_keys=True)}",
            flush=True,
        )
    if any(not bool(item.top_chunk_valid.all().item()) for item in work):
        raise RuntimeError(
            "generated prefill work has invalid top chunks; increase key_len/q suffix "
            "or reduce top_chunks"
        )

    results: dict[str, dict[str, float]] = {}
    results["work_stats"] = work_stats
    if reverse_stats is not None:
        results["reverse_mapping_stats"] = reverse_stats
    if results_build is not None:
        results["union_build_gpu_sort"] = results_build
    if _kernel_enabled(args, "gqa"):
        _, results["dsa_moonshot_gqa_triton"] = _measure_cuda(
            "dsa_moonshot_gqa_triton",
            lambda: _run_dsa_moonshot_gqa(page_table_attn, tensors, work),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "splitk"):
        _, results["dsa_moonshot_splitk_triton"] = _measure_cuda(
            "dsa_moonshot_splitk_triton",
            lambda: _run_dsa_moonshot_splitk(
                page_table_attn,
                tensors,
                work,
                args.split_top_chunks,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "union"):
        _, results["dsa_moonshot_union_triton"] = _measure_cuda(
            "dsa_moonshot_union_triton",
            lambda: _run_dsa_moonshot_union(
                page_table_attn,
                tensors,
                union_work,
                args.union_chunks_per_iter,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "unionqh"):
        _, results["dsa_moonshot_union_qh_triton"] = _measure_cuda(
            "dsa_moonshot_union_qh_triton",
            lambda: _run_dsa_moonshot_union_qh(
                page_table_attn,
                tensors,
                union_work,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "wideunion"):
        _, results["dsa_moonshot_wide_union_triton"] = _measure_cuda(
            "dsa_moonshot_wide_union_triton",
            lambda: _run_dsa_moonshot_wide_union(
                page_table_attn,
                tensors,
                wide_union_work,
                args.union_chunks_per_iter,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "sharedfa"):
        _, results["dsa_shared_run_page_table_fa"] = _measure_cuda(
            "dsa_shared_run_page_table_fa",
            lambda: _run_dsa_shared_run_page_table_fa(
                page_table_attn,
                tensors,
                shared_work,
            ),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "apply"):
        _, results["dsa_apply_only_page_table"] = _measure_cuda(
            "dsa_apply_only_page_table",
            lambda: _run_dsa_apply_only(page_table_attn, tensors, work),
            warmup=args.warmup,
            iters=args.iters,
        )
    if _kernel_enabled(args, "full"):
        _, results["dsa_full_current"] = _measure_cuda(
            "dsa_full_current",
            lambda: _run_dsa_full_path(page_table_attn, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if not args.skip_dense_fa and _kernel_enabled(args, "dense"):
        _, results["dense_full_flashattention"] = _measure_cuda(
            "dense_full_flashattention",
            lambda: _run_dense_flashattention(args, tensors),
            warmup=args.warmup,
            iters=args.iters,
        )
    if "dense_full_flashattention" in results:
        dense = results["dense_full_flashattention"]["min_ms"]
        speedups = {}
        for name, stats in results.items():
            if name in {
                "dense_full_flashattention",
                "work_stats",
                "reverse_mapping_stats",
            }:
                continue
            speedups[name] = dense / stats["min_ms"]
        results["speedup_vs_dense_min"] = speedups
        print(f"speedup_vs_dense_min: {json.dumps(speedups, sort_keys=True)}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-indexer-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-chunks", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=512)
    parser.add_argument("--split-top-chunks", type=int, default=64)
    parser.add_argument("--union-rows", type=int, default=8)
    parser.add_argument("--union-chunks-per-iter", type=int, default=4)
    parser.add_argument("--union-build", choices=("cpu", "gpu-sort"), default="cpu")
    parser.add_argument("--time-union-build", action="store_true")
    parser.add_argument("--reverse-stats", action="store_true")
    parser.add_argument("--shared-noise-chunks", type=int, default=32)
    parser.add_argument(
        "--kernels",
        default="all",
        help=(
            "Comma-separated kernels to time. Use names: dense,gqa,splitk,union,"
            "unionqh,wideunion,apply,full,sharedfa or all."
        ),
    )
    parser.add_argument(
        "--top-chunk-pattern",
        choices=("indexer", "shared-current-chunk", "shared-noise-current-chunk"),
        default="indexer",
        help=(
            "Controls precomputed top chunks for benchmark-only apply kernels. "
            "The correctness check always uses indexer semantics."
        ),
    )
    parser.add_argument("--key-len", type=int, default=131072)
    parser.add_argument("--q-len", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--skip-dense-fa", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--check-key-len", type=int, default=4096)
    parser.add_argument("--check-q-len", type=int, default=128)
    parser.add_argument("--check-top-chunks", type=int, default=32)
    parser.add_argument("--check-query-chunk-size", type=int, default=64)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if not args.correctness and not args.bench:
        args.correctness = True
        args.bench = True
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    return args


def main() -> None:
    args = parse_args()
    if nemotron_h.flash_attn_varlen_func is None:
        raise RuntimeError("vLLM FlashAttention is not importable in this environment")

    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "results": {},
    }
    if args.correctness:
        payload["results"]["correctness"] = _run_correctness(args)
    if args.bench:
        payload["results"]["bench"] = _run_bench(args)

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
