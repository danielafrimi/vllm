# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import nemotron_h
from vllm.model_executor.models import nemotron_h_dsa_triton_summaries
from vllm.model_executor.models import nemotron_h_dsa_triton_scoring
from vllm.model_executor.models.nemotron_h_dsa_attention_legacy import (
    NemotronHDSALegacyAttention,
)

if (
    nemotron_h_dsa_triton_scoring.triton is not None
    and nemotron_h_dsa_triton_scoring.tl is not None
):
    triton = nemotron_h_dsa_triton_scoring.triton
    tl = nemotron_h_dsa_triton_scoring.tl

    @triton.jit
    def _dsa_chunk_score_kernel_constexpr_reference(
        query,
        chunk_reps,
        current_chunks,
        logits,
        stride_q_r: tl.constexpr,
        stride_q_d: tl.constexpr,
        stride_rep_c: tl.constexpr,
        stride_rep_d: tl.constexpr,
        stride_logits_r: tl.constexpr,
        stride_logits_c: tl.constexpr,
        q_indexer_dim: tl.constexpr,
        max_prior_chunks: tl.constexpr,
        score_scale: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk_block = tl.program_id(1)
        chunk_offsets = chunk_block * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < q_indexer_dim

        q_vals = tl.load(
            query + row * stride_q_r + dim_offsets * stride_q_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        reps = tl.load(
            chunk_reps
            + chunk_offsets[:, None] * stride_rep_c
            + dim_offsets[None, :] * stride_rep_d,
            mask=(chunk_offsets[:, None] < max_prior_chunks) & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(reps * q_vals[None, :], axis=1) * score_scale
        current_chunk = tl.load(current_chunks + row).to(tl.int64)
        valid_chunk = (chunk_offsets < max_prior_chunks) & (
            chunk_offsets < current_chunk
        )
        scores = tl.where(valid_chunk, scores, -float("inf"))
        tl.store(
            logits + row * stride_logits_r + chunk_offsets * stride_logits_c,
            scores,
            mask=chunk_offsets < max_prior_chunks,
        )
else:
    _dsa_chunk_score_kernel_constexpr_reference = None


def _make_chunked_dsa_attn() -> NemotronHDSALegacyAttention:
    attn = NemotronHDSALegacyAttention.__new__(NemotronHDSALegacyAttention)
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
    attn.q_indexer_use_flattened_prefill_page_table_fa = False
    attn.q_indexer_use_flattened_decode_page_table_fa = False
    attn.q_indexer_use_full_attention_short_seq = False
    attn.q_indexer_use_triton_batched_summaries = False
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


def test_dsa_chunk_representatives_gather_from_page_blocks():
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


def test_dsa_triton_batched_summaries_returns_none_for_cpu_inputs():
    result = nemotron_h_dsa_triton_summaries.dsa_block_summaries_triton(
        key_cache=torch.zeros(4, 4, 1, 2),
        block_table=torch.zeros(2, 3, dtype=torch.long),
        seq_lens=torch.tensor([5, 9], dtype=torch.long),
        q_indexer_dim=2,
    )

    assert result is None


def test_dsa_triton_batched_summaries_match_current_representatives():
    if not torch.cuda.is_available() or not nemotron_h_dsa_triton_summaries.HAS_TRITON:
        return

    torch.manual_seed(35)
    dtypes = [torch.float32, torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        attn = _make_chunked_dsa_attn()
        attn.num_kv_heads = 2
        attn.head_dim = 5
        attn.q_indexer_dim = 3
        block_size = attn.q_indexer_chunk_size
        seq_lens = torch.tensor([1, 4, 5, 8, 10], device="cuda")
        global_max_chunks = 6
        block_table = torch.arange(
            seq_lens.numel() * global_max_chunks,
            device="cuda",
            dtype=torch.long,
        ).view(seq_lens.numel(), global_max_chunks)
        packed_key_cache = torch.randn(
            int(block_table.max().item()) + 1,
            block_size,
            attn.num_kv_heads,
            attn.head_dim,
            device="cuda",
            dtype=dtype,
        )
        padded_key_storage = torch.randn(
            int(block_table.max().item()) + 1,
            block_size * 2,
            attn.num_kv_heads,
            attn.head_dim,
            device="cuda",
            dtype=dtype,
        )
        padded_key_cache = padded_key_storage[:, :block_size, :, :]

        for key_cache in (packed_key_cache, padded_key_cache):
            actual = nemotron_h_dsa_triton_summaries.dsa_block_summaries_triton(
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                q_indexer_dim=attn.q_indexer_dim,
            )

            assert actual is not None
            assert actual.dtype == torch.bfloat16
            assert actual.shape == (
                seq_lens.numel(),
                global_max_chunks,
                attn.num_kv_heads,
                attn.q_indexer_dim,
            )
            for seq_idx, key_len in enumerate(seq_lens.tolist()):
                expected = attn._get_indexer_chunk_representatives(
                    key_states=None,
                    key_cache=key_cache,
                    block_table=block_table[seq_idx],
                    key_len=int(key_len),
                )
                num_chunks = math.ceil(int(key_len) / block_size)
                torch.testing.assert_close(
                    actual[seq_idx, :num_chunks],
                    expected.to(torch.bfloat16),
                    atol=1e-2,
                    rtol=1e-2,
                )

            capped_chunks = max(
                math.ceil(int(key_len) / block_size)
                for key_len in seq_lens.tolist()
            )
            capped_actual = (
                nemotron_h_dsa_triton_summaries.dsa_block_summaries_triton(
                    key_cache=key_cache,
                    block_table=block_table,
                    seq_lens=seq_lens,
                    q_indexer_dim=attn.q_indexer_dim,
                    max_chunks=capped_chunks,
                )
            )
            assert capped_actual is not None
            assert capped_actual.shape == (
                seq_lens.numel(),
                capped_chunks,
                attn.num_kv_heads,
                attn.q_indexer_dim,
            )
            torch.testing.assert_close(
                capped_actual,
                actual[:, :capped_chunks],
                atol=1e-5,
                rtol=1e-5,
            )


def test_dsa_batched_summary_helper_uses_live_table_width(monkeypatch):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_use_triton_batched_summaries = True
    key_cache = torch.zeros(16, attn.q_indexer_chunk_size, 1, 2)
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
    ):
        calls.append((tuple(block_table.shape), seq_lens.tolist(), q_indexer_dim))
        return expected_output.to(device=block_table.device)

    monkeypatch.setattr(
        nemotron_h,
        "dsa_block_summaries_triton",
        fake_dsa_block_summaries_triton,
    )

    representatives = attn._get_triton_batched_chunk_representatives(
        key_cache=key_cache,
        block_table=block_table,
        active_seq_infos=[(0, 0, 1, 5), (1, 1, 2, 9)],
        cache_info=("NHD", attn.q_indexer_chunk_size),
    )

    assert calls == [((2, 3), [5, 9], attn.q_indexer_dim)]
    assert representatives is not None
    torch.testing.assert_close(representatives[0], expected_output[0, :2])
    torch.testing.assert_close(representatives[1], expected_output[1, :3])


def test_dsa_chunked_sequence_uses_precomputed_chunk_representatives(monkeypatch):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_chunk_top_k = 2
    attn.q_indexer_chunked_query_chunk_size = 8
    torch.manual_seed(36)

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

    reference = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=key_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
    )
    chunk_representatives = attn._build_indexer_chunk_representatives(
        key_states[..., : attn.q_indexer_dim]
    )

    def unexpected_get_indexer_chunk_representatives(**kwargs):
        raise AssertionError("precomputed chunk representatives should be used")

    monkeypatch.setattr(
        attn,
        "_get_indexer_chunk_representatives",
        unexpected_get_indexer_chunk_representatives,
    )

    output = attn._forward_dsa_chunked_sequence(
        query_states=query_states,
        indexer_query_states=indexer_query_states,
        key_states=None,
        key_len=key_len,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        positions=positions,
        chunk_representatives=chunk_representatives,
    )

    torch.testing.assert_close(output, reference)


def test_dsa_active_sequence_infos_prefers_cpu_metadata():
    class PoisonTensor:
        def numel(self):
            raise AssertionError("device metadata should not be touched")

        def __getitem__(self, index):
            raise AssertionError("device metadata should not be touched")

    attn = _make_chunked_dsa_attn()
    attn_metadata = SimpleNamespace(
        num_actual_tokens=5,
        query_start_loc=PoisonTensor(),
        seq_lens=PoisonTensor(),
        query_start_loc_cpu=torch.tensor([0, 2, 5, 5], dtype=torch.int32),
        seq_lens_cpu=torch.tensor([10, 11, 0], dtype=torch.int32),
    )

    infos = attn._dsa_active_sequence_infos(attn_metadata)

    assert infos == [(0, 0, 2, 10), (1, 2, 5, 11)]


def test_dsa_triton_scoring_returns_none_for_cpu_inputs():
    result = nemotron_h_dsa_triton_scoring.dsa_score_topk_triton(
        score_query_states=torch.zeros(2, 4),
        chunk_representatives=torch.zeros(8, 4),
        current_chunks=torch.tensor([0, 4], dtype=torch.long),
        chunk_top_k=2,
        logit_scale=1.0,
        q_indexer_dim=4,
    )

    assert result is None


def test_dsa_torch_scoring_matches_torch_topk_on_cpu():
    torch.manual_seed(32)
    q_indexer_dim = 4
    chunk_top_k = 3
    score_query_states = torch.randn(3, q_indexer_dim, dtype=torch.bfloat16)
    chunk_representatives = torch.randn(5, q_indexer_dim, dtype=torch.float32)
    current_chunks = torch.tensor([0, 2, 7], dtype=torch.long)
    chunk_ids = torch.arange(5, dtype=torch.int32)

    result = nemotron_h_dsa_triton_scoring.dsa_score_topk_torch(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        current_chunks=current_chunks,
        chunk_top_k=chunk_top_k,
        logit_scale=1.25,
        q_indexer_dim=q_indexer_dim,
        chunk_ids=chunk_ids,
        return_logits=True,
    )

    assert result is not None
    top_chunk_indices, top_chunk_valid, chunk_logits = result
    assert chunk_logits is not None
    selectable_counts = current_chunks.clamp(max=chunk_representatives.shape[0])
    reference_logits = (
        score_query_states.float() @ chunk_representatives.transpose(0, 1)
    ) * (1.25 / math.sqrt(q_indexer_dim))
    valid = chunk_ids[None, :].to(dtype=torch.long) < selectable_counts[:, None]
    reference_logits = reference_logits.masked_fill(
        ~valid,
        torch.finfo(reference_logits.dtype).min,
    )
    score_logits = nemotron_h_dsa_triton_scoring.dsa_score_logits_torch(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        current_chunks=current_chunks,
        logit_scale=1.25,
        q_indexer_dim=q_indexer_dim,
        chunk_ids=chunk_ids,
    )
    assert score_logits is not None
    torch.testing.assert_close(score_logits, reference_logits)
    torch.testing.assert_close(chunk_logits, reference_logits)
    assert top_chunk_indices[~top_chunk_valid].eq(0).all()
    for row, selectable_count in enumerate(selectable_counts.tolist()):
        valid_count = min(selectable_count, chunk_top_k)
        assert top_chunk_valid[row, :valid_count].all()
        assert not top_chunk_valid[row, valid_count:].any()
        if valid_count == 0:
            continue
        expected = set(
            reference_logits[row, :selectable_count]
            .topk(k=valid_count)
            .indices.tolist()
        )
        actual = set(top_chunk_indices[row, :valid_count].tolist())
        assert actual == expected


def test_dsa_torch_scoring_handles_varying_shapes_and_chunk_ids():
    torch.manual_seed(35)
    q_indexer_dim = 5
    cases = [
        (2, 3, torch.tensor([1, 3], dtype=torch.long), 2),
        (4, 7, torch.tensor([0, 2, 5, 9], dtype=torch.long), 2),
        (3, 1, torch.tensor([-1, 0, 1], dtype=torch.long), 4),
    ]
    for num_rows, max_prior_chunks, current_chunks, chunk_top_k in cases:
        score_query_states = torch.randn(num_rows, q_indexer_dim)
        chunk_representatives = torch.randn(max_prior_chunks, q_indexer_dim)
        chunk_ids = torch.arange(max_prior_chunks, dtype=torch.int32)

        result = nemotron_h_dsa_triton_scoring.dsa_score_topk_torch(
            score_query_states=score_query_states,
            chunk_representatives=chunk_representatives,
            current_chunks=current_chunks,
            chunk_top_k=chunk_top_k,
            logit_scale=1.0,
            q_indexer_dim=q_indexer_dim,
            chunk_ids=chunk_ids,
            return_logits=True,
        )

        assert result is not None
        top_chunk_indices, top_chunk_valid, chunk_logits = result
        assert chunk_logits is not None
        expected_width = min(chunk_top_k, max_prior_chunks)
        assert top_chunk_indices.shape == (num_rows, expected_width)
        assert top_chunk_valid.shape == (num_rows, expected_width)
        selectable_counts = current_chunks.clamp(
            min=0, max=max_prior_chunks)
        valid = (
            chunk_ids[None, :].to(dtype=torch.long)
            < selectable_counts[:, None]
        )
        reference_logits = (
            score_query_states.float() @ chunk_representatives.transpose(0, 1)
        ) * (1.0 / math.sqrt(q_indexer_dim))
        reference_logits = reference_logits.masked_fill(
            ~valid,
            torch.finfo(reference_logits.dtype).min,
        )
        torch.testing.assert_close(chunk_logits, reference_logits)
        assert top_chunk_indices[~top_chunk_valid].eq(0).all()


def test_dsa_triton_scoring_runtime_args_are_not_constexpr():
    if (
        nemotron_h_dsa_triton_scoring.tl is None
        or not hasattr(nemotron_h_dsa_triton_scoring, "_dsa_chunk_score_kernel")
    ):
        return

    kernel = nemotron_h_dsa_triton_scoring._dsa_chunk_score_kernel
    fn = getattr(kernel, "fn", kernel)
    annotations = getattr(fn, "__annotations__", {})
    for name in {
        "stride_q_r",
        "stride_q_d",
        "stride_rep_c",
        "stride_rep_d",
        "stride_logits_r",
        "stride_logits_c",
        "q_indexer_dim",
        "max_prior_chunks",
        "score_scale",
    }:
        assert name not in annotations
    constexpr_annotations = {
        nemotron_h_dsa_triton_scoring.tl.constexpr,
        "tl.constexpr",
    }
    assert annotations["BLOCK_CHUNKS"] in constexpr_annotations
    assert annotations["BLOCK_D"] in constexpr_annotations


def _triton_kernel_cache_size(kernel) -> int | None:
    if not hasattr(kernel, "device_caches"):
        return None
    try:
        from triton.runtime.driver import driver
    except ImportError:
        return None

    device = driver.active.get_current_device()
    cache_entry = kernel.device_caches.get(device)
    if cache_entry is None:
        return 0
    kernel_cache = cache_entry[0]
    return len(kernel_cache)


def _clear_triton_kernel_cache(kernel) -> None:
    if hasattr(kernel, "device_caches"):
        kernel.device_caches.clear()


def _run_dsa_chunk_score_kernel(
    kernel,
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    score_scale: float,
    block_chunks: int,
) -> torch.Tensor:
    q_indexer_dim = int(score_query_states.shape[1])
    max_prior_chunks = int(chunk_representatives.shape[0])
    logits = torch.empty(
        score_query_states.shape[0],
        max_prior_chunks,
        device=score_query_states.device,
        dtype=torch.float32,
    )
    block_d = nemotron_h_dsa_triton_scoring.triton.next_power_of_2(q_indexer_dim)
    kernel[(score_query_states.shape[0],
            nemotron_h_dsa_triton_scoring.triton.cdiv(
                max_prior_chunks, block_chunks))](
                    score_query_states,
                    chunk_representatives,
                    current_chunks,
                    logits,
                    score_query_states.stride(0),
                    score_query_states.stride(1),
                    chunk_representatives.stride(0),
                    chunk_representatives.stride(1),
                    logits.stride(0),
                    logits.stride(1),
                    q_indexer_dim,
                    max_prior_chunks,
                    score_scale,
                    BLOCK_CHUNKS=block_chunks,
                    BLOCK_D=block_d,
                    num_warps=4,
                    num_stages=2,
                )
    return logits


def test_dsa_triton_score_kernel_runtime_args_do_not_recompile():
    new_kernel = getattr(
        nemotron_h_dsa_triton_scoring,
        "_dsa_chunk_score_kernel",
        None,
    )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton JIT cache validation")
    if nemotron_h_dsa_triton_scoring.triton is None:
        pytest.skip("Triton is unavailable")
    if _dsa_chunk_score_kernel_constexpr_reference is None or new_kernel is None:
        pytest.skip("DSA scoring kernels are unavailable")
    if not hasattr(new_kernel, "device_caches") or not hasattr(
            _dsa_chunk_score_kernel_constexpr_reference, "device_caches"):
        pytest.skip("Triton JIT cache introspection is unavailable")

    torch.manual_seed(36)
    q_indexer_dim = 8
    block_chunks = 8
    cases = [
        (2, 5, 1.0),
        (4, 11, 0.75),
        (1, 7, 1.25),
        (3, 13, 0.5),
        (5, 3, 1.5),
    ]
    _clear_triton_kernel_cache(new_kernel)
    _clear_triton_kernel_cache(_dsa_chunk_score_kernel_constexpr_reference)

    new_cache_sizes: list[int] = []
    for num_rows, max_prior_chunks, score_scale in cases:
        score_query_states = torch.randn(
            num_rows,
            q_indexer_dim,
            device="cuda",
            dtype=torch.bfloat16,
        ).contiguous()
        chunk_representatives = torch.randn(
            max_prior_chunks,
            q_indexer_dim,
            device="cuda",
            dtype=torch.float32,
        ).contiguous()
        current_chunks = torch.randint(
            0,
            max_prior_chunks + 3,
            (num_rows, ),
            device="cuda",
            dtype=torch.int32,
        )

        new_logits = _run_dsa_chunk_score_kernel(
            new_kernel,
            score_query_states=score_query_states,
            chunk_representatives=chunk_representatives,
            current_chunks=current_chunks,
            score_scale=score_scale,
            block_chunks=block_chunks,
        )
        old_logits = _run_dsa_chunk_score_kernel(
            _dsa_chunk_score_kernel_constexpr_reference,
            score_query_states=score_query_states,
            chunk_representatives=chunk_representatives,
            current_chunks=current_chunks,
            score_scale=score_scale,
            block_chunks=block_chunks,
        )
        torch.cuda.synchronize()

        selectable_counts = current_chunks.clamp(max=max_prior_chunks).to(
            dtype=torch.long)
        valid = (
            torch.arange(max_prior_chunks, device="cuda")[None, :]
            < selectable_counts[:, None]
        )
        torch.testing.assert_close(new_logits[valid], old_logits[valid])
        assert torch.isneginf(old_logits[~valid]).all()
        assert new_logits[~valid].eq(torch.finfo(new_logits.dtype).min).all()

        cache_size = _triton_kernel_cache_size(new_kernel)
        assert cache_size is not None
        new_cache_sizes.append(cache_size)

    assert new_cache_sizes[0] == 1
    assert new_cache_sizes == [new_cache_sizes[0]] * len(new_cache_sizes)

    old_cache_size = _triton_kernel_cache_size(
        _dsa_chunk_score_kernel_constexpr_reference)
    assert old_cache_size is not None
    assert old_cache_size > new_cache_sizes[0]


def test_dsa_triton_scoring_matches_torch_topk_on_cuda():
    if (
        not torch.cuda.is_available()
        or nemotron_h_dsa_triton_scoring.triton is None
        or not nemotron_h_dsa_triton_scoring._has_top_k_per_row_prefill()
    ):
        return

    torch.manual_seed(33)
    q_indexer_dim = 8
    chunk_top_k = 3
    score_query_states = torch.randn(
        4,
        q_indexer_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    chunk_representatives = torch.randn(
        6,
        q_indexer_dim,
        device="cuda",
        dtype=torch.float32,
    )
    current_chunks = torch.tensor([0, 1, 4, 6], device="cuda", dtype=torch.long)

    result = nemotron_h_dsa_triton_scoring.dsa_score_topk_triton(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        current_chunks=current_chunks,
        chunk_top_k=chunk_top_k,
        logit_scale=1.5,
        q_indexer_dim=q_indexer_dim,
        return_logits=True,
    )

    assert result is not None
    top_chunk_indices, top_chunk_valid, chunk_logits = result
    assert chunk_logits is not None
    reference_logits = (
        score_query_states.float() @ chunk_representatives.transpose(0, 1)
    ) * (1.5 / math.sqrt(q_indexer_dim))
    valid = torch.arange(6, device="cuda")[None, :] < current_chunks[:, None]
    reference_logits = reference_logits.masked_fill(
        ~valid,
        torch.finfo(reference_logits.dtype).min,
    )
    torch.testing.assert_close(chunk_logits, reference_logits)

    for row, current_chunk in enumerate(current_chunks.tolist()):
        valid_count = min(current_chunk, chunk_top_k)
        assert top_chunk_valid[row, :valid_count].all()
        assert not top_chunk_valid[row, valid_count:].any()
        if valid_count == 0:
            continue
        expected = set(
            reference_logits[row, :current_chunk]
            .topk(k=valid_count)
            .indices.cpu()
            .tolist()
        )
        actual = set(top_chunk_indices[row, :valid_count].cpu().tolist())
        assert actual == expected


def test_dsa_chunked_sequence_uses_torch_scoring_helper(monkeypatch):
    attn = _make_chunked_dsa_attn()
    attn.q_indexer_chunk_top_k = 2
    attn.q_indexer_chunked_query_chunk_size = 8
    torch.manual_seed(34)

    key_len = 12
    block_size = attn.q_indexer_chunk_size
    key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
    query_states = torch.randn(8, attn.num_heads, attn.head_dim)
    indexer_query_states = torch.randn(8, attn.num_kv_heads, attn.q_indexer_dim)
    positions = torch.arange(4, 12)
    block_table = torch.arange(math.ceil(key_len / block_size), dtype=torch.int32)
    key_cache = _pack_nhd_cache(key_states, block_size, block_table)
    value_cache = _pack_nhd_cache(value_states, block_size, block_table)

    reference = attn._forward_dsa_chunked_sequence(
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

    def fake_dsa_score_topk_torch(
        *,
        score_query_states,
        chunk_representatives,
        current_chunks,
        chunk_top_k,
        logit_scale,
        q_indexer_dim,
        chunk_ids=None,
        return_logits=False,
    ):
        calls.append(chunk_ids is not None)
        logits = (
            score_query_states.float()
            @ chunk_representatives.transpose(0, 1)
        ) * (logit_scale / math.sqrt(q_indexer_dim))
        if chunk_ids is None:
            chunk_ids = torch.arange(
                chunk_representatives.shape[0],
                device=logits.device,
                dtype=current_chunks.dtype,
            )
        selectable_counts = current_chunks.clamp(
            min=0,
            max=chunk_representatives.shape[0],
        )
        valid = chunk_ids[None, :] < selectable_counts[:, None]
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        top_k = min(chunk_top_k, chunk_representatives.shape[0])
        top_indices = logits.topk(k=top_k, dim=-1).indices
        top_valid = valid.gather(dim=-1, index=top_indices)
        top_indices = top_indices.masked_fill(~top_valid, 0)
        return top_indices, top_valid, logits if return_logits else None

    monkeypatch.setattr(
        nemotron_h,
        "dsa_score_topk_torch",
        fake_dsa_score_topk_torch,
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

    torch.testing.assert_close(output, reference)
    assert calls
    assert all(calls)


def test_dsa_chunked_recall_matches_causal_reference():
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


def test_dsa_chunked_flattened_page_table_fa_single_kv_head_uses_cache_view(
    monkeypatch,
):
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 1
    attn.num_heads = 2

    block_size = attn.q_indexer_chunk_size
    query_states = torch.randn(1, attn.num_heads, attn.head_dim)
    key_cache = torch.randn(2, block_size, attn.num_kv_heads, attn.head_dim)
    value_cache = torch.randn_like(key_cache)
    block_table = torch.tensor([1, 0], dtype=torch.int32)
    top_chunk_indices = torch.zeros(1, attn.num_kv_heads, 1, dtype=torch.long)
    top_chunk_valid = torch.ones_like(top_chunk_indices, dtype=torch.bool)
    current_chunks = torch.ones(1, dtype=torch.long)
    query_positions = torch.tensor([block_size * 2 - 1], dtype=torch.long)
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
        assert k is key_cache
        assert v is value_cache
        assert cu_seqlens_q.tolist() == [0, 1]
        assert block_table.tolist() == [[1, 0]]
        assert seqused_k.tolist() == [block_size * 2]
        out.zero_()
        calls.append(q.shape[0])
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )

    output = attn._forward_dsa_chunked_flattened_decode_page_table_fa_sequence(
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        attn_metadata=None,
        top_chunk_indices=top_chunk_indices,
        top_chunk_valid=top_chunk_valid,
        current_chunks=current_chunks,
        query_positions=query_positions,
        key_len=block_size * 2,
        softmax_scale=1.0 / math.sqrt(attn.head_dim),
    )

    assert output is not None
    assert output.shape == query_states.shape
    assert calls == [1]


def test_dsa_single_kv_unified_page_table_fa_mixes_dense_and_sparse(
    monkeypatch,
):
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 1
    attn.num_heads = 2
    attn.q_indexer_chunk_top_k = 1
    attn.q_indexer_use_page_table_fa = True
    attn.q_indexer_use_prefill_page_table_fa = True
    attn.q_indexer_use_flattened_prefill_page_table_fa = True
    attn.q_indexer_use_full_attention_short_seq = True
    attn.total_num_kv_heads = 1
    object.__setattr__(
        attn,
        "_local_kv_head_indices",
        torch.tensor([0], dtype=torch.long),
    )

    block_size = attn.q_indexer_chunk_size
    query_states = torch.randn(6, attn.num_heads, attn.head_dim)
    hidden_states = torch.randn(6, 4)
    key_cache = torch.randn(4, block_size, attn.num_kv_heads, attn.head_dim)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor(
        [
            [0, 0, 0],
            [1, 2, 3],
        ],
        dtype=torch.int32,
    )
    positions = torch.tensor([0, 1, 2, 7, 8, 9], dtype=torch.long)
    indexer_query_states = torch.zeros(6, attn.num_kv_heads, attn.q_indexer_dim)
    indexer_query_states[:, 0, 0] = 1.0

    def fake_indexer_q_proj(x):
        assert x.shape[0] == 6
        return indexer_query_states.view(6, -1), None

    object.__setattr__(attn, "indexer_q_proj", fake_indexer_q_proj)
    representatives = torch.zeros(3, attn.num_kv_heads, attn.q_indexer_dim)
    representatives[0, 0, 0] = 1.0
    representatives[1, 0, 0] = 2.0
    output = torch.empty_like(query_states)
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
        assert q.shape == query_states.shape
        assert k is key_cache
        assert v is value_cache
        assert out.data_ptr() == output.data_ptr()
        assert cu_seqlens_q.tolist() == [0, 3, 4, 5, 6]
        assert max_seqlen_q == 3
        assert seqused_k.tolist() == [3, 8, 5, 6]
        assert max_seqlen_k == 8
        assert dropout_p == 0.0
        assert causal is True
        assert "fa_version" in kwargs
        assert block_table.tolist() == [
            [0, 0],
            [1, 2],
            [2, 3],
            [2, 3],
        ]
        out.copy_(q + 123.0)
        calls.append(True)
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )

    handled = attn._forward_dsa_chunked_unified_page_table_fa_bucket(
        hidden_states=hidden_states,
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        attn_metadata=None,
        positions=positions,
        active_seq_infos=[(0, 0, 3, 3), (1, 3, 6, 10)],
        batched_chunk_representatives={1: representatives},
        output=output,
    )

    assert handled == {0, 1}
    assert calls == [True]
    torch.testing.assert_close(output, query_states + 123.0)


def test_dsa_single_kv_unified_page_table_fa_pads_sparse_plan(
    monkeypatch,
):
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 1
    attn.num_heads = 1
    attn.q_indexer_chunk_top_k = 3
    attn.q_indexer_use_page_table_fa = True
    attn.q_indexer_use_prefill_page_table_fa = True
    attn.q_indexer_use_flattened_prefill_page_table_fa = True
    attn.total_num_kv_heads = 1
    object.__setattr__(
        attn,
        "_local_kv_head_indices",
        torch.tensor([0], dtype=torch.long),
    )

    block_size = attn.q_indexer_chunk_size
    query_len = 16
    query_states = torch.randn(query_len, attn.num_heads, attn.head_dim)
    hidden_states = torch.randn(query_len, 4)
    key_cache = torch.randn(15, block_size, attn.num_kv_heads, attn.head_dim)
    value_cache = torch.randn_like(key_cache)
    block_tables = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.int32)
    positions = torch.arange(4, 20, dtype=torch.long)
    indexer_query_states = torch.zeros(
        query_len,
        attn.num_kv_heads,
        attn.q_indexer_dim,
    )
    indexer_query_states[:, 0, 0] = 1.0

    def fake_indexer_q_proj(x):
        assert x.shape[0] == query_len
        return indexer_query_states.view(query_len, -1), None

    object.__setattr__(attn, "indexer_q_proj", fake_indexer_q_proj)
    representatives = torch.zeros(5, attn.num_kv_heads, attn.q_indexer_dim)
    representatives[:, 0, 0] = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    output = torch.empty_like(query_states)

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
        assert cu_seqlens_q.tolist() == list(range(query_len + 1))
        assert max_seqlen_q == 1
        assert seqused_k.tolist() == [
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            13,
            14,
            15,
            16,
        ]
        assert max_seqlen_k == 16
        assert causal is True
        block_rows = block_table.tolist()
        assert block_rows[0] == [10, 11, 0, 0]
        assert block_rows[4] == [10, 11, 12, 0]
        assert block_rows[8] == [10, 11, 12, 13]
        assert block_rows[-1] == [10, 11, 12, 14]
        out.copy_(q)
        return out

    monkeypatch.setattr(
        nemotron_h,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
    )

    handled = attn._forward_dsa_chunked_unified_page_table_fa_bucket(
        hidden_states=hidden_states,
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        attn_metadata=None,
        positions=positions,
        active_seq_infos=[(0, 0, query_len, 20)],
        batched_chunk_representatives={0: representatives},
        output=output,
    )

    assert handled == {0}
    torch.testing.assert_close(output, query_states)


def test_dsa_multi_kv_unified_dispatch_reuses_one_kv_bucket():
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    object.__setattr__(
        attn,
        "_local_kv_head_indices",
        torch.arange(attn.num_kv_heads, dtype=torch.long),
    )

    block_size = attn.q_indexer_chunk_size
    query_states = torch.randn(3, attn.num_heads, attn.head_dim)
    output = torch.zeros_like(query_states)
    key_cache = torch.randn(5, block_size, attn.num_kv_heads, attn.head_dim)
    value_cache = torch.randn_like(key_cache)
    representatives = {
        0: torch.randn(2, attn.num_kv_heads, attn.q_indexer_dim),
    }
    calls = []

    def fake_one_kv_bucket(**kwargs):
        local_indices = kwargs["local_kv_head_indices"]
        kv_head_idx = int(local_indices[0].item())
        calls.append(
            (
                tuple(kwargs["query_states"].shape),
                tuple(kwargs["key_cache"].shape),
                tuple(local_indices.tolist()),
                tuple(kwargs["batched_chunk_representatives"][0].shape),
            )
        )
        kwargs["output"].copy_(
            kwargs["query_states"] + float((kv_head_idx + 1) * 100)
        )
        return {0, 1}

    object.__setattr__(
        attn,
        "_forward_dsa_chunked_one_kv_head_page_table_fa_bucket",
        fake_one_kv_bucket,
    )

    handled = attn._forward_dsa_chunked_unified_page_table_fa_bucket(
        hidden_states=torch.zeros(3, 4),
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        attn_metadata=None,
        positions=torch.arange(3),
        active_seq_infos=[(0, 0, 2, 5), (1, 2, 3, 6)],
        batched_chunk_representatives=representatives,
        output=output,
    )

    assert calls == [
        ((3, 2, attn.head_dim), (5, block_size, 1, attn.head_dim), (0,), (2, 1, 2)),
        ((3, 2, attn.head_dim), (5, block_size, 1, attn.head_dim), (1,), (2, 1, 2)),
    ]
    assert handled == {0, 1}
    torch.testing.assert_close(output[:, :2], query_states[:, :2] + 100.0)
    torch.testing.assert_close(output[:, 2:], query_states[:, 2:] + 200.0)


def test_dsa_multi_kv_flattened_decode_falls_back_to_gather_path():
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    attn.q_indexer_chunk_top_k = 1
    torch.manual_seed(6)

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

    attn.q_indexer_use_page_table_fa = True
    attn.q_indexer_use_flattened_decode_page_table_fa = True
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


def test_dsa_multi_kv_unified_bucket_reuses_single_kv_path(monkeypatch):
    attn = _make_chunked_dsa_attn()
    attn.num_kv_heads = 2
    attn.num_heads = 4
    attn.q_indexer_chunk_top_k = 1
    torch.manual_seed(7)

    block_size = attn.q_indexer_chunk_size
    group_size = attn.num_heads // attn.num_kv_heads
    seq_specs = [(0, 10), (1, 11)]
    block_tables = torch.tensor([[2, 0, 3], [5, 4, 1]], dtype=torch.int32)
    num_physical_blocks = int(block_tables.max().item()) + 1
    key_cache = torch.zeros(
        num_physical_blocks,
        block_size,
        attn.num_kv_heads,
        attn.head_dim,
    )
    value_cache = torch.zeros_like(key_cache)
    key_states_by_seq = []
    value_states_by_seq = []
    for seq_idx, key_len in seq_specs:
        key_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
        value_states = torch.randn(key_len, attn.num_kv_heads, attn.head_dim)
        key_states_by_seq.append(key_states)
        value_states_by_seq.append(value_states)
        for token in range(key_len):
            block_id = int(block_tables[seq_idx, token // block_size].item())
            key_cache[block_id, token % block_size] = key_states[token]
            value_cache[block_id, token % block_size] = value_states[token]

    query_states = torch.randn(2, attn.num_heads, attn.head_dim)
    indexer_query_states = torch.randn(2, attn.num_kv_heads, attn.q_indexer_dim)
    hidden_states = torch.zeros(2, 4)
    positions = torch.tensor([9, 10], dtype=torch.long)
    representatives = {
        seq_idx: attn._build_indexer_chunk_representatives(key_states)
        for seq_idx, key_states in enumerate(key_states_by_seq)
    }
    reference = torch.cat(
        [
            attn._forward_dsa_chunked_sequence(
                query_states=query_states[row : row + 1],
                indexer_query_states=indexer_query_states[row : row + 1],
                key_states=key_states_by_seq[seq_idx],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_tables[seq_idx],
                attn_metadata=None,
                positions=positions[row : row + 1],
                key_len=key_len,
                chunk_representatives=representatives[seq_idx],
            )
            for row, (seq_idx, key_len) in enumerate(seq_specs)
        ],
        dim=0,
    )

    attn.q_indexer_use_page_table_fa = True
    attn.q_indexer_use_prefill_page_table_fa = True
    attn.q_indexer_use_flattened_prefill_page_table_fa = True

    def fake_indexer_q_proj(x):
        assert x.shape[0] == 2
        return indexer_query_states.reshape(2, -1), None

    object.__setattr__(attn, "indexer_q_proj", fake_indexer_q_proj)
    object.__setattr__(
        attn,
        "_local_kv_head_indices",
        torch.arange(attn.num_kv_heads, dtype=torch.long),
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
        assert q.shape == (2, group_size, attn.head_dim)
        assert k.shape == (num_physical_blocks, block_size, 1, attn.head_dim)
        assert max_seqlen_q == 1
        assert dropout_p == 0.0
        assert causal is True
        assert "fa_version" in kwargs
        assert cu_seqlens_q.tolist() == [0, 1, 2]
        assert max_seqlen_k == int(seqused_k.max().item())
        calls.append(
            (
                block_table.detach().cpu().clone(),
                seqused_k.detach().cpu().clone(),
            )
        )
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

    output = torch.empty_like(query_states)
    handled = attn._forward_dsa_chunked_unified_page_table_fa_bucket(
        hidden_states=hidden_states,
        query_states=query_states,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_tables,
        attn_metadata=None,
        positions=positions,
        active_seq_infos=[(0, 0, 1, 10), (1, 1, 2, 11)],
        batched_chunk_representatives=representatives,
        output=output,
    )

    assert handled == {0, 1}
    assert len(calls) == attn.num_kv_heads
    for group_idx, (actual_table, actual_seqused_k) in enumerate(calls):
        expected_page_tables = []
        for row, (seq_idx, _) in enumerate(seq_specs):
            current_chunk = int(positions[row].item()) // block_size
            logits = torch.mv(
                representatives[seq_idx][:current_chunk, group_idx],
                indexer_query_states[row, group_idx].float(),
            )
            top_chunk = int(logits.topk(k=1).indices[0].item())
            expected_page_tables.append(
                [
                    int(block_tables[seq_idx, top_chunk].item()),
                    int(block_tables[seq_idx, current_chunk].item()),
                ]
            )
        assert actual_table.tolist() == expected_page_tables
        expected_seqused_k = []
        for row, _ in enumerate(seq_specs):
            current_chunk = int(positions[row].item()) // block_size
            tail_len = int(positions[row].item()) - current_chunk * block_size + 1
            expected_seqused_k.append(block_size + tail_len)
        assert actual_seqused_k.tolist() == expected_seqused_k
    torch.testing.assert_close(output, reference)
