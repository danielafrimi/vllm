# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random

import pytest
import torch

from vllm.model_executor.models import (
    nemotron_h_chunked_dsa_components_efficient as efficient_components,
)
from vllm.model_executor.models import (
    nemotron_h_dsa_triton_scoring,
)

try:
    from vllm.triton_utils import triton
except ImportError:
    triton = None


def test_dynamic_top_k_segments_allow_gaps_but_not_overlaps():
    row_top_k = torch.tensor([1, 1, 3, 3], dtype=torch.int32)
    validate = nemotron_h_dsa_triton_scoring._dsa_validate_dynamic_top_k

    assert validate(
        row_top_k=row_top_k,
        top_k_segments=[(0, 2, 1), (2, 4, 3)],
        num_rows=4,
        device=row_top_k.device,
    )
    assert validate(
        row_top_k=torch.tensor([1, 0, 3, 3], dtype=torch.int32),
        top_k_segments=[(0, 1, 1), (2, 4, 3)],
        num_rows=4,
        device=row_top_k.device,
    )
    assert not validate(
        row_top_k=row_top_k,
        top_k_segments=[(0, 3, 1), (2, 4, 3)],
        num_rows=4,
        device=row_top_k.device,
    )
    assert not validate(
        row_top_k=row_top_k,
        top_k_segments=None,
        num_rows=4,
        device=row_top_k.device,
    )


def test_dynamic_top_k_selects_each_segment_at_its_exact_k(monkeypatch):
    min_logit = torch.finfo(torch.float32).min
    logits = torch.tensor(
        [
            [0.0, 9.0, 8.0, 7.0, 6.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [9.0, 8.0, 7.0, 6.0, 5.0],
            [0.0, 1.0, 8.0, 9.0, 7.0],
            [6.0, 5.0, min_logit, min_logit, min_logit],
        ],
        dtype=torch.float32,
    )
    selectable_counts = torch.tensor([5, 5, 5, 5, 2], dtype=torch.int32)
    row_top_k = torch.tensor([1, 1, 0, 3, 3], dtype=torch.int32)
    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "_has_top_k_per_row_prefill",
        lambda: False,
    )

    indices, selected_counts = (
        nemotron_h_dsa_triton_scoring._dsa_dynamic_top_k_from_logits(
            logits=logits,
            selectable_counts=selectable_counts,
            row_top_k=row_top_k,
            top_k_segments=[(0, 2, 1), (3, 5, 3)],
            max_top_k=3,
        )
    )

    torch.testing.assert_close(
        selected_counts,
        torch.tensor([1, 1, 0, 3, 2], dtype=torch.int32),
    )
    torch.testing.assert_close(
        indices[:2, 1:],
        torch.full((2, 2), -1, dtype=torch.int32),
    )
    torch.testing.assert_close(
        indices[2],
        torch.full((3,), -1, dtype=torch.int32),
    )
    assert indices[0, 0].item() == 1
    assert indices[1, 0].item() == 0
    assert set(indices[3, :3].tolist()) == {2, 3, 4}
    assert set(indices[4, :2].tolist()) == {0, 1}


def test_dynamic_top_k_uses_safe_fallback_above_cuda_shared_memory_limit(
    monkeypatch,
):
    top_k = nemotron_h_dsa_triton_scoring._DSA_CUDA_PREFILL_TOP_K_MAX + 64
    num_chunks = top_k + 808
    min_logit = torch.finfo(torch.float32).min
    logits = torch.randn(2, num_chunks, dtype=torch.float32)
    selectable_counts = torch.tensor(
        [num_chunks, top_k - 192],
        dtype=torch.int32,
    )
    logits[1, int(selectable_counts[1].item()) :].fill_(min_logit)
    row_top_k = torch.full((2,), top_k, dtype=torch.int32)
    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "_has_top_k_per_row_prefill",
        lambda: True,
    )
    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "_run_top_k_per_row_prefill",
        lambda *_args, **_kwargs: pytest.fail(
            "unsafe K must not launch the CUDA prefill top-k kernel"
        ),
    )

    indices, selected_counts = (
        nemotron_h_dsa_triton_scoring._dsa_dynamic_top_k_from_logits(
            logits=logits,
            selectable_counts=selectable_counts,
            row_top_k=row_top_k,
            top_k_segments=[(0, 2, top_k)],
            max_top_k=top_k,
        )
    )

    torch.testing.assert_close(
        selected_counts,
        torch.tensor([top_k, top_k - 192], dtype=torch.int32),
    )
    assert bool((indices[0] < num_chunks).all())
    assert bool((indices[1, : top_k - 192] < top_k - 192).all())


def test_dynamic_top_k_tile_plan_initializes_invalid_logits_for_unsafe_k(
    monkeypatch,
):
    top_k = nemotron_h_dsa_triton_scoring._DSA_CUDA_PREFILL_TOP_K_MAX + 64
    observed: dict[str, bool] = {}

    def fake_score_logits_tile_plan_triton(
        *, initialize_invalid: bool, **_kwargs: object
    ) -> torch.Tensor:
        observed["initialize_invalid"] = initialize_invalid
        return torch.zeros(2, top_k + 32, dtype=torch.float32)

    def fake_dynamic_top_k_from_logits(
        *, selectable_counts: torch.Tensor, **_kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(2, top_k, dtype=torch.int32),
            selectable_counts.clamp(max=top_k),
        )

    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "_has_top_k_per_row_prefill",
        lambda: True,
    )
    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "dsa_batched_score_logits_tile_plan_triton",
        fake_score_logits_tile_plan_triton,
    )
    monkeypatch.setattr(
        nemotron_h_dsa_triton_scoring,
        "_dsa_dynamic_top_k_from_logits",
        fake_dynamic_top_k_from_logits,
    )

    result = nemotron_h_dsa_triton_scoring.dsa_batched_score_topk_tile_plan_triton(
        score_query_states=torch.zeros(2, 16),
        chunk_representatives=torch.zeros(1, 1, 1, 16),
        tile_plan=torch.zeros(0, 8, dtype=torch.int32),
        current_chunks=torch.tensor([top_k + 32, top_k - 1], dtype=torch.int32),
        row_num_prior_chunks=torch.tensor([top_k + 32, top_k + 32], dtype=torch.int32),
        total_rows=2,
        chunk_size=16,
        chunk_top_k=top_k,
        logit_scale=1.0,
        q_indexer_dim=16,
        max_prior_chunks=top_k + 32,
        row_top_k=torch.full((2,), top_k, dtype=torch.int32),
        top_k_segments=[(0, 2, top_k)],
    )

    assert result is not None
    assert observed == {"initialize_invalid": True}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the large-K launch-limit regression",
)
def test_dynamic_top_k_cuda_stress_at_8192():
    top_k = 8192
    num_chunks = 65536
    device = torch.device("cuda")
    selectable_counts = torch.tensor(
        [num_chunks, num_chunks - 128, top_k - 192],
        device=device,
        dtype=torch.int32,
    )
    logits = torch.randn(3, num_chunks, device=device, dtype=torch.float32)
    chunk_ids = torch.arange(num_chunks, device=device, dtype=torch.int32)
    logits.masked_fill_(
        chunk_ids[None, :] >= selectable_counts[:, None],
        torch.finfo(torch.float32).min,
    )
    row_top_k = torch.full((3,), top_k, device=device, dtype=torch.int32)

    indices, selected_counts = (
        nemotron_h_dsa_triton_scoring._dsa_dynamic_top_k_from_logits(
            logits=logits,
            selectable_counts=selectable_counts,
            row_top_k=row_top_k,
            top_k_segments=[(0, 3, top_k)],
            max_top_k=top_k,
        )
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        selected_counts,
        torch.tensor(
            [top_k, top_k, top_k - 192],
            device=device,
            dtype=torch.int32,
        ),
    )
    valid_slots = (
        torch.arange(top_k, device=device)[None, :] < (selected_counts[:, None])
    )
    assert bool((indices.masked_select(valid_slots) >= 0).all())
    assert bool(
        (
            indices.masked_select(valid_slots)
            < selectable_counts[:, None].expand_as(indices).masked_select(valid_slots)
        ).all()
    )


def _key_len_for_query(
    *,
    query_len: int,
    seq_idx: int,
    chunk_size: int,
) -> int:
    prior_chunks = 1 + (seq_idx % 5)
    tail = (3 * seq_idx + query_len) % chunk_size
    return max(query_len, prior_chunks * chunk_size + tail + query_len // 2)


def _mixed_q_lens(
    batch_size: int,
    *,
    profile: tuple[int, ...],
    shuffle_seed: int,
) -> list[int]:
    q_lens = [profile[i % len(profile)] for i in range(batch_size)]
    random.Random(shuffle_seed).shuffle(q_lens)
    return q_lens


def _make_trial_inputs(
    *,
    batch_size: int,
    q_lens: list[int],
    num_kv_heads: int,
    q_indexer_dim: int,
    chunk_size: int,
    seed: int,
) -> dict[str, object]:
    device = torch.device("cuda")
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
    # Include the first-token case where no prior chunks are selectable.
    key_lens[0] = q_lens[0]

    num_chunks_by_seq = [
        math.ceil(key_len / chunk_size) if key_len > 0 else 0 for key_len in key_lens
    ]
    max_chunks = max(num_chunks_by_seq)
    representatives = torch.randn(
        batch_size,
        max_chunks,
        num_kv_heads,
        q_indexer_dim,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    indexer_queries_by_seq = [
        torch.randn(
            q_len,
            num_kv_heads,
            q_indexer_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        for q_len in q_lens
    ]

    flat_queries: list[torch.Tensor] = []
    row_seq_ids: list[torch.Tensor] = []
    row_group_ids: list[torch.Tensor] = []
    row_current_chunks: list[torch.Tensor] = []
    row_prior_chunks: list[torch.Tensor] = []

    for seq_idx, (q_len, key_len, num_chunks) in enumerate(
        zip(q_lens, key_lens, num_chunks_by_seq)
    ):
        positions = torch.arange(
            key_len - q_len,
            key_len,
            device=device,
            dtype=torch.long,
        )
        current_chunks = torch.div(
            positions,
            chunk_size,
            rounding_mode="floor",
        ).clamp(min=0, max=max(num_chunks - 1, 0))
        prior_chunks = max(num_chunks - 1, 0)
        for group_idx in range(num_kv_heads):
            flat_queries.append(indexer_queries_by_seq[seq_idx][:, group_idx])
            row_seq_ids.append(
                torch.full((q_len,), seq_idx, device=device, dtype=torch.int32)
            )
            row_group_ids.append(
                torch.full(
                    (q_len,),
                    group_idx,
                    device=device,
                    dtype=torch.int32,
                )
            )
            row_current_chunks.append(current_chunks.to(torch.int32))
            row_prior_chunks.append(
                torch.full(
                    (q_len,),
                    prior_chunks,
                    device=device,
                    dtype=torch.int32,
                )
            )

    return {
        "q_lens": q_lens,
        "key_lens": key_lens,
        "num_chunks_by_seq": num_chunks_by_seq,
        "representatives": representatives,
        "indexer_queries_by_seq": indexer_queries_by_seq,
        "flat_queries": torch.cat(flat_queries, dim=0).contiguous(),
        "row_seq_ids": torch.cat(row_seq_ids, dim=0).contiguous(),
        "row_group_ids": torch.cat(row_group_ids, dim=0).contiguous(),
        "row_current_chunks": torch.cat(row_current_chunks, dim=0).contiguous(),
        "row_prior_chunks": torch.cat(row_prior_chunks, dim=0).contiguous(),
    }


def _run_trial_batched_scoring(
    *,
    flat_queries: torch.Tensor,
    representatives: torch.Tensor,
    row_current_chunks: torch.Tensor,
    row_seq_ids: torch.Tensor,
    row_group_ids: torch.Tensor,
    row_prior_chunks: torch.Tensor,
    q_indexer_dim: int,
    logit_scale: float,
) -> torch.Tensor:
    max_prior_chunks = int(row_prior_chunks.max().item())
    logits = nemotron_h_dsa_triton_scoring.dsa_batched_score_logits_triton(
        score_query_states=flat_queries,
        chunk_representatives=representatives,
        current_chunks=row_current_chunks,
        row_seq_ids=row_seq_ids,
        row_group_ids=row_group_ids,
        row_num_prior_chunks=row_prior_chunks,
        logit_scale=logit_scale,
        q_indexer_dim=q_indexer_dim,
        max_prior_chunks=max_prior_chunks,
    )
    assert logits is not None
    return logits


def _make_trial_row_plan_parts(
    *,
    representatives: torch.Tensor,
    q_lens: list[int],
    key_lens: list[int],
    num_chunks_by_seq: list[int],
) -> list[tuple[int, int, int, int, int, int, int]]:
    parts: list[tuple[int, int, int, int, int, int, int]] = []
    row_start = 0
    num_kv_heads = int(representatives.shape[2])
    for seq_idx, (q_len, key_len, num_chunks) in enumerate(
        zip(q_lens, key_lens, num_chunks_by_seq)
    ):
        prior_chunks = max(num_chunks - 1, 0)
        query_position_start = key_len - q_len
        for group_idx in range(num_kv_heads):
            parts.append(
                (
                    row_start,
                    q_len,
                    seq_idx,
                    seq_idx,
                    group_idx,
                    prior_chunks,
                    query_position_start,
                )
            )
            row_start += q_len
    return parts


def _make_trial_row_plan(
    *,
    representatives: torch.Tensor,
    q_lens: list[int],
    key_lens: list[int],
    num_chunks_by_seq: list[int],
) -> torch.Tensor:
    parts = _make_trial_row_plan_parts(
        representatives=representatives,
        q_lens=q_lens,
        key_lens=key_lens,
        num_chunks_by_seq=num_chunks_by_seq,
    )
    return torch.tensor(parts, device=representatives.device, dtype=torch.int32)


def _make_trial_tile_plan(
    *,
    representatives: torch.Tensor,
    q_lens: list[int],
    key_lens: list[int],
    num_chunks_by_seq: list[int],
) -> tuple[torch.Tensor, int, int, int, int]:
    row_plan_parts = _make_trial_row_plan_parts(
        representatives=representatives,
        q_lens=q_lens,
        key_lens=key_lens,
        num_chunks_by_seq=num_chunks_by_seq,
    )
    (
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = nemotron_h_dsa_triton_scoring.dsa_score_tile_plan_config()
    tile_plan_parts = nemotron_h_dsa_triton_scoring.dsa_build_score_tile_plan_parts(
        row_plan_parts,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    return (
        torch.tensor(
            tile_plan_parts,
            device=representatives.device,
            dtype=torch.int32,
        ),
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for DSA tile-plan metadata",
)
@pytest.mark.parametrize(
    "q_lens",
    [
        [1, 1, 1, 1],
        [2, 4, 5],
        [1, 3, 17, 70, 129],
    ],
)
def test_trial_batched_dsa_gpu_tile_plan_builder_matches_cpu_reference(
    q_lens: list[int],
):
    inputs = _make_trial_inputs(
        batch_size=len(q_lens),
        q_lens=q_lens,
        num_kv_heads=2,
        q_indexer_dim=16,
        chunk_size=8,
        seed=19021 + sum(q_lens),
    )
    row_plan_parts = _make_trial_row_plan_parts(
        representatives=inputs["representatives"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        num_chunks_by_seq=inputs["num_chunks_by_seq"],
    )
    (
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = nemotron_h_dsa_triton_scoring.dsa_score_tile_plan_config()
    expected_parts = nemotron_h_dsa_triton_scoring.dsa_build_score_tile_plan_parts(
        row_plan_parts,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    row_plan_with_tiles: list[tuple[int, ...]] = []
    tile_offset = 0
    max_tiles_per_row_plan = 0
    for row_plan_part in row_plan_parts:
        tile_count, _, _, _ = (
            nemotron_h_dsa_triton_scoring.dsa_count_score_tile_plan_parts(
                (row_plan_part,),
                small_block_rows=small_block_rows,
                large_block_rows=large_block_rows,
                block_chunks=block_chunks,
                decode_block_chunks=decode_block_chunks,
            )
        )
        row_plan_with_tiles.append((*row_plan_part, tile_offset, tile_count))
        tile_offset += tile_count
        max_tiles_per_row_plan = max(max_tiles_per_row_plan, tile_count)

    row_plan = torch.tensor(
        row_plan_with_tiles,
        device=inputs["representatives"].device,
        dtype=torch.int32,
    )
    actual = nemotron_h_dsa_triton_scoring.dsa_build_score_tile_plan_triton(
        row_plan_with_tiles=row_plan,
        total_tiles=len(expected_parts),
        max_tiles_per_row_plan=max_tiles_per_row_plan,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    assert actual is not None
    torch.testing.assert_close(
        actual.cpu(),
        torch.tensor(expected_parts, dtype=torch.int32),
    )


def _current_efficient_scoring_reference(
    *,
    representatives: torch.Tensor,
    indexer_queries_by_seq: list[torch.Tensor],
    q_lens: list[int],
    key_lens: list[int],
    chunk_size: int,
    logit_scale: float,
) -> torch.Tensor:
    q_indexer_dim = int(representatives.shape[-1])
    num_kv_heads = int(representatives.shape[2])
    max_prior_chunks = max(
        max(math.ceil(key_len / chunk_size) - 1, 0) for key_len in key_lens
    )
    provider = efficient_components.EfficientChunkedDSAScoringProvider(
        q_indexer_dim=q_indexer_dim,
        logit_scale=logit_scale,
    )

    padded_logits: list[torch.Tensor] = []
    for seq_idx, (q_len, key_len) in enumerate(zip(q_lens, key_lens)):
        num_chunks = math.ceil(key_len / chunk_size) if key_len > 0 else 0
        prior_chunks = max(num_chunks - 1, 0)
        positions = torch.arange(
            key_len - q_len,
            key_len,
            device=representatives.device,
            dtype=torch.long,
        )
        current_chunks = torch.div(
            positions,
            chunk_size,
            rounding_mode="floor",
        ).clamp(min=0, max=max(num_chunks - 1, 0))
        for group_idx in range(num_kv_heads):
            result = provider(
                score_query_states=indexer_queries_by_seq[seq_idx][:, group_idx],
                representative_state=representatives[seq_idx, :num_chunks],
                current_chunks=current_chunks,
                max_prior_chunks=prior_chunks,
                group_idx=group_idx,
            )
            scores = provider.get_scores(result)
            assert scores is not None
            logits, _ = scores
            if prior_chunks < max_prior_chunks:
                pad = torch.full(
                    (q_len, max_prior_chunks - prior_chunks),
                    torch.finfo(logits.dtype).min,
                    device=logits.device,
                    dtype=logits.dtype,
                )
                logits = torch.cat((logits, pad), dim=-1)
            padded_logits.append(logits)

    return torch.cat(padded_logits, dim=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for trial DSA batched scoring",
)
@pytest.mark.parametrize(
    ("profile", "batch_size", "shuffle_seed"),
    [
        ((1,), 7, 11),
        ((4, 5, 6), 9, 13),
        ((65, 96, 129), 6, 17),
        ((1, 4, 5, 6, 17, 64, 96), 14, 19),
    ],
)
def test_trial_batched_dsa_tile_plan_scoring_matches_row_kernel(
    profile: tuple[int, ...],
    batch_size: int,
    shuffle_seed: int,
):
    case = {
        "batch_size": batch_size,
        "q_lens": _mixed_q_lens(
            batch_size,
            profile=profile,
            shuffle_seed=shuffle_seed,
        ),
        "num_kv_heads": 3,
        "q_indexer_dim": 16,
        "chunk_size": 8,
        "logit_scale": 1.25,
        "seed": 8101 + shuffle_seed,
    }
    inputs = _make_trial_inputs(
        **{
            k: case[k]
            for k in (
                "batch_size",
                "q_lens",
                "num_kv_heads",
                "q_indexer_dim",
                "chunk_size",
                "seed",
            )
        }
    )
    max_prior_chunks = int(inputs["row_prior_chunks"].max().item())
    (
        tile_plan,
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = _make_trial_tile_plan(
        representatives=inputs["representatives"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        num_chunks_by_seq=inputs["num_chunks_by_seq"],
    )
    modes = set(tile_plan[:, 7].detach().cpu().tolist())
    if 1 in profile:
        assert 0 in modes
    if any(1 < q_len <= small_block_rows for q_len in profile):
        assert 1 in modes
    if any(q_len > small_block_rows for q_len in profile):
        assert 2 in modes

    bf16_representatives = inputs["representatives"].to(torch.bfloat16)
    actual = nemotron_h_dsa_triton_scoring.dsa_batched_score_logits_tile_plan_triton(
        score_query_states=inputs["flat_queries"],
        chunk_representatives=bf16_representatives,
        tile_plan=tile_plan,
        current_chunks=inputs["row_current_chunks"],
        total_rows=int(inputs["flat_queries"].shape[0]),
        chunk_size=case["chunk_size"],
        logit_scale=case["logit_scale"],
        q_indexer_dim=case["q_indexer_dim"],
        max_prior_chunks=max_prior_chunks,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    assert actual is not None
    expected = _run_trial_batched_scoring(
        flat_queries=inputs["flat_queries"],
        representatives=bf16_representatives,
        row_current_chunks=inputs["row_current_chunks"],
        row_seq_ids=inputs["row_seq_ids"],
        row_group_ids=inputs["row_group_ids"],
        row_prior_chunks=inputs["row_prior_chunks"],
        q_indexer_dim=case["q_indexer_dim"],
        logit_scale=case["logit_scale"],
    )

    # The tile-plan kernel intentionally rounds dot operands to BF16 and
    # accumulates into FP32, while the row reference uses FP32 operands.
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for trial DSA batched scoring",
)
def test_trial_batched_dsa_tile_plan_topk_matches_row_topk():
    case = {
        "batch_size": 7,
        "q_lens": _mixed_q_lens(
            7,
            profile=(1, 3, 9, 17, 41),
            shuffle_seed=621,
        ),
        "num_kv_heads": 2,
        "q_indexer_dim": 16,
        "chunk_size": 8,
        "logit_scale": 1.25,
        "seed": 8123,
        "chunk_top_k": 3,
    }
    inputs = _make_trial_inputs(
        **{
            k: case[k]
            for k in (
                "batch_size",
                "q_lens",
                "num_kv_heads",
                "q_indexer_dim",
                "chunk_size",
                "seed",
            )
        }
    )
    max_prior_chunks = int(inputs["row_prior_chunks"].max().item())
    (
        tile_plan,
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = _make_trial_tile_plan(
        representatives=inputs["representatives"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        num_chunks_by_seq=inputs["num_chunks_by_seq"],
    )

    actual = nemotron_h_dsa_triton_scoring.dsa_batched_score_topk_tile_plan_triton(
        score_query_states=inputs["flat_queries"],
        chunk_representatives=inputs["representatives"],
        tile_plan=tile_plan,
        current_chunks=inputs["row_current_chunks"],
        row_num_prior_chunks=inputs["row_prior_chunks"],
        total_rows=int(inputs["flat_queries"].shape[0]),
        chunk_size=case["chunk_size"],
        chunk_top_k=case["chunk_top_k"],
        logit_scale=case["logit_scale"],
        q_indexer_dim=case["q_indexer_dim"],
        max_prior_chunks=max_prior_chunks,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    expected = nemotron_h_dsa_triton_scoring.dsa_batched_score_topk_triton(
        score_query_states=inputs["flat_queries"],
        chunk_representatives=inputs["representatives"],
        current_chunks=inputs["row_current_chunks"],
        row_seq_ids=inputs["row_seq_ids"],
        row_group_ids=inputs["row_group_ids"],
        row_num_prior_chunks=inputs["row_prior_chunks"],
        chunk_top_k=case["chunk_top_k"],
        logit_scale=case["logit_scale"],
        q_indexer_dim=case["q_indexer_dim"],
        max_prior_chunks=max_prior_chunks,
    )
    assert actual is not None
    assert expected is not None
    actual_indices, actual_counts, _ = actual
    expected_indices, expected_counts, _ = expected
    torch.testing.assert_close(actual_counts, expected_counts)

    slot_ids = torch.arange(
        actual_indices.shape[1],
        device=actual_indices.device,
        dtype=torch.int32,
    )
    actual_valid = slot_ids[None, :] < actual_counts[:, None].clamp(
        max=actual_indices.shape[1]
    )
    expected_valid = slot_ids[None, :] < expected_counts[:, None].clamp(
        max=expected_indices.shape[1]
    )
    torch.testing.assert_close(actual_valid, expected_valid)
    torch.testing.assert_close(
        actual_indices.masked_fill(~actual_valid, -1).sort(dim=-1).values,
        expected_indices.masked_fill(~expected_valid, -1).sort(dim=-1).values,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or triton is None
    or not nemotron_h_dsa_triton_scoring._has_top_k_per_row_prefill(),
    reason="CUDA, Triton, and the CUDA prefill top-k op are required",
)
def test_trial_batched_dsa_tile_plan_supports_exact_mixed_topk_segments():
    q_lens = [4, 4, 4]
    inputs = _make_trial_inputs(
        batch_size=len(q_lens),
        q_lens=q_lens,
        num_kv_heads=1,
        q_indexer_dim=16,
        chunk_size=8,
        seed=31741,
    )
    max_prior_chunks = int(inputs["row_prior_chunks"].max().item())
    (
        tile_plan,
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = _make_trial_tile_plan(
        representatives=inputs["representatives"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        num_chunks_by_seq=inputs["num_chunks_by_seq"],
    )
    row_top_k = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1, 3, 3, 3, 3],
        device="cuda",
        dtype=torch.int32,
    )
    top_k_segments = [(4, 8, 1), (8, 12, 3)]

    result = nemotron_h_dsa_triton_scoring.dsa_batched_score_topk_tile_plan_triton(
        score_query_states=inputs["flat_queries"],
        chunk_representatives=inputs["representatives"],
        tile_plan=tile_plan,
        current_chunks=inputs["row_current_chunks"],
        row_num_prior_chunks=inputs["row_prior_chunks"],
        total_rows=12,
        chunk_size=8,
        chunk_top_k=3,
        logit_scale=1.25,
        q_indexer_dim=16,
        max_prior_chunks=max_prior_chunks,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
        return_logits=True,
        row_top_k=row_top_k,
        top_k_segments=top_k_segments,
    )
    assert result is not None
    indices, selected_counts, logits = result
    assert logits is not None
    torch.testing.assert_close(
        indices[:4],
        torch.full((4, 3), -1, device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(
        selected_counts[:4],
        torch.zeros(4, device="cuda", dtype=torch.int32),
    )

    for row_start, row_end, requested_top_k in top_k_segments:
        for row in range(row_start, row_end):
            selectable_count = min(
                int(inputs["row_current_chunks"][row].item()),
                int(inputs["row_prior_chunks"][row].item()),
            )
            expected_count = min(selectable_count, requested_top_k)
            assert int(selected_counts[row].item()) == expected_count
            expected = logits[row, :selectable_count].topk(expected_count).indices
            actual = indices[row, :expected_count].to(torch.long)
            torch.testing.assert_close(
                actual.sort().values,
                expected.sort().values,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for trial DSA batched scoring",
)
def test_trial_batched_dsa_tile_plan_scoring_handles_pure_decode():
    inputs = _make_trial_inputs(
        batch_size=4,
        q_lens=[1, 1, 1, 1],
        num_kv_heads=2,
        q_indexer_dim=16,
        chunk_size=8,
        seed=9417,
    )
    (
        tile_plan,
        small_block_rows,
        large_block_rows,
        block_chunks,
        decode_block_chunks,
    ) = _make_trial_tile_plan(
        representatives=inputs["representatives"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        num_chunks_by_seq=inputs["num_chunks_by_seq"],
    )
    assert set(tile_plan[:, 7].detach().cpu().tolist()) == {0}

    actual = nemotron_h_dsa_triton_scoring.dsa_batched_score_logits_tile_plan_triton(
        score_query_states=inputs["flat_queries"],
        chunk_representatives=inputs["representatives"],
        tile_plan=tile_plan,
        current_chunks=inputs["row_current_chunks"],
        total_rows=int(inputs["flat_queries"].shape[0]),
        chunk_size=8,
        logit_scale=1.0,
        q_indexer_dim=16,
        max_prior_chunks=int(inputs["row_prior_chunks"].max().item()),
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
    )
    assert actual is not None
    expected = _run_trial_batched_scoring(
        flat_queries=inputs["flat_queries"],
        representatives=inputs["representatives"],
        row_current_chunks=inputs["row_current_chunks"],
        row_seq_ids=inputs["row_seq_ids"],
        row_group_ids=inputs["row_group_ids"],
        row_prior_chunks=inputs["row_prior_chunks"],
        q_indexer_dim=16,
        logit_scale=1.0,
    )

    # The tile-plan kernel intentionally rounds dot operands to BF16 and
    # accumulates into FP32, while the row reference uses FP32 operands.
    torch.testing.assert_close(actual, expected, atol=4e-2, rtol=2e-2)


def _clear_triton_kernel_cache(kernel) -> None:
    if hasattr(kernel, "device_caches"):
        kernel.device_caches.clear()


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
    return len(cache_entry[0])


def test_efficient_batched_selection_passes_summary_tensor_directly(
    monkeypatch,
):
    representatives = torch.randn(2, 5, 2, 3, dtype=torch.float32)
    representative_state = efficient_components._TritonBatchedChunkRepresentatives(
        representatives=representatives,
        local_by_seq={10: 0, 11: 1},
        num_chunks_by_seq={10: 4, 11: 5},
    )
    indexer_q = torch.randn(5, 1, 3, dtype=torch.bfloat16)
    current_chunks_0 = torch.tensor([3, 3], dtype=torch.long)
    current_chunks_1 = torch.tensor([4, 4, 4], dtype=torch.long)
    sparse_infos = [
        (10, 0, 2, 32, 4, 30, current_chunks_0),
        (11, 2, 5, 40, 5, 37, current_chunks_1),
    ]
    block_table = torch.arange(10, dtype=torch.int32).view(2, 5)
    observed: dict[str, object] = {}

    def fake_score_topk(
        *,
        score_query_states: torch.Tensor,
        chunk_representatives: torch.Tensor,
        tile_plan: torch.Tensor,
        current_chunks: torch.Tensor,
        row_num_prior_chunks: torch.Tensor,
        chunk_top_k: int,
        **_: object,
    ):
        observed["direct_representatives"] = chunk_representatives is representatives
        torch.testing.assert_close(score_query_states, indexer_q[:, 0])
        torch.testing.assert_close(
            current_chunks,
            torch.tensor([3, 3, 4, 4, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(
            tile_plan,
            torch.tensor(
                [
                    [0, 2, 0, 1, 3, 30, 0, 1],
                    [2, 3, 1, 1, 4, 37, 0, 1],
                ],
                dtype=torch.int32,
            ),
        )
        torch.testing.assert_close(
            row_num_prior_chunks,
            torch.tensor([3, 3, 4, 4, 4], dtype=torch.int32),
        )
        return (
            torch.zeros(5, chunk_top_k, dtype=torch.int32),
            torch.full((5,), chunk_top_k, dtype=torch.int32),
            None,
        )

    def fake_build_score_metadata_triton(
        *,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        num_actual_tokens: int,
        active_seq_count: int,
        num_sparse_plans: int,
        total_rows: int,
        chunk_size: int,
        representative_group_idx: int,
        dense_decode_threshold: int,
        dense_prefill_threshold: int,
        chunk_top_k: int,
        max_q_len: int,
        **_: object,
    ):
        torch.testing.assert_close(
            query_start_loc,
            torch.tensor([0, 2, 5], dtype=torch.int32),
        )
        torch.testing.assert_close(
            seq_lens,
            torch.tensor([32, 40], dtype=torch.int32),
        )
        assert num_actual_tokens == 5
        assert active_seq_count == 2
        assert num_sparse_plans == 2
        assert total_rows == 5
        assert chunk_size == 8
        assert representative_group_idx == 1
        assert dense_decode_threshold == -1
        assert dense_prefill_threshold == -1
        assert chunk_top_k == 2
        assert max_q_len == 3
        row_plan = torch.tensor(
            [
                [0, 2, 0, 10, 1, 3, 30, 0, 1],
                [2, 3, 1, 11, 1, 4, 37, 1, 1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(
            row_plan,
            torch.tensor(
                [
                    [0, 2, 0, 10, 1, 3, 30, 0, 1],
                    [2, 3, 1, 11, 1, 4, 37, 1, 1],
                ],
                dtype=torch.int32,
            ),
        )
        return row_plan, (
            torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
            torch.tensor([10, 10, 11, 11, 11], dtype=torch.int32),
            torch.ones(5, dtype=torch.int32),
            torch.tensor([3, 3, 4, 4, 4], dtype=torch.int32),
            torch.tensor([3, 3, 4, 4, 4], dtype=torch.int32),
            torch.tensor([7, 8, 6, 7, 8], dtype=torch.int32),
        )

    def fake_build_score_tile_plan_triton(
        *,
        row_plan_with_tiles: torch.Tensor,
        total_tiles: int,
        max_tiles_per_row_plan: int,
        **_: object,
    ):
        assert total_tiles == 2
        assert max_tiles_per_row_plan == 1
        torch.testing.assert_close(
            row_plan_with_tiles.cpu(),
            torch.tensor(
                [
                    [0, 2, 0, 10, 1, 3, 30, 0, 1],
                    [2, 3, 1, 11, 1, 4, 37, 1, 1],
                ],
                dtype=torch.int32,
            ),
        )
        return torch.tensor(
            [
                [0, 2, 0, 1, 3, 30, 0, 1],
                [2, 3, 1, 1, 4, 37, 0, 1],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_tile_plan_triton",
        fake_build_score_tile_plan_triton,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_score_topk_tile_plan_triton",
        fake_score_topk,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_metadata_triton",
        fake_build_score_metadata_triton,
    )
    bundle = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=3,
        chunk_size=8,
        num_kv_heads=1,
        head_dim=16,
        logit_scale=1.0,
        chunk_top_k=2,
    )

    actual = bundle.try_select_blocks_batched(
        indexer_q=indexer_q,
        sparse_infos=sparse_infos,
        batched_chunk_representatives=representative_state,
        block_table=block_table,
        representative_group_idx=1,
        query_start_loc=torch.tensor([0, 2, 5], dtype=torch.int32),
        seq_lens=torch.tensor([32, 40], dtype=torch.int32),
        num_actual_tokens=5,
        active_seq_count=2,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert observed["direct_representatives"] is True
    assert set(actual) == {10, 11}


def test_efficient_batched_selection_accepts_original_seq_representatives(
    monkeypatch,
):
    representatives = torch.randn(3, 5, 1, 3, dtype=torch.float32)
    representative_state = efficient_components._TritonBatchedChunkRepresentatives(
        representatives=representatives,
        local_by_seq={0: 0, 2: 2},
        num_chunks_by_seq={0: 4, 2: 5},
        seq_id_layout="original",
    )
    indexer_q = torch.randn(6, 1, 3, dtype=torch.bfloat16)
    sparse_infos = [
        (0, 0, 2, 32, 4, 30, torch.tensor([3, 3], dtype=torch.int32)),
        (2, 3, 6, 40, 5, 37, torch.tensor([4, 4, 4], dtype=torch.int32)),
    ]
    block_table = torch.arange(3 * 5, dtype=torch.int32).view(3, 5)
    observed: dict[str, object] = {}

    def fake_build_score_metadata_triton(
        *,
        representatives_use_original_seq_ids: bool,
        **_: object,
    ):
        observed["original_ids"] = representatives_use_original_seq_ids
        return torch.tensor(
            [
                [0, 2, 0, 0, 0, 3, 30, 0, 1],
                [3, 3, 2, 2, 0, 4, 37, 1, 1],
            ],
            dtype=torch.int32,
        ), (
            torch.tensor([0, 0, 0, 2, 2, 2], dtype=torch.int32),
            torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32),
            torch.zeros(6, dtype=torch.int32),
            torch.tensor([3, 3, 0, 4, 4, 4], dtype=torch.int32),
            torch.tensor([3, 3, 0, 4, 4, 4], dtype=torch.int32),
            torch.ones(6, dtype=torch.int32),
        )

    def fake_build_score_tile_plan_triton(
        *,
        row_plan_with_tiles: torch.Tensor,
        **_: object,
    ):
        torch.testing.assert_close(
            row_plan_with_tiles,
            torch.tensor(
                [
                    [0, 2, 0, 0, 0, 3, 30, 0, 1],
                    [3, 3, 2, 2, 0, 4, 37, 1, 1],
                ],
                dtype=torch.int32,
            ),
        )
        return torch.tensor(
            [
                [0, 2, 0, 0, 3, 30, 0, 1],
                [3, 3, 2, 0, 4, 37, 0, 1],
            ],
            dtype=torch.int32,
        )

    def fake_score_topk(
        *,
        chunk_representatives: torch.Tensor,
        tile_plan: torch.Tensor,
        chunk_top_k: int,
        **_: object,
    ):
        observed["direct_representatives"] = chunk_representatives is representatives
        torch.testing.assert_close(
            tile_plan,
            torch.tensor(
                [
                    [0, 2, 0, 0, 3, 30, 0, 1],
                    [3, 3, 2, 0, 4, 37, 0, 1],
                ],
                dtype=torch.int32,
            ),
        )
        return (
            torch.zeros(6, chunk_top_k, dtype=torch.int32),
            torch.full((6,), chunk_top_k, dtype=torch.int32),
            None,
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_metadata_triton",
        fake_build_score_metadata_triton,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_tile_plan_triton",
        fake_build_score_tile_plan_triton,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_score_topk_tile_plan_triton",
        fake_score_topk,
    )
    bundle = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=3,
        chunk_size=8,
        num_kv_heads=1,
        head_dim=16,
        logit_scale=1.0,
        chunk_top_k=2,
    )

    actual = bundle.try_select_blocks_batched(
        indexer_q=indexer_q,
        sparse_infos=sparse_infos,
        batched_chunk_representatives=representative_state,
        block_table=block_table,
        query_start_loc=torch.tensor([0, 2, 3, 6], dtype=torch.int32),
        seq_lens=torch.tensor([32, 1, 40], dtype=torch.int32),
        num_actual_tokens=6,
        active_seq_count=3,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert observed["original_ids"] is True
    assert observed["direct_representatives"] is True
    assert set(actual) == {0, 2}


def test_efficient_batched_selection_builds_compact_score_tile_plan(
    monkeypatch,
):
    representatives = torch.randn(4, 9, 1, 16, dtype=torch.float32)
    representative_state = efficient_components._TritonBatchedChunkRepresentatives(
        representatives=representatives,
        local_by_seq={10: 0, 11: 1, 12: 2, 13: 3},
        num_chunks_by_seq={10: 5, 11: 6, 12: 7, 13: 9},
    )
    q_lens = [1, 4, 6, 70]
    indexer_q = torch.randn(sum(q_lens), 1, 16, dtype=torch.bfloat16)
    sparse_infos = []
    q_start = 0
    for seq_idx, q_len, num_chunks, query_position_start in (
        (10, 1, 5, 40),
        (11, 4, 6, 45),
        (12, 6, 7, 51),
        (13, 70, 9, 80),
    ):
        q_end = q_start + q_len
        sparse_infos.append(
            (
                seq_idx,
                q_start,
                q_end,
                query_position_start + q_len,
                num_chunks,
                query_position_start,
                torch.full((q_len,), num_chunks - 1, dtype=torch.int32),
            )
        )
        q_start = q_end
    block_table = torch.arange(4 * 9, dtype=torch.int32).view(4, 9)
    observed: dict[str, torch.Tensor] = {}

    def fake_build_score_metadata_triton(
        *,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        num_actual_tokens: int,
        active_seq_count: int,
        num_sparse_plans: int,
        total_rows: int,
        chunk_size: int,
        representative_group_idx: int,
        dense_decode_threshold: int,
        dense_prefill_threshold: int,
        chunk_top_k: int,
        max_q_len: int,
        **_: object,
    ):
        torch.testing.assert_close(
            query_start_loc,
            torch.tensor([0, 1, 5, 11, 81], dtype=torch.int32),
        )
        torch.testing.assert_close(
            seq_lens,
            torch.tensor([41, 49, 57, 150], dtype=torch.int32),
        )
        assert num_actual_tokens == sum(q_lens)
        assert active_seq_count == 4
        assert num_sparse_plans == 4
        assert total_rows == sum(q_lens)
        assert chunk_size == 8
        assert representative_group_idx == 0
        assert dense_decode_threshold == -1
        assert dense_prefill_threshold == -1
        assert chunk_top_k == 2
        assert max_q_len == 70
        row_plan = torch.tensor(
            [
                [0, 1, 0, 10, 0, 4, 40, 0, 1],
                [1, 4, 1, 11, 0, 5, 45, 1, 1],
                [5, 6, 2, 12, 0, 6, 51, 2, 1],
                [11, 70, 3, 13, 0, 8, 80, 3, 2],
            ],
            dtype=torch.int32,
        )
        observed["row_plan"] = row_plan.cpu()
        return row_plan, (
            torch.zeros(total_rows, dtype=torch.int32),
            torch.zeros(total_rows, dtype=torch.int32),
            torch.zeros(total_rows, dtype=torch.int32),
            torch.full((total_rows,), 8, dtype=torch.int32),
            torch.full((total_rows,), 8, dtype=torch.int32),
            torch.ones(total_rows, dtype=torch.int32),
        )

    def fake_score_topk_tile_plan(
        *,
        tile_plan: torch.Tensor,
        total_rows: int,
        chunk_top_k: int,
        small_block_rows: int,
        large_block_rows: int,
        block_chunks: int,
        decode_block_chunks: int,
        **_: object,
    ):
        assert total_rows == sum(q_lens)
        assert small_block_rows == 4
        assert large_block_rows == 64
        assert block_chunks == 128
        assert decode_block_chunks == 128
        observed["tile_plan"] = tile_plan.cpu()
        return (
            torch.zeros(total_rows, chunk_top_k, dtype=torch.int32),
            torch.full((total_rows,), chunk_top_k, dtype=torch.int32),
            None,
        )

    def fake_build_score_tile_plan_triton(
        *,
        row_plan_with_tiles: torch.Tensor,
        total_tiles: int,
        max_tiles_per_row_plan: int,
        small_block_rows: int,
        large_block_rows: int,
        block_chunks: int,
        decode_block_chunks: int,
    ):
        assert total_tiles == 5
        assert max_tiles_per_row_plan == 2
        assert small_block_rows == 4
        assert large_block_rows == 64
        assert block_chunks == 128
        assert decode_block_chunks == 128
        torch.testing.assert_close(row_plan_with_tiles.cpu(), observed["row_plan"])
        return torch.tensor(
            [
                [0, 1, 0, 0, 4, 40, 0, 0],
                [1, 4, 1, 0, 5, 45, 0, 1],
                [5, 6, 2, 0, 6, 51, 0, 2],
                [11, 64, 3, 0, 8, 80, 0, 2],
                [75, 6, 3, 0, 8, 144, 0, 2],
            ],
            dtype=torch.int32,
        )

    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_metadata_triton",
        fake_build_score_metadata_triton,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_build_score_tile_plan_triton",
        fake_build_score_tile_plan_triton,
    )
    monkeypatch.setattr(
        efficient_components,
        "dsa_batched_score_topk_tile_plan_triton",
        fake_score_topk_tile_plan,
    )
    bundle = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=16,
        chunk_size=8,
        num_kv_heads=1,
        head_dim=16,
        logit_scale=1.0,
        chunk_top_k=2,
    )

    actual = bundle.try_select_blocks_batched(
        indexer_q=indexer_q,
        sparse_infos=sparse_infos,
        batched_chunk_representatives=representative_state,
        block_table=block_table,
        query_start_loc=torch.tensor([0, 1, 5, 11, 81], dtype=torch.int32),
        seq_lens=torch.tensor([41, 49, 57, 150], dtype=torch.int32),
        num_actual_tokens=sum(q_lens),
        active_seq_count=4,
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )

    assert set(actual) == {10, 11, 12, 13}
    torch.testing.assert_close(
        observed["row_plan"],
        torch.tensor(
            [
                [0, 1, 0, 10, 0, 4, 40, 0, 1],
                [1, 4, 1, 11, 0, 5, 45, 1, 1],
                [5, 6, 2, 12, 0, 6, 51, 2, 1],
                [11, 70, 3, 13, 0, 8, 80, 3, 2],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        observed["tile_plan"],
        torch.tensor(
            [
                [0, 1, 0, 0, 4, 40, 0, 0],
                [1, 4, 1, 0, 5, 45, 0, 1],
                [5, 6, 2, 0, 6, 51, 0, 2],
                [11, 64, 3, 0, 8, 80, 0, 2],
                [75, 6, 3, 0, 8, 144, 0, 2],
            ],
            dtype=torch.int32,
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for trial DSA batched scoring",
)
def test_trial_batched_dsa_scoring_matches_current_efficient_variable_batch():
    case = {
        "batch_size": 12,
        "q_lens": _mixed_q_lens(
            12,
            profile=(1, 4, 4, 17, 33, 64, 96),
            shuffle_seed=31991,
        ),
        "num_kv_heads": 4,
        "q_indexer_dim": 7,
        "chunk_size": 8,
        "logit_scale": 1.25,
        "seed": 62003,
    }
    inputs = _make_trial_inputs(
        **{
            k: case[k]
            for k in (
                "batch_size",
                "q_lens",
                "num_kv_heads",
                "q_indexer_dim",
                "chunk_size",
                "seed",
            )
        }
    )

    actual = _run_trial_batched_scoring(
        flat_queries=inputs["flat_queries"],
        representatives=inputs["representatives"],
        row_current_chunks=inputs["row_current_chunks"],
        row_seq_ids=inputs["row_seq_ids"],
        row_group_ids=inputs["row_group_ids"],
        row_prior_chunks=inputs["row_prior_chunks"],
        q_indexer_dim=case["q_indexer_dim"],
        logit_scale=case["logit_scale"],
    )
    expected = _current_efficient_scoring_reference(
        representatives=inputs["representatives"],
        indexer_queries_by_seq=inputs["indexer_queries_by_seq"],
        q_lens=inputs["q_lens"],
        key_lens=inputs["key_lens"],
        chunk_size=case["chunk_size"],
        logit_scale=case["logit_scale"],
    )

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for DSA batched scoring",
)
def test_efficient_provider_batched_triton_selection_matches_loop(monkeypatch):
    case = {
        "batch_size": 10,
        "q_lens": _mixed_q_lens(
            10,
            profile=(1, 3, 9, 17, 41),
            shuffle_seed=117,
        ),
        "num_kv_heads": 1,
        "q_indexer_dim": 7,
        "chunk_size": 8,
        "logit_scale": 1.25,
        "seed": 9211,
        "chunk_top_k": 3,
    }
    inputs = _make_trial_inputs(
        **{
            k: case[k]
            for k in (
                "batch_size",
                "q_lens",
                "num_kv_heads",
                "q_indexer_dim",
                "chunk_size",
                "seed",
            )
        }
    )
    chunk_values = torch.arange(
        inputs["representatives"].shape[1],
        device=torch.device("cuda"),
        dtype=torch.float32,
    ).view(1, -1, 1, 1)
    dim_values = torch.arange(
        case["q_indexer_dim"],
        device=torch.device("cuda"),
        dtype=torch.float32,
    ).view(1, 1, 1, -1)
    inputs["representatives"].copy_(chunk_values * 10.0 + dim_values)
    for queries in inputs["indexer_queries_by_seq"]:
        queries.fill_(1.0)

    monkeypatch.delenv("VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING", raising=False)
    bundle = efficient_components.EfficientChunkedDSAProviderBundle(
        q_indexer_dim=case["q_indexer_dim"],
        chunk_size=case["chunk_size"],
        num_kv_heads=1,
        head_dim=16,
        logit_scale=case["logit_scale"],
        chunk_top_k=case["chunk_top_k"],
    )
    assert bundle.q_indexer_use_triton_scoring

    sparse_infos: list[tuple[int, int, int, int, int, int, torch.Tensor]] = []
    representatives_by_seq: dict[int, torch.Tensor] = {}
    q_start = 0
    query_start_locs = [0]
    for seq_idx, (q_len, key_len, num_chunks) in enumerate(
        zip(
            inputs["q_lens"],
            inputs["key_lens"],
            inputs["num_chunks_by_seq"],
        )
    ):
        q_end = q_start + q_len
        query_position_start = key_len - q_len
        seq_positions = torch.arange(
            query_position_start,
            key_len,
            device=torch.device("cuda"),
            dtype=torch.long,
        )
        current_chunks = torch.div(
            seq_positions,
            case["chunk_size"],
            rounding_mode="floor",
        ).clamp(min=0, max=max(num_chunks - 1, 0))
        sparse_infos.append(
            (
                seq_idx,
                q_start,
                q_end,
                key_len,
                num_chunks,
                query_position_start,
                current_chunks,
            )
        )
        representatives_by_seq[seq_idx] = inputs["representatives"][
            seq_idx, :num_chunks
        ]
        q_start = q_end
        query_start_locs.append(q_start)

    indexer_q = torch.cat(inputs["indexer_queries_by_seq"], dim=0).contiguous()
    max_chunks = max(inputs["num_chunks_by_seq"])
    block_table = torch.arange(
        case["batch_size"] * max_chunks,
        device=torch.device("cuda"),
        dtype=torch.int32,
    ).view(case["batch_size"], max_chunks)

    actual = bundle.try_select_blocks_batched(
        indexer_q=indexer_q,
        sparse_infos=sparse_infos,
        batched_chunk_representatives=representatives_by_seq,
        block_table=block_table,
        query_start_loc=torch.tensor(
            query_start_locs,
            device=torch.device("cuda"),
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor(
            inputs["key_lens"],
            device=torch.device("cuda"),
            dtype=torch.int32,
        ),
        num_actual_tokens=sum(inputs["q_lens"]),
        active_seq_count=case["batch_size"],
        dense_decode_threshold=-1,
        dense_prefill_threshold=-1,
    )
    if actual is None:
        pytest.skip("batched Triton scoring provider is unavailable")

    assert set(actual) == {info[0] for info in sparse_infos}
    for (
        seq_idx,
        q_start,
        q_end,
        _,
        num_chunks,
        _,
        current_chunks,
    ) in sparse_infos:
        max_prior_chunks = max(num_chunks - 1, 0)
        chunk_top_k = min(case["chunk_top_k"], max_prior_chunks)
        if max_prior_chunks <= 0 or chunk_top_k <= 0:
            assert actual[seq_idx] is None
            continue

        expected = bundle.select_blocks(
            score_query_states=indexer_q[q_start:q_end, 0],
            representative_state=representatives_by_seq[seq_idx],
            current_chunks=current_chunks,
            max_prior_chunks=max_prior_chunks,
            block_top_k=chunk_top_k,
            block_table=block_table[seq_idx],
            chunk_ids=torch.arange(
                max_prior_chunks,
                device=torch.device("cuda"),
                dtype=current_chunks.dtype,
            ),
            seq_idx=seq_idx,
            group_idx=0,
        )
        actual_selected = bundle.get_selected_blocks(actual[seq_idx])
        expected_selected = bundle.get_selected_blocks(expected)
        torch.testing.assert_close(actual_selected[1], expected_selected[1])
        actual_sorted = (
            actual_selected[0].masked_fill(~actual_selected[1], -1).sort(dim=-1).values
        )
        expected_sorted = (
            expected_selected[0]
            .masked_fill(~expected_selected[1], -1)
            .sort(dim=-1)
            .values
        )
        torch.testing.assert_close(
            actual_sorted,
            expected_sorted,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or triton is None,
    reason="CUDA and Triton are required for trial DSA batched scoring",
)
def test_trial_batched_dsa_scoring_runtime_args_do_not_recompile():
    kernel = getattr(
        nemotron_h_dsa_triton_scoring,
        "_dsa_batched_chunk_score_kernel",
        None,
    )
    assert kernel is not None
    if not hasattr(kernel, "device_caches"):
        pytest.skip("Triton JIT cache introspection is unavailable")

    cases = [
        {
            "batch_size": 5,
            "q_lens": [1, 4, 17, 8, 33],
            "num_kv_heads": 2,
            "q_indexer_dim": 6,
            "chunk_size": 4,
            "logit_scale": 1.0,
            "seed": 7001,
        },
        {
            "batch_size": 9,
            "q_lens": _mixed_q_lens(
                9,
                profile=(1, 4, 64, 8, 33),
                shuffle_seed=17,
            ),
            "num_kv_heads": 3,
            "q_indexer_dim": 7,
            "chunk_size": 8,
            "logit_scale": 0.75,
            "seed": 7002,
        },
        {
            "batch_size": 7,
            "q_lens": _mixed_q_lens(
                7,
                profile=(1, 4, 96, 17),
                shuffle_seed=23,
            ),
            "num_kv_heads": 4,
            "q_indexer_dim": 5,
            "chunk_size": 4,
            "logit_scale": 1.5,
            "seed": 7003,
        },
    ]

    _clear_triton_kernel_cache(kernel)
    cache_sizes: list[int] = []
    for case in cases:
        inputs = _make_trial_inputs(
            **{
                k: case[k]
                for k in (
                    "batch_size",
                    "q_lens",
                    "num_kv_heads",
                    "q_indexer_dim",
                    "chunk_size",
                    "seed",
                )
            }
        )
        actual = _run_trial_batched_scoring(
            flat_queries=inputs["flat_queries"],
            representatives=inputs["representatives"],
            row_current_chunks=inputs["row_current_chunks"],
            row_seq_ids=inputs["row_seq_ids"],
            row_group_ids=inputs["row_group_ids"],
            row_prior_chunks=inputs["row_prior_chunks"],
            q_indexer_dim=case["q_indexer_dim"],
            logit_scale=case["logit_scale"],
        )
        expected = _current_efficient_scoring_reference(
            representatives=inputs["representatives"],
            indexer_queries_by_seq=inputs["indexer_queries_by_seq"],
            q_lens=inputs["q_lens"],
            key_lens=inputs["key_lens"],
            chunk_size=case["chunk_size"],
            logit_scale=case["logit_scale"],
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.cuda.synchronize()

        cache_size = _triton_kernel_cache_size(kernel)
        assert cache_size is not None
        cache_sizes.append(cache_size)

    assert cache_sizes[0] == 1
    assert cache_sizes == [cache_sizes[0]] * len(cache_sizes)
