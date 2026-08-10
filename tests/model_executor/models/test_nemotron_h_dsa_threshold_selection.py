# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient import (
    _EfficientChunkBlockSelection,
    _EfficientChunkScores,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    TorchChunkedDSABlockTableProvider,
    _TorchChunkBlockSelection,
    _TorchChunkScores,
)
from vllm.model_executor.models.nemotron_h_dsa_threshold_selection import (
    ThresholdChunkedDSABlockSelectionProvider,
    triton,
)


def test_threshold_selector_compacts_torch_scores_with_capacity():
    provider = ThresholdChunkedDSABlockSelectionProvider(
        base_threshold=0.5,
        prefer_triton=False,
    )
    score_state = _TorchChunkScores(
        chunk_logits=torch.tensor(
            [
                [0.1, 0.7, 0.9, 0.8],
                [0.6, 0.2, 0.95, 0.75],
            ],
            dtype=torch.float32,
        ),
        chunk_valid=torch.tensor(
            [
                [True, True, False, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )

    result = provider(
        score_state=score_state,
        block_top_k=3,
        max_prior_chunks=4,
    )

    assert isinstance(result, _TorchChunkBlockSelection)
    selected, selected_valid = provider.get_selected_blocks(result)
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [1, 3, 0],
                [0, 2, 3],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        selected_valid,
        torch.tensor(
            [
                [True, True, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        ),
    )


def test_threshold_selector_can_use_row_length_schedule():
    provider = ThresholdChunkedDSABlockSelectionProvider(
        base_threshold=0.0,
        length_scale=0.1,
        length_source="row_chunks",
        prefer_triton=False,
    )
    score_state = _TorchChunkScores(
        chunk_logits=torch.tensor(
            [
                [0.05, 0.11, 0.2],
                [0.25, 0.31, 0.29],
            ],
            dtype=torch.float32,
        ),
        chunk_valid=torch.ones(2, 3, dtype=torch.bool),
    )

    result = provider(
        score_state=score_state,
        block_top_k=3,
        current_chunks=torch.tensor([0, 2], dtype=torch.long),
        max_prior_chunks=3,
    )

    selected, selected_valid = provider.get_selected_blocks(result)
    torch.testing.assert_close(
        selected,
        torch.tensor(
            [
                [1, 2, 0],
                [1, 0, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        selected_valid,
        torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ],
            dtype=torch.bool,
        ),
    )


def test_threshold_selection_state_feeds_torch_block_table_provider():
    selection_provider = ThresholdChunkedDSABlockSelectionProvider(
        base_threshold=0.5,
        prefer_triton=False,
    )
    score_state = _TorchChunkScores(
        chunk_logits=torch.tensor(
            [
                [0.6, 0.7],
                [0.4, 0.9],
            ],
            dtype=torch.float32,
        ),
        chunk_valid=torch.ones(2, 2, dtype=torch.bool),
    )
    selection_state = selection_provider(
        score_state=score_state,
        block_top_k=2,
        max_prior_chunks=2,
    )
    block_table_provider = TorchChunkedDSABlockTableProvider()

    block_table_state = block_table_provider(
        block_table=torch.tensor([10, 11, 12, 13], dtype=torch.int32),
        chunk_size=4,
        key_len=10,
        q_len=2,
        dense=False,
        mode="prefill",
        selection_state=selection_state,
        current_chunks=torch.tensor([2, 2], dtype=torch.long),
        query_position_start=8,
    )

    assert block_table_provider.is_available(block_table_state)
    page_table = block_table_provider.get_page_table(block_table_state)
    assert page_table is not None
    pages, request_lens, seqused_k, max_seqlen_q, max_seqlen_k = page_table
    torch.testing.assert_close(
        pages,
        torch.tensor(
            [
                [10, 11, 12],
                [11, 12, 0],
            ],
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(request_lens, torch.ones(2, dtype=torch.int32))
    torch.testing.assert_close(seqused_k, torch.tensor([9, 6], dtype=torch.int32))
    assert max_seqlen_q == 1
    assert max_seqlen_k == 10


def test_threshold_selector_returns_efficient_selection_for_efficient_scores():
    provider = ThresholdChunkedDSABlockSelectionProvider(
        base_threshold=0.5,
        prefer_triton=False,
    )
    score_state = _EfficientChunkScores(
        chunk_logits=torch.tensor([[0.25, 0.75]], dtype=torch.float32),
        chunk_valid=torch.tensor([[True, True]], dtype=torch.bool),
    )

    result = provider(
        score_state=score_state,
        block_top_k=2,
        max_prior_chunks=2,
    )

    assert isinstance(result, _EfficientChunkBlockSelection)
    selected, selected_valid = provider.get_selected_blocks(result)
    torch.testing.assert_close(selected, torch.tensor([[1, 0]]))
    torch.testing.assert_close(
        selected_valid,
        torch.tensor([[True, False]], dtype=torch.bool),
    )


def test_triton_compactor_matches_torch_fallback_on_cuda():
    if not torch.cuda.is_available() or triton is None:
        pytest.skip("CUDA and Triton are required for the compact kernel")

    provider = ThresholdChunkedDSABlockSelectionProvider(prefer_triton=True)
    mask = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, True, False],
        ],
        device="cuda",
        dtype=torch.bool,
    )

    actual, actual_valid = provider._compact_selected_mask(mask, capacity=2)
    expected = provider._compact_selected_mask_torch(mask, capacity=2)
    expected_valid = torch.tensor(
        [
            [True, True],
            [True, True],
        ],
        device="cuda",
        dtype=torch.bool,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_valid, expected_valid)
