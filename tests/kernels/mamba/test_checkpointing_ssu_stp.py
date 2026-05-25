# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""STP parity tests for FlashInfer checkpointing SSU.

These tests compare the new FlashInfer ``checkpointing_ssu`` path against the
old FlashInfer ``selective_state_update`` path in the single-token decode flow
that vLLM uses without speculative decoding.
"""

from __future__ import annotations

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import FlashInferSSUBackend
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_FLASHINFER = False


requires_flashinfer = pytest.mark.skipif(
    not HAS_FLASHINFER or not torch.cuda.is_available(),
    reason="FlashInfer Mamba kernels require CUDA and flashinfer",
)


def _make_backend(
    *,
    checkpoint_interval: int = 1,
    enable_stochastic_rounding: bool = False,
) -> FlashInferSSUBackend:
    return FlashInferSSUBackend(
        MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            checkpoint_interval=checkpoint_interval,
            enable_stochastic_rounding=enable_stochastic_rounding,
            stochastic_rounding_philox_rounds=10,
        )
    )


def _make_weights(
    *,
    nheads: int,
    head_dim: int,
    dstate: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # A is per-head and broadcast over the head_dim/dstate axes.
    A = -torch.rand(nheads, device=device, dtype=torch.float32)
    A = A[:, None, None].expand(nheads, head_dim, dstate)
    D = torch.zeros(nheads, device=device, dtype=dtype)[:, None].expand(
        nheads, head_dim
    )
    dt_bias = torch.zeros(nheads, device=device, dtype=dtype)[:, None].expand(
        nheads, head_dim
    )
    return A, D, dt_bias


def _make_nonzero_weights(
    *,
    nheads: int,
    head_dim: int,
    dstate: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    A = -torch.rand(
        nheads,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    A = A[:, None, None].expand(nheads, head_dim, dstate)
    D = (0.1 * torch.randn(
        nheads,
        device=device,
        dtype=dtype,
        generator=generator,
    ))[:, None].expand(nheads, head_dim)
    dt_bias = (0.1 * torch.randn(
        nheads,
        device=device,
        dtype=dtype,
        generator=generator,
    ))[:, None].expand(nheads, head_dim)
    return A, D, dt_bias


def _make_checkpointing_cache(
    *,
    cache_size: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    max_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    return {
        "old_x": torch.zeros(
            cache_size,
            max_window,
            nheads,
            head_dim,
            device=device,
            dtype=dtype,
        ),
        "old_B": torch.zeros(
            cache_size,
            2,
            max_window,
            ngroups,
            dstate,
            device=device,
            dtype=dtype,
        ),
        "old_dt": torch.zeros(
            cache_size,
            2,
            nheads,
            max_window,
            device=device,
            dtype=torch.float32,
        ),
        "old_cumAdt": torch.zeros(
            cache_size,
            2,
            nheads,
            max_window,
            device=device,
            dtype=torch.float32,
        ),
        "cache_buf_idx": torch.zeros(cache_size, device=device, dtype=torch.int32),
        "prev_num_accepted_tokens": torch.zeros(
            cache_size, device=device, dtype=torch.int32
        ),
    }


def _make_decode_inputs(
    *,
    batch_size: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = 0.1 * torch.randn(
        batch_size,
        nheads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    dt_base = 0.1 * torch.randn(
        batch_size,
        nheads,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    dt = dt_base.unsqueeze(-1).expand(batch_size, nheads, head_dim)
    B = 0.1 * torch.randn(
        batch_size,
        ngroups,
        dstate,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    C = 0.1 * torch.randn(
        batch_size,
        ngroups,
        dstate,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return x, dt, B, C


def _make_strided_decode_inputs(
    *,
    batch_size: int,
    row_stride: int,
    nheads: int,
    head_dim: int,
    dstate: int,
    ngroups: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_dense, dt_dense, B_dense, C_dense = _make_decode_inputs(
        batch_size=batch_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    if scale != 0.1:
        factor = scale / 0.1
        x_dense = x_dense * factor
        dt_dense = dt_dense * factor
        B_dense = B_dense * factor
        C_dense = C_dense * factor

    x_base = torch.empty((batch_size, row_stride), device=device, dtype=dtype)
    dt_base = torch.empty((batch_size, row_stride), device=device, dtype=dtype)
    B_base = torch.empty((batch_size, row_stride), device=device, dtype=dtype)
    C_base = torch.empty((batch_size, row_stride), device=device, dtype=dtype)
    x = torch.as_strided(x_base, x_dense.shape, (row_stride, head_dim, 1))
    dt = torch.as_strided(dt_base, dt_dense.shape, (row_stride, 1, 0))
    B = torch.as_strided(B_base, B_dense.shape, (row_stride, dstate, 1))
    C = torch.as_strided(C_base, C_dense.shape, (row_stride, dstate, 1))
    x.copy_(x_dense)
    dt_base[:, :nheads].copy_(dt_dense[..., 0])
    B.copy_(B_dense)
    C.copy_(C_dense)
    return x, dt, B, C


def _call_backend(
    backend: FlashInferSSUBackend,
    *,
    state: torch.Tensor,
    cache: dict[str, torch.Tensor] | None,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    state_batch_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    dst_state_batch_indices: torch.Tensor | None = None,
    max_seqlen: int = 1,
) -> torch.Tensor:
    out = torch.empty_like(x)
    kwargs = {}
    if cache is not None:
        kwargs = {
            "old_x": cache["old_x"],
            "old_B": cache["old_B"],
            "old_dt": cache["old_dt"],
            "old_cumAdt": cache["old_cumAdt"],
            "cache_buf_idx": cache["cache_buf_idx"],
            "prev_num_accepted_tokens": cache["prev_num_accepted_tokens"],
        }
    backend(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=NULL_BLOCK_ID,
        out=out,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        **kwargs,
    )
    return out


@requires_flashinfer
@pytest.mark.parametrize("metadata_max_seqlen", [1, 8])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("use_z", [False, True])
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_outputs_match_old_flashinfer(
    batch_size: int,
    metadata_max_seqlen: int,
    use_z: bool,
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """The new STP path should produce the same token outputs as old FI SSU.

    With ``checkpoint_interval=1``, checkpointing SSU still keeps one replay
    token between calls, so the HBM state is not expected to equal the old
    materialized state at every step. The observable decode outputs should
    match, and this test exercises several consecutive STP decode iterations
    so the replay tracker is used.
    """

    torch.manual_seed(0)
    device = torch.device("cuda")
    cache_size = 8
    nheads = 4
    head_dim = 64
    dstate = 128
    ngroups = 1
    max_window = 1

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )

    initial_state = 0.1 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )

    slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    state_index_table = torch.stack((slots - 1, slots), dim=1)
    state_batch_indices = state_index_table[:, 1:2]
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)

    for step in range(5):
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=100 + step,
        )
        z = torch.tanh(2 * x) if use_z else None
        torch.manual_seed(1000 + step)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            z=z,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=metadata_max_seqlen,
        )
        torch.manual_seed(1000 + step)
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            z=z,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=metadata_max_seqlen,
        )

        torch.testing.assert_close(
            new_out.float(),
            old_out.float(),
            atol=3e-2,
            rtol=3e-2,
            msg=f"STP output mismatch at decode step {step}",
        )


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_copies_replay_state_to_destination_slot(
    dtype: torch.dtype,
) -> None:
    """Slot moves must preserve checkpoint state and live replay buffers."""

    torch.manual_seed(0)
    device = torch.device("cuda")
    cache_size = 5
    nheads = 4
    head_dim = 64
    dstate = 128
    ngroups = 1
    max_window = 1

    old_backend = _make_backend()
    new_backend = _make_backend()
    initial_state = 0.1 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    src = torch.tensor([1], device=device, dtype=torch.int32)
    dst = torch.tensor([2], device=device, dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], device=device, dtype=torch.int32)

    x, dt, B, C = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=99,
    )
    _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=src,
        cu_seqlens=cu_seqlens,
    )
    _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=src,
        cu_seqlens=cu_seqlens,
    )
    assert int(cache["prev_num_accepted_tokens"][1].item()) == 1
    src_slot = int(src.item())
    source_snapshot = (
        new_state[src_slot].clone(),
        cache["old_x"][src_slot].clone(),
        cache["old_B"][src_slot].clone(),
        cache["old_dt"][src_slot].clone(),
        cache["old_cumAdt"][src_slot].clone(),
        cache["cache_buf_idx"][src_slot].clone(),
        cache["prev_num_accepted_tokens"][src_slot].clone(),
    )

    x, dt, B, C = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=100,
    )
    old_out = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        cu_seqlens=cu_seqlens,
    )
    new_out = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=src,
        dst_state_batch_indices=dst,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(new_out.float(), old_out.float(), atol=3e-2, rtol=3e-2)
    for actual, expected in zip(
        (
            new_state[src_slot],
            cache["old_x"][src_slot],
            cache["old_B"][src_slot],
            cache["old_dt"][src_slot],
            cache["old_cumAdt"][src_slot],
            cache["cache_buf_idx"][src_slot],
            cache["prev_num_accepted_tokens"][src_slot],
        ),
        source_snapshot,
    ):
        torch.testing.assert_close(actual, expected)


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_large_batch_outputs_match_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """Large CUDA-graph decode batches must match the old FI STP path."""

    device = torch.device("cuda")
    batch_size = 24
    cache_size = batch_size * 2 + 8
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    generator = torch.Generator(device=device).manual_seed(123)

    old_backend = _make_backend()
    new_backend = _make_backend(checkpoint_interval=max_window)
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    state_batch_indices = slots[:, None]
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)

    for step in range(12):
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=200 + step,
        )
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=8,
        )
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_batch_indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=8,
        )
        max_abs_error = (new_out.float() - old_out.float()).abs().max()
        assert max_abs_error <= 6e-2, (
            f"large-batch STP output mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][slots.to(torch.long)]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_large_batch_slot_moves_match_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """Large STP decode with separate source/destination cache slots."""

    device = torch.device("cuda")
    batch_size = 8
    cache_size = batch_size * 2 + 8
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    generator = torch.Generator(device=device).manual_seed(321)

    old_backend = _make_backend()
    new_backend = _make_backend(checkpoint_interval=max_window)
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    odd_slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    even_slots = odd_slots + 1
    src_slots = odd_slots
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)

    for step in range(8):
        dst_slots = even_slots if step % 2 == 0 else odd_slots
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=500 + step,
        )
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=cu_seqlens,
        )
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=cu_seqlens,
        )
        max_abs_error = (new_out.float() - old_out.float()).abs().max()
        assert max_abs_error <= 6e-2, (
            f"slot-move STP output mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][dst_slots.to(torch.long)]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked
        src_slots = dst_slots


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_window6_strided_long_decode_matches_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Production-like STP decode should match over a long replay lifecycle.

    This is the cheap detector for the GSM8K failure mode: window=6 grouped
    replay, non-zero D/dt_bias, padded row strides, slot moves, and many decode
    steps after a prefill-like reset.
    """

    device = torch.device("cuda")
    batch_size = 12
    cache_size = batch_size * 2 + 8
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(9753)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        checkpoint_interval=max_window,
        enable_stochastic_rounding=enable_stochastic_rounding,
    )
    initial_state = 0.02 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=9754,
    )
    odd_slots = torch.arange(batch_size, device=device, dtype=torch.int32) * 2 + 1
    even_slots = odd_slots + 1
    src_slots = odd_slots
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)

    for step in range(48):
        dst_slots = even_slots if step % 2 == 0 else odd_slots
        x, dt, B, C = _make_strided_decode_inputs(
            batch_size=batch_size,
            row_stride=row_stride,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=9800 + step,
            scale=0.2,
        )
        torch.manual_seed(9900 + step)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=cu_seqlens,
        )
        torch.manual_seed(9900 + step)
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=cu_seqlens,
        )
        max_abs_error = (new_out.float() - old_out.float()).abs().max()
        assert max_abs_error <= 6e-2, (
            f"window-6 strided STP output mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][dst_slots.to(torch.long)]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked
        src_slots = dst_slots


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_window6_padded_graph_batch_matches_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Window-6 STP replay must ignore inactive CUDA-graph padding rows."""

    device = torch.device("cuda")
    active_batch_size = 12
    padded_batch_size = 32
    cache_size = active_batch_size * 2 + 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(7531)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        checkpoint_interval=max_window,
        enable_stochastic_rounding=enable_stochastic_rounding,
    )
    initial_state = 0.02 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=7532,
    )
    odd_slots = torch.arange(
        active_batch_size, device=device, dtype=torch.int32
    ) * 2 + 1
    even_slots = odd_slots + 1
    active_src = odd_slots

    for step in range(24):
        active_dst = even_slots if step % 2 == 0 else odd_slots
        src_slots = torch.full(
            (padded_batch_size,),
            NULL_BLOCK_ID,
            device=device,
            dtype=torch.int32,
        )
        dst_slots = torch.full_like(src_slots, NULL_BLOCK_ID)
        src_slots[:active_batch_size] = active_src
        dst_slots[:active_batch_size] = active_dst
        x, dt, B, C = _make_strided_decode_inputs(
            batch_size=padded_batch_size,
            row_stride=row_stride,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=7600 + step,
            scale=0.2,
        )
        torch.manual_seed(7700 + step)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=None,
        )
        torch.manual_seed(7700 + step)
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=None,
        )
        max_abs_error = (
            new_out[:active_batch_size].float()
            - old_out[:active_batch_size].float()
        ).abs().max()
        assert max_abs_error <= 6e-2, (
            f"window-6 padded graph-batch STP output mismatch at step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][active_dst.to(torch.long)]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked
        active_src = active_dst


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_cudagraph_replay_block_table_matches_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """CUDA graph replay with server-style block-table gathers matches old FI.

    Serving does not only call the op eagerly: it captures persistent Mamba
    metadata buffers, gathers source/destination slots from them inside the
    graph, and then replays the graph with new tensor contents. This catches
    bugs that eager old-vs-new tests miss.
    """

    device = torch.device("cuda")
    active_batch_size = 12
    padded_batch_size = 32
    cache_size = active_batch_size * 2 + 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    generator = torch.Generator(device=device).manual_seed(2468)

    old_backend = _make_backend()
    new_backend = _make_backend(checkpoint_interval=max_window)
    initial_state = 0.02 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=2469,
    )

    odd_slots = torch.arange(
        active_batch_size, device=device, dtype=torch.int32
    ) * 2 + 1
    even_slots = odd_slots + 1
    state_index_table = torch.full(
        (padded_batch_size, 2),
        NULL_BLOCK_ID,
        device=device,
        dtype=torch.int32,
    )
    state_index_table[:active_batch_size, 0] = odd_slots
    state_index_table[:active_batch_size, 1] = even_slots
    src_block_idx = torch.zeros(padded_batch_size, device=device, dtype=torch.int64)
    dst_block_idx = torch.ones(padded_batch_size, device=device, dtype=torch.int64)

    static_x = torch.empty(
        padded_batch_size, nheads, head_dim, device=device, dtype=dtype
    )
    static_dt_base = torch.empty(
        padded_batch_size, nheads, device=device, dtype=dtype
    )
    static_dt = static_dt_base[:, :, None].expand(-1, -1, head_dim)
    static_B = torch.empty(
        padded_batch_size, ngroups, dstate, device=device, dtype=dtype
    )
    static_C = torch.empty_like(static_B)
    graph_out = torch.empty_like(static_x)

    warm_x, warm_dt, warm_B, warm_C = _make_decode_inputs(
        batch_size=padded_batch_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=2470,
    )
    warm_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
    warm_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
    warm_state = new_state.clone()
    warm_cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    _call_backend(
        new_backend,
        state=warm_state,
        cache=warm_cache,
        x=warm_x,
        dt=warm_dt,
        A=A,
        B=warm_B,
        C=warm_C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=warm_src,
        dst_state_batch_indices=warm_dst,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
        graph_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
        new_backend(
            new_state,
            static_x,
            static_dt,
            A,
            static_B,
            static_C,
            D,
            dt_bias,
            dt_softplus=True,
            state_batch_indices=graph_src,
            dst_state_batch_indices=graph_dst,
            null_block_id=NULL_BLOCK_ID,
            out=graph_out,
            old_x=cache["old_x"],
            old_B=cache["old_B"],
            old_dt=cache["old_dt"],
            old_cumAdt=cache["old_cumAdt"],
            cache_buf_idx=cache["cache_buf_idx"],
            prev_num_accepted_tokens=cache["prev_num_accepted_tokens"],
        )

    # Capture executes the body once. Real serving should start requests from
    # prefill-initialized slots, so reset this synthetic state before replaying.
    new_state.copy_(initial_state)
    for tensor in cache.values():
        tensor.zero_()

    active_src_col = 0
    for step in range(24):
        active_dst_col = 1 - active_src_col
        src_block_idx.fill_(active_src_col)
        dst_block_idx.fill_(active_dst_col)
        x, dt, B, C = _make_decode_inputs(
            batch_size=padded_batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=2500 + step,
        )
        static_x.copy_(x)
        static_dt_base.copy_(dt[..., 0])
        static_B.copy_(B)
        static_C.copy_(C)

        src_slots = state_index_table[:, active_src_col]
        dst_slots = state_index_table[:, active_dst_col]
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=static_x,
            dt=static_dt,
            A=A,
            B=static_B,
            C=static_C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=src_slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=None,
        )
        graph.replay()
        torch.cuda.synchronize()
        max_abs_error = (
            graph_out[:active_batch_size].float()
            - old_out[:active_batch_size].float()
        ).abs().max()
        assert max_abs_error <= 6e-2, (
            f"CUDA graph STP replay mismatch at step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][
            dst_slots[:active_batch_size].to(torch.long)
        ]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked
        active_src_col = active_dst_col


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_server_like_2d_indices_match_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Server STP passes state indices as [batch, 1] for mamba_cache_mode=none.

    Keep this at the production STP window size.  Window 1 has a known direct
    FlashInfer-kernel parity issue; the STP branch target is interval/window 6.
    """

    device = torch.device("cuda")
    active_batch_size = 12
    padded_batch_size = 32
    cache_size = active_batch_size * 2 + 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(3571)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        checkpoint_interval=max_window,
        enable_stochastic_rounding=enable_stochastic_rounding,
    )
    initial_state = 0.02 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=3572,
    )
    active_slots = torch.arange(
        active_batch_size, device=device, dtype=torch.int32
    ) + 1

    for step in range(24):
        slots = torch.full(
            (padded_batch_size, 1),
            NULL_BLOCK_ID,
            device=device,
            dtype=torch.int32,
        )
        slots[:active_batch_size, 0] = active_slots
        x, dt, B, C = _make_strided_decode_inputs(
            batch_size=padded_batch_size,
            row_stride=row_stride,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=3600 + step,
            scale=0.2,
        )
        torch.manual_seed(3700 + step)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=slots,
            dst_state_batch_indices=slots,
            cu_seqlens=None,
        )
        torch.manual_seed(3700 + step)
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=slots,
            dst_state_batch_indices=slots,
            cu_seqlens=None,
        )
        max_abs_error = (
            new_out[:active_batch_size].float()
            - old_out[:active_batch_size].float()
        ).abs().max()
        assert max_abs_error <= 6e-2, (
            f"server-like 2D-index STP mismatch at step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][active_slots.to(torch.long)]
        expected_tracked = step % max_window + 1
        assert int(tracked_tokens.max().item()) == expected_tracked


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_mixed_lifecycle_batch2_outputs_match_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """A newly joined request must batch with a replaying request correctly."""

    device = torch.device("cuda")
    cache_size = 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 1
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(5678)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    slot_a = torch.tensor([1], device=device, dtype=torch.int32)
    slots_ab = torch.tensor([1, 7], device=device, dtype=torch.int32)
    cu_a = torch.tensor([0, 1], device=device, dtype=torch.int32)
    cu_ab = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)

    x_a, dt_a, B_a, C_a = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=401,
    )
    old_out_a = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x_a,
        dt=dt_a,
        A=A,
        B=B_a,
        C=C_a,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slot_a,
        cu_seqlens=cu_a,
    )
    new_out_a = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x_a,
        dt=dt_a,
        A=A,
        B=B_a,
        C=C_a,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slot_a,
        cu_seqlens=cu_a,
    )
    torch.testing.assert_close(
        new_out_a.float(), old_out_a.float(), atol=6e-2, rtol=6e-2
    )
    assert int(cache["prev_num_accepted_tokens"][1].item()) == 1
    assert int(cache["prev_num_accepted_tokens"][7].item()) == 0

    x_dense, dt_dense, B_dense, C_dense = _make_decode_inputs(
        batch_size=2,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=402,
    )
    x_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    dt_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    B_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    C_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    x = torch.as_strided(x_base, x_dense.shape, (row_stride, head_dim, 1))
    dt = torch.as_strided(dt_base, dt_dense.shape, (row_stride, 1, 0))
    B = torch.as_strided(B_base, B_dense.shape, (row_stride, dstate, 1))
    C = torch.as_strided(C_base, C_dense.shape, (row_stride, dstate, 1))
    x.copy_(x_dense)
    dt_base[:, :nheads].copy_(dt_dense[..., 0])
    B.copy_(B_dense)
    C.copy_(C_dense)

    old_out = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slots_ab,
        cu_seqlens=cu_ab,
    )
    new_out = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slots_ab,
        cu_seqlens=cu_ab,
    )
    max_abs_error = (new_out.float() - old_out.float()).abs().max()
    assert max_abs_error <= 6e-2, (
        "mixed-lifecycle batch-2 STP output mismatch: "
        f"max_abs_error={max_abs_error.item()}"
    )


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_nonvarlen_mixed_lifecycle_batch2_matches_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Non-varlen decode must handle a new request batched with replay."""

    device = torch.device("cuda")
    cache_size = 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 1
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(8765)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    slot_a = torch.tensor([1], device=device, dtype=torch.int32)
    slots_ab = torch.tensor([1, 7], device=device, dtype=torch.int32)

    x_a, dt_a, B_a, C_a = _make_decode_inputs(
        batch_size=1,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=801,
    )
    old_out_a = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x_a,
        dt=dt_a,
        A=A,
        B=B_a,
        C=C_a,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slot_a,
        cu_seqlens=None,
    )
    new_out_a = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x_a,
        dt=dt_a,
        A=A,
        B=B_a,
        C=C_a,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slot_a,
        cu_seqlens=None,
    )
    torch.testing.assert_close(
        new_out_a.float(), old_out_a.float(), atol=6e-2, rtol=6e-2
    )
    assert int(cache["prev_num_accepted_tokens"][1].item()) == 1
    assert int(cache["prev_num_accepted_tokens"][7].item()) == 0

    x_dense, dt_dense, B_dense, C_dense = _make_decode_inputs(
        batch_size=2,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=802,
    )
    x_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    dt_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    B_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    C_base = torch.empty((2, row_stride), device=device, dtype=dtype)
    x = torch.as_strided(x_base, x_dense.shape, (row_stride, head_dim, 1))
    dt = torch.as_strided(dt_base, dt_dense.shape, (row_stride, 1, 0))
    B = torch.as_strided(B_base, B_dense.shape, (row_stride, dstate, 1))
    C = torch.as_strided(C_base, C_dense.shape, (row_stride, dstate, 1))
    x.copy_(x_dense)
    dt_base[:, :nheads].copy_(dt_dense[..., 0])
    B.copy_(B_dense)
    C.copy_(C_dense)

    old_out = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slots_ab,
        cu_seqlens=None,
    )
    new_out = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=slots_ab,
        cu_seqlens=None,
    )
    max_abs_error = (new_out.float() - old_out.float()).abs().max()
    assert max_abs_error <= 6e-2, (
        "non-varlen mixed-lifecycle batch-2 STP output mismatch: "
        f"max_abs_error={max_abs_error.item()}"
    )


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_vllm_padded_cache_layout_matches_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """Checkpointing cache tensors may be as_strided into padded vLLM pages."""

    device = torch.device("cuda")
    batch_size = 2
    cache_size = 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 1
    page_pad = 4096
    generator = torch.Generator(device=device).manual_seed(1357)

    old_backend = _make_backend()
    new_backend = _make_backend()
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    state_slot_el = nheads * head_dim * dstate
    raw_state = torch.empty(
        cache_size * (state_slot_el + page_pad),
        device=device,
        dtype=torch.float16,
    )
    new_state = torch.as_strided(
        raw_state,
        (cache_size, nheads, head_dim, dstate),
        (state_slot_el + page_pad, head_dim * dstate, dstate, 1),
    )
    new_state.copy_(initial_state)

    old_x_slot_el = max_window * nheads * head_dim
    old_B_slot_el = 2 * max_window * ngroups * dstate
    old_dt_slot_el = 2 * nheads * max_window
    raw_old_x = torch.zeros(
        cache_size * (old_x_slot_el + page_pad),
        device=device,
        dtype=dtype,
    )
    raw_old_B = torch.zeros(
        cache_size * (old_B_slot_el + page_pad),
        device=device,
        dtype=dtype,
    )
    raw_old_dt = torch.zeros(
        cache_size * (old_dt_slot_el + page_pad),
        device=device,
        dtype=torch.float32,
    )
    raw_old_cumAdt = torch.zeros_like(raw_old_dt)
    cache = {
        "old_x": torch.as_strided(
            raw_old_x,
            (cache_size, max_window, nheads, head_dim),
            (old_x_slot_el + page_pad, nheads * head_dim, head_dim, 1),
        ),
        "old_B": torch.as_strided(
            raw_old_B,
            (cache_size, 2, max_window, ngroups, dstate),
            (old_B_slot_el + page_pad, max_window * ngroups * dstate,
             ngroups * dstate, dstate, 1),
        ),
        "old_dt": torch.as_strided(
            raw_old_dt,
            (cache_size, 2, nheads, max_window),
            (old_dt_slot_el + page_pad, nheads * max_window, max_window, 1),
        ),
        "old_cumAdt": torch.as_strided(
            raw_old_cumAdt,
            (cache_size, 2, nheads, max_window),
            (old_dt_slot_el + page_pad, nheads * max_window, max_window, 1),
        ),
        "cache_buf_idx": torch.zeros(cache_size, device=device, dtype=torch.int32),
        "prev_num_accepted_tokens": torch.zeros(
            cache_size, device=device, dtype=torch.int32
        ),
    }
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    slots = torch.tensor([1, 7], device=device, dtype=torch.int32)

    for step in range(4):
        dst_slots = slots + 1
        x, dt, B, C = _make_decode_inputs(
            batch_size=batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=901 + step,
        )
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=None,
        )
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=slots,
            dst_state_batch_indices=dst_slots,
            cu_seqlens=None,
        )
        max_abs_error = (new_out.float() - old_out.float()).abs().max()
        assert max_abs_error <= 6e-2, (
            f"vLLM padded cache layout mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )
        slots = dst_slots


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_fixed_padded_batch_outputs_match_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Fixed CUDA-graph-shaped decode batches must ignore padded slots."""

    device = torch.device("cuda")
    active_batch_size = 2
    padded_batch_size = 8
    cache_size = 32
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 1
    generator = torch.Generator(device=device).manual_seed(1234)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
    )
    active_slots = torch.tensor([1, 5], device=device, dtype=torch.int32)
    padded_slots = torch.full(
        (padded_batch_size,), NULL_BLOCK_ID, device=device, dtype=torch.int32
    )
    padded_slots[:active_batch_size] = active_slots

    for step in range(12):
        x, dt, B, C = _make_decode_inputs(
            batch_size=padded_batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=300 + step,
        )
        torch.manual_seed(5000 + step)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=padded_slots,
            cu_seqlens=None,
        )
        torch.manual_seed(5000 + step)
        new_out = _call_backend(
            new_backend,
            state=new_state,
            cache=cache,
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=padded_slots,
            cu_seqlens=None,
        )
        max_abs_error = (
            new_out[:active_batch_size].float()
            - old_out[:active_batch_size].float()
        ).abs().max()
        assert max_abs_error <= 6e-2, (
            f"fixed padded-batch STP output mismatch at decode step {step}: "
            f"max_abs_error={max_abs_error.item()}"
        )


@requires_flashinfer
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_cudagraph_replay_varying_active_rows_matches_old_flashinfer(
    dtype: torch.dtype,
) -> None:
    """CUDA graph replay must tolerate server-like active-row churn.

    The full-graph runner captures padded Mamba tensors and then replays them
    with different active request counts and slot mappings. A fixed active set
    is too friendly: stale padded rows can hide until requests finish and new
    requests reuse lower rows in the same captured shape.
    """

    device = torch.device("cuda")
    padded_batch_size = 32
    cache_size = 160
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    generator = torch.Generator(device=device).manual_seed(7531)

    old_backend = _make_backend()
    new_backend = _make_backend(checkpoint_interval=max_window)
    initial_state = 0.015 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=7532,
    )

    state_index_table = torch.full(
        (padded_batch_size, 2),
        NULL_BLOCK_ID,
        device=device,
        dtype=torch.int32,
    )
    src_block_idx = torch.zeros(padded_batch_size, device=device, dtype=torch.int64)
    dst_block_idx = torch.ones(padded_batch_size, device=device, dtype=torch.int64)
    static_x = torch.empty(
        padded_batch_size, nheads, head_dim, device=device, dtype=dtype
    )
    static_dt_base = torch.empty(
        padded_batch_size, nheads, device=device, dtype=dtype
    )
    static_dt = static_dt_base[:, :, None].expand(-1, -1, head_dim)
    static_B = torch.empty(
        padded_batch_size, ngroups, dstate, device=device, dtype=dtype
    )
    static_C = torch.empty_like(static_B)
    graph_out = torch.empty_like(static_x)

    request_schedules = [
        list(range(32)),
        list(range(5, 32)),
        [
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            16,
            17,
            18,
            19,
            20,
        ],
        [40, 41, 42, 43, 44, 45, 46, 47],
        [
            2,
            4,
            6,
            8,
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
            44,
            46,
            48,
            50,
            52,
            54,
            56,
            58,
            60,
            62,
        ],
        [
            7,
            11,
            15,
            19,
            23,
            27,
            31,
            35,
            39,
            43,
            47,
            51,
            55,
            59,
            61,
            63,
            3,
            5,
            9,
            13,
            17,
            21,
            25,
            29,
        ],
        [64, 65, 66, 67, 68, 69, 70, 71, 72],
        list(range(24, 56)),
    ]
    req_src_col: dict[int, int] = {}

    def fill_state_table(request_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        state_index_table.fill_(NULL_BLOCK_ID)
        src_slots_cpu = []
        dst_slots_cpu = []
        for row, req_id in enumerate(request_ids):
            base_slot = req_id * 2 + 1
            src_col = req_src_col.setdefault(req_id, 0)
            src_slot = base_slot + src_col
            dst_slot = base_slot + (1 - src_col)
            state_index_table[row, 0] = src_slot
            state_index_table[row, 1] = dst_slot
            src_slots_cpu.append(src_slot)
            dst_slots_cpu.append(dst_slot)
        return (
            torch.tensor(src_slots_cpu, device=device, dtype=torch.int32),
            torch.tensor(dst_slots_cpu, device=device, dtype=torch.int32),
        )

    def advance_requests(request_ids: list[int]) -> None:
        for req_id in request_ids:
            req_src_col[req_id] = 1 - req_src_col[req_id]

    fill_state_table(request_schedules[0])
    warm_x, warm_dt, warm_B, warm_C = _make_decode_inputs(
        batch_size=padded_batch_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        device=device,
        dtype=dtype,
        seed=7533,
    )
    static_x.copy_(warm_x)
    static_dt_base.copy_(warm_dt[..., 0])
    static_B.copy_(warm_B)
    static_C.copy_(warm_C)

    warm_state = new_state.clone()
    warm_cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    warm_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
    warm_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
    _call_backend(
        new_backend,
        state=warm_state,
        cache=warm_cache,
        x=warm_x,
        dt=warm_dt,
        A=A,
        B=warm_B,
        C=warm_C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=warm_src,
        dst_state_batch_indices=warm_dst,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
        graph_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
        new_backend(
            new_state,
            static_x,
            static_dt,
            A,
            static_B,
            static_C,
            D,
            dt_bias,
            dt_softplus=True,
            state_batch_indices=graph_src,
            dst_state_batch_indices=graph_dst,
            null_block_id=NULL_BLOCK_ID,
            out=graph_out,
            old_x=cache["old_x"],
            old_B=cache["old_B"],
            old_dt=cache["old_dt"],
            old_cumAdt=cache["old_cumAdt"],
            cache_buf_idx=cache["cache_buf_idx"],
            prev_num_accepted_tokens=cache["prev_num_accepted_tokens"],
        )

    new_state.copy_(initial_state)
    for tensor in cache.values():
        tensor.zero_()
    req_src_col.clear()

    for step, request_ids in enumerate(request_schedules):
        _, dst_slots = fill_state_table(request_ids)
        x, dt, B, C = _make_decode_inputs(
            batch_size=padded_batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=7600 + step,
        )
        static_x.copy_(x)
        static_dt_base.copy_(dt[..., 0])
        static_B.copy_(B)
        static_C.copy_(C)

        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=static_x,
            dt=static_dt,
            A=A,
            B=static_B,
            C=static_C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_index_table[:, 0],
            dst_state_batch_indices=state_index_table[:, 1],
            cu_seqlens=None,
        )
        graph.replay()
        torch.cuda.synchronize()
        active_count = len(request_ids)
        max_abs_error = (
            graph_out[:active_count].float() - old_out[:active_count].float()
        ).abs().max()
        assert max_abs_error <= 6e-2, (
            f"varying-active-row CUDA graph STP mismatch at step {step}: "
            f"active_count={active_count} max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][dst_slots.to(torch.long)]
        assert int(tracked_tokens.min().item()) >= 1
        assert int(tracked_tokens.max().item()) <= max_window
        advance_requests(request_ids)


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_cudagraph_replay_conv_then_ssu_varying_active_rows_matches_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Captured conv-update plus checkpointing SSU matches old SSU.

    The serving Mamba2 decode path transforms the packed hidden/B/C tensor with
    ``causal_conv1d_update`` before it calls SSU. SSU-only graph replay can pass
    while the combined captured sequence still mishandles padded or churned
    decode rows.
    """

    device = torch.device("cuda")
    padded_batch_size = 32
    cache_size = 160
    dim = 128 * 64 + 2 * 8 * 128
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    generator = torch.Generator(device=device).manual_seed(8642)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        checkpoint_interval=max_window,
        enable_stochastic_rounding=enable_stochastic_rounding,
    )
    initial_state = 0.012 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    old_conv_state = 0.01 * torch.randn(
        cache_size,
        dim,
        3,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    new_conv_state = old_conv_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A, D, dt_bias = _make_nonzero_weights(
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        device=device,
        dtype=dtype,
        seed=8643,
    )
    conv_weight = 0.02 * torch.randn(
        dim,
        4,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    conv_bias = 0.02 * torch.randn(
        dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    state_index_table = torch.full(
        (padded_batch_size, 2),
        NULL_BLOCK_ID,
        device=device,
        dtype=torch.int32,
    )
    src_block_idx = torch.zeros(padded_batch_size, device=device, dtype=torch.int64)
    dst_block_idx = torch.ones(padded_batch_size, device=device, dtype=torch.int64)
    static_hidden_B_C = torch.empty(padded_batch_size, dim, device=device, dtype=dtype)
    static_dt_base = torch.empty(
        padded_batch_size, nheads, device=device, dtype=dtype
    )
    static_dt = static_dt_base[:, :, None].expand(-1, -1, head_dim)
    graph_out = torch.empty(
        padded_batch_size, nheads, head_dim, device=device, dtype=dtype
    )

    request_schedules = [
        list(range(32)),
        list(range(5, 32)),
        [8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39,
         16, 17, 18, 19, 20],
        [40, 41, 42, 43, 44, 45, 46, 47],
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
         34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62],
        [7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 61, 63,
         3, 5, 9, 13, 17, 21, 25, 29],
        [64, 65, 66, 67, 68, 69, 70, 71, 72],
        list(range(24, 56)),
    ]
    req_src_col: dict[int, int] = {}

    def fill_state_table(request_ids: list[int]) -> torch.Tensor:
        state_index_table.fill_(NULL_BLOCK_ID)
        dst_slots_cpu = []
        for row, req_id in enumerate(request_ids):
            base_slot = req_id * 2 + 1
            src_col = req_src_col.setdefault(req_id, 0)
            state_index_table[row, 0] = base_slot + src_col
            state_index_table[row, 1] = base_slot + (1 - src_col)
            dst_slots_cpu.append(base_slot + (1 - src_col))
        return torch.tensor(dst_slots_cpu, device=device, dtype=torch.int32)

    def advance_requests(request_ids: list[int]) -> None:
        for req_id in request_ids:
            req_src_col[req_id] = 1 - req_src_col[req_id]

    def make_packed(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, dt, B, C = _make_decode_inputs(
            batch_size=padded_batch_size,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        packed = torch.cat(
            [
                x.reshape(padded_batch_size, -1),
                B.reshape(padded_batch_size, -1),
                C.reshape(padded_batch_size, -1),
            ],
            dim=1,
        )
        return packed, dt

    def split_conv_out(
        hidden_B_C: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, B, C = torch.split(
            hidden_B_C,
            [
                nheads * head_dim,
                ngroups * dstate,
                ngroups * dstate,
            ],
            dim=1,
        )
        return (
            hidden.view(-1, nheads, head_dim),
            B.view(-1, ngroups, dstate),
            C.view(-1, ngroups, dstate),
        )

    fill_state_table(request_schedules[0])
    warm_packed, warm_dt = make_packed(8644)
    static_hidden_B_C.copy_(warm_packed)
    static_dt_base.copy_(warm_dt[..., 0])
    warm_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
    warm_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
    warm_hidden_B_C = causal_conv1d_update(
        warm_packed,
        new_conv_state.clone(),
        conv_weight,
        conv_bias,
        activation="silu",
        conv_state_indices=state_index_table,
        block_idx_last_scheduled_token=dst_block_idx.to(torch.int32),
        initial_state_idx=src_block_idx.to(torch.int32),
    )
    warm_x, warm_B, warm_C = split_conv_out(warm_hidden_B_C)
    warm_state = new_state.clone()
    warm_cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    _call_backend(
        new_backend,
        state=warm_state,
        cache=warm_cache,
        x=warm_x,
        dt=warm_dt,
        A=A,
        B=warm_B,
        C=warm_C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=warm_src,
        dst_state_batch_indices=warm_dst,
        cu_seqlens=None,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_hidden_B_C = causal_conv1d_update(
            static_hidden_B_C,
            new_conv_state,
            conv_weight,
            conv_bias,
            activation="silu",
            conv_state_indices=state_index_table,
            block_idx_last_scheduled_token=dst_block_idx.to(torch.int32),
            initial_state_idx=src_block_idx.to(torch.int32),
        )
        graph_x, graph_B, graph_C = split_conv_out(graph_hidden_B_C)
        graph_src = state_index_table.gather(1, src_block_idx[:, None]).squeeze(1)
        graph_dst = state_index_table.gather(1, dst_block_idx[:, None]).squeeze(1)
        new_backend(
            new_state,
            graph_x,
            static_dt,
            A,
            graph_B,
            graph_C,
            D,
            dt_bias,
            dt_softplus=True,
            state_batch_indices=graph_src,
            dst_state_batch_indices=graph_dst,
            null_block_id=NULL_BLOCK_ID,
            out=graph_out,
            old_x=cache["old_x"],
            old_B=cache["old_B"],
            old_dt=cache["old_dt"],
            old_cumAdt=cache["old_cumAdt"],
            cache_buf_idx=cache["cache_buf_idx"],
            prev_num_accepted_tokens=cache["prev_num_accepted_tokens"],
        )

    new_state.copy_(initial_state)
    new_conv_state.copy_(old_conv_state)
    for tensor in cache.values():
        tensor.zero_()
    req_src_col.clear()

    for step, request_ids in enumerate(request_schedules):
        dst_slots = fill_state_table(request_ids)
        packed, dt = make_packed(8700 + step)
        static_hidden_B_C.copy_(packed)
        static_dt_base.copy_(dt[..., 0])

        old_hidden_B_C = causal_conv1d_update(
            packed.clone(),
            old_conv_state,
            conv_weight,
            conv_bias,
            activation="silu",
            conv_state_indices=state_index_table,
            block_idx_last_scheduled_token=dst_block_idx.to(torch.int32),
            initial_state_idx=src_block_idx.to(torch.int32),
        )
        old_x, old_B, old_C = split_conv_out(old_hidden_B_C)
        old_out = _call_backend(
            old_backend,
            state=old_state,
            cache=None,
            x=old_x,
            dt=dt,
            A=A,
            B=old_B,
            C=old_C,
            D=D,
            dt_bias=dt_bias,
            state_batch_indices=state_index_table[:, 0],
            dst_state_batch_indices=state_index_table[:, 1],
            cu_seqlens=None,
        )
        graph.replay()
        torch.cuda.synchronize()
        active_count = len(request_ids)
        max_abs_error = (
            graph_out[:active_count].float() - old_out[:active_count].float()
        ).abs().max()
        assert max_abs_error <= 8e-2, (
            f"conv+SSU varying-active CUDA graph mismatch at step {step}: "
            f"active_count={active_count} max_abs_error={max_abs_error.item()}"
        )
        tracked_tokens = cache["prev_num_accepted_tokens"][dst_slots.to(torch.long)]
        assert int(tracked_tokens.min().item()) >= 1
        assert int(tracked_tokens.max().item()) <= max_window
        advance_requests(request_ids)


@requires_flashinfer
@pytest.mark.parametrize("enable_stochastic_rounding", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_checkpointing_ssu_stp_overlapping_cache_slot_copy_matches_old_flashinfer(
    enable_stochastic_rounding: bool,
    dtype: torch.dtype,
) -> None:
    """Batched cache-slot moves must be safe when source/destination overlap."""

    device = torch.device("cuda")
    cache_size = 16
    nheads = 128
    head_dim = 64
    dstate = 128
    ngroups = 8
    max_window = 6
    row_stride = 18560
    generator = torch.Generator(device=device).manual_seed(2468)

    old_backend = _make_backend(
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    new_backend = _make_backend(
        checkpoint_interval=max_window,
        enable_stochastic_rounding=enable_stochastic_rounding
    )
    initial_state = 0.01 * torch.randn(
        cache_size,
        nheads,
        head_dim,
        dstate,
        device=device,
        dtype=torch.float16,
        generator=generator,
    )
    old_state = initial_state.clone()
    new_state = initial_state.clone()
    cache = _make_checkpointing_cache(
        cache_size=cache_size,
        nheads=nheads,
        head_dim=head_dim,
        dstate=dstate,
        ngroups=ngroups,
        max_window=max_window,
        device=device,
        dtype=dtype,
    )
    A = -torch.rand(nheads, device=device, dtype=torch.float32, generator=generator)
    A = A[:, None, None].expand(nheads, head_dim, dstate)
    D = (0.1 * torch.randn(
        nheads, device=device, dtype=dtype, generator=generator
    ))[:, None].expand(nheads, head_dim)
    dt_bias = (0.1 * torch.randn(
        nheads, device=device, dtype=dtype, generator=generator
    ))[:, None].expand(nheads, head_dim)

    def make_strided_inputs(seed: int) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        x_dense, dt_dense, B_dense, C_dense = _make_decode_inputs(
            batch_size=2,
            nheads=nheads,
            head_dim=head_dim,
            dstate=dstate,
            ngroups=ngroups,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        x_base = torch.empty((2, row_stride), device=device, dtype=dtype)
        dt_base = torch.empty((2, row_stride), device=device, dtype=dtype)
        B_base = torch.empty((2, row_stride), device=device, dtype=dtype)
        C_base = torch.empty((2, row_stride), device=device, dtype=dtype)
        x = torch.as_strided(x_base, x_dense.shape, (row_stride, head_dim, 1))
        dt = torch.as_strided(dt_base, dt_dense.shape, (row_stride, 1, 0))
        B = torch.as_strided(B_base, B_dense.shape, (row_stride, dstate, 1))
        C = torch.as_strided(C_base, C_dense.shape, (row_stride, dstate, 1))
        x.copy_(x_dense)
        dt_base[:, :nheads].copy_(dt_dense[..., 0])
        B.copy_(B_dense)
        C.copy_(C_dense)
        return x, dt, B, C

    init_src = torch.tensor([0, 4], device=device, dtype=torch.int32)
    init_dst = torch.tensor([1, 2], device=device, dtype=torch.int32)
    x, dt, B, C = make_strided_inputs(701)
    torch.manual_seed(9001)
    old_init = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=init_src,
        dst_state_batch_indices=init_dst,
        cu_seqlens=None,
    )
    torch.manual_seed(9001)
    new_init = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=init_src,
        dst_state_batch_indices=init_dst,
        cu_seqlens=None,
    )
    torch.testing.assert_close(
        new_init.float(), old_init.float(), atol=6e-2, rtol=6e-2
    )

    # Slot 2 is both the destination of request 0 and the source of request 1.
    # A batched copy must preserve slot 2's old contents until request 1 reads it.
    overlap_src = torch.tensor([1, 2], device=device, dtype=torch.int32)
    overlap_dst = torch.tensor([2, 3], device=device, dtype=torch.int32)
    x, dt, B, C = make_strided_inputs(702)
    torch.manual_seed(9002)
    old_overlap = _call_backend(
        old_backend,
        state=old_state,
        cache=None,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=overlap_src,
        dst_state_batch_indices=overlap_dst,
        cu_seqlens=None,
    )
    torch.manual_seed(9002)
    new_overlap = _call_backend(
        new_backend,
        state=new_state,
        cache=cache,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        state_batch_indices=overlap_src,
        dst_state_batch_indices=overlap_dst,
        cu_seqlens=None,
    )
    max_abs_error = (new_overlap.float() - old_overlap.float()).abs().max()
    assert max_abs_error <= 6e-2, (
        "overlapping cache-slot STP output mismatch: "
        f"max_abs_error={max_abs_error.item()}"
    )
