# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit
def _update_checkpointing_trackers_kernel(
    cache_buf_idx,
    prev_num_accepted_tokens,
    state_batch_indices,
    cu_seqlens,
    fixed_seq_len: tl.constexpr,
    max_window: tl.constexpr,
    pad_slot_id: tl.constexpr,
    n_slots: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
    if HAS_CU_SEQLENS:
        seq_lens = tl.load(cu_seqlens + offsets + 1, mask=mask, other=0) - tl.load(
            cu_seqlens + offsets, mask=mask, other=0
        )
    else:
        seq_lens = tl.full((BLOCK,), fixed_seq_len, tl.int32)
    prev = tl.load(prev_num_accepted_tokens + slots, mask=valid, other=0)
    must_checkpoint = prev + seq_lens > max_window
    old_buf = tl.load(cache_buf_idx + slots, mask=valid, other=0)
    new_buf = tl.where(must_checkpoint, 1 - old_buf, old_buf)
    new_prev = tl.minimum(
        tl.where(must_checkpoint, seq_lens, prev + seq_lens), max_window
    )
    tl.store(cache_buf_idx + slots, new_buf, mask=valid)
    tl.store(prev_num_accepted_tokens + slots, new_prev, mask=valid)


@triton.jit
def _reset_checkpointing_trackers_kernel(
    cache_buf_idx,
    prev_num_accepted_tokens,
    state_batch_indices,
    pad_slot_id: tl.constexpr,
    n_slots: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_slots
    slots = tl.load(state_batch_indices + offsets, mask=mask, other=pad_slot_id)
    valid = mask & (slots != pad_slot_id)
    tl.store(cache_buf_idx + slots, 0, mask=valid)
    tl.store(prev_num_accepted_tokens + slots, 0, mask=valid)


@triton.jit
def _fixup_old_cumAdt_append_kernel(
    old_cumAdt,
    state_batch_indices,
    cache_buf_idx,
    prev_num_accepted_tokens,
    cu_seqlens,
    stride_cache,
    stride_dbuf,
    stride_head,
    fixed_seq_len: tl.constexpr,
    max_window: tl.constexpr,
    nheads: tl.constexpr,
    pad_slot_id: tl.constexpr,
    n_seqs,
    HAS_CU_SEQLENS: tl.constexpr,
    BLOCK_H: tl.constexpr,
) -> None:
    seq = tl.program_id(0)
    head_block = tl.program_id(1)
    if seq >= n_seqs:
        return

    slot = tl.load(state_batch_indices + seq).to(tl.int64)
    if slot == pad_slot_id:
        return

    if HAS_CU_SEQLENS:
        seq_len = tl.load(cu_seqlens + seq + 1) - tl.load(cu_seqlens + seq)
    else:
        seq_len = fixed_seq_len

    prev_k = tl.load(prev_num_accepted_tokens + slot)
    is_no_ckpt_append = ((prev_k.to(tl.int64) + seq_len.to(tl.int64)) <= max_window) & (
        prev_k > 0
    )
    if not is_no_ckpt_append:
        return

    buf = tl.load(cache_buf_idx + slot).to(tl.int64)
    head_offs = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = head_offs < nheads
    base_ptrs = (
        old_cumAdt
        + slot * stride_cache
        + buf * stride_dbuf
        + head_offs.to(tl.int64) * stride_head
    )
    total_old = tl.load(base_ptrs + (prev_k - 1).to(tl.int64), mask=h_mask, other=0.0)

    for t in tl.static_range(max_window):
        in_range = (t >= prev_k) & (t < (prev_k + seq_len))
        ptrs = base_ptrs + t
        cur = tl.load(ptrs, mask=h_mask & in_range, other=0.0)
        tl.store(ptrs, cur + total_old, mask=h_mask & in_range)


@triton.jit
def _copy_checkpointing_slots_kernel(
    tensor,
    src_indices,
    dst_indices,
    slot_size: tl.constexpr,
    slot_stride: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    slot = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slot_size
    src = tl.load(src_indices + slot)
    dst = tl.load(dst_indices + slot)
    valid = (src != pad_slot_id) & (dst != pad_slot_id) & (src != dst)
    values = tl.load(tensor + src * slot_stride + offsets, mask=mask & valid)
    tl.store(tensor + dst * slot_stride + offsets, values, mask=mask & valid)


@triton.jit
def _gather_checkpointing_slots_kernel(
    tensor,
    scratch,
    src_indices,
    dst_indices,
    slot_size: tl.constexpr,
    tensor_slot_stride: tl.constexpr,
    scratch_slot_stride: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    slot = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slot_size
    src = tl.load(src_indices + slot)
    dst = tl.load(dst_indices + slot)
    valid = (src != pad_slot_id) & (dst != pad_slot_id) & (src != dst)
    values = tl.load(
        tensor + src * tensor_slot_stride + offsets,
        mask=mask & valid,
    )
    tl.store(
        scratch + slot * scratch_slot_stride + offsets,
        values,
        mask=mask & valid,
    )


@triton.jit
def _scatter_checkpointing_slots_kernel(
    tensor,
    scratch,
    src_indices,
    dst_indices,
    slot_size: tl.constexpr,
    tensor_slot_stride: tl.constexpr,
    scratch_slot_stride: tl.constexpr,
    pad_slot_id: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    slot = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < slot_size
    src = tl.load(src_indices + slot)
    dst = tl.load(dst_indices + slot)
    valid = (src != pad_slot_id) & (dst != pad_slot_id) & (src != dst)
    values = tl.load(
        scratch + slot * scratch_slot_stride + offsets,
        mask=mask & valid,
    )
    tl.store(
        tensor + dst * tensor_slot_stride + offsets,
        values,
        mask=mask & valid,
    )


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        self._copy_scratch: dict[tuple[object, ...], torch.Tensor] = {}
        try:
            from flashinfer.mamba import checkpointing_ssu as _fi_checkpointing_ssu
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install a FlashInfer build with Mamba checkpointing SSU."
            ) from e
        self._kernel = _fi_ssu
        self._checkpointing_kernel = _fi_checkpointing_ssu

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        is_blackwell: bool = False,
        old_x: torch.Tensor | None = None,
        old_B: torch.Tensor | None = None,
        old_dt: torch.Tensor | None = None,
        old_cumAdt: torch.Tensor | None = None,
        cache_buf_idx: torch.Tensor | None = None,
        prev_num_accepted_tokens: torch.Tensor | None = None,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), dtype=torch.int64, device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )

        checkpointing_args = (
            old_x,
            old_B,
            old_dt,
            old_cumAdt,
            cache_buf_idx,
            prev_num_accepted_tokens,
        )
        has_checkpointing_cache = all(arg is not None for arg in checkpointing_args)
        state_indices = self._checkpointing_state_indices(state_batch_indices)
        simple_decode = state_indices is not None and x.size(0) == state_indices.numel()
        non_spec_varlen = state_indices is not None and cu_seqlens is not None
        num_accepted_tokens_for_kernel = (
            None if simple_decode or non_spec_varlen else num_accepted_tokens
        )
        can_checkpoint = (
            state_indices is not None
            and simple_decode
            and has_checkpointing_cache
            and state.dtype in (torch.float16, torch.bfloat16, torch.float32)
        )
        if can_checkpoint:
            assert old_x is not None
            assert old_B is not None
            assert old_dt is not None
            assert old_cumAdt is not None
            assert cache_buf_idx is not None
            assert prev_num_accepted_tokens is not None
            assert state_indices is not None
            kernel_state_indices = state_indices
            dst_indices = None
            if dst_state_batch_indices is not state_batch_indices:
                dst_indices = self._checkpointing_state_indices(
                    dst_state_batch_indices
                )
            if (
                dst_indices is not None
                and dst_indices.numel() == kernel_state_indices.numel()
            ):
                self._copy_checkpointing_slots(
                    (
                        state,
                        old_x,
                        old_B,
                        old_dt,
                        old_cumAdt,
                        cache_buf_idx,
                        prev_num_accepted_tokens,
                    ),
                    kernel_state_indices,
                    dst_indices,
                    null_block_id,
                )
                kernel_state_indices = dst_indices
            ckpt_cu_seqlens = None
            checkpoint_window = old_x.size(1)
            kernel_old_x = old_x
            kernel_old_B = old_B
            kernel_old_dt = old_dt
            kernel_old_cumAdt = old_cumAdt
            x_ckpt, dt_ckpt, B_ckpt, C_ckpt, z_ckpt, out_ckpt, ckpt_max_seqlen = (
                self._reshape_checkpointing_inputs(
                    x,
                    dt,
                    B,
                    C,
                    z,
                    out,
                    kernel_state_indices,
                    ckpt_cu_seqlens,
                    max_seqlen,
                    checkpoint_window,
                )
            )
            kernel_max_seqlen = (
                ckpt_max_seqlen if ckpt_cu_seqlens is not None else None
            )
            self._run_checkpointing_kernel(
                state,
                kernel_old_x,
                kernel_old_B,
                kernel_old_dt,
                kernel_old_cumAdt,
                cache_buf_idx,
                prev_num_accepted_tokens,
                x_ckpt,
                dt_ckpt,
                A,
                B_ckpt,
                C_ckpt,
                out_ckpt,
                D,
                z_ckpt,
                dt_bias,
                dt_softplus,
                kernel_state_indices,
                null_block_id,
                rand_seed,
                ckpt_cu_seqlens,
                kernel_max_seqlen,
            )
            if checkpoint_window > 1:
                self._fixup_old_cumAdt_after_append(
                    kernel_old_cumAdt,
                    kernel_state_indices,
                    cache_buf_idx,
                    prev_num_accepted_tokens,
                    ckpt_cu_seqlens,
                    kernel_max_seqlen or x_ckpt.size(1),
                    checkpoint_window,
                    null_block_id,
                )
            self._update_checkpointing_trackers(
                cache_buf_idx,
                prev_num_accepted_tokens,
                kernel_state_indices,
                ckpt_cu_seqlens,
                ckpt_max_seqlen,
                checkpoint_window,
                null_block_id,
            )
            return

        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens_for_kernel,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            algorithm="simple",
        )
        if (
            num_accepted_tokens is None
            and cache_buf_idx is not None
            and prev_num_accepted_tokens is not None
        ):
            reset_indices = self._checkpointing_state_indices(dst_state_batch_indices)
            if reset_indices is None:
                reset_indices = state_indices
            if reset_indices is not None:
                self._reset_checkpointing_trackers(
                    cache_buf_idx,
                    prev_num_accepted_tokens,
                    reset_indices,
                    null_block_id,
                )

    def _run_checkpointing_kernel(
        self,
        state: torch.Tensor,
        old_x: torch.Tensor,
        old_B: torch.Tensor,
        old_dt: torch.Tensor,
        old_cumAdt: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        x_ckpt: torch.Tensor,
        dt_ckpt: torch.Tensor,
        A: torch.Tensor,
        B_ckpt: torch.Tensor,
        C_ckpt: torch.Tensor,
        out_ckpt: torch.Tensor,
        D: torch.Tensor | None,
        z_ckpt: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        dt_softplus: bool,
        kernel_state_indices: torch.Tensor,
        null_block_id: int,
        rand_seed: torch.Tensor | None,
        ckpt_cu_seqlens: torch.Tensor | None,
        kernel_max_seqlen: int | None,
    ) -> None:
        self._checkpointing_kernel(
            state,
            old_x,
            old_B,
            old_dt,
            old_cumAdt,
            cache_buf_idx,
            prev_num_accepted_tokens,
            x_ckpt,
            dt_ckpt,
            A,
            B_ckpt,
            C_ckpt,
            out_ckpt,
            D=D,
            z=z_ckpt,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=kernel_state_indices,
            pad_slot_id=null_block_id,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds
            or 10,
            cu_seqlens=ckpt_cu_seqlens,
            max_seqlen=kernel_max_seqlen,
        )

    @staticmethod
    def _checkpointing_state_indices(
        state_batch_indices: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if state_batch_indices is None:
            return None
        if state_batch_indices.dim() == 1:
            return state_batch_indices.to(torch.int32).contiguous()
        if state_batch_indices.dim() == 2 and state_batch_indices.size(1) == 1:
            return state_batch_indices[:, 0].to(torch.int32).contiguous()
        return None

    @staticmethod
    def _reshape_checkpointing_inputs(
        x: torch.Tensor,
        dt: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        z: torch.Tensor | None,
        out: torch.Tensor | None,
        state_batch_indices: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        max_window: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        int,
    ]:
        assert out is not None
        if dt.stride(-1) != 0:
            dt = dt[..., :1].contiguous().expand_as(dt)
        if cu_seqlens is not None:
            x = x.contiguous()
            B = B.contiguous()
            C = C.contiguous()
            if z is not None:
                z = z.contiguous()
            return (
                x.unsqueeze(0),
                dt.unsqueeze(0),
                B.unsqueeze(0),
                C.unsqueeze(0),
                z.unsqueeze(0) if z is not None else None,
                out.unsqueeze(0),
                min(max_seqlen or max_window, max_window),
            )
        batch = state_batch_indices.numel()
        x = x.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        if z is not None:
            z = z.contiguous()
        tokens_per_batch = x.shape[0] // batch
        z_ckpt = None
        if z is not None:
            z_ckpt = z.view(batch, tokens_per_batch, *z.shape[1:])
        return (
            x.view(batch, tokens_per_batch, *x.shape[1:]),
            dt.view(batch, tokens_per_batch, *dt.shape[1:]),
            B.view(batch, tokens_per_batch, *B.shape[1:]),
            C.view(batch, tokens_per_batch, *C.shape[1:]),
            z_ckpt,
            out.view(batch, tokens_per_batch, *out.shape[1:]),
            tokens_per_batch,
        )

    @staticmethod
    def _fixup_old_cumAdt_after_append(
        old_cumAdt: torch.Tensor,
        state_batch_indices: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int,
        max_window: int,
        pad_slot_id: int,
    ) -> None:
        n_slots = state_batch_indices.numel()
        if n_slots == 0:
            return
        n_heads = old_cumAdt.size(2)
        block_h = 1
        while block_h < n_heads and block_h < 64:
            block_h *= 2
        state_batch_indices = state_batch_indices.to(
            device=old_cumAdt.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        device_cu_seqlens = (
            cu_seqlens.to(
                device=old_cumAdt.device,
                dtype=torch.int32,
                non_blocking=True,
            ).contiguous()
            if cu_seqlens is not None
            else state_batch_indices
        )
        _fixup_old_cumAdt_append_kernel[
            (n_slots, triton.cdiv(n_heads, block_h))
        ](
            old_cumAdt,
            state_batch_indices,
            cache_buf_idx,
            prev_num_accepted_tokens,
            device_cu_seqlens,
            old_cumAdt.stride(0),
            old_cumAdt.stride(1),
            old_cumAdt.stride(2),
            max_seqlen,
            max_window,
            n_heads,
            pad_slot_id,
            n_slots,
            cu_seqlens is not None,
            BLOCK_H=block_h,
        )

    @staticmethod
    def _update_checkpointing_trackers(
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int,
        max_window: int,
        pad_slot_id: int,
    ) -> None:
        block = 128
        n_slots = state_batch_indices.numel()
        _update_checkpointing_trackers_kernel[(triton.cdiv(n_slots, block),)](
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            cu_seqlens,
            max_seqlen,
            max_window,
            pad_slot_id,
            n_slots,
            cu_seqlens is not None,
            BLOCK=block,
        )

    @staticmethod
    def _reset_checkpointing_trackers(
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        state_batch_indices: torch.Tensor,
        pad_slot_id: int,
    ) -> None:
        block = 128
        n_slots = state_batch_indices.numel()
        _reset_checkpointing_trackers_kernel[(triton.cdiv(n_slots, block),)](
            cache_buf_idx,
            prev_num_accepted_tokens,
            state_batch_indices,
            pad_slot_id,
            n_slots,
            BLOCK=block,
        )

    def _copy_checkpointing_slots(
        self,
        tensors: tuple[torch.Tensor, ...],
        src_indices: torch.Tensor,
        dst_indices: torch.Tensor,
        pad_slot_id: int,
    ) -> None:
        block = 256
        n_slots = src_indices.numel()
        for tensor in tensors:
            slot_size = tensor[0].numel()
            slot_stride = tensor.stride(0)
            scratch = self._get_copy_scratch(tensor, n_slots)
            scratch_stride = scratch.stride(0)
            _gather_checkpointing_slots_kernel[
                (n_slots, triton.cdiv(slot_size, block))
            ](
                tensor,
                scratch,
                src_indices,
                dst_indices,
                slot_size,
                slot_stride,
                scratch_stride,
                pad_slot_id,
                BLOCK=block,
            )
            _scatter_checkpointing_slots_kernel[
                (n_slots, triton.cdiv(slot_size, block))
            ](
                tensor,
                scratch,
                src_indices,
                dst_indices,
                slot_size,
                slot_stride,
                scratch_stride,
                pad_slot_id,
                BLOCK=block,
            )

    def _get_copy_scratch(
        self,
        tensor: torch.Tensor,
        n_slots: int,
    ) -> torch.Tensor:
        key = (
            tensor.device.type,
            tensor.device.index,
            tensor.dtype,
            tuple(tensor.shape[1:]),
            n_slots,
        )
        scratch = self._copy_scratch.get(key)
        if scratch is not None:
            return scratch
        if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Checkpointing slot-copy scratch is unavailable during CUDA "
                "graph capture. Warm up this tensor shape and n_slots before "
                "capture."
            )
        scratch = torch.empty(
            (n_slots, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        self._copy_scratch[key] = scratch
        return scratch


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Initialize the global Mamba SSU backend.

    No-op if `kv_cache_config` contains no specs that call
    selective_state_update.
    """
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssu_backend

    backend = mamba_config.backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_cls = _BACKEND_REGISTRY[backend]
    if isinstance(_mamba_ssu_backend, backend_cls):
        return

    _mamba_ssu_backend = backend_cls(mamba_config)
    logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    is_blackwell: bool = False,
    old_x: torch.Tensor | None = None,
    old_B: torch.Tensor | None = None,
    old_dt: torch.Tensor | None = None,
    old_cumAdt: torch.Tensor | None = None,
    cache_buf_idx: torch.Tensor | None = None,
    prev_num_accepted_tokens: torch.Tensor | None = None,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        is_blackwell=is_blackwell,
        old_x=old_x,
        old_B=old_B,
        old_dt=old_dt,
        old_cumAdt=old_cumAdt,
        cache_buf_idx=cache_buf_idx,
        prev_num_accepted_tokens=prev_num_accepted_tokens,
    )
