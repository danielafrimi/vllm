# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod
import os

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)

_SSU_DEBUG_MAX_CALLS = int(os.environ.get("MAMBA_SSU_DEBUG_CALLS", "0"))
_ssu_debug_calls = 0
_CKPT_MAX_BATCH_ENV = os.environ.get("MAMBA_CKPT_MAX_BATCH")
_CKPT_MAX_BATCH = (
    int(_CKPT_MAX_BATCH_ENV) if _CKPT_MAX_BATCH_ENV not in (None, "") else None
)
# Toggle the `_fixup_old_cumAdt_after_append` Triton pass that runs after the
# FlashInfer checkpointing kernel writes a fresh append. The wrapper was
# originally written assuming the kernel writes per-window-relative cumAdt
# values that need the prefix sum from `prev_k - 1` added back in. If the
# kernel actually writes absolute (already-prefixed) values, this pass
# double-adds and corrupts state -- consistent with the eager-mode accuracy
# gap measured on 2026-05-25. Setting `MAMBA_DISABLE_CUMADT_FIXUP=1` skips
# the fixup at runtime so we can A/B without redeploying.
_DISABLE_CUMADT_FIXUP = (
    os.environ.get("MAMBA_DISABLE_CUMADT_FIXUP", "0") not in ("", "0", "false", "False")
)

# Per-call state-signature logging, gated on `MAMBA_LOG_STATE_HASH=1`.
# Emits one log line per dispatch call with cheap deterministic stats
# (mean / std / abs-sum, plus shape) for each named tensor. The log shape
# is identical across Triton (old) and FlashInfer (new/checkpointing)
# backends, so two runs can be diff'd line-by-line to find the first call
# that diverges. Skipped during CUDA-graph capture.
_LOG_STATE_HASH = (
    os.environ.get("MAMBA_LOG_STATE_HASH", "0") not in ("", "0", "false", "False")
)
_LOG_STATE_MAX_CALLS = int(os.environ.get("MAMBA_LOG_STATE_HASH_MAX_CALLS", "200000"))
_LOG_STATE_MAX_ROWS = int(os.environ.get("MAMBA_LOG_STATE_HASH_MAX_ROWS", "8"))
_log_state_calls = 0


# Per-call kernel-parity check: gated on MAMBA_KERNEL_PARITY_CHECK=1. After
# the FlashInfer `checkpointing_ssu` runs (the new kernel), we replay the
# SAME inputs through `selective_state_update` (the old kernel that scores
# 0.94/0.94 on GSM8K limit50) on cloned `state` and `out` tensors, then
# compare the two `out` results. This is a direct kernel-vs-kernel A/B
# at the inner boundary -- it isolates kernel math from every wrapper-side
# concern (slot copy, cumAdt fixup, tracker updates). Skipped during CUDA
# graph capture.
_KERNEL_PARITY_CHECK = (
    os.environ.get("MAMBA_KERNEL_PARITY_CHECK", "0")
    not in ("", "0", "false", "False")
)
_KERNEL_PARITY_MAX_CALLS = int(
    os.environ.get("MAMBA_KERNEL_PARITY_MAX_CALLS", "200000")
)
_kernel_parity_calls = 0


# A tensor with more than this many elements is reduced via a slot-aware
# slice (when slots are known) or a chunked accumulator, so we don't
# materialize a full fp32 copy of the cache. The state cache for a 120B
# model can easily hit billions of elements; full upcast OOMs the engine.
_LARGE_TENSOR_NUMEL = 1 << 22  # 4M elements


def _tensor_signature(
    t: torch.Tensor | None,
    slots: list[int] | None = None,
) -> str:
    """Deterministic, OOB-safe, OOM-safe per-tensor signature.

    Three modes depending on tensor size:

    * Small tensor (≤ _LARGE_TENSOR_NUMEL): cast to fp32 once and compute
      mean / std / abs-sum directly.
    * Large tensor with known per-slot leading dim and ``slots`` provided:
      restrict to ``tensor[slots]`` (clipped to valid range) before stats,
      which is what we actually care about for divergence diagnosis (the
      kernel only touched those rows).
    * Large tensor without slots: chunked fp32 accumulator over view(-1),
      bounded by chunk-size memory.

    All paths are wrapped in try/except so logger glitches never crash the
    server.
    """
    if t is None:
        return "None"
    try:
        if t.numel() == 0:
            return f"shape={tuple(t.shape)} m=0 s=0 a=0"
        x = t.detach()

        # Slot-aware slice for big per-slot tensors (state, old_x, old_B,
        # ...). This gives us the rows actually touched by this dispatch
        # call — exactly what we want to diff.
        if (
            x.numel() > _LARGE_TENSOR_NUMEL
            and x.dim() >= 1
            and slots
        ):
            try:
                cap = int(x.shape[0])
                valid = sorted({int(s) for s in slots if 0 <= int(s) < cap})
                if valid:
                    idx = torch.tensor(valid, device=x.device, dtype=torch.long)
                    x = x.index_select(0, idx)
            except Exception:  # noqa: BLE001
                pass

        if x.numel() <= _LARGE_TENSOR_NUMEL:
            if x.is_floating_point():
                x = x.to(dtype=torch.float32)
            else:
                x = x.to(dtype=torch.float64)
            m = x.mean().item()
            s = x.std().item() if x.numel() > 1 else 0.0
            a = x.abs().sum().item()
            return (
                f"shape={tuple(t.shape)} m={m:+.6e} s={s:.6e} a={a:.6e}"
            )

        # Chunked fp32 accumulator. chunk_numel ~= 4M elements -> 16 MB
        # fp32 working set, far below any GPU we care about.
        flat = x.contiguous().view(-1)
        n = flat.numel()
        chunk = 1 << 22
        sum_x = 0.0
        sum_x2 = 0.0
        sum_abs = 0.0
        for i in range(0, n, chunk):
            c = flat[i : i + chunk].to(dtype=torch.float32)
            sum_x += c.sum().item()
            sum_x2 += (c * c).sum().item()
            sum_abs += c.abs().sum().item()
        m = sum_x / n
        var = max(0.0, sum_x2 / n - m * m)
        s = var**0.5
        return (
            f"shape={tuple(t.shape)} m={m:+.6e} s={s:.6e} a={sum_abs:.6e}"
        )
    except Exception as e:  # noqa: BLE001
        return f"err={type(e).__name__}:{e}"


def _maybe_log_state_call(
    phase: str,
    state_batch_indices: torch.Tensor | None,
    tensors: dict[str, torch.Tensor | None],
    layer_name: str | None = None,
) -> None:
    """Emit one structured stats line per kernel call.

    Two runs (old vs new kernel, eager mode, same prompt order, same seed)
    will produce identical logs up to the first call that diverges. ``diff``
    the two files to localize the bug to a specific (call_id, tensor) pair.

    ``layer_name`` (e.g. ``self.prefix`` from ``mamba_mixer2``) is included
    in the log line so divergence is reported as
    ``call=N layer=model.layers.17.mixer`` rather than a raw counter.
    Pre/post lines for the same dispatch share the same counter only when
    paired explicitly — we only bump on the ``pre`` phase.
    """
    global _log_state_calls
    if not _LOG_STATE_HASH:
        return
    if _log_state_calls >= _LOG_STATE_MAX_CALLS:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if phase == "pre":
        _log_state_calls += 1

    full_slots: list[int] | None = None
    sbi_repr: list[int] | None = None
    if state_batch_indices is not None:
        try:
            flat = state_batch_indices
            if flat.dim() == 2 and flat.size(1) == 1:
                flat = flat[:, 0]
            if flat.dim() == 1:
                full_slots = flat.detach().cpu().tolist()
                sbi_repr = full_slots[:_LOG_STATE_MAX_ROWS]
        except Exception:  # noqa: BLE001
            sbi_repr = None

    # The "state" tensor has shape (state_cache_size, ...) — the call only
    # touches the rows in state_batch_indices. Passing the FULL slot list
    # (not the display-truncated sbi_repr) into _tensor_signature triggers
    # a slot-aware slice that stays small even when the cache pool is
    # multi-GB. Small per-call tensors (x, dt, B, C, out) don't trigger
    # the large-tensor branch and just get the regular fast path.
    parts = [
        f"name={name} {_tensor_signature(t, slots=full_slots)}"
        for name, t in tensors.items()
    ]
    logger.warning(
        "MAMBA_LOG_STATE call=%d phase=%s layer=%s slots=%s | %s",
        _log_state_calls,
        phase,
        layer_name if layer_name is not None else "?",
        sbi_repr,
        " | ".join(parts),
    )


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
        log_layer_name: str | None = None,
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
        log_layer_name: str | None = None,
    ) -> None:
        del log_layer_name  # unused by Triton backend
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
        log_layer_name: str | None = None,
    ) -> None:
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
            and (
                _CKPT_MAX_BATCH is None
                or state_indices.numel() <= _CKPT_MAX_BATCH
            )
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
            self._maybe_log_checkpointing_call(
                "before",
                kernel_state_indices,
                cache_buf_idx,
                prev_num_accepted_tokens,
                checkpoint_window,
                x_ckpt,
                B_ckpt,
                old_x,
                ckpt_cu_seqlens,
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
                None,
                ckpt_cu_seqlens,
                kernel_max_seqlen,
            )
            self._maybe_log_checkpointing_call(
                "after_kernel",
                kernel_state_indices,
                cache_buf_idx,
                prev_num_accepted_tokens,
                checkpoint_window,
                x_ckpt,
                B_ckpt,
                old_x,
                ckpt_cu_seqlens,
            )
            self._maybe_run_kernel_parity_check(
                state=state,
                x_ckpt=x_ckpt,
                dt_ckpt=dt_ckpt,
                A=A,
                B_ckpt=B_ckpt,
                C_ckpt=C_ckpt,
                D=D,
                z_ckpt=z_ckpt,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                kernel_state_indices=kernel_state_indices,
                null_block_id=null_block_id,
                out_new=out_ckpt,
                ckpt_cu_seqlens=ckpt_cu_seqlens,
                layer_name=log_layer_name,
            )
            if checkpoint_window > 1 and not _DISABLE_CUMADT_FIXUP:
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
            self._maybe_log_checkpointing_call(
                "after_tracker",
                kernel_state_indices,
                cache_buf_idx,
                prev_num_accepted_tokens,
                checkpoint_window,
                x_ckpt,
                B_ckpt,
                old_x,
                ckpt_cu_seqlens,
            )
            return

        rand_seed = (
            torch.randint(0, 2**32, (1,), dtype=torch.int64, device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )

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
        should_reset_checkpointing_trackers = (
            not simple_decode
            and cache_buf_idx is not None
            and prev_num_accepted_tokens is not None
        )
        if should_reset_checkpointing_trackers:
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
    def _maybe_log_checkpointing_call(
        phase: str,
        kernel_state_indices: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        checkpoint_window: int,
        x_ckpt: torch.Tensor,
        B_ckpt: torch.Tensor,
        old_x: torch.Tensor,
        ckpt_cu_seqlens: torch.Tensor | None,
    ) -> None:
        global _ssu_debug_calls
        if _ssu_debug_calls >= _SSU_DEBUG_MAX_CALLS:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        max_rows = 8
        slots = kernel_state_indices[:max_rows].detach()
        if slots.numel() > 0 and torch.all(slots == 0).item():
            return
        valid_slots = slots[slots >= 0].to(torch.long)
        cache_sample = []
        prev_sample = []
        if valid_slots.numel() > 0:
            cache_sample = cache_buf_idx[valid_slots].detach().cpu().tolist()
            prev_sample = prev_num_accepted_tokens[valid_slots].detach().cpu().tolist()
        _ssu_debug_calls += 1
        logger.warning(
            "MAMBA_SSU_DEBUG call=%d phase=%s slots=%s cache=%s prev=%s "
            "window=%d x=%s B=%s old_x=%s cu=%s",
            _ssu_debug_calls,
            phase,
            slots.cpu().tolist(),
            cache_sample,
            prev_sample,
            checkpoint_window,
            tuple(x_ckpt.shape),
            tuple(B_ckpt.shape),
            tuple(old_x.shape),
            ckpt_cu_seqlens[:max_rows + 1].detach().cpu().tolist()
            if ckpt_cu_seqlens is not None else None,
        )

    def _maybe_run_kernel_parity_check(
        self,
        *,
        state: torch.Tensor,
        x_ckpt: torch.Tensor,
        dt_ckpt: torch.Tensor,
        A: torch.Tensor,
        B_ckpt: torch.Tensor,
        C_ckpt: torch.Tensor,
        D: torch.Tensor,
        z_ckpt: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        dt_softplus: bool,
        kernel_state_indices: torch.Tensor,
        null_block_id: int,
        out_new: torch.Tensor,
        ckpt_cu_seqlens: torch.Tensor | None,
        layer_name: str | None,
    ) -> None:
        """Run the OLD ``selective_state_update`` on cloned state with the
        SAME inputs the new ``checkpointing_ssu`` kernel saw, then log the
        per-call divergence of ``out``.

        Bypasses every wrapper-side concern (slot copy, cumAdt fixup,
        tracker updates) so we can attribute a per-call ``out`` mismatch
        squarely to the kernel implementations. Both kernels are FlashInfer
        in this backend, so any disagreement here is a precision/ordering
        gap between ``selective_state_update`` (target: 0.94/0.94) and
        ``checkpointing_ssu`` (current: ≤0.88/0.88). Skipped under CUDA
        graph capture; bounded by ``MAMBA_KERNEL_PARITY_MAX_CALLS``.
        """
        global _kernel_parity_calls
        if not _KERNEL_PARITY_CHECK:
            return
        if _kernel_parity_calls >= _KERNEL_PARITY_MAX_CALLS:
            return
        if torch.cuda.is_current_stream_capturing():
            return
        _kernel_parity_calls += 1
        try:
            state_ref = state.detach().clone()
            # The wrapper's normal fallback to the old ``selective_state_update``
            # passes 3D ``(batch, nheads, head_dim)`` tensors; the new kernel
            # eats 4D ``(batch, T=1, nheads, head_dim)``. Squeeze T=1 so the
            # parity comparison hits the same old-kernel code path the wrapper
            # would otherwise use - removes a 4D-vs-3D asymmetry as a possible
            # source of mismatch and makes any residual disagreement
            # attributable to the kernels themselves.
            def _sq(t: torch.Tensor | None) -> torch.Tensor | None:
                if t is None:
                    return None
                if t.dim() >= 2 and t.size(1) == 1:
                    return t.squeeze(1)
                return t

            x_ref_in = _sq(x_ckpt)
            dt_ref_in = _sq(dt_ckpt)
            B_ref_in = _sq(B_ckpt)
            C_ref_in = _sq(C_ckpt)
            z_ref_in = _sq(z_ckpt)
            out_ref = torch.empty_like(_sq(out_new))
            self._kernel(
                state_ref,
                x_ref_in,
                dt_ref_in,
                A,
                B_ref_in,
                C_ref_in,
                D=D,
                z=z_ref_in,
                dt_bias=dt_bias,
                dt_softplus=dt_softplus,
                state_batch_indices=kernel_state_indices,
                cu_seqlens=ckpt_cu_seqlens,
                num_accepted_tokens=None,
                cache_steps=0,
                pad_slot_id=null_block_id,
                out=out_ref,
                rand_seed=None,
                philox_rounds=(
                    self._mamba_config.stochastic_rounding_philox_rounds or 10
                ),
                algorithm="simple",
            )
            out_new_cmp = _sq(out_new).detach().float()
            out_ref_cmp = out_ref.detach().float()
            diff = (out_new_cmp - out_ref_cmp).abs()
            denom = out_ref_cmp.abs().clamp_min(1e-6)
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            max_rel = float((diff / denom).max().item())
            slot_preview = (
                kernel_state_indices.detach().cpu().tolist()[:8]
                if kernel_state_indices.numel() > 0 else []
            )
            # Top-K worst per-element disagreements: index + new/ref values.
            # Discriminates "few wild outliers" (suggests indexing/race bug)
            # from "uniform precision gap" (suggests bf16 rounding noise).
            flat_diff = diff.flatten()
            k = min(5, flat_diff.numel())
            top_vals, top_idx = torch.topk(flat_diff, k)
            flat_new = out_new_cmp.flatten()
            flat_ref = out_ref_cmp.flatten()
            shape = tuple(out_new_cmp.shape)
            offenders: list[str] = []
            for j in range(k):
                lin = int(top_idx[j].item())
                coords = []
                rem = lin
                for s in reversed(shape):
                    coords.append(rem % s)
                    rem //= s
                coords = list(reversed(coords))
                new_v = float(flat_new[lin].item())
                ref_v = float(flat_ref[lin].item())
                offenders.append(
                    f"idx={coords} new={new_v:+.4e} ref={ref_v:+.4e} "
                    f"diff={float(top_vals[j].item()):+.4e}"
                )
            # Quantile of abs diff to see the bulk-vs-tail distribution.
            q = torch.quantile(
                flat_diff,
                torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat_diff.device),
            )
            n_above_1pct = int(
                (diff > 0.01 * out_ref_cmp.abs().clamp_min(1e-6)).sum().item()
            )
            logger.warning(
                "MAMBA_KERNEL_PARITY call=%d layer=%s slots=%s | "
                "out: max_abs=%.3e max_rel=%.3e mean_abs=%.3e "
                "p50=%.2e p90=%.2e p99=%.2e p999=%.2e "
                "n_rel>1%%=%d/%d shape=%s dtype=%s",
                _kernel_parity_calls,
                layer_name if layer_name else "?",
                slot_preview,
                max_abs,
                max_rel,
                mean_abs,
                float(q[0].item()),
                float(q[1].item()),
                float(q[2].item()),
                float(q[3].item()),
                n_above_1pct,
                int(flat_diff.numel()),
                tuple(out_new.shape),
                str(out_new.dtype),
            )
            logger.warning(
                "MAMBA_KERNEL_PARITY call=%d layer=%s top%d offenders: %s",
                _kernel_parity_calls,
                layer_name if layer_name else "?",
                k,
                " || ".join(offenders),
            )
        except Exception as e:  # noqa: BLE001
            # Never break serving from a diagnostic. Common skip reasons:
            # the old kernel rejects an arg combo the new kernel accepts
            # (rare in simple_decode), or shape juggling errors.
            logger.warning(
                "MAMBA_KERNEL_PARITY call=%d layer=%s FAILED: %s",
                _kernel_parity_calls,
                layer_name if layer_name else "?",
                e,
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
    log_layer_name: str | None = None,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer). The
    ``log_layer_name`` kwarg is only consumed by the optional state-hash
    logger (``MAMBA_LOG_STATE_HASH=1``) so divergence is reported with the
    originating Mamba layer name attached.
    """
    _maybe_log_state_call(
        "pre",
        state_batch_indices,
        {
            "state": state,
            "x": x,
            "dt": dt,
            "B": B,
            "C": C,
        },
        layer_name=log_layer_name,
    )
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
        log_layer_name=log_layer_name,
    )
    _maybe_log_state_call(
        "post",
        state_batch_indices,
        {
            "out": out,
            "state": state,
        },
        layer_name=log_layer_name,
    )
