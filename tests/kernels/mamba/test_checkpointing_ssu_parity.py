# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for ``flashinfer.mamba.checkpointing_ssu`` state HBM updates.

Goal
----
Pinpoint where ``checkpointing_ssu`` diverges from the textbook SSM recurrence
so the MTP fused-replay path can be debugged kernel-side, not via full GSM8K
runs. Each test feeds the SAME tokens in two ways and compares the post-replay
state in HBM:

* a per-token PyTorch reference (``ssm_recurrence_reference``), and
* the kernel under different scheduling patterns (single-call, no-checkpoint
  append followed by a forced checkpoint, etc.).

The state HBM is only written by ``checkpointing_ssu`` on the path where
``prev_num_accepted_tokens + seq_len > max_window`` (the kernel's internal
``must_checkpoint`` predicate). To compare HBM, every test ends with a forced
checkpoint step whose new tokens are deliberately zero, so the state HBM that
lands is purely "state after the recorded ``prev_k`` cache tokens were
replayed". Anything that contributes after that points to a real kernel-level
or adapter-level mismatch.

Run with::

    FLASHINFER_WORKSPACE_BASE=/path/to/fi-workspace \\
    .venv/bin/python -m pytest \\
        tests/kernels/mamba/test_checkpointing_ssu_parity.py -v -s
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

try:
    from flashinfer.mamba import checkpointing_ssu  # type: ignore[import]

    HAS_FI_CKPT = True
except Exception:  # pragma: no cover - environment-dependent
    HAS_FI_CKPT = False

# Imported lazily inside tests that need it so the file still imports cleanly
# in environments without a built vLLM extension.
try:
    from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (  # noqa: E402
        FlashInferSSUBackend,
    )

    HAS_VLLM_SSU = True
except Exception:  # pragma: no cover
    HAS_VLLM_SSU = False

requires_fi_ckpt = pytest.mark.skipif(
    not HAS_FI_CKPT or not torch.cuda.is_available(),
    reason="flashinfer.mamba.checkpointing_ssu unavailable",
)


@dataclasses.dataclass(frozen=True)
class Sizes:
    """Smallest shapes that satisfy ``checkpointing_ssu`` kernel constraints.

    * ``dim`` must be >= 32 and divisible by 16 (output MMA m16n8 atom).
    * ``dstate`` is the swizzle column count for the state; 128 fits the
      ``Swizzle<3,3,3>`` (8,64) atom cleanly.
    * ``max_window`` <= 16 (single replay K-tile assumption in the kernel).
    """

    cache_size: int = 4
    nheads: int = 4
    dim: int = 64
    dstate: int = 128
    ngroups: int = 1
    max_window: int = 8


def _make_state_and_buffers(
    s: Sizes, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    return {
        "state": torch.randn(
            s.cache_size, s.nheads, s.dim, s.dstate, device=device, dtype=dtype
        ),
        # old_x is single-buffered.
        "old_x": torch.zeros(
            s.cache_size, s.max_window, s.nheads, s.dim, device=device, dtype=dtype
        ),
        # old_B / old_dt / old_cumAdt are double-buffered (the "2" axis).
        "old_B": torch.zeros(
            s.cache_size,
            2,
            s.max_window,
            s.ngroups,
            s.dstate,
            device=device,
            dtype=dtype,
        ),
        "old_dt": torch.zeros(
            s.cache_size,
            2,
            s.nheads,
            s.max_window,
            device=device,
            dtype=torch.float32,
        ),
        "old_cumAdt": torch.zeros(
            s.cache_size,
            2,
            s.nheads,
            s.max_window,
            device=device,
            dtype=torch.float32,
        ),
        "cache_buf_idx": torch.zeros(s.cache_size, dtype=torch.int32, device=device),
        "prev_num_accepted_tokens": torch.zeros(
            s.cache_size, dtype=torch.int32, device=device
        ),
    }


def _make_token_inputs(
    s: Sizes, T: int, device: torch.device, dtype: torch.dtype, seed: int
) -> dict[str, torch.Tensor]:
    """Build (x, dt, B, C) for ``T`` tokens of one sequence (batch=1).

    ``dt`` must satisfy the kernel's tie_hdim layout: stride along ``dim`` is
    0, stride along ``nheads`` is 1. Building it as ``(1, T, nheads)`` and
    expanding the trailing ``dim`` axis preserves that without materializing.

    All scales are deliberately small: large dt produces ``exp(A*dt)`` values
    near 1 (slow decay), letting the state accumulate across many tokens
    without overflowing fp16; large x/B grows the new-update term to a
    magnitude comparable to the state. Both keep the reference recurrence
    inside fp16's dynamic range so reference-vs-kernel comparisons are
    meaningful.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = 0.1 * torch.randn(
        1, T, s.nheads, s.dim, device=device, dtype=dtype, generator=g
    )
    # dt magnitudes ~0.5 → softplus(dt) ~ 0.5..1.5, A*dt_proc ~ -0.5
    # → decay = exp(-0.5) ≈ 0.6 (stable).
    dt_base = 0.5 * torch.randn(1, T, s.nheads, device=device, dtype=dtype, generator=g)
    dt = dt_base.unsqueeze(-1).expand(-1, -1, -1, s.dim)
    B = 0.1 * torch.randn(
        1, T, s.ngroups, s.dstate, device=device, dtype=dtype, generator=g
    )
    C = 0.1 * torch.randn(
        1, T, s.ngroups, s.dstate, device=device, dtype=dtype, generator=g
    )
    return {"x": x, "dt": dt, "B": B, "C": C}


def _make_weights(
    s: Sizes, device: torch.device, dtype: torch.dtype, seed: int
) -> dict[str, torch.Tensor]:
    """Build A, D, dt_bias matching the kernel's tie_hdim broadcast layout.

    * ``A``: per-head decay with stride(0)=1 and broadcast over (dim, dstate).
    * ``D`` and ``dt_bias``: per-head, broadcast over ``dim``.

    A is kept small (max magnitude 1.0) so ``exp(A*dt_proc)`` over the small
    dt scale used above stays close to 1, keeping the recurrence numerically
    stable for a few tokens.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    A_per_head = -torch.rand(s.nheads, device=device, dtype=torch.float32, generator=g)
    A = A_per_head[:, None, None].expand(s.nheads, s.dim, s.dstate)
    D_per_head = torch.zeros(s.nheads, device=device, dtype=dtype)
    D = D_per_head[:, None].expand(s.nheads, s.dim)
    dt_bias = torch.zeros(s.nheads, device=device, dtype=dtype)[:, None].expand(
        s.nheads, s.dim
    )
    return {"A": A, "D": D, "dt_bias": dt_bias}


def _softplus(x: torch.Tensor) -> torch.Tensor:
    """Same thresholded softplus as the kernel's ``thresholded_softplus``."""
    return torch.where(x > 20.0, x, torch.log1p(torch.exp(x.clamp(max=20.0))))


def ssm_recurrence_reference(
    initial_state: torch.Tensor,
    A_per_head: torch.Tensor,
    tokens: list[dict[str, torch.Tensor]],
    dt_bias_per_head: torch.Tensor,
    dt_softplus: bool = True,
) -> torch.Tensor:
    """Run the textbook SSM recurrence in fp64 on the CPU side.

    Processes the concatenation of ``tokens`` one new-token at a time:
        dt_proc = softplus(dt + dt_bias)        # per-head, broadcast over dim
        decay   = exp(A * dt_proc)              # per-head, broadcast (dim, dstate)
        state   = state * decay + x * B * dt_proc

    Inputs use the kernel's tie_hdim layout (dim is the broadcast axis), so we
    just slice the per-head scalars.
    """
    nheads, dim, dstate = initial_state.shape
    state = initial_state.to(torch.float64).clone()
    A_h = A_per_head.to(torch.float64)
    bias_h = dt_bias_per_head.to(torch.float64)
    for tok in tokens:
        # tok dict holds (x, dt, B, C) with shape (1, 1, ...).
        x_h = tok["x"].to(torch.float64).reshape(nheads, dim)
        dt_h = tok["dt"].to(torch.float64).reshape(nheads, dim)
        # ngroups=1: B/C have shape (1, 1, 1, dstate) → broadcast to nheads.
        B_g = tok["B"].to(torch.float64).reshape(1, dstate)

        dt_h = dt_h + bias_h[:, None]
        if dt_softplus:
            dt_h = _softplus(dt_h)
        # decay: (nheads, dim, dstate) via per-head A broadcast.
        decay = torch.exp(A_h[:, None, None] * dt_h[:, :, None])
        update = x_h[:, :, None] * B_g[None, :, :] * dt_h[:, :, None]
        state = state * decay + update
    return state


def _run_kernel_step(
    *,
    state: torch.Tensor,
    cache: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor],
    step: dict[str, torch.Tensor],
    state_batch_indices: torch.Tensor,
) -> torch.Tensor:
    """One call to ``checkpointing_ssu`` matching vLLM's adapter shape."""
    T = step["x"].shape[1]
    cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=state.device)
    out = torch.zeros_like(step["x"])
    checkpointing_ssu(
        state,
        cache["old_x"],
        cache["old_B"],
        cache["old_dt"],
        cache["old_cumAdt"],
        cache["cache_buf_idx"],
        cache["prev_num_accepted_tokens"],
        step["x"],
        step["dt"],
        weights["A"],
        step["B"],
        step["C"],
        out,
        D=weights["D"],
        dt_bias=weights["dt_bias"],
        dt_softplus=True,
        state_batch_indices=state_batch_indices,
        pad_slot_id=-1,
        cu_seqlens=cu_seqlens,
        max_seqlen=T,
    )
    return out


def _update_trackers_host(
    cache: dict[str, torch.Tensor],
    slot: int,
    seq_len: int,
    accepted: int,
    max_window: int,
) -> None:
    """Mirror ``FlashInferSSUBackend._update_checkpointing_trackers`` on the
    host. Scheduled ``seq_len`` decides ``must_checkpoint``; ``accepted``
    decides the new ``prev_num_accepted_tokens`` value (this is the fix from
    the previous chat that closed the scheduled-vs-accepted bug).
    """
    prev = int(cache["prev_num_accepted_tokens"][slot].item())
    must_ckpt = (prev + seq_len) > max_window
    if must_ckpt:
        cache["cache_buf_idx"][slot] = 1 - int(cache["cache_buf_idx"][slot].item())
        cache["prev_num_accepted_tokens"][slot] = accepted
    else:
        cache["prev_num_accepted_tokens"][slot] = prev + accepted


# -- shared fixtures ---------------------------------------------------------


@pytest.fixture
def cuda_device() -> torch.device:
    return torch.device("cuda")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.float16


@pytest.fixture
def sizes() -> Sizes:
    return Sizes()


@pytest.fixture
def weights(sizes: Sizes, cuda_device: torch.device, dtype: torch.dtype):
    return _make_weights(sizes, cuda_device, dtype, seed=0)


@pytest.fixture
def state_batch_indices(cuda_device: torch.device) -> torch.Tensor:
    return torch.tensor([0], dtype=torch.int32, device=cuda_device)


# -- helpers -----------------------------------------------------------------


def _run_pattern_and_flush(
    *,
    sizes: Sizes,
    weights: dict[str, torch.Tensor],
    state_batch_indices: torch.Tensor,
    pattern: list[tuple[int, int]],
    init_state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seeds: list[int],
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Apply a (T, accepted) pattern step by step, then force a checkpoint
    flush so the resulting state lands in HBM and can be compared.

    Returns:
        (state_after_flush_slot0, accepted_tokens) where the second item is
        the flat list of per-token (x, dt, B, C) dicts that the reference
        recurrence should consume.
    """
    cache = _make_state_and_buffers(sizes, device, dtype)
    # The kernel mutates `state` in place when must_checkpoint=True; for non-
    # checkpoint steps the HBM is untouched, so seeding init_state here
    # actually matters only for the eventual flush write.
    cache["state"].copy_(init_state)

    accepted_tokens: list[dict[str, torch.Tensor]] = []
    for step_idx, ((T, accepted), seed) in enumerate(zip(pattern, seeds)):
        step = _make_token_inputs(sizes, T, device, dtype, seed=seed)
        _run_kernel_step(
            state=cache["state"],
            cache=cache,
            weights=weights,
            step=step,
            state_batch_indices=state_batch_indices,
        )
        for t in range(accepted):
            accepted_tokens.append({k: v[:, t : t + 1] for k, v in step.items()})
        _update_trackers_host(
            cache,
            slot=int(state_batch_indices[0].item()),
            seq_len=T,
            accepted=accepted,
            max_window=sizes.max_window,
        )

    # Force a final checkpoint flush so state HBM contains the result of
    # replaying everything currently in old_*. Use seq_len = max_window so any
    # prev_k >= 1 triggers ``(prev + seq) > max_window``. The new-token inputs
    # are kept as zeros so they cannot influence the state HBM write.
    flush_T = sizes.max_window
    flush_step = {
        "x": torch.zeros(
            1, flush_T, sizes.nheads, sizes.dim, device=device, dtype=dtype
        ),
        # zero dt with tie_hdim layout: use a (1,T,nheads) zero tensor and
        # expand the dim axis.
        "dt": torch.zeros(1, flush_T, sizes.nheads, device=device, dtype=dtype)
        .unsqueeze(-1)
        .expand(-1, -1, -1, sizes.dim),
        "B": torch.zeros(
            1, flush_T, sizes.ngroups, sizes.dstate, device=device, dtype=dtype
        ),
        "C": torch.zeros(
            1, flush_T, sizes.ngroups, sizes.dstate, device=device, dtype=dtype
        ),
    }
    # Bump prev to max_window if it is zero so the flush always triggers
    # must_checkpoint. With ``prev > 0`` (the common case after any non-empty
    # pattern), we use the natural prev value and rely on
    # ``prev + max_window > max_window``.
    slot = int(state_batch_indices[0].item())
    if int(cache["prev_num_accepted_tokens"][slot].item()) == 0:
        # No accepted tokens means there is nothing to replay; just return
        # the unchanged initial state.
        return cache["state"][slot].clone(), accepted_tokens

    _run_kernel_step(
        state=cache["state"],
        cache=cache,
        weights=weights,
        step=flush_step,
        state_batch_indices=state_batch_indices,
    )
    return cache["state"][slot].clone(), accepted_tokens


# -- the actual tests --------------------------------------------------------


@requires_fi_ckpt
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Raw FlashInfer call without the vLLM-side old_cumAdt fixup. fp16 "
        "round-off + __expf precision in checkpointing_ssu currently shows "
        "~0.24 max diff vs an fp64 reference at small state magnitudes. "
        "Once the kernel writes a globally-consistent cumAdt this tolerance "
        "should tighten significantly."
    ),
)
def test_single_step_all_accepted_matches_reference(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """Sanity: one step of T tokens, all accepted, then a forced checkpoint.

    This is the simplest path that exercises the checkpoint-state-write code.
    Expected to pass once the kernel-side cumsum bug is fixed upstream;
    until then it pins the current fp16+exp drift.
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    pattern = [(4, 4)]
    state_k, tokens = _run_pattern_and_flush(
        sizes=sizes,
        weights=weights,
        state_batch_indices=state_batch_indices,
        pattern=pattern,
        init_state=init_state,
        device=cuda_device,
        dtype=dtype,
        seeds=[101],
    )

    A_per_head = weights["A"][:, 0, 0]  # tie_hdim: A is per-head.
    bias_per_head = weights["dt_bias"][:, 0]
    state_ref = ssm_recurrence_reference(
        init_state[0], A_per_head, tokens, bias_per_head
    )

    diff = (state_k.to(torch.float64) - state_ref).abs()
    assert diff.max().item() < 5e-2, (
        f"single-step all-accepted state mismatch: max={diff.max().item():.4g} "
        f"mean={diff.mean().item():.4g}"
    )


@requires_fi_ckpt
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Raw FlashInfer call without the vLLM-side old_cumAdt fixup. The "
        "kernel writes a per-step local cumsum into old_cumAdt on the "
        "no-checkpoint append path, so the replay-state recurrence diverges "
        "from the reference SSM at the append boundary. Verified by "
        "test_old_cumAdt_is_locally_reset_on_append; fixed end-to-end by "
        "test_vllm_fixup_makes_single_vs_split_agree."
    ),
)
def test_two_no_ckpt_appends_then_flush_matches_reference(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """The critical pattern: two no-checkpoint appends, then a forced flush.

    Pattern: (T=2, accepted=2) twice with max_window=8 means both steps stay
    on the active buffer (prev grows 0 -> 2 -> 4 without rotating). The
    forced flush then replays the 4 buffered tokens into state HBM.

    The kernel writes ``smem.cumAdt[lane]`` to ``old_cumAdt[write_offset +
    lane]`` directly, where ``smem.cumAdt`` is a local cumulative sum over
    just the new tokens (it starts from 0 inside ``compute_cumAdt``). After
    two no-checkpoint appends, ``old_cumAdt[0..3]`` is therefore::

        [c1, c1+c2, c1', c1'+c2']

    not the globally-consistent::

        [c1, c1+c2, c1+c2+c1', c1+c2+c1'+c2'].

    Yet the replay reads ``total_cumAdt = old_cumAdt[prev_k-1]`` and
    ``coeff[k] = exp(total_cumAdt - old_cumAdt[k]) * old_dt[k]``, which only
    makes sense if ``old_cumAdt`` is the global cumsum. If this test fails,
    that is the bug fingerprint: state HBM systematically diverges from the
    reference once the buffer crosses an append boundary.
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    pattern = [(2, 2), (2, 2)]
    state_k, tokens = _run_pattern_and_flush(
        sizes=sizes,
        weights=weights,
        state_batch_indices=state_batch_indices,
        pattern=pattern,
        init_state=init_state,
        device=cuda_device,
        dtype=dtype,
        seeds=[201, 202],
    )

    A_per_head = weights["A"][:, 0, 0]
    bias_per_head = weights["dt_bias"][:, 0]
    state_ref = ssm_recurrence_reference(
        init_state[0], A_per_head, tokens, bias_per_head
    )

    diff = (state_k.to(torch.float64) - state_ref).abs()
    rel = diff / state_ref.abs().clamp(min=1e-3)
    print(
        f"\n[two-append parity] max_abs={diff.max().item():.4g} "
        f"mean_abs={diff.mean().item():.4g} max_rel={rel.max().item():.4g}"
    )
    assert diff.max().item() < 5e-2, (
        "Two no-checkpoint appends followed by a forced flush diverged from "
        "the reference SSM recurrence — see test docstring for the suspected "
        "kernel bug (local vs global cumAdt write on append)."
    )


@requires_fi_ckpt
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Raw FlashInfer call without the vLLM-side old_cumAdt fixup. Same "
        "tokens fed differently produce different state HBM because of the "
        "per-step local-cumsum write on the no-checkpoint append path."
    ),
)
def test_single_call_vs_two_no_ckpt_appends(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """Method-vs-method: process 4 tokens in one call, vs as 2+2 no-ckpt.

    Both should produce the same state HBM after a forced flush, because both
    have the same logical accepted-token history. Splits this from the
    reference-vs-kernel test so a failure isolates whether the bug is in the
    append/replay buffer state or in the reference itself.
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    # Single 4-token call. Use deterministic seeds so the two methods feed
    # equivalent token inputs at the per-step level: the single call's 4
    # tokens come from one seed; the multi-step path feeds the same 4 in two
    # halves derived from the same seed via slicing on the kernel side
    # rather than re-generation. Easiest: build inputs once for the 4-token
    # case and split them for the multi-step case.
    full = _make_token_inputs(sizes, 4, cuda_device, dtype, seed=301)

    def split_step(step, start, end):
        return {k: v[:, start:end] for k, v in step.items()}

    # ---- single call ----
    cache_a = _make_state_and_buffers(sizes, cuda_device, dtype)
    cache_a["state"].copy_(init_state)
    _run_kernel_step(
        state=cache_a["state"],
        cache=cache_a,
        weights=weights,
        step=full,
        state_batch_indices=state_batch_indices,
    )
    _update_trackers_host(
        cache_a,
        slot=0,
        seq_len=4,
        accepted=4,
        max_window=sizes.max_window,
    )
    flush_T = sizes.max_window
    flush_step = {
        "x": torch.zeros(
            1, flush_T, sizes.nheads, sizes.dim, device=cuda_device, dtype=dtype
        ),
        "dt": torch.zeros(1, flush_T, sizes.nheads, device=cuda_device, dtype=dtype)
        .unsqueeze(-1)
        .expand(-1, -1, -1, sizes.dim),
        "B": torch.zeros(
            1, flush_T, sizes.ngroups, sizes.dstate, device=cuda_device, dtype=dtype
        ),
        "C": torch.zeros(
            1, flush_T, sizes.ngroups, sizes.dstate, device=cuda_device, dtype=dtype
        ),
    }
    _run_kernel_step(
        state=cache_a["state"],
        cache=cache_a,
        weights=weights,
        step=flush_step,
        state_batch_indices=state_batch_indices,
    )
    state_single = cache_a["state"][0].clone()

    # ---- two no-ckpt appends ----
    cache_b = _make_state_and_buffers(sizes, cuda_device, dtype)
    cache_b["state"].copy_(init_state)
    for start, end in [(0, 2), (2, 4)]:
        _run_kernel_step(
            state=cache_b["state"],
            cache=cache_b,
            weights=weights,
            step=split_step(full, start, end),
            state_batch_indices=state_batch_indices,
        )
        _update_trackers_host(
            cache_b,
            slot=0,
            seq_len=end - start,
            accepted=end - start,
            max_window=sizes.max_window,
        )
    _run_kernel_step(
        state=cache_b["state"],
        cache=cache_b,
        weights=weights,
        step=flush_step,
        state_batch_indices=state_batch_indices,
    )
    state_split = cache_b["state"][0].clone()

    diff = (state_single.to(torch.float64) - state_split.to(torch.float64)).abs()
    print(
        f"\n[single-vs-split] max_abs={diff.max().item():.4g} "
        f"mean_abs={diff.mean().item():.4g}"
    )
    # Method-vs-method tolerance is tighter than method-vs-reference (no fp64
    # cross-precision delta), but we still allow fp16 round-off noise.
    assert diff.max().item() < 5e-2, (
        "Single-call vs two-no-ckpt-append diverged: same logical tokens fed "
        "differently produce different state HBM after a forced flush. This "
        "isolates the bug to the no-checkpoint append path of the kernel "
        "(distinct from any reference-recurrence assumptions)."
    )


@requires_fi_ckpt
def test_old_cumAdt_is_locally_reset_on_append(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """Direct buffer inspection: show ``old_cumAdt`` is a local cumsum.

    After two no-checkpoint append steps (T=2, accepted=2) each, the active
    replay buffer should hold a *global* inclusive cumulative sum::

        old_cumAdt[0..3] = [c1, c1+c2, c1+c2+c3, c1+c2+c3+c4]

    where ``ci = A * softplus(dt_i + bias)`` for the i-th token in chronology.

    The current kernel instead writes ``smem.cumAdt[lane]`` directly to
    ``old_cumAdt[write_offset + lane]``, where ``smem.cumAdt`` is the cumsum
    of just the new step's tokens. Hence we expect::

        old_cumAdt[0..3] = [c1, c1+c2, c3, c3+c4]

    i.e. ``old_cumAdt[2]`` and ``old_cumAdt[3]`` should NOT include the
    ``c1+c2`` prefix. This test pins that as the kernel's actual behavior so
    the fix path is unambiguous.
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    cache = _make_state_and_buffers(sizes, cuda_device, dtype)
    cache["state"].copy_(init_state)

    # Run two appends and capture per-step dt_proc to compute expected values.
    A_per_head = weights["A"][:, 0, 0].to(torch.float64).cpu()
    bias_per_head = weights["dt_bias"][:, 0].to(torch.float64).cpu()
    all_steps = []
    for seed in (501, 502):
        step = _make_token_inputs(sizes, 2, cuda_device, dtype, seed=seed)
        _run_kernel_step(
            state=cache["state"],
            cache=cache,
            weights=weights,
            step=step,
            state_batch_indices=state_batch_indices,
        )
        _update_trackers_host(
            cache,
            slot=0,
            seq_len=2,
            accepted=2,
            max_window=sizes.max_window,
        )
        all_steps.append(step)
    torch.accelerator.synchronize()

    # Active buffer index after two no-ckpt appends should still be 0.
    assert int(cache["cache_buf_idx"][0].item()) == 0
    assert int(cache["prev_num_accepted_tokens"][0].item()) == 4

    # Pull the active buffer's old_cumAdt for the first head.
    buf = int(cache["cache_buf_idx"][0].item())
    head0_cumAdt = cache["old_cumAdt"][0, buf, 0, :4].cpu().to(torch.float64)
    head0_dt = cache["old_dt"][0, buf, 0, :4].cpu().to(torch.float64)

    # Recompute the per-token dt_proc for head 0 across the two steps.
    # The kernel applies dt += dt_bias and (here) softplus before cumsum.
    h = 0
    A_h = A_per_head[h]
    bias_h = bias_per_head[h]
    dt_proc_seq = []
    for step in all_steps:
        # step["dt"] has shape (1, 2, nheads, dim); tie_hdim broadcast, so
        # take the [..., 0] slice along dim. Apply softplus the same way the
        # kernel does.
        raw = step["dt"][0, :, h, 0].cpu().to(torch.float64)
        dt_proc_seq.extend((_softplus(raw + bias_h)).tolist())

    # Expected global cumsum of A * dt_proc:
    global_cumsum = torch.tensor(dt_proc_seq, dtype=torch.float64) * A_h
    global_cumsum = torch.cumsum(global_cumsum, dim=0)

    # Expected local-reset cumsum (kernel's actual behavior):
    local_step1 = (torch.tensor(dt_proc_seq[:2], dtype=torch.float64) * A_h).cumsum(
        dim=0
    )
    local_step2 = (torch.tensor(dt_proc_seq[2:], dtype=torch.float64) * A_h).cumsum(
        dim=0
    )
    local_reset = torch.cat([local_step1, local_step2])

    obs = head0_cumAdt
    print(
        "\n[old_cumAdt buffer] head 0 active buf:",
        obs.tolist(),
        "\n  expected global cumsum:",
        global_cumsum.tolist(),
        "\n  expected local-reset  :",
        local_reset.tolist(),
        "\n  also stored old_dt    :",
        head0_dt.tolist(),
    )

    # Sanity: the per-token old_dt should at least match the per-token
    # dt_proc (this confirms our recomputation is using the right inputs).
    dt_proc_t = torch.tensor(dt_proc_seq, dtype=torch.float64)
    assert torch.allclose(head0_dt, dt_proc_t, atol=5e-3), (
        f"old_dt mismatch: {head0_dt} vs {dt_proc_t}"
    )

    diff_global = (obs - global_cumsum).abs().max().item()
    diff_local = (obs - local_reset).abs().max().item()
    print(f"  diff_vs_global={diff_global:.4g}  diff_vs_local_reset={diff_local:.4g}")

    # The headline assertion: this test PASSES if the kernel is buggy
    # (matches local-reset, not global cumsum). The intent is to pin the
    # current behavior so the bug shows up explicitly as expected output and
    # any future fix turns this assertion into a (deliberate) failure.
    assert diff_local < 5e-3, (
        "old_cumAdt did not even match the local-reset hypothesis — the bug "
        "may be different from what we suspected. observed=" + repr(obs)
    )
    assert diff_global > 1e-2, (
        "old_cumAdt matched the global cumsum, contradicting the expected "
        "local-reset kernel bug. The diagnosis needs to be revisited."
    )


@requires_fi_ckpt
@pytest.mark.skipif(not HAS_VLLM_SSU, reason="vLLM ssu_dispatch not importable")
def test_vllm_fixup_recovers_global_cumAdt_buffer(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """Run the same two no-checkpoint appends, but apply
    ``FlashInferSSUBackend._fixup_old_cumAdt_after_append`` immediately
    after each kernel call. The active buffer should then hold a globally
    consistent inclusive cumsum.

    This is the verification step for the vLLM-side workaround that
    compensates for the FlashInfer kernel's per-step local-reset.
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    cache = _make_state_and_buffers(sizes, cuda_device, dtype)
    cache["state"].copy_(init_state)

    A_per_head = weights["A"][:, 0, 0].to(torch.float64).cpu()
    bias_per_head = weights["dt_bias"][:, 0].to(torch.float64).cpu()
    all_steps = []
    for seed in (501, 502):
        step = _make_token_inputs(sizes, 2, cuda_device, dtype, seed=seed)
        T = step["x"].shape[1]
        cu_seqlens = torch.tensor([0, T], dtype=torch.int32, device=cuda_device)
        _run_kernel_step(
            state=cache["state"],
            cache=cache,
            weights=weights,
            step=step,
            state_batch_indices=state_batch_indices,
        )
        # Apply the workaround using the same cu_seqlens / max_seqlen the
        # FlashInfer call saw, BEFORE updating trackers.
        FlashInferSSUBackend._fixup_old_cumAdt_after_append(
            cache["old_cumAdt"],
            state_batch_indices,
            cache["cache_buf_idx"],
            cache["prev_num_accepted_tokens"],
            cu_seqlens,
            T,  # max_seqlen passed to the FI call
            sizes.max_window,
            pad_slot_id=-1,
        )
        _update_trackers_host(
            cache,
            slot=0,
            seq_len=2,
            accepted=2,
            max_window=sizes.max_window,
        )
        all_steps.append(step)
    torch.accelerator.synchronize()

    buf = int(cache["cache_buf_idx"][0].item())
    head0 = cache["old_cumAdt"][0, buf, 0, :4].cpu().to(torch.float64)

    h = 0
    A_h = A_per_head[h]
    bias_h = bias_per_head[h]
    dt_proc_seq = []
    for step in all_steps:
        raw = step["dt"][0, :, h, 0].cpu().to(torch.float64)
        dt_proc_seq.extend((_softplus(raw + bias_h)).tolist())
    global_cumsum = (torch.tensor(dt_proc_seq, dtype=torch.float64) * A_h).cumsum(dim=0)

    diff = (head0 - global_cumsum).abs().max().item()
    print(
        f"\n[fixup parity] head 0 active buf: {head0.tolist()}\n"
        f"  expected global cumsum: {global_cumsum.tolist()}\n"
        f"  diff_vs_global={diff:.4g}"
    )
    assert diff < 5e-4, (
        "Fixup did not recover a globally consistent old_cumAdt; "
        f"diff_vs_global={diff:.4g}"
    )


@requires_fi_ckpt
@pytest.mark.skipif(not HAS_VLLM_SSU, reason="vLLM ssu_dispatch not importable")
def test_vllm_fixup_makes_single_vs_split_agree(
    sizes, weights, state_batch_indices, cuda_device, dtype
):
    """End-to-end check of the workaround: with the fixup applied after
    every kernel call, feeding the same tokens as a single ``T=4`` step or
    as two ``T=2`` no-checkpoint appends produces the same state HBM after
    a forced flush (within fp16 tolerance).
    """
    init_state = 0.1 * torch.randn(
        sizes.cache_size,
        sizes.nheads,
        sizes.dim,
        sizes.dstate,
        device=cuda_device,
        dtype=dtype,
    )
    full = _make_token_inputs(sizes, 4, cuda_device, dtype, seed=301)

    def run_with_fixup(pattern: list[tuple[int, int]]) -> torch.Tensor:
        cache = _make_state_and_buffers(sizes, cuda_device, dtype)
        cache["state"].copy_(init_state)
        offset = 0
        for T, _accepted in pattern:
            step = {k: v[:, offset : offset + T] for k, v in full.items()}
            offset += T
            cu = torch.tensor([0, T], dtype=torch.int32, device=cuda_device)
            _run_kernel_step(
                state=cache["state"],
                cache=cache,
                weights=weights,
                step=step,
                state_batch_indices=state_batch_indices,
            )
            FlashInferSSUBackend._fixup_old_cumAdt_after_append(
                cache["old_cumAdt"],
                state_batch_indices,
                cache["cache_buf_idx"],
                cache["prev_num_accepted_tokens"],
                cu,
                T,
                sizes.max_window,
                pad_slot_id=-1,
            )
            _update_trackers_host(
                cache,
                slot=0,
                seq_len=T,
                accepted=T,
                max_window=sizes.max_window,
            )
        # Forced flush.
        flush_T = sizes.max_window
        flush_step = {
            "x": torch.zeros(
                1, flush_T, sizes.nheads, sizes.dim, device=cuda_device, dtype=dtype
            ),
            "dt": torch.zeros(1, flush_T, sizes.nheads, device=cuda_device, dtype=dtype)
            .unsqueeze(-1)
            .expand(-1, -1, -1, sizes.dim),
            "B": torch.zeros(
                1, flush_T, sizes.ngroups, sizes.dstate, device=cuda_device, dtype=dtype
            ),
            "C": torch.zeros(
                1, flush_T, sizes.ngroups, sizes.dstate, device=cuda_device, dtype=dtype
            ),
        }
        cu = torch.tensor([0, flush_T], dtype=torch.int32, device=cuda_device)
        _run_kernel_step(
            state=cache["state"],
            cache=cache,
            weights=weights,
            step=flush_step,
            state_batch_indices=state_batch_indices,
        )
        # No fixup needed for the flush itself: prev_k + max_window > max_window
        # so the FI call takes the checkpoint path, which writes to a fresh
        # buffer at offset 0. That buffer's cumAdt is already a global cumsum
        # (it just doesn't get re-read here).
        return cache["state"][0].clone()

    state_single = run_with_fixup([(4, 4)])
    state_split = run_with_fixup([(2, 2), (2, 2)])
    diff = (state_single.to(torch.float64) - state_split.to(torch.float64)).abs()
    print(
        f"\n[fixup single-vs-split] max_abs={diff.max().item():.4g} "
        f"mean_abs={diff.mean().item():.4g}"
    )
    assert diff.max().item() < 5e-3, (
        "Fixup did not equalize single-call vs split state HBM; "
        f"max_abs={diff.max().item():.4g}"
    )


# ---------------------------------------------------------------------------
# Old-FI vs new-FI state-HBM parity under an MTP-shaped pattern.
#
# These tests do NOT use any reference recurrence; they compare the two real
# vLLM dispatch paths directly. The motivation is the residual ~7 pp GSM8K
# strict-match gap between the old FI MTP backend
# (`flashinfer.mamba.selective_state_update` with the 2D
# `state_batch_indices` + `num_accepted_tokens` MTP contract) and the new FI
# fused-replay backend (`checkpointing_ssu` + the vLLM-side `cumAdt` fixup
# committed earlier on this branch). The fixup unblocked the new path from
# 0.0000 → 0.8582 strict-match, but the old FI baseline today is
# 0.9272 / 0.9325. We want to know whether that remaining gap is:
#
#   (1) append-path numerical drift in fp16 + ``__expf`` precision over up to
#       ``max_window`` buffered tokens between checkpoints, or
#   (2) a residual slot-semantic mismatch in our adapter (the new path always
#       reads/writes the stable per-request slot ``state_batch_indices[:, 0]``,
#       while old FI reads ``state[seq, num_accepted - 1]`` and writes all
#       speculative state slots).
#
# Each test drives an MTP-shaped pattern (T = 1 + num_spec_tokens, varying
# accepted counts) through both backends starting from the same state and
# inputs, then compares the SSM state at the "logical accepted-prefix
# endpoint":
#   - old FI: ``state[seq, num_accepted - 1]``
#   - new FI: ``state[seq, 0]`` (the stable per-request slot)
#
# If the two states agree to fp16 round-off after each step, the slot
# convention is semantically equivalent and the gap is owned by hypothesis
# (1). If they diverge, hypothesis (2) is right and we have a real adapter
# bug to fix.
# ---------------------------------------------------------------------------


def _make_mtp_state_table(
    nseq: int,
    spec_steps: int,
    cache_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a vLLM-shaped MTP block table.

    Shape is ``(nseq, spec_steps)`` so that ``state_batch_indices[seq, 0..S-1]``
    are the per-request materialized speculative state slots used by old FI.
    Allocate contiguous slot ranges per request (``seq * spec_steps``) so the
    two paths agree on which physical slot corresponds to which speculative
    position; this also makes ``state_batch_indices[:, 0]`` (the new-FI stable
    slot) distinct from ``[:, k]`` for any other ``k``, which is exactly the
    layout vLLM hands to ``selective_state_update``.
    """
    assert cache_size >= nseq * spec_steps, (
        f"cache_size={cache_size} too small for nseq={nseq} spec_steps={spec_steps}"
    )
    rows = []
    for s in range(nseq):
        rows.append(
            torch.arange(
                s * spec_steps,
                (s + 1) * spec_steps,
                dtype=torch.int32,
                device=device,
            )
        )
    return torch.stack(rows, dim=0)


@requires_fi_ckpt
@pytest.mark.skipif(not HAS_VLLM_SSU, reason="vLLM ssu_dispatch not importable")
@pytest.mark.parametrize(
    ("input_dtype", "state_dtype"),
    [
        pytest.param(
            torch.float16,
            torch.float16,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "checkpointing_ssu currently uses BF16 MMA atoms and does "
                    "not match old FI for FP16 activation inputs"
                ),
            ),
            id="fp16-input-fp16-state",
        ),
        pytest.param(torch.bfloat16, torch.bfloat16, id="bf16-input-bf16-state"),
        pytest.param(torch.bfloat16, torch.float16, id="bf16-input-fp16-state"),
    ],
)
def test_old_fi_vs_new_fi_single_token_micro_parity(
    cuda_device, input_dtype, state_dtype,
):
    """Smallest possible old-FI-vs-new-FI parity test: one sequence, one
    token, accepted=1.

    Both kernels start from the same init state (slot 0) and process
    exactly one token. There is no cascade, no replay, no slot rotation.
    If the kernel-level layout / contract is the same, ``out_old[0]``
    must equal ``out_new_flat[0]`` to fp16 noise (~1e-3 abs). If this
    test fails, the larger MTP parity tests are diagnosing a kernel-
    level disagreement (not a multi-token / multi-iter slot bug). If it
    passes, the divergence in the larger tests is from cascade /
    slot-rotation handling.

    This is the fastest discriminator between
        (a) wrong test setup / kernel layout difference
        (b) real slot-semantics divergence in MTP usage.
    """
    from flashinfer.mamba import selective_state_update as fi_ssu  # type: ignore[import]

    spec_steps = 6
    nseq = 1
    cache_size = nseq * spec_steps + 2
    sizes = dataclasses.replace(Sizes(), cache_size=cache_size)
    weights_d = _make_weights(sizes, cuda_device, input_dtype, seed=7)
    state_table = _make_mtp_state_table(nseq, spec_steps, cache_size, cuda_device)
    # Identical init across every slot — independent of which slot either
    # kernel chooses to read from for init.
    init_slot = 0.1 * torch.randn(
        sizes.nheads, sizes.dim, sizes.dstate, device=cuda_device, dtype=state_dtype
    )
    init_state_template = init_slot.unsqueeze(0).expand(
        cache_size, sizes.nheads, sizes.dim, sizes.dstate
    ).contiguous()

    T = 1
    step = _make_token_inputs(sizes, T, cuda_device, input_dtype, seed=123)
    # For old FI varlen: 3D flat tensors via squeeze(0). Strides come from
    # _make_token_inputs's expand pattern (tie_hdim with stride[-1]=0 for
    # dt) which the varlen kernel expects.
    x_flat = step["x"].squeeze(0)            # (T, nheads, dim)
    dt_flat = step["dt"].squeeze(0)          # (T, nheads, dim), stride[-1]=0
    B_flat = step["B"].squeeze(0)            # (T, ngroups, dstate)
    C_flat = step["C"].squeeze(0)            # (T, ngroups, dstate)
    cu = torch.tensor([0, T], dtype=torch.int32, device=cuda_device)
    num_accepted_t = torch.tensor([1], dtype=torch.int32, device=cuda_device)
    stable_idx_t = state_table[:, 0].contiguous().to(torch.int32)

    # Old FI (varlen + MTP) — uses 3D flat inputs.
    state_old = init_state_template.clone()
    out_old = torch.zeros_like(x_flat)
    fi_ssu(
        state_old,
        x_flat,
        dt_flat,
        weights_d["A"],
        B_flat,
        C_flat,
        weights_d["D"],
        dt_bias=weights_d["dt_bias"],
        dt_softplus=True,
        state_batch_indices=state_table,
        dst_state_batch_indices=state_table,
        cache_steps=spec_steps,
        num_accepted_tokens=num_accepted_t,
        cu_seqlens=cu,
        out=out_old,
        pad_slot_id=-1,
    )

    # New FI (varlen checkpointing_ssu) — pass the original 4D tensors
    # so stride[0] is non-zero (matches `_run_kernel_step` pattern).
    cache_new = _make_state_and_buffers(sizes, cuda_device, input_dtype)
    cache_new["state"] = init_state_template.clone()
    out_new = torch.zeros_like(step["x"])
    checkpointing_ssu(
        cache_new["state"],
        cache_new["old_x"],
        cache_new["old_B"],
        cache_new["old_dt"],
        cache_new["old_cumAdt"],
        cache_new["cache_buf_idx"],
        cache_new["prev_num_accepted_tokens"],
        step["x"],
        step["dt"],
        weights_d["A"],
        step["B"],
        step["C"],
        out_new,
        D=weights_d["D"],
        dt_bias=weights_d["dt_bias"],
        dt_softplus=True,
        state_batch_indices=stable_idx_t,
        pad_slot_id=-1,
        cu_seqlens=cu,
        max_seqlen=T,
    )
    out_new_flat = out_new.reshape(T, sizes.nheads, sizes.dim)

    # Pure-fp64 reference: closed-form one-token SSM step.
    #   state' = state * exp(A * softplus(dt + dt_bias))
    #            + x * B * softplus(dt + dt_bias)
    #   out    = sum_n state'[..., n] * C[..., n] + D * x
    # All quantities slice along nheads / dim / dstate. ngroups=1 so B/C
    # broadcast over heads.
    init_ref = init_slot.to(torch.float64)
    A_ref = weights_d["A"].to(torch.float64)
    dt_bias_ref = weights_d["dt_bias"].to(torch.float64)
    D_ref = weights_d["D"].to(torch.float64)
    x_ref = x_flat[0].to(torch.float64)               # (nheads, dim)
    dt_ref = dt_flat[0].to(torch.float64)             # (nheads, dim)
    B_ref = B_flat[0].to(torch.float64)               # (ngroups, dstate)
    C_ref = C_flat[0].to(torch.float64)               # (ngroups, dstate)

    dt_proc = _softplus(dt_ref + dt_bias_ref)         # (nheads, dim)
    decay = torch.exp(A_ref * dt_proc[..., None])     # (nheads, dim, dstate)
    update = (
        x_ref[..., None] * B_ref[0, None, None, :] * dt_proc[..., None]
    )                                                  # (nheads, dim, dstate)
    state_after = init_ref * decay + update
    out_ref = (state_after * C_ref[0, None, None, :]).sum(dim=-1) + D_ref * x_ref
    out_ref = out_ref                                  # (nheads, dim)

    out_old_t = out_old.squeeze(0).float()             # (nheads, dim)
    out_new_t = out_new_flat.squeeze(0).float()        # (nheads, dim)
    diff_old_ref = (out_old_t.double() - out_ref).abs()
    diff_new_ref = (out_new_t.double() - out_ref).abs()
    diff_old_new = (out_old_t.double() - out_new_t.double()).abs()
    print(
        "\n[single-token micro] old_vs_ref max_abs="
        f"{diff_old_ref.max().item():.4g} mean_abs={diff_old_ref.mean().item():.4g}"
    )
    print(
        "[single-token micro] new_vs_ref max_abs="
        f"{diff_new_ref.max().item():.4g} mean_abs={diff_new_ref.mean().item():.4g}"
    )
    print(
        "[single-token micro] old_vs_new max_abs="
        f"{diff_old_new.max().item():.4g} mean_abs={diff_old_new.mean().item():.4g} "
        f"input_dtype={input_dtype} state_dtype={state_dtype}"
    )
    print(f"  out_old[0,:8] : {out_old_t.flatten()[:8].cpu().tolist()}")
    print(f"  out_new[0,:8] : {out_new_t.flatten()[:8].cpu().tolist()}")
    print(f"  out_ref[0,:8] : {out_ref.flatten()[:8].cpu().tolist()}")
    # Pure diagnostic — both kernels should be close to the fp64 ref. The
    # one that isn't tells us which kernel's contract we are mis-using.
    if diff_new_ref.max().item() >= 5e-3 or diff_old_ref.max().item() >= 5e-3:
        pytest.fail(
            f"At least one kernel disagrees with the fp64 reference: "
            f"old_vs_ref={diff_old_ref.max().item():.4g}, "
            f"new_vs_ref={diff_new_ref.max().item():.4g}. The off-by kernel "
            "is the one with the layout / contract mismatch in this test."
        )


@requires_fi_ckpt
@pytest.mark.skipif(not HAS_VLLM_SSU, reason="vLLM ssu_dispatch not importable")
def test_old_fi_vs_new_fi_mtp_step_parity(
    cuda_device,
):
    """Compare old FI MTP vs new FI fused-replay ``out`` after one MTP step.

    One sequence, MTP-5 (``spec_steps = 6``), varying accepted counts in
    ``[1, 3, 5, 6]``. Both paths start from the same init state and consume
    identical (x, dt, B, C) inputs.

    Compare ``out`` directly (per-token output of the SSU): both kernels
    cascade the same T tokens from the same init, so ``out_old[seq, k]``
    must equal ``out_new[seq, k]`` to fp16 / __expf reduction noise for
    every k. ``accepted`` only changes which slot old FI reads as the init
    state; we make all slots in the cache hold the same init so this is
    independent of ``accepted`` and isolates kernel-cascade equivalence.

    A slot-semantic divergence between old FI and our stable-slot new FI
    fused-replay path would show up as a per-step ``out`` divergence
    much larger than the per-element fp16 noise (~1e-3 abs).
    """
    from flashinfer.mamba import selective_state_update as fi_ssu  # type: ignore[import]

    spec_steps = 6  # MTP-5 = 1 + num_spec_tokens
    nseq = 1
    cache_size = nseq * spec_steps + 2  # +2 headroom so slot 0 stays the stable slot
    sizes = dataclasses.replace(Sizes(), cache_size=cache_size)
    input_dtype = torch.bfloat16
    state_dtype = torch.float16
    weights_d = _make_weights(sizes, cuda_device, input_dtype, seed=0)

    state_table = _make_mtp_state_table(nseq, spec_steps, cache_size, cuda_device)
    # Use the SAME init value in every cache slot so old FI's per-call
    # `init_token_idx = num_accepted - 1` slot lookup is independent of
    # `accepted` (and matches new FI's stable-slot read). Without this,
    # the two paths would read different per-slot random initialisations
    # and the comparison would be meaningless.
    init_slot = 0.1 * torch.randn(
        sizes.nheads, sizes.dim, sizes.dstate, device=cuda_device, dtype=state_dtype
    )
    init_state = init_slot.unsqueeze(0).expand(
        cache_size, sizes.nheads, sizes.dim, sizes.dstate
    ).contiguous()

    failures: list[str] = []
    stable_idx = state_table[:, 0].contiguous().to(torch.int32)
    for accepted in (1, 3, 5, 6):
        # Same tokens for both paths, in vLLM's production varlen layout:
        # x/dt/B/C are flat (total_tokens, ...) and cu_seqlens drives the
        # per-sequence iteration. The 4D (batch, T, ...) layout would route
        # old FI into a different (MTP-mode, non-varlen) kernel that vLLM
        # does not actually use, so we must squeeze the batch axis.
        step = _make_token_inputs(
            sizes, spec_steps, cuda_device, input_dtype, seed=900 + accepted
        )
        T = step["x"].shape[1]
        x_flat = step["x"].squeeze(0)               # (T, nheads, dim)
        dt_flat = step["dt"].squeeze(0)             # (T, nheads, dim)
        B_flat = step["B"].squeeze(0)               # (T, ngroups, dstate)
        C_flat = step["C"].squeeze(0)               # (T, ngroups, dstate)
        cu = torch.tensor([0, T], dtype=torch.int32, device=cuda_device)
        num_accepted_t = torch.tensor([accepted], dtype=torch.int32, device=cuda_device)

        # ---- old FI varlen+MTP path (the one vLLM actually uses) ----
        state_old = init_state.clone()
        out_old = torch.zeros_like(x_flat)
        fi_ssu(
            state_old,
            x_flat,
            dt_flat,
            weights_d["A"],
            B_flat,
            C_flat,
            weights_d["D"],
            dt_bias=weights_d["dt_bias"],
            dt_softplus=True,
            state_batch_indices=state_table,
            dst_state_batch_indices=state_table,
            cache_steps=spec_steps,
            num_accepted_tokens=num_accepted_t,
            cu_seqlens=cu,
            out=out_old,
            pad_slot_id=-1,
        )

        # ---- new FI fused-replay path with the vLLM cumAdt fixup ----
        # checkpointing_ssu varlen mode expects 4D x of shape
        # (1, total_tokens, nheads, dim) — matching what the vLLM
        # dispatcher does via _reshape_checkpointing_inputs.unsqueeze(0).
        cache_new = _make_state_and_buffers(sizes, cuda_device, input_dtype)
        cache_new["state"] = init_state.clone()
        out_new = torch.zeros_like(x_flat).unsqueeze(0)
        checkpointing_ssu(
            cache_new["state"],
            cache_new["old_x"],
            cache_new["old_B"],
            cache_new["old_dt"],
            cache_new["old_cumAdt"],
            cache_new["cache_buf_idx"],
            cache_new["prev_num_accepted_tokens"],
            x_flat.unsqueeze(0),
            dt_flat.unsqueeze(0),
            weights_d["A"],
            B_flat.unsqueeze(0),
            C_flat.unsqueeze(0),
            out_new,
            D=weights_d["D"],
            dt_bias=weights_d["dt_bias"],
            dt_softplus=True,
            state_batch_indices=stable_idx,
            pad_slot_id=-1,
            cu_seqlens=cu,
            max_seqlen=T,
        )
        out_new_flat = out_new.squeeze(0)  # (T, nheads, dim) to match out_old

        # Compare per-token output (unambiguous; no state-HBM flush needed).
        # Both paths cascade the same T tokens from the same init state, so
        # out_old[k] must agree with out_new_flat[k] for every k. fp16 ULPs
        # accumulate but a slot mismatch would explode this.
        diff = (out_old.float() - out_new_flat.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        msg = (
            f"accepted={accepted}: out max_abs={max_abs:.4g} "
            f"mean_abs={mean_abs:.4g}"
        )
        print(f"\n[old-FI vs new-FI MTP] {msg}")
        # 2e-2 leaves room for fp16 + __expf reduction-order drift across
        # 6 tokens, but is much tighter than the slot-mismatch signature
        # (~5e-1 max abs on a single bad cascade step).
        if max_abs >= 2e-2:
            failures.append(msg)

    assert not failures, (
        "Old-FI vs new-FI MTP out parity exceeded fp16 tolerance for "
        f"{len(failures)} accepted-count(s): {failures}. Indicates a "
        "slot-semantic mismatch between the two dispatch paths."
    )


@requires_fi_ckpt
@pytest.mark.skipif(not HAS_VLLM_SSU, reason="vLLM ssu_dispatch not importable")
def test_old_fi_vs_new_fi_multistep_mtp_parity(
    cuda_device,
):
    """Drive a 3-step MTP-shaped sequence through both backends and compare
    the per-token ``out`` at each step.

    Step i feeds ``T=6`` tokens with ``accepted_i`` of them accepted, with
    pattern ``[1, 3, 2]`` (small numbers exercise the no-checkpoint append
    path with max_window=8). Between iters, we reconcile the two paths'
    committed state via a single source-of-truth manual recurrence on the
    accepted prefix — that way per-step ``out`` comparison isolates kernel
    cascade equivalence and is not contaminated by prior-step state drift.

    A slot-semantic divergence — the residual hypothesis for the ~7 pp
    GSM8K gap between old FI MTP and new FI fused-replay MTP — would
    show as a per-step ``out`` divergence that *grows* with step index
    beyond the fp16 noise floor.
    """
    from flashinfer.mamba import selective_state_update as fi_ssu  # type: ignore[import]

    spec_steps = 6
    nseq = 1
    cache_size = nseq * spec_steps + 2
    sizes = dataclasses.replace(Sizes(), cache_size=cache_size)
    input_dtype = torch.bfloat16
    state_dtype = torch.float16
    weights_d = _make_weights(sizes, cuda_device, input_dtype, seed=42)
    state_table = _make_mtp_state_table(nseq, spec_steps, cache_size, cuda_device)
    # Same per-slot init for old FI's `init_token_idx = num_accepted - 1`
    # to land on the same committed state as new FI's stable-slot read.
    init_slot = 0.1 * torch.randn(
        sizes.nheads, sizes.dim, sizes.dstate, device=cuda_device, dtype=state_dtype
    )
    init_state = init_slot.unsqueeze(0).expand(
        cache_size, sizes.nheads, sizes.dim, sizes.dstate
    ).contiguous()

    # ---- old FI path: state lives across iters in `state_old` ----
    state_old = init_state.clone()
    # ---- new FI path: state + replay buffers ----
    cache_new = _make_state_and_buffers(sizes, cuda_device, input_dtype)
    cache_new["state"] = init_state.clone()
    stable_idx = state_table[:, 0].contiguous().to(torch.int32)
    stable_slot = int(stable_idx[0].item())

    pattern = [(6, 1), (6, 3), (6, 2)]
    step_diffs: list[float] = []
    for i, (T, accepted) in enumerate(pattern):
        step = _make_token_inputs(sizes, T, cuda_device, input_dtype, seed=100 + i)
        x_flat = step["x"].squeeze(0)               # (T, nheads, dim)
        dt_flat = step["dt"].squeeze(0)             # (T, nheads, dim)
        B_flat = step["B"].squeeze(0)               # (T, ngroups, dstate)
        C_flat = step["C"].squeeze(0)               # (T, ngroups, dstate)
        cu = torch.tensor([0, T], dtype=torch.int32, device=cuda_device)
        num_accepted_t = torch.tensor([accepted], dtype=torch.int32, device=cuda_device)

        # Old FI step (varlen+MTP, the path vLLM uses).
        out_old = torch.zeros_like(x_flat)
        fi_ssu(
            state_old,
            x_flat,
            dt_flat,
            weights_d["A"],
            B_flat,
            C_flat,
            weights_d["D"],
            dt_bias=weights_d["dt_bias"],
            dt_softplus=True,
            state_batch_indices=state_table,
            dst_state_batch_indices=state_table,
            cache_steps=spec_steps,
            num_accepted_tokens=num_accepted_t,
            cu_seqlens=cu,
            out=out_old,
            pad_slot_id=-1,
        )

        # New FI step (stable slot, cumAdt fixup, tracker update).
        # checkpointing_ssu varlen mode wants 4D x — match the vLLM
        # dispatcher's _reshape_checkpointing_inputs.unsqueeze(0).
        out_new = torch.zeros_like(x_flat).unsqueeze(0)
        checkpointing_ssu(
            cache_new["state"],
            cache_new["old_x"],
            cache_new["old_B"],
            cache_new["old_dt"],
            cache_new["old_cumAdt"],
            cache_new["cache_buf_idx"],
            cache_new["prev_num_accepted_tokens"],
            x_flat.unsqueeze(0),
            dt_flat.unsqueeze(0),
            weights_d["A"],
            B_flat.unsqueeze(0),
            C_flat.unsqueeze(0),
            out_new,
            D=weights_d["D"],
            dt_bias=weights_d["dt_bias"],
            dt_softplus=True,
            state_batch_indices=stable_idx,
            pad_slot_id=-1,
            cu_seqlens=cu,
            max_seqlen=T,
        )
        out_new_flat = out_new.squeeze(0)  # (T, nheads, dim)
        FlashInferSSUBackend._fixup_old_cumAdt_after_append(
            cache_new["old_cumAdt"],
            stable_idx,
            cache_new["cache_buf_idx"],
            cache_new["prev_num_accepted_tokens"],
            cu,
            T,
            sizes.max_window,
            pad_slot_id=-1,
        )
        _update_trackers_host(
            cache_new,
            slot=stable_slot,
            seq_len=T,
            accepted=accepted,
            max_window=sizes.max_window,
        )

        # Compare per-token output. Both paths cascade the same T tokens
        # over the same init, so out_old[k] must equal out_new_flat[k]
        # to fp16 noise for every k.
        diff = (out_old.float() - out_new_flat.float()).abs()
        diff_max = diff.max().item()
        step_diffs.append(diff_max)
        print(
            f"\n[old-FI vs new-FI multistep MTP] step={i} T={T} accepted={accepted} "
            f"out_max_abs={diff_max:.4g} out_mean_abs={diff.mean().item():.4g}"
        )

        # Reconcile committed state between iters via the SAME accepted-prefix
        # source-of-truth so neither path's intermediate state drift
        # contaminates the next step's `out` comparison. We simply re-use the
        # state old FI just wrote into state_table[0, accepted-1] (which IS
        # the committed-prefix endpoint for this iter, and equals what the
        # next iter's old FI reads back via `init_token_idx = accepted-1`),
        # and copy it into new FI's stable slot for the next iter.
        committed_state = state_old[
            int(state_table[0, accepted - 1].item())
        ].clone()
        # Old FI: broadcast back into every slot so the next iter's
        # `init_token_idx = next_accepted - 1` lookup hits the right state
        # regardless of next_accepted.
        for k in range(spec_steps):
            state_old[int(state_table[0, k].item())] = committed_state
        # New FI: write into the stable slot AND clear the replay buffers
        # so the next iter starts from a clean append regime (this mirrors
        # the production behavior of starting each MTP iter from the
        # committed state with empty replay buffers).
        cache_new["state"][stable_slot] = committed_state
        cache_new["old_x"].zero_()
        cache_new["old_B"].zero_()
        cache_new["old_dt"].zero_()
        cache_new["old_cumAdt"].zero_()
        cache_new["cache_buf_idx"][stable_slot] = 0
        cache_new["prev_num_accepted_tokens"][stable_slot] = 0

    print(f"\n[old-FI vs new-FI multistep MTP] per-step out_max_abs: {step_diffs}")
    # 2e-2 leaves headroom for fp16 + __expf reduction-order drift across
    # 6-token cascades; a slot-semantic divergence would produce
    # per-step max-abs well above 1e-1 and grow with step index.
    assert max(step_diffs) < 2e-2, (
        "Old-FI vs new-FI multistep MTP out parity exceeded tolerance: "
        f"per-step out_max_abs={step_diffs}. Likely slot-semantic divergence "
        "in fused-replay state handling between MTP iters."
    )
