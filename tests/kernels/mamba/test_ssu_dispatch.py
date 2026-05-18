# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


def _kv_cache_config_with_ssu(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2,
) -> KVCacheConfig:
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_type=mamba_type,
    )
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )


def test_default_backend_is_triton():
    initialize_mamba_ssu_backend(MambaConfig(), _kv_cache_config_with_ssu())
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)
    assert backend.name == "triton"


def test_explicit_triton_backend():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_backend_init():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, FlashInferSSUBackend)
    assert backend.name == "flashinfer"


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()
    mod._mamba_ssu_backend = old


@pytest.mark.parametrize(
    "mamba_type",
    [
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
        MambaAttentionBackendEnum.SHORT_CONV,
    ],
)
def test_init_is_noop_for_non_ssu_mamba_type(mamba_type):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    try:
        initialize_mamba_ssu_backend(
            MambaConfig(), _kv_cache_config_with_ssu(mamba_type)
        )
        assert mod._mamba_ssu_backend is None
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_mamba_ssu_backend()
    finally:
        mod._mamba_ssu_backend = old


@pytest.mark.skipif(HAS_FLASHINFER, reason="flashinfer is installed")
def test_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferSSUBackend(MambaConfig())


def test_triton_basic_call():
    set_random_seed(0)
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    device = "cuda"
    batch_size = 2
    dim = 64
    dstate = 16

    state = torch.randn(batch_size, dim, dstate, device=device)
    x = torch.randn(batch_size, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpointing_tracker_uses_accepted_count_for_replay():
    device = "cuda"
    cache_size = 16
    cache_buf_idx = torch.zeros(cache_size, dtype=torch.int32, device=device)
    prev_num_accepted_tokens = torch.zeros(cache_size, dtype=torch.int32, device=device)

    state_batch_indices = torch.tensor([3, 7, 11], dtype=torch.int32, device=device)
    prev_num_accepted_tokens[state_batch_indices] = torch.tensor(
        [0, 2, 15], dtype=torch.int32, device=device
    )
    num_accepted_tokens = torch.tensor([1, 3, 1], dtype=torch.int32, device=device)
    cu_seqlens = torch.tensor([0, 6, 10, 12], dtype=torch.int32, device=device)
    x = torch.empty((1, 12, 1, 1), device=device)

    FlashInferSSUBackend._update_checkpointing_trackers(
        cache_buf_idx,
        prev_num_accepted_tokens,
        state_batch_indices,
        num_accepted_tokens,
        cu_seqlens,
        x,
        max_seqlen=6,
        max_window=16,
        pad_slot_id=NULL_BLOCK_ID,
    )
    torch.accelerator.synchronize()

    assert prev_num_accepted_tokens[state_batch_indices].cpu().tolist() == [1, 5, 1]
    assert cache_buf_idx[state_batch_indices].cpu().tolist() == [0, 0, 1]


def test_checkpointing_state_indices_use_stable_request_slot():
    state_batch_indices = torch.tensor(
        [
            [3, 4, 5, 6],
            [7, 8, 9, 10],
        ],
        dtype=torch.int32,
    )
    num_accepted_tokens = torch.tensor([1, 3], dtype=torch.int32)

    state_indices = FlashInferSSUBackend._checkpointing_state_indices(
        state_batch_indices, num_accepted_tokens
    )

    assert state_indices.tolist() == [3, 7]
