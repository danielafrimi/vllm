#!/usr/bin/env python
"""Temporary CUDA JIT prototype for batched DSA KV-page summaries.

This intentionally lives outside the production tree. It compares a Python CPU
reference loop against one CUDA launch that summarizes variable-length
sequences from a padded block table.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import site

import torch


THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(THIS_DIR / ".torch_extensions"))

if "CUDA_HOME" not in os.environ:
    for site_dir in site.getsitepackages():
        cuda_home = Path(site_dir) / "nvidia" / "cu13"
        if (cuda_home / "bin" / "nvcc").exists():
            os.environ["CUDA_HOME"] = str(cuda_home)
            os.environ["PATH"] = (
                f"{cuda_home / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
            )
            os.environ["LD_LIBRARY_PATH"] = (
                f"{cuda_home / 'lib'}{os.pathsep}"
                f"{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
            break

from torch.utils.cpp_extension import load

try:
    import ninja

    ninja_bin_dir = getattr(ninja, "BIN_DIR", None)
    if ninja_bin_dir is not None:
        os.environ["PATH"] = f"{ninja_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
except ImportError:
    pass


CPP_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

void launch_dsa_block_summaries_raw(const void* key_cache,
                                    const void* block_table,
                                    const int64_t* seq_lens,
                                    float* output,
                                    int dtype_code,
                                    int index_code,
                                    int64_t batch,
                                    int64_t max_chunks,
                                    int64_t block_size,
                                    int64_t kv_heads,
                                    int64_t head_dim,
                                    int64_t q_indexer_dim,
                                    cudaStream_t stream);

torch::Tensor dsa_block_summaries(torch::Tensor key_cache,
                                  torch::Tensor block_table,
                                  torch::Tensor seq_lens,
                                  int64_t q_indexer_dim) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be CUDA");
  TORCH_CHECK(block_table.is_cuda(), "block_table must be CUDA");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be CUDA");
  TORCH_CHECK(key_cache.dim() == 4,
              "key_cache must be [num_blocks, block_size, kv_heads, head_dim]");
  TORCH_CHECK(block_table.dim() == 2,
              "block_table must be [batch, max_blocks]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [batch]");
  TORCH_CHECK(block_table.size(0) == seq_lens.size(0),
              "block_table batch must match seq_lens");
  TORCH_CHECK(q_indexer_dim > 0 && q_indexer_dim <= key_cache.size(3),
              "q_indexer_dim must be in (0, head_dim]");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt ||
                  block_table.scalar_type() == torch::kLong,
              "block_table must be int32 or int64");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kLong,
              "seq_lens must be int64");

  int dtype_code = -1;
  if (key_cache.scalar_type() == torch::kFloat32) {
    dtype_code = 0;
  } else if (key_cache.scalar_type() == torch::kHalf) {
    dtype_code = 1;
  } else if (key_cache.scalar_type() == torch::kBFloat16) {
    dtype_code = 2;
  }
  TORCH_CHECK(dtype_code >= 0, "key_cache must be fp32, fp16, or bf16");
  const int index_code = block_table.scalar_type() == torch::kInt ? 0 : 1;

  auto output = torch::empty({block_table.size(0), block_table.size(1),
                              key_cache.size(2), q_indexer_dim},
                             key_cache.options().dtype(torch::kFloat32));
  launch_dsa_block_summaries_raw(
      key_cache.data_ptr(),
      block_table.data_ptr(),
      seq_lens.data_ptr<int64_t>(),
      output.data_ptr<float>(),
      dtype_code,
      index_code,
      block_table.size(0),
      block_table.size(1),
      key_cache.size(1),
      key_cache.size(2),
      key_cache.size(3),
      q_indexer_dim,
      at::cuda::getCurrentCUDAStream());
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dsa_block_summaries", &dsa_block_summaries,
        "Temporary batched DSA KV-page summary kernel");
}

"""


CUDA_SRC = r"""
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float load_key_value(const void* base,
                                                int dtype_code,
                                                int64_t index) {
  if (dtype_code == 0) {
    return static_cast<const float*>(base)[index];
  }
  if (dtype_code == 1) {
    return __half2float(static_cast<const __half*>(base)[index]);
  }
  return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
}

__device__ __forceinline__ int64_t load_block_id(const void* block_table,
                                                 int index_code,
                                                 int64_t index) {
  if (index_code == 0) {
    return static_cast<int64_t>(static_cast<const int32_t*>(block_table)[index]);
  }
  return static_cast<const int64_t*>(block_table)[index];
}

__global__ void dsa_block_summaries_kernel(
    const void* __restrict__ key_cache,
    const void* __restrict__ block_table,
    const int64_t* __restrict__ seq_lens,
    float* __restrict__ output,
    int dtype_code,
    int index_code,
    int64_t batch,
    int64_t max_chunks,
    int64_t block_size,
    int64_t kv_heads,
    int64_t head_dim,
    int64_t q_indexer_dim) {
  const int64_t linear = blockIdx.x;
  const int64_t kv_head = blockIdx.y;
  const int64_t seq = linear / max_chunks;
  const int64_t chunk = linear - seq * max_chunks;

  if (seq >= batch || chunk >= max_chunks || kv_head >= kv_heads) {
    return;
  }

  const int64_t seq_len = seq_lens[seq];
  const int64_t num_chunks = (seq_len + block_size - 1) / block_size;
  float inv_len = 0.0f;
  int64_t valid_len = 0;
  int64_t physical_block = 0;

  if (chunk < num_chunks) {
    valid_len = block_size;
    const int64_t remaining = seq_len - chunk * block_size;
    if (remaining < block_size) {
      valid_len = remaining;
    }
    physical_block = load_block_id(block_table, index_code,
                                   seq * max_chunks + chunk);
    inv_len = valid_len > 0 ? 1.0f / static_cast<float>(valid_len) : 0.0f;
  }

  for (int64_t dim = threadIdx.x; dim < q_indexer_dim; dim += blockDim.x) {
    float acc = 0.0f;
    if (valid_len > 0) {
      for (int64_t offset = 0; offset < valid_len; ++offset) {
        const int64_t key_idx =
            (((static_cast<int64_t>(physical_block) * block_size + offset) *
                  kv_heads +
              kv_head) *
                 head_dim +
             dim);
        acc += load_key_value(key_cache, dtype_code, key_idx);
      }
      acc *= inv_len;
    }
    const int64_t out_idx =
        (((seq * max_chunks + chunk) * kv_heads + kv_head) * q_indexer_dim +
         dim);
    output[out_idx] = acc;
  }
}

void launch_dsa_block_summaries_raw(const void* key_cache,
                                    const void* block_table,
                                    const int64_t* seq_lens,
                                    float* output,
                                    int dtype_code,
                                    int index_code,
                                    int64_t batch,
                                    int64_t max_chunks,
                                    int64_t block_size,
                                    int64_t kv_heads,
                                    int64_t head_dim,
                                    int64_t q_indexer_dim,
                                    cudaStream_t stream) {
  const dim3 grid(batch * max_chunks, kv_heads);
  const int threads = 128;
  dsa_block_summaries_kernel<<<grid, threads, 0, stream>>>(
      key_cache,
      block_table,
      seq_lens,
      output,
      dtype_code,
      index_code,
      batch,
      max_chunks,
      block_size,
      kv_heads,
      head_dim,
      q_indexer_dim);
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--num-physical-blocks", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--q-indexer-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def load_extension():
    source_dir = THIS_DIR / ".jit_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    cpp_path = source_dir / "temp_dsa_block_summaries.cpp"
    cuda_path = source_dir / "temp_dsa_block_summaries.cu"
    cpp_path.write_text(CPP_SRC)
    cuda_path.write_text(CUDA_SRC)
    extra_ldflags = []
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is not None:
        cuda_lib = Path(cuda_home) / "lib"
        versioned_cudart = cuda_lib / "libcudart.so.13"
        if versioned_cudart.exists() and not (cuda_lib / "libcudart.so").exists():
            link_dir = THIS_DIR / ".jit_lib"
            link_dir.mkdir(parents=True, exist_ok=True)
            cudart_link = link_dir / "libcudart.so"
            if not cudart_link.exists():
                cudart_link.symlink_to(versioned_cudart)
            extra_ldflags.extend([
                f"-L{link_dir}",
                f"-Wl,-rpath,{cuda_lib}",
            ])
    return load(
        name="temp_dsa_block_summaries_ext",
        sources=[str(cpp_path), str(cuda_path)],
        extra_cuda_cflags=["-O3", "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"],
        extra_ldflags=extra_ldflags,
        verbose=False,
    )


def make_variable_lengths(
    *,
    batch: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    max_seq_len = batch * block_size + block_size // 2
    lengths = [
        1,
        max(1, block_size - 3),
        block_size,
        block_size + 1,
        block_size + 5,
        2 * block_size - 1,
        2 * block_size,
        2 * block_size + 1,
        max_seq_len,
    ]
    while len(lengths) < batch:
        i = len(lengths)
        lengths.append((i * 11) % max_seq_len + 1)
    return torch.tensor(lengths[:batch], dtype=torch.long, device=device)


def make_block_table(
    *,
    batch: int,
    max_chunks: int,
    num_physical_blocks: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    block_table = torch.empty(
        batch,
        max_chunks,
        device=device,
        dtype=torch.int64,
    )
    for seq in range(batch):
        block_table[seq] = torch.randperm(
            num_physical_blocks,
            device=device,
            generator=generator,
            dtype=torch.long,
        )[:max_chunks]
    return block_table


def cpu_reference(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indexer_dim: int,
) -> torch.Tensor:
    key_cpu = key_cache.detach().cpu()
    table_cpu = block_table.detach().cpu()
    lens_cpu = seq_lens.detach().cpu()
    batch, max_chunks = table_cpu.shape
    _, block_size, kv_heads, _ = key_cpu.shape
    out = torch.zeros(batch, max_chunks, kv_heads, q_indexer_dim, dtype=torch.float32)

    for seq in range(batch):
        key_len = int(lens_cpu[seq].item())
        num_chunks = (key_len + block_size - 1) // block_size
        for chunk in range(num_chunks):
            physical_block = int(table_cpu[seq, chunk].item())
            valid_len = min(block_size, key_len - chunk * block_size)
            for kv_head in range(kv_heads):
                acc = torch.zeros(q_indexer_dim, dtype=torch.float32)
                for offset in range(valid_len):
                    acc += key_cpu[
                        physical_block,
                        offset,
                        kv_head,
                        :q_indexer_dim,
                    ].float()
                out[seq, chunk, kv_head] = acc / valid_len
    return out


def run_one_case(
    *,
    ext,
    batch: int,
    num_physical_blocks: int,
    block_size: int,
    kv_heads: int,
    head_dim: int,
    q_indexer_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    verbose: bool = False,
) -> tuple[list[int], tuple[int, ...], float]:
    if q_indexer_dim > head_dim:
        raise ValueError("q_indexer_dim must be <= head_dim")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    seq_lens = make_variable_lengths(
        batch=batch,
        block_size=block_size,
        device=device,
    )
    max_chunks = int(torch.div(
        seq_lens.max() + block_size - 1,
        block_size,
        rounding_mode="floor",
    ).item())
    if max_chunks > num_physical_blocks:
        raise ValueError("--num-physical-blocks must be >= max sequence chunks")

    key_cache = torch.randn(
        num_physical_blocks,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    block_table = make_block_table(
        batch=batch,
        max_chunks=max_chunks,
        num_physical_blocks=num_physical_blocks,
        device=device,
        generator=generator,
    )

    expected = cpu_reference(key_cache, block_table, seq_lens, q_indexer_dim)
    actual = ext.dsa_block_summaries(
        key_cache,
        block_table,
        seq_lens,
        q_indexer_dim,
    ).detach().cpu()

    max_abs = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    if verbose:
        print("block_table=")
        print(block_table.detach().cpu())
    return seq_lens.detach().cpu().tolist(), tuple(actual.shape), max_abs


def sweep_cases(args: argparse.Namespace) -> list[dict[str, int | str]]:
    if not args.sweep:
        return [{
            "name": "single",
            "batch": args.batch,
            "num_physical_blocks": args.num_physical_blocks,
            "block_size": args.block_size,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_indexer_dim": args.q_indexer_dim,
        }]
    return [
        {
            "name": "tiny_block_scalar_dim",
            "batch": 3,
            "num_physical_blocks": 16,
            "block_size": 1,
            "kv_heads": 1,
            "head_dim": 1,
            "q_indexer_dim": 1,
        },
        {
            "name": "small_odd_dim",
            "batch": 7,
            "num_physical_blocks": 64,
            "block_size": 3,
            "kv_heads": 2,
            "head_dim": 17,
            "q_indexer_dim": 13,
        },
        {
            "name": "exact_and_partial_16",
            "batch": 8,
            "num_physical_blocks": 128,
            "block_size": 16,
            "kv_heads": 4,
            "head_dim": 80,
            "q_indexer_dim": 64,
        },
        {
            "name": "qdim_equals_headdim",
            "batch": 11,
            "num_physical_blocks": 256,
            "block_size": 16,
            "kv_heads": 8,
            "head_dim": 64,
            "q_indexer_dim": 64,
        },
        {
            "name": "large_batch_many_chunks",
            "batch": 33,
            "num_physical_blocks": 2048,
            "block_size": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "q_indexer_dim": 96,
        },
        {
            "name": "block32",
            "batch": 17,
            "num_physical_blocks": 1024,
            "block_size": 32,
            "kv_heads": 4,
            "head_dim": 128,
            "q_indexer_dim": 80,
        },
        {
            "name": "fp16_shape_stress",
            "batch": 19,
            "num_physical_blocks": 1024,
            "block_size": 16,
            "kv_heads": 6,
            "head_dim": 96,
            "q_indexer_dim": 95,
            "dtype": "fp16",
        },
        {
            "name": "fp32_reference_dtype",
            "batch": 9,
            "num_physical_blocks": 256,
            "block_size": 8,
            "kv_heads": 3,
            "head_dim": 33,
            "q_indexer_dim": 17,
            "dtype": "fp32",
        },
    ]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required for this temporary prototype")
    device = torch.device(args.device)
    ext = load_extension()

    for case_idx, case in enumerate(sweep_cases(args)):
        dtype = dtype_from_name(str(case.get("dtype", args.dtype)))
        seq_lens, shape, max_abs = run_one_case(
            ext=ext,
            batch=int(case["batch"]),
            num_physical_blocks=int(case["num_physical_blocks"]),
            block_size=int(case["block_size"]),
            kv_heads=int(case["kv_heads"]),
            head_dim=int(case["head_dim"]),
            q_indexer_dim=int(case["q_indexer_dim"]),
            dtype=dtype,
            device=device,
            seed=args.seed + case_idx * 1009,
            verbose=args.verbose,
        )
        print(
            "PASS case="
            f"{case['name']} dtype={dtype} seq_lens={seq_lens} "
            f"output_shape={shape} max_abs_diff={max_abs:.6g}",
            flush=True,
        )

    print("PASS temporary CUDA block-summary JIT prototype")


if __name__ == "__main__":
    main()
