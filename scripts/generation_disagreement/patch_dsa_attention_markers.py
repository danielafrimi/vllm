#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Instrument an installed DSA attention module with path markers for a run."""

from __future__ import annotations

import hashlib
import py_compile
import sys
from pathlib import Path


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(f"patch context not found:\n{old[:400]}")
    return text.replace(old, new, 1)


def _replace_once_if_present(text: str, old: str, new: str) -> tuple[str, bool]:
    if old not in text:
        return text, False
    return text.replace(old, new, 1), True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_dsa_attention_markers.py <module.py>")
    path = Path(sys.argv[1])
    text = path.read_text()
    before = _sha256(text)
    if "DSA_PATH_MARKER" in text and "_dsa_log_path_marker" in text:
        print(f"dsa_marker_patch already_instrumented path={path} sha256={before}")
        return

    if "_DSA_PATH_DEBUG_PRINT_LIMIT_ENV" not in text:
        chunk_top_k_env = (
            '_DSA_CHUNK_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K"\n'
        )
        if chunk_top_k_env in text:
            text = _replace_once(
                text,
                chunk_top_k_env,
                chunk_top_k_env
                + """_DSA_PATH_DEBUG_PRINT_LIMIT_ENV = "VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT"
_DSA_PATH_DEBUG_COUNTS: dict[str, int] = {}
""",
            )
        else:
            text = _replace_once(
                text,
                """_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV = (
    "VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS"
)
""",
                """_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV = (
    "VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS"
)
_DSA_CHUNK_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K"
_DSA_PATH_DEBUG_PRINT_LIMIT_ENV = "VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT"
_DSA_PATH_DEBUG_COUNTS: dict[str, int] = {}
""",
            )
    text = _replace_once(
        text,
        """def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)
""",
        """def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _dsa_log_path_marker(marker: str, **fields: typing.Any) -> None:
    limit = _env_int(_DSA_PATH_DEBUG_PRINT_LIMIT_ENV, 0)
    if limit <= 0:
        return
    count = _DSA_PATH_DEBUG_COUNTS.get(marker, 0)
    if count >= limit:
        return
    _DSA_PATH_DEBUG_COUNTS[marker] = count + 1
    details = " ".join(
        f"{key}={value}" for key, value in sorted(fields.items()))
    print(
        f"DSA_PATH_MARKER marker={marker} count={count + 1} {details}",
        flush=True,
    )
""",
    )
    if "_DSA_CHUNK_TOP_K_ENV,\n" not in text:
        text = _replace_once(
            text,
            """        self.q_indexer_chunk_top_k = int(
            _coalesce(
                getattr(config, "q_indexer_chunk_top_k", None),
                default_chunk_top_k,
            ))
        self.q_indexer_chunked_query_chunk_size = int(
""",
            """        self.q_indexer_chunk_top_k = int(
            _coalesce(
                getattr(config, "q_indexer_chunk_top_k", None),
                default_chunk_top_k,
            ))
        chunk_top_k_override = os.environ.get(_DSA_CHUNK_TOP_K_ENV)
        if chunk_top_k_override is not None:
            self.q_indexer_chunk_top_k = int(chunk_top_k_override)
        self.q_indexer_chunked_query_chunk_size = int(
""",
        )
    text = _replace_once(
        text,
        """        if self.q_indexer_dense_prefill_kv_threshold_tokens <= 0:
            raise ValueError(
                f"{_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV} must be positive: "
                f"{self.q_indexer_dense_prefill_kv_threshold_tokens}"
            )

        self.indexer_q_proj = ReplicatedLinear(
""",
        """        if self.q_indexer_dense_prefill_kv_threshold_tokens <= 0:
            raise ValueError(
                f"{_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV} must be positive: "
                f"{self.q_indexer_dense_prefill_kv_threshold_tokens}"
            )
        _dsa_log_path_marker(
            "config",
            chunk_size=self.q_indexer_chunk_size,
            chunk_top_k=self.q_indexer_chunk_top_k,
            dense_prefill_threshold=(
                self.q_indexer_dense_prefill_kv_threshold_tokens),
            layer_idx=self.layer_idx,
            use_flattened_decode=(
                self.q_indexer_use_flattened_decode_page_table_fa),
            use_flattened_prefill=(
                self.q_indexer_use_flattened_prefill_page_table_fa),
            use_full_attention_short_seq=(
                self.q_indexer_use_full_attention_short_seq),
            use_page_table_fa=self.q_indexer_use_page_table_fa,
            use_prefill_page_table_fa=(
                self.q_indexer_use_prefill_page_table_fa),
        )

        self.indexer_q_proj = ReplicatedLinear(
""",
    )
    text = _replace_once(
        text,
        """        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        flash_attn(
            q=query_states.contiguous(),
""",
        """        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        _dsa_log_path_marker(
            "dense_full_page_table_sequence",
            key_len=key_len,
            query_len=query_len,
        )
        flash_attn(
            q=query_states.contiguous(),
""",
    )
    text, patched_bucket_loop = _replace_once_if_present(
        text,
        """        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            seq_block_table = block_table[seq_idx]
            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                num_pages = math.ceil(key_len / chunk_size)
""",
        """        dense_request_count = 0
        sparse_request_count = 0
        sparse_decode_request_count = 0
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            seq_block_table = block_table[seq_idx]
            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                dense_request_count += 1
                num_pages = math.ceil(key_len / chunk_size)
""",
    )
    if not patched_bucket_loop:
        text = _replace_once(
            text,
            """        if block_table.device != device:
            block_table = block_table.to(device=device)

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
""",
            """        if block_table.device != device:
            block_table = block_table.to(device=device)

        dense_request_count = 0
        sparse_request_count = 0
        sparse_decode_request_count = 0
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
""",
        )
        text = _replace_once(
            text,
            """            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                dense_reason = self._dsa_full_page_table_fa_fallback_reason(
""",
            """            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                dense_request_count += 1
                dense_reason = self._dsa_full_page_table_fa_fallback_reason(
""",
        )
    text = _replace_once(
        text,
        """            request_lens_parts.append(
                torch.ones(q_len, device=device, dtype=torch.int32))
            max_seqlen_q = max(max_seqlen_q, 1)
""",
        """            request_lens_parts.append(
                torch.ones(q_len, device=device, dtype=torch.int32))
            sparse_request_count += q_len
            if q_len == 1:
                sparse_decode_request_count += 1
            max_seqlen_q = max(max_seqlen_q, 1)
""",
    )
    text = _replace_once(
        text,
        """        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)
        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        flash_attn(
            q=query_states[:total_rows].contiguous(),
""",
        """        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)
        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        if dense_request_count:
            _dsa_log_path_marker(
                "dense_prefill_page_table_bucket",
                dense_requests=dense_request_count,
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                table_elems=table_elems,
            )
        if sparse_request_count:
            _dsa_log_path_marker(
                "sparse_prefill_page_table_bucket",
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                sparse_requests=sparse_request_count,
                table_elems=table_elems,
            )
        if sparse_decode_request_count:
            _dsa_log_path_marker(
                "sparse_decode",
                decode_requests=sparse_decode_request_count,
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                table_elems=table_elems,
            )
        flash_attn(
            q=query_states[:total_rows].contiguous(),
""",
    )
    text = _replace_once(
        text,
        """        output = torch.empty_like(query_states)
        flash_attn(
            q=query_states.contiguous(),
""",
        """        output = torch.empty_like(query_states)
        _dsa_log_path_marker(
            "sparse_decode_page_table_fa",
            key_len=key_len,
            max_seqlen_k=max_seqlen_k,
            top_chunks=valid_top_count,
        )
        flash_attn(
            q=query_states.contiguous(),
""",
    )
    text = _replace_once(
        text,
        """                if flat_output is not None:
                    output[query_start:query_end] = flat_output
                    continue

            for group_idx in range(self.num_kv_heads):
""",
        """                if flat_output is not None:
                    output[query_start:query_end] = flat_output
                    continue

            if chunk_len == 1:
                _dsa_log_path_marker(
                    "sparse_decode",
                    chunk_top_k=chunk_top_k,
                    key_len=key_len,
                    max_prior_chunks=max_prior_chunks,
                    num_kv_heads=self.num_kv_heads,
                )
            for group_idx in range(self.num_kv_heads):
""",
    )

    path.write_text(text)
    py_compile.compile(str(path), doraise=True)
    after = _sha256(text)
    print(
        "dsa_marker_patch applied "
        f"path={path} before_sha256={before} after_sha256={after}"
    )


if __name__ == "__main__":
    main()
