# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models import nemotron_h


@pytest.mark.parametrize("alias", ["moonshot", "vanilla"])
def test_moonshot_attention_aliases(monkeypatch: pytest.MonkeyPatch, alias: str):
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS", alias)

    assert (
        nemotron_h._get_dsa_attention_class()
        is nemotron_h.NemotronHDSASelectiveAttention
    )


@pytest.mark.parametrize(
    ("alias", "provider"),
    [
        ("refactored-efficient", "efficient"),
        ("refactored-pytorch", "pytorch"),
    ],
)
def test_refactored_attention_alias_selects_provider(
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
    provider: str,
):
    monkeypatch.setenv("VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS", alias)
    monkeypatch.delenv("VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS", raising=False)

    attention_cls = nemotron_h._get_dsa_attention_class()

    assert attention_cls.__name__ == "NemotronHDSARefactoredAttention"
    assert nemotron_h.os.environ["VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS"] == provider


def test_moonshot_attention_is_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS", raising=False)

    assert (
        nemotron_h._get_dsa_attention_class()
        is nemotron_h.NemotronHDSASelectiveAttention
    )
