"""Shared fixtures.

Tests that need a real tokenizer skip themselves when the model files are not
reachable (offline machine, cold cache), so the pure-logic tests still run.
"""

from __future__ import annotations

import pytest

from twinner.config import Config


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    except Exception as exc:  # noqa: BLE001 - any download/cache failure means skip
        pytest.skip(f"bert-base-uncased tokenizer unavailable: {exc}")


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def entity_map() -> dict[str, str]:
    return {"SENSOR": "sensorType", "TWIN": "digitalTwinID"}
