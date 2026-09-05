"""Tests for configuration loading and the synthetic data generator."""

from __future__ import annotations

import csv

import pytest
import yaml

from twinner.config import Config
from twinner.synth import generate, write_dataset


class TestConfig:
    def test_defaults_are_usable_without_a_file(self):
        config = Config()
        assert config.model.encoder == "bert-base-uncased"
        assert config.data.entities == {"SENSOR": "sensorType", "TWIN": "digitalTwinID"}

    def test_loads_from_yaml(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            yaml.safe_dump(
                {"model": {"encoder": "distilbert-base-uncased", "max_length": 96}}
            ),
            encoding="utf-8",
        )
        config = Config.load(path)
        assert config.model.encoder == "distilbert-base-uncased"
        assert config.model.max_length == 96
        assert config.train.epochs == Config().train.epochs  # untouched sections keep defaults

    def test_rejects_unknown_keys_instead_of_silently_ignoring_them(self):
        with pytest.raises(ValueError, match="Unknown key"):
            Config.from_dict({"model": {"encodr": "typo"}})

    def test_rejects_unknown_sections(self):
        with pytest.raises(ValueError, match="Unknown config section"):
            Config.from_dict({"trainn": {}})

    def test_missing_explicit_config_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Config.load(tmp_path / "nope.yaml")

    def test_overrides_apply_by_dotted_path(self):
        config = Config().apply_overrides({"train.epochs": 3, "serve.port": 9001})
        assert config.train.epochs == 3
        assert config.serve.port == 9001

    def test_none_overrides_are_ignored(self):
        config = Config().apply_overrides({"train.epochs": None})
        assert config.train.epochs == Config().train.epochs

    def test_unknown_override_raises(self):
        with pytest.raises(ValueError, match="Unknown config override"):
            Config().apply_overrides({"train.nonsense": 1})

    def test_serve_binds_loopback_by_default(self):
        # Binding every interface should always be an explicit decision.
        assert Config().serve.host == "127.0.0.1"
        assert Config().serve.cors_origins == []


class TestSynth:
    def test_generates_the_requested_number_of_unique_rows(self):
        rows = generate(size=200, seed=1)
        assert len(rows) == 200
        assert len({row["text"] for row in rows}) == 200

    def test_is_deterministic_for_a_seed(self):
        assert generate(size=50, seed=7) == generate(size=50, seed=7)

    def test_every_row_has_the_expected_columns(self):
        for row in generate(size=50, seed=3):
            assert set(row) == {"text", "intent", "entities"}
            assert "sensorType:" in row["entities"]
            assert "digitalTwinID:" in row["entities"]

    def test_includes_utterances_without_a_twin_id(self):
        rows = generate(size=500, seed=5)
        assert any("not present" in row["entities"] for row in rows)

    def test_entity_values_appear_verbatim_in_the_text(self):
        # If a generated value cannot be found in its own utterance, every
        # example built from that template would silently lose its labels.
        for row in generate(size=300, seed=11):
            for part in row["entities"].split(";"):
                value = part.split(":", 1)[1].strip()
                if value != "not present":
                    assert value.lower() in row["text"].lower(), row

    def test_writes_a_readable_csv(self, tmp_path):
        path = write_dataset(tmp_path / "nested" / "out.csv", size=25, seed=2)
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 25
        assert list(rows[0]) == ["text", "intent", "entities"]

    def test_oversized_request_degrades_instead_of_hanging(self):
        # More rows than the templates can produce: should return what it can.
        rows = generate(size=100_000, seed=4)
        assert 0 < len(rows) < 100_000
