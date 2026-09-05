"""Typed configuration loaded from YAML, with CLI-friendly overrides.

Every tunable lives in one file (``configs/default.yaml``) so that an
experiment is fully described by its config plus the data it points at.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


@dataclass
class DataConfig:
    raw_csv: str = "data/raw/sensor_queries.csv"
    processed_dir: str = "data/processed"
    text_column: str = "text"
    intent_column: str = "intent"
    entity_column: str = "entities"
    # Entity type -> the key used for it inside the ``entities`` column.
    # Add a pair here to teach the pipeline a new entity; nothing else changes.
    entities: dict[str, str] = field(
        default_factory=lambda: {"SENSOR": "sensorType", "TWIN": "digitalTwinID"}
    )
    # Values that mean "this entity is absent from the utterance".
    null_values: list[str] = field(
        default_factory=lambda: ["not present", "none", "n/a", "null", "-"]
    )
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    split_seed: int = 42


@dataclass
class ModelConfig:
    encoder: str = "bert-base-uncased"
    max_length: int = 64
    dropout: float = 0.1
    # Relative weight of the intent objective inside the joint loss.
    intent_loss_weight: float = 1.0


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    # Stop after this many epochs without an improved validation score.
    patience: int = 3
    seed: int = 42
    device: str = "auto"
    output_dir: str = "artifacts"
    checkpoint_name: str = "twinner.pt"


@dataclass
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    # Empty means "same origin only". Widen deliberately, never with "*".
    cors_origins: list[str] = field(default_factory=list)
    max_batch_size: int = 64
    max_text_length: int = 2000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.train.output_dir) / self.train.checkpoint_name

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load a config, falling back to built-in defaults when no file exists."""
        candidate = Path(path) if path else DEFAULT_CONFIG_PATH
        if not candidate.exists():
            if path:
                raise FileNotFoundError(f"Config file not found: {candidate}")
            return cls()
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        return cls.from_dict(payload)

    # `from __future__ import annotations` turns field types into strings, so the
    # section classes are resolved through an explicit map rather than `f.type`.
    _SECTIONS: ClassVar[dict[str, type]] = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "serve": ServeConfig,
    }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        unknown_sections = set(payload) - set(cls._SECTIONS)
        if unknown_sections:
            raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")

        kwargs: dict[str, Any] = {}
        for name, section_cls in cls._SECTIONS.items():
            section = payload.get(name) or {}
            if not isinstance(section, dict):
                raise TypeError(f"Config section '{name}' must be a mapping")
            unknown = set(section) - {sf.name for sf in fields(section_cls)}
            if unknown:
                raise ValueError(
                    f"Unknown key(s) in config section '{name}': {sorted(unknown)}"
                )
            kwargs[name] = section_cls(**section)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def apply_overrides(self, overrides: dict[str, Any]) -> Config:
        """Apply ``{"train.epochs": 3}``-style overrides coming from the CLI."""
        for dotted, value in overrides.items():
            if value is None:
                continue
            section_name, _, key = dotted.partition(".")
            section = getattr(self, section_name, None)
            if not is_dataclass(section) or not hasattr(section, key):
                raise ValueError(f"Unknown config override: {dotted}")
            setattr(section, key, value)
        return self
