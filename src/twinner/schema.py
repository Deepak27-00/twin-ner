"""Label schema shared by preprocessing, training and inference.

Entity tagging uses the BILOU scheme (Begin / Inside / Last / Outside / Unit).
The `U-` tag for single-token entities is what makes BILOU stricter than BIO:
without it a one-token entity is indistinguishable from the start of a longer
one, which is the single most common source of ragged span boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Token positions that must not contribute to the loss ([CLS], [SEP], padding,
# and the continuation pieces we choose not to supervise).
IGNORE_INDEX = -100

OUTSIDE = "O"
_PREFIXES = ("B", "I", "L", "U")


@dataclass
class LabelScheme:
    """Bidirectional maps between human-readable labels and model indices."""

    entity_types: list[str] = field(default_factory=list)
    intents: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.entity_types = sorted(dict.fromkeys(self.entity_types))
        self.intents = sorted(dict.fromkeys(self.intents))

        self.tags: list[str] = [OUTSIDE]
        for entity_type in self.entity_types:
            self.tags.extend(f"{prefix}-{entity_type}" for prefix in _PREFIXES)

        self.tag_to_id = {tag: idx for idx, tag in enumerate(self.tags)}
        self.id_to_tag = {idx: tag for tag, idx in self.tag_to_id.items()}
        self.intent_to_id = {name: idx for idx, name in enumerate(self.intents)}
        self.id_to_intent = {idx: name for name, idx in self.intent_to_id.items()}

    @property
    def num_tags(self) -> int:
        return len(self.tags)

    @property
    def num_intents(self) -> int:
        return len(self.intents)

    def to_dict(self) -> dict:
        return {"entity_types": self.entity_types, "intents": self.intents}

    @classmethod
    def from_dict(cls, payload: dict) -> LabelScheme:
        return cls(
            entity_types=list(payload["entity_types"]),
            intents=list(payload["intents"]),
        )


def split_tag(tag: str) -> tuple[str, str | None]:
    """Return the (prefix, entity_type) pair for a tag; ``("O", None)`` for outside."""
    if tag == OUTSIDE or "-" not in tag:
        return OUTSIDE, None
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type
