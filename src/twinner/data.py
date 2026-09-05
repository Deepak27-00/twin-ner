"""Dataset loading, entity/character alignment and BILOU tagging.

The alignment here is offset-based: entity values are located as character
spans in the *original* text, then projected onto tokens using the tokenizer's
offset mapping. That is what keeps "Cam004" a single entity instead of the
three word-pieces "cam", "##00", "##4", and it means the predicted surface form
can be sliced straight out of the input string, casing intact.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .schema import IGNORE_INDEX, OUTSIDE, LabelScheme

logger = logging.getLogger(__name__)


@dataclass
class Example:
    """One labelled utterance."""

    text: str
    intent: str
    entities: dict[str, str]  # entity type -> surface value


def parse_entity_string(raw: str, field_map: dict[str, str]) -> dict[str, str]:
    """Parse "sensorType: motion; digitalTwinID:Cam004" into entity values.

    ``field_map`` maps entity type (SENSOR) to the dataset's key (sensorType).
    Keys are matched case-insensitively and whitespace is tolerated on both
    sides of the separators.
    """
    lookup = {key.strip().lower(): entity_type for entity_type, key in field_map.items()}
    parsed: dict[str, str] = {}
    if not isinstance(raw, str):
        return parsed

    for part in raw.split(";"):
        key, sep, value = part.partition(":")
        if not sep:
            continue
        entity_type = lookup.get(key.strip().lower())
        if entity_type is not None:
            parsed[entity_type] = value.strip()
    return parsed


def find_span(text: str, value: str, null_values: set[str]) -> tuple[int, int] | None:
    """Locate ``value`` in ``text`` as a character span, or None if absent.

    Matching is case-insensitive and prefers occurrences sitting on word
    boundaries, so looking for "cam001" does not latch onto the middle of
    "webcam001x".
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in null_values:
        return None

    haystack, needle = text.lower(), value.lower()
    fallback: tuple[int, int] | None = None
    for match in re.finditer(re.escape(needle), haystack):
        start, end = match.span()
        left_clear = start == 0 or not text[start - 1].isalnum()
        right_clear = end == len(text) or not text[end].isalnum()
        if left_clear and right_clear:
            return start, end
        if fallback is None:
            fallback = (start, end)
    return fallback


def assign_bilou(
    offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
    spans: dict[str, tuple[int, int]],
    scheme: LabelScheme,
) -> list[str | None]:
    """Project character spans onto tokens as BILOU tags.

    Returns one entry per token: a tag string, or None for special tokens and
    padding, which must be excluded from the loss.
    """
    tags: list[str | None] = [None if special else OUTSIDE for special in special_tokens_mask]

    for entity_type, (span_start, span_end) in spans.items():
        covered = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if not special_tokens_mask[idx]
            and start < end  # skip zero-width offsets
            and start < span_end
            and end > span_start
        ]
        if not covered:
            logger.debug(
                "Span %s (%d,%d) matched no tokens", entity_type, span_start, span_end
            )
            continue
        if any(tags[idx] != OUTSIDE for idx in covered):
            logger.warning(
                "Overlapping entities near chars %d-%d; keeping the first match",
                span_start,
                span_end,
            )
            continue

        if len(covered) == 1:
            tags[covered[0]] = "U-" + entity_type
        else:
            tags[covered[0]] = "B-" + entity_type
            tags[covered[-1]] = "L-" + entity_type
            for idx in covered[1:-1]:
                tags[idx] = "I-" + entity_type

    unknown = {tag for tag in tags if tag is not None and tag not in scheme.tag_to_id}
    if unknown:
        raise ValueError("Produced tags outside the label scheme: " + str(sorted(unknown)))
    return tags


def load_examples(csv_path: str | Path, cfg) -> list[Example]:
    """Read the raw CSV into Example objects, validating the columns."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run 'twinner synth' to generate one, "
            "or point data.raw_csv at your own CSV."
        )

    frame = pd.read_csv(path)
    required = {cfg.text_column, cfg.intent_column, cfg.entity_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            f"Found: {list(frame.columns)}"
        )

    examples: list[Example] = []
    for _, row in frame.iterrows():
        text = row[cfg.text_column]
        if not isinstance(text, str) or not text.strip():
            continue
        examples.append(
            Example(
                text=text.strip(),
                intent=str(row[cfg.intent_column]).strip(),
                entities=parse_entity_string(row[cfg.entity_column], cfg.entities),
            )
        )
    if not examples:
        raise ValueError(f"No usable rows found in {path}")
    return examples


def build_scheme(examples: list[Example], entity_types: list[str]) -> LabelScheme:
    """Derive the label scheme from configured entity types and observed intents."""
    return LabelScheme(
        entity_types=entity_types,
        intents=sorted({example.intent for example in examples}),
    )


def stratified_split(
    examples: list[Example], cfg
) -> tuple[list[Example], list[Example], list[Example]]:
    """Split into train/val/test, stratified by intent where class counts allow."""
    from sklearn.model_selection import train_test_split

    total = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total})")

    labels = [example.intent for example in examples]
    counts = pd.Series(labels).value_counts()
    # Stratifying needs at least one example per class in every split.
    stratify = labels if counts.min() >= 3 else None
    if stratify is None:
        logger.warning("Rarest intent has <3 examples; falling back to a random split")

    holdout_ratio = cfg.val_ratio + cfg.test_ratio
    train, holdout = train_test_split(
        examples,
        test_size=holdout_ratio,
        random_state=cfg.split_seed,
        stratify=stratify,
    )

    holdout_labels = [example.intent for example in holdout]
    holdout_counts = pd.Series(holdout_labels).value_counts()
    stratify_holdout = holdout_labels if holdout_counts.min() >= 2 else None
    val, test = train_test_split(
        holdout,
        test_size=cfg.test_ratio / holdout_ratio,
        random_state=cfg.split_seed,
        stratify=stratify_holdout,
    )
    return train, val, test


class TwinNERDataset(Dataset):
    """Tokenises on the fly and returns tensors ready for the joint model."""

    def __init__(self, examples, tokenizer, scheme, max_length, null_values):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.scheme = scheme
        self.max_length = max_length
        self.null_values = {value.lower() for value in null_values}

    def __len__(self) -> int:
        return len(self.examples)

    def encode(self, example: Example) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            example.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        offsets = [tuple(pair) for pair in encoding["offset_mapping"][0].tolist()]
        special_mask = encoding["special_tokens_mask"][0].tolist()

        spans: dict[str, tuple[int, int]] = {}
        for entity_type, value in example.entities.items():
            span = find_span(example.text, value, self.null_values)
            if span is not None:
                spans[entity_type] = span

        tags = assign_bilou(offsets, special_mask, spans, self.scheme)
        # IGNORE_INDEX keeps padding and special tokens out of the loss entirely,
        # instead of teaching the model that padding belongs to the "O" class.
        label_ids = [
            IGNORE_INDEX if tag is None else self.scheme.tag_to_id[tag] for tag in tags
        ]

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "intent_label": torch.tensor(
                self.scheme.intent_to_id.get(example.intent, 0), dtype=torch.long
            ),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.encode(self.examples[index])


def prepare_splits(config, save: bool = True) -> tuple[dict, LabelScheme]:
    """Load, split and (optionally) persist the dataset splits.

    Splits are written to disk so that training, evaluation and any later error
    analysis all look at exactly the same rows -- the cheapest defence against
    accidentally reporting scores on data the model was trained on.
    """
    examples = load_examples(config.data.raw_csv, config.data)
    scheme = build_scheme(examples, list(config.data.entities))
    train, val, test = stratified_split(examples, config.data)
    splits = {"train": train, "val": val, "test": test}

    if save:
        out_dir = Path(config.data.processed_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            pd.DataFrame(
                [
                    {
                        "text": row.text,
                        "intent": row.intent,
                        **{f"entity_{k}": v for k, v in sorted(row.entities.items())},
                    }
                    for row in rows
                ]
            ).to_csv(out_dir / f"{name}.csv", index=False)

    return splits, scheme
