"""Inference: turn raw text into an intent and typed entity spans.

Entity values are recovered by slicing the original string with character
offsets rather than by joining word-pieces. That is why this returns "Cam004"
and not "cam 00 4", and why casing and punctuation survive intact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .config import Config
from .metrics import decode_spans
from .model import JointIntentNERModel, load_tokenizer, resolve_device


@dataclass
class Entity:
    type: str
    value: str
    start: int  # character offset into the input text
    end: int
    confidence: float


@dataclass
class Prediction:
    text: str
    intent: str
    intent_confidence: float
    entities: list[Entity] = field(default_factory=list)

    @property
    def entities_by_type(self) -> dict[str, list[str]]:
        """Convenience view for callers that just want 'the sensor' and 'the twin'."""
        grouped: dict[str, list[str]] = {}
        for entity in self.entities:
            grouped.setdefault(entity.type, []).append(entity.value)
        return grouped

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent": {"label": self.intent, "confidence": round(self.intent_confidence, 4)},
            "entities": [
                {**asdict(entity), "confidence": round(entity.confidence, 4)}
                for entity in self.entities
            ],
            "entities_by_type": self.entities_by_type,
        }


class Predictor:
    """Loads a checkpoint once and answers queries against it."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        max_length: int | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.model, self.scheme, self.metadata = JointIntentNERModel.load(
            checkpoint, self.device
        )
        self.tokenizer = load_tokenizer(self.model.encoder_name)
        self.max_length = max_length or int(self.metadata.get("max_length", 64))

    @classmethod
    def from_config(cls, config: Config) -> Predictor:
        # max_length is deliberately not passed: the checkpoint records the
        # length it was trained with, which is what inference must honour even
        # if the config has since been edited.
        return cls(config.checkpoint_path, device=config.train.device)

    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Predict for several utterances in one forward pass."""
        cleaned = [text if isinstance(text, str) else "" for text in texts]
        if not cleaned:
            return []

        encoding = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }
        outputs = self.model(**inputs)

        tag_probs = outputs.tag_logits.softmax(dim=-1).cpu()
        intent_probs = outputs.intent_logits.softmax(dim=-1).cpu()

        predictions: list[Prediction] = []
        for row, text in enumerate(cleaned):
            offsets = encoding["offset_mapping"][row].tolist()
            special_mask = encoding["special_tokens_mask"][row].tolist()
            attention = encoding["attention_mask"][row].tolist()

            # Keep only real content tokens, carrying their offsets and scores along.
            kept_tags: list[str] = []
            kept_offsets: list[tuple[int, int]] = []
            kept_scores: list[float] = []
            for index, (is_special, is_attended) in enumerate(
                zip(special_mask, attention, strict=True)
            ):
                if is_special or not is_attended:
                    continue
                start, end = offsets[index]
                if start == end:
                    continue
                best = int(tag_probs[row, index].argmax())
                kept_tags.append(self.scheme.id_to_tag[best])
                kept_offsets.append((start, end))
                kept_scores.append(float(tag_probs[row, index, best]))

            entities = []
            for span in decode_spans(kept_tags):
                char_start = kept_offsets[span.start][0]
                char_end = kept_offsets[span.end - 1][1]
                token_scores = kept_scores[span.start : span.end]
                entities.append(
                    Entity(
                        type=span.entity_type,
                        value=text[char_start:char_end],
                        start=char_start,
                        end=char_end,
                        confidence=sum(token_scores) / len(token_scores),
                    )
                )

            intent_index = int(intent_probs[row].argmax())
            predictions.append(
                Prediction(
                    text=text,
                    intent=self.scheme.id_to_intent[intent_index],
                    intent_confidence=float(intent_probs[row, intent_index]),
                    entities=entities,
                )
            )
        return predictions

    def predict(self, text: str) -> Prediction:
        """Predict for a single utterance."""
        return self.predict_batch([text])[0]


def format_prediction(prediction: Prediction) -> str:
    """Render a prediction for the terminal, underlining entities in the text."""
    lines = [
        "",
        f'  "{prediction.text}"',
        f"  intent   {prediction.intent}  ({prediction.intent_confidence:.1%})",
    ]
    if prediction.entities:
        lines.append("  entities")
        width = max(len(entity.type) for entity in prediction.entities)
        for entity in sorted(prediction.entities, key=lambda e: e.start):
            lines.append(
                f"    {entity.type:<{width}}  {entity.value}"
                f"  [{entity.start}:{entity.end}]  ({entity.confidence:.1%})"
            )
    else:
        lines.append("  entities  (none found)")
    lines.append("")
    return "\n".join(lines)


def interactive_loop(predictor: Predictor) -> None:
    """Simple REPL for poking at the model by hand."""
    print("\nTwin-NER interactive mode. Type a query, or 'quit' to exit.\n")
    while True:
        try:
            text = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break
        print(format_prediction(predictor.predict(text)))
