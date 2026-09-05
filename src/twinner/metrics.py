"""Span-level evaluation.

Token accuracy flatters a NER model badly: in these utterances roughly 80% of
tokens are "O", so a model that predicts nothing at all still scores ~0.8.
What matters is whether a *whole span* was found with the right type and the
right boundaries, so precision/recall/F1 are computed over decoded spans.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .schema import OUTSIDE, split_tag


@dataclass(frozen=True)
class Span:
    entity_type: str
    start: int  # inclusive token index
    end: int  # exclusive token index


def decode_spans(tags: list[str]) -> list[Span]:
    """Decode a BILOU tag sequence into spans, tolerating malformed output.

    A trained model can emit sequences no annotation scheme would produce
    (an ``I-`` with no ``B-`` before it, a type switching mid-span). Rather than
    dropping those, the decoder closes the current span and starts a new one,
    which is the behaviour that keeps a demo useful on messy input.
    """
    spans: list[Span] = []
    current_type: str | None = None
    current_start = 0

    def close(end: int) -> None:
        nonlocal current_type
        if current_type is not None:
            spans.append(Span(current_type, current_start, end))
            current_type = None

    for index, tag in enumerate(tags):
        prefix, entity_type = split_tag(tag)

        if prefix == OUTSIDE:
            close(index)
            continue

        if prefix == "U":
            close(index)
            spans.append(Span(entity_type, index, index + 1))
            continue

        if prefix == "B":
            close(index)
            current_type, current_start = entity_type, index
            continue

        # I- or L-: continue the open span when the type agrees, otherwise treat
        # it as the start of a new one.
        if current_type != entity_type:
            close(index)
            current_type, current_start = entity_type, index

        if prefix == "L":
            close(index + 1)

    close(len(tags))
    return spans


@dataclass
class PRF:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0
    predicted: int = 0

    @classmethod
    def from_counts(cls, tp: int, fp: int, fn: int) -> PRF:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(precision, recall, f1, support=tp + fn, predicted=tp + fp)


@dataclass
class EvalReport:
    entity_micro: PRF = field(default_factory=PRF)
    entity_macro_f1: float = 0.0
    per_entity: dict[str, PRF] = field(default_factory=dict)
    intent_accuracy: float = 0.0
    intent_per_class: dict[str, PRF] = field(default_factory=dict)
    exact_match: float = 0.0  # intent and every entity correct in one utterance
    num_examples: int = 0

    @property
    def joint_score(self) -> float:
        """Single number for model selection: entity F1 and intent accuracy."""
        return 0.5 * self.entity_micro.f1 + 0.5 * self.intent_accuracy

    def to_dict(self) -> dict:
        return {
            "entity_micro": vars(self.entity_micro),
            "entity_macro_f1": self.entity_macro_f1,
            "per_entity": {k: vars(v) for k, v in self.per_entity.items()},
            "intent_accuracy": self.intent_accuracy,
            "intent_per_class": {k: vars(v) for k, v in self.intent_per_class.items()},
            "exact_match": self.exact_match,
            "joint_score": self.joint_score,
            "num_examples": self.num_examples,
        }


def evaluate_predictions(
    true_tags: list[list[str]],
    pred_tags: list[list[str]],
    true_intents: list[str],
    pred_intents: list[str],
) -> EvalReport:
    """Compute span-level entity metrics and intent metrics over a whole split."""
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    entity_types: set[str] = set()

    per_example_entities_correct: list[bool] = []

    for gold_seq, pred_seq in zip(true_tags, pred_tags, strict=True):
        gold = set(decode_spans(gold_seq))
        pred = set(decode_spans(pred_seq))
        per_example_entities_correct.append(gold == pred)

        for span in gold | pred:
            entity_types.add(span.entity_type)
        for span in gold & pred:
            tp[span.entity_type] += 1
        for span in pred - gold:
            fp[span.entity_type] += 1
        for span in gold - pred:
            fn[span.entity_type] += 1

    per_entity = {
        entity_type: PRF.from_counts(tp[entity_type], fp[entity_type], fn[entity_type])
        for entity_type in sorted(entity_types)
    }
    micro = PRF.from_counts(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    macro_f1 = (
        sum(scores.f1 for scores in per_entity.values()) / len(per_entity)
        if per_entity
        else 0.0
    )

    intent_tp: dict[str, int] = defaultdict(int)
    intent_fp: dict[str, int] = defaultdict(int)
    intent_fn: dict[str, int] = defaultdict(int)
    correct_intents = 0
    for gold_intent, pred_intent in zip(true_intents, pred_intents, strict=True):
        if gold_intent == pred_intent:
            correct_intents += 1
            intent_tp[gold_intent] += 1
        else:
            intent_fp[pred_intent] += 1
            intent_fn[gold_intent] += 1

    intent_labels = sorted(set(true_intents) | set(pred_intents))
    intent_per_class = {
        label: PRF.from_counts(intent_tp[label], intent_fp[label], intent_fn[label])
        for label in intent_labels
    }

    total = len(true_intents) or 1
    exact = sum(
        1
        for entities_ok, gold_intent, pred_intent in zip(
            per_example_entities_correct, true_intents, pred_intents, strict=True
        )
        if entities_ok and gold_intent == pred_intent
    )

    return EvalReport(
        entity_micro=micro,
        entity_macro_f1=macro_f1,
        per_entity=per_entity,
        intent_accuracy=correct_intents / total,
        intent_per_class=intent_per_class,
        exact_match=exact / total,
        num_examples=len(true_intents),
    )


def format_report(report: EvalReport, title: str = "Evaluation") -> str:
    """Render a report as a readable console table."""
    lines = [
        "",
        f"  {title}  ({report.num_examples} examples)",
        "  " + "-" * 62,
        f"  {'entity':<14}{'precision':>11}{'recall':>10}{'f1':>9}{'support':>10}",
    ]
    for name, scores in report.per_entity.items():
        lines.append(
            f"  {name:<14}{scores.precision:>11.3f}{scores.recall:>10.3f}"
            f"{scores.f1:>9.3f}{scores.support:>10d}"
        )
    micro = report.entity_micro
    lines += [
        "  " + "-" * 62,
        f"  {'micro avg':<14}{micro.precision:>11.3f}{micro.recall:>10.3f}"
        f"{micro.f1:>9.3f}{micro.support:>10d}",
        f"  {'macro f1':<14}{report.entity_macro_f1:>11.3f}",
        "",
        f"  intent accuracy      {report.intent_accuracy:.3f}",
        f"  exact match          {report.exact_match:.3f}   (intent + all entities)",
        f"  joint score          {report.joint_score:.3f}",
        "",
    ]
    return "\n".join(lines)
