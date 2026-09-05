"""Running a trained model over a split and scoring it."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import TwinNERDataset, prepare_splits
from .metrics import EvalReport, evaluate_predictions, format_report
from .model import JointIntentNERModel, load_tokenizer, resolve_device
from .schema import IGNORE_INDEX, LabelScheme

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_predictions(
    model: JointIntentNERModel,
    loader: DataLoader,
    scheme: LabelScheme,
    device: torch.device,
) -> tuple[list[list[str]], list[list[str]], list[str], list[str], float]:
    """Run the model over a loader and return aligned gold/predicted labels.

    Positions marked with IGNORE_INDEX (special tokens, padding) are dropped from
    both sequences, so metrics only ever see real content tokens.
    """
    model.eval()
    true_tags: list[list[str]] = []
    pred_tags: list[list[str]] = []
    true_intents: list[str] = []
    pred_intents: list[str] = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        if outputs.loss is not None:
            total_loss += outputs.loss.item()
            num_batches += 1

        tag_predictions = outputs.tag_logits.argmax(dim=-1).cpu()
        intent_predictions = outputs.intent_logits.argmax(dim=-1).cpu()
        gold_labels = batch["labels"].cpu()
        gold_intents = batch["intent_label"].cpu()

        for row in range(gold_labels.size(0)):
            keep = gold_labels[row] != IGNORE_INDEX
            true_tags.append(
                [scheme.id_to_tag[int(i)] for i in gold_labels[row][keep].tolist()]
            )
            pred_tags.append(
                [scheme.id_to_tag[int(i)] for i in tag_predictions[row][keep].tolist()]
            )
            true_intents.append(scheme.id_to_intent[int(gold_intents[row])])
            pred_intents.append(scheme.id_to_intent[int(intent_predictions[row])])

    average_loss = total_loss / num_batches if num_batches else float("nan")
    return true_tags, pred_tags, true_intents, pred_intents, average_loss


def score_loader(
    model: JointIntentNERModel,
    loader: DataLoader,
    scheme: LabelScheme,
    device: torch.device,
) -> tuple[EvalReport, float]:
    """Evaluate one loader, returning the report and the mean loss."""
    true_tags, pred_tags, true_intents, pred_intents, loss = collect_predictions(
        model, loader, scheme, device
    )
    report = evaluate_predictions(true_tags, pred_tags, true_intents, pred_intents)
    return report, loss


def run_evaluation(config: Config, split: str = "test") -> EvalReport:
    """CLI entry point: score the saved checkpoint on one split."""
    device = resolve_device(config.train.device)
    model, scheme, extra = JointIntentNERModel.load(config.checkpoint_path, device)

    trained_on = extra.get("dataset")
    if trained_on and str(config.data.raw_csv) != trained_on:
        # Scoring a checkpoint against a corpus it never saw silently produces
        # numbers that mean nothing, so say so loudly rather than printing them.
        print(
            f"\n  WARNING: this checkpoint was trained on {trained_on}, but you are"
            f" evaluating against {config.data.raw_csv}.\n"
            f"           Pass --data {trained_on} to score the split it was trained on."
        )

    splits, _ = prepare_splits(config, save=False)
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Choose from {sorted(splits)}.")

    tokenizer = load_tokenizer(model.encoder_name)
    dataset = TwinNERDataset(
        splits[split],
        tokenizer,
        scheme,
        config.model.max_length,
        config.data.null_values,
    )
    loader = DataLoader(dataset, batch_size=config.train.eval_batch_size)

    report, loss = score_loader(model, loader, scheme, device)
    print(format_report(report, f"Test-set evaluation ({split} split)"))
    print(f"  loss                 {loss:.4f}")
    if extra.get("trained_at"):
        print(f"  checkpoint trained   {extra['trained_at']}")
    if trained_on:
        print(f"  trained on           {trained_on}")

    output_path = Path(config.train.output_dir) / f"metrics_{split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(f"  metrics written to   {output_path}\n")
    return report
