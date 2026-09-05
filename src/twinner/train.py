"""Training loop with validation, early stopping and reproducible seeding."""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .data import TwinNERDataset, prepare_splits
from .evaluate import score_loader
from .metrics import format_report
from .model import JointIntentNERModel, load_tokenizer, resolve_device

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches, so runs are comparable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config: Config) -> dict:
    """Fine-tune the joint model and save the best checkpoint by validation score.

    "Best" is the joint score (entity F1 and intent accuracy, evenly weighted)
    on the validation split -- never the training loss, which keeps improving
    long after the model has started memorising.
    """
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)
    print(f"\nDevice: {device}")

    splits, scheme = prepare_splits(config, save=True)
    print(
        f"Data:   {len(splits['train'])} train / {len(splits['val'])} val / "
        f"{len(splits['test'])} test"
    )
    print(f"Labels: {scheme.num_tags} BILOU tags, {scheme.num_intents} intents")

    tokenizer = load_tokenizer(config.model.encoder)

    def make_loader(name: str, batch_size: int, shuffle: bool) -> DataLoader:
        dataset = TwinNERDataset(
            splits[name],
            tokenizer,
            scheme,
            config.model.max_length,
            config.data.null_values,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader("train", config.train.batch_size, shuffle=True)
    val_loader = make_loader("val", config.train.eval_batch_size, shuffle=False)

    model = JointIntentNERModel(
        encoder_name=config.model.encoder,
        num_tags=scheme.num_tags,
        num_intents=scheme.num_intents,
        dropout=config.model.dropout,
        intent_loss_weight=config.model.intent_loss_weight,
    ).to(device)

    # No weight decay on biases or LayerNorm gains: decaying them costs accuracy
    # and is the standard exception in transformer fine-tuning recipes.
    no_decay = ("bias", "LayerNorm.weight")
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(marker in n for marker in no_decay)
            ],
            "weight_decay": config.train.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(marker in n for marker in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(grouped_parameters, lr=config.train.learning_rate)

    total_steps = max(1, len(train_loader) * config.train.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.train.warmup_ratio),
        num_training_steps=total_steps,
    )

    history: list[dict] = []
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config.train.epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        train_loss = running_loss / max(1, len(train_loader))
        report, val_loss = score_loader(model, val_loader, scheme, device)

        print(
            f"epoch {epoch:>2}/{config.train.epochs}  "
            f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"entity_f1 {report.entity_micro.f1:.3f}  "
            f"intent_acc {report.intent_accuracy:.3f}  "
            f"joint {report.joint_score:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **report.to_dict(),
            }
        )

        if report.joint_score > best_score:
            best_score = report.joint_score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save(
                config.checkpoint_path,
                scheme,
                extra={
                    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "val_joint_score": best_score,
                    "val_entity_f1": report.entity_micro.f1,
                    "val_intent_accuracy": report.intent_accuracy,
                    "max_length": config.model.max_length,
                    "train_size": len(splits["train"]),
                    # Recorded so `twinner evaluate` can warn when it is about
                    # to score this checkpoint against a different corpus.
                    "dataset": str(config.data.raw_csv),
                    "split_seed": config.data.split_seed,
                },
            )
            print(f"          saved checkpoint (best so far) -> {config.checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.patience:
                print(
                    f"\nEarly stop: no improvement for {config.train.patience} epochs "
                    f"(best was epoch {best_epoch})."
                )
                break

    elapsed = time.time() - started
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.used.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    print(f"\nTraining finished in {elapsed / 60:.1f} min. Best epoch: {best_epoch}.")

    # Final word on the held-out test split, using the best checkpoint rather
    # than whatever the last epoch happened to leave in memory.
    best_model, best_scheme, _ = JointIntentNERModel.load(config.checkpoint_path, device)
    test_loader = make_loader("test", config.train.eval_batch_size, shuffle=False)
    test_report, test_loss = score_loader(best_model, test_loader, best_scheme, device)
    print(format_report(test_report, "Held-out test set"))
    (output_dir / "metrics_test.json").write_text(
        json.dumps(test_report.to_dict(), indent=2), encoding="utf-8"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_joint_score": best_score,
        "test": test_report.to_dict(),
        "test_loss": test_loss,
        "history": history,
        "checkpoint": str(config.checkpoint_path),
    }
