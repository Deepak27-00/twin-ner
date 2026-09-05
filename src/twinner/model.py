"""The joint intent + entity model, and self-describing checkpoints.

A chatbot needs two answers from one utterance: *what is being asked*
(the intent) and *what it is being asked about* (the entities). Running two
separate encoders doubles the latency and the memory for no benefit, so a
single transformer feeds two heads:

    encoder -> per-token states -> tag head    -> BILOU tags
            -> [CLS] state      -> intent head -> intent label

Checkpoints store the label scheme and the encoder name alongside the weights,
so inference never has to re-declare a tag vocabulary that must "match
training" by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .schema import IGNORE_INDEX, LabelScheme


@dataclass
class ModelOutput:
    tag_logits: torch.Tensor
    intent_logits: torch.Tensor
    loss: torch.Tensor | None = None
    tag_loss: torch.Tensor | None = None
    intent_loss: torch.Tensor | None = None


class JointIntentNERModel(nn.Module):
    """Transformer encoder with a token-classification head and an intent head."""

    def __init__(
        self,
        encoder_name: str,
        num_tags: int,
        num_intents: int,
        dropout: float = 0.1,
        intent_loss_weight: float = 1.0,
        *,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.intent_loss_weight = intent_loss_weight

        if pretrained:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        else:
            # Used when restoring a checkpoint: the pretrained weights are about
            # to be overwritten, so only the architecture config is needed.
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(encoder_name))

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.tag_head = nn.Linear(hidden_size, num_tags)
        self.intent_head = nn.Linear(hidden_size, num_intents)

        self.num_tags = num_tags
        self.num_intents = num_intents

        self.tag_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        self.intent_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        intent_label: torch.Tensor | None = None,
    ) -> ModelOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(encoded.last_hidden_state)

        tag_logits = self.tag_head(sequence_output)
        # [CLS] is BERT's sentence-level summary slot: the natural place to read
        # a single label for the whole utterance from.
        intent_logits = self.intent_head(sequence_output[:, 0, :])

        loss = tag_loss = intent_loss = None
        if labels is not None and intent_label is not None:
            tag_loss = self.tag_loss_fn(
                tag_logits.reshape(-1, self.num_tags), labels.reshape(-1)
            )
            intent_loss = self.intent_loss_fn(intent_logits, intent_label)
            loss = tag_loss + self.intent_loss_weight * intent_loss

        return ModelOutput(
            tag_logits=tag_logits,
            intent_logits=intent_logits,
            loss=loss,
            tag_loss=tag_loss,
            intent_loss=intent_loss,
        )

    def save(self, path: str | Path, scheme: LabelScheme, extra: dict | None = None) -> Path:
        """Write a self-contained checkpoint: weights plus everything to rebuild."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "state_dict": self.state_dict(),
                "encoder_name": self.encoder_name,
                "scheme": scheme.to_dict(),
                "dropout": float(self.dropout.p),
                "intent_loss_weight": float(self.intent_loss_weight),
                "extra": extra or {},
            },
            destination,
        )
        return destination

    @classmethod
    def load(
        cls, path: str | Path, device: torch.device | str = "cpu"
    ) -> tuple[JointIntentNERModel, LabelScheme, dict]:
        """Restore a checkpoint together with its label scheme and metadata."""
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint_path}. Train one with 'twinner train'."
            )

        # weights_only=True refuses to unpickle arbitrary objects; our checkpoints
        # hold only tensors and plain containers, so nothing is lost by being strict.
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        scheme = LabelScheme.from_dict(checkpoint["scheme"])

        model = cls(
            encoder_name=checkpoint["encoder_name"],
            num_tags=scheme.num_tags,
            num_intents=scheme.num_intents,
            dropout=checkpoint.get("dropout", 0.1),
            intent_loss_weight=checkpoint.get("intent_loss_weight", 1.0),
            pretrained=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        return model, scheme, checkpoint.get("extra", {})


def load_tokenizer(encoder_name: str):
    """Fast tokenizer, required because the pipeline depends on offset mappings."""
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{encoder_name} has no fast tokenizer, but offset mapping is required "
            "for character-accurate entity spans."
        )
    return tokenizer


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the best available device, honouring an explicit preference."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
