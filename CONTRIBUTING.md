# Contributing

Thanks for taking a look. Issues and pull requests are welcome.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check src tests
ruff format src tests
pytest
```

API tests skip unless a checkpoint exists. To exercise them, train a small model
first:

```bash
twinner synth --size 400 --out data/raw/dev.csv
twinner train --data data/raw/dev.csv --epochs 1
pytest tests/test_api.py
```

## Guidelines

- Keep the label scheme data-driven. Adding an entity type should stay a config
  change, not a code change.
- Anything touching tokenisation, alignment or span decoding needs a test —
  these are the parts that fail silently rather than loudly.
- Comments should explain *why*, not restate the code.
- No model weights, datasets over a few MB, or generated artifacts in commits.
