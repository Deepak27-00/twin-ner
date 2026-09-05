# Twin-NER

**One BERT pass. Two answers: what the user wants, and what they want it about.**

A joint intent-classification and named-entity-recognition model for digital-twin
and IoT assistants. Ask it *"Fetch the latest motion data from Cam004"* and it
returns the intent (`GET_LATEST_DATA`) together with the entities it found
(`SENSOR: motion`, `TWIN: Cam004`) — each with a confidence score and the exact
character offsets it occupies in your string.

[![CI](https://github.com/Deepak27-00/twin-ner/actions/workflows/ci.yml/badge.svg)](https://github.com/Deepak27-00/twin-ner/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

---

## Contents

- [Why joint](#why-joint) · [Quickstart](#quickstart) · [Results](#results)
- [How it works](#how-it-works) · [Design notes](#design-notes)
- [CLI](#cli) · [REST API](#rest-api) · [Web demo](#web-demo)
- [Bring your own data](#bring-your-own-data) · [Configuration](#configuration)
- [Project layout](#project-layout) · [Development](#development) · [Limitations](#limitations)

---

## Why joint

A chatbot needs two things from every utterance: the **intent** (what action to
take) and the **entities** (what to take it on). The usual approach runs two
models, which means loading two transformers, paying for two forward passes,
and maintaining two training pipelines.

Twin-NER runs one encoder and reads two heads off it. Intent and entities are
learned together, so the shared representation is trained by both objectives at
once — and inference costs a single forward pass:

```
                                       ┌─ per-token states ─→ tag head    ─→ BILOU tags   ─→ entity spans
"Fetch motion from Cam004" ─→ encoder ─┤
                                       └─ [CLS] state       ─→ intent head ─→ GET_LATEST_DATA
```

## Quickstart

```bash
git clone https://github.com/Deepak27-00/twin-ner.git
cd twin-ner

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

# The CPU wheel is a fraction of the CUDA build and is plenty for this model.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Train on the bundled dataset and try it — no GPU required:

```bash
twinner train                                   # ~3 minutes on a laptop CPU
twinner predict "Fetch the latest motion data from Cam004"
```

```
  "Fetch the latest motion data from Cam004"
  intent   GET_LATEST_DATA  (98.7%)
  entities
    SENSOR  motion  [17:23]  (84.7%)
    TWIN    Cam004  [34:40]  (84.4%)
```

(Confidences sit in the 80s because the bundled corpus is only 250 rows. Train
on `twinner synth` output and they move above 99% — see [Results](#results).)

Then open the browser demo:

```bash
twinner serve          # http://127.0.0.1:8000
```

## Results

Trained on 3,000 generated utterances (`twinner synth --size 3000`), scored on a
held-out 15% test split. Entity scores are **span-level**: a prediction counts
only if the type *and* both boundaries are exactly right.

<!-- METRICS:START -->
| Metric | Score |
| --- | --- |
| Entity F1 (micro, span-level) | **1.000** |
| Entity F1 (macro) | **1.000** |
| Intent accuracy | **1.000** |
| Exact match (intent + every entity) | **1.000** |

<sub>450 held-out utterances · 872 gold entity spans · best epoch 1 of 4 ·
21 min on a laptop CPU, no GPU.</sub>
<!-- METRICS:END -->

Reproduce:

```bash
twinner synth --size 3000 --out data/raw/sensor_queries_large.csv
twinner train --data data/raw/sensor_queries_large.csv --epochs 4
twinner evaluate --data data/raw/sensor_queries_large.csv --split test
```

> **Read these numbers honestly: the task is saturated.** The corpus is
> template-generated, so a perfect score is the *expected* outcome, not an
> achievement — validation hit 1.000 after a single epoch. What it demonstrates
> is that the pipeline is wired correctly end to end: entity values align to the
> right tokens, padding stays out of the loss, spans decode back to exactly the
> right characters, and the scorer is strict enough that a single off-by-one
> boundary would show up as both a false positive and a false negative.
>
> It says nothing about real user traffic, where phrasing is unconstrained and
> entity vocabularies are open. Use this as a working reference implementation
> to point at your own labelled data, not as a benchmark result.

### The same numbers, out of distribution

Perfect in-distribution scores are cheap. Here is the more honest measurement:
the model trained on the bundled 250-row corpus, scored against the *varied*
3,000-row corpus it never saw — different phrasings, ten extra sensor types,
twin IDs in naming conventions it was never shown.

| Evaluated on | Entity F1 | Intent accuracy | Exact match |
| --- | --- | --- | --- |
| Its own test split (in-distribution) | 1.000 | 1.000 | 1.000 |
| Varied corpus (out-of-distribution) | **0.560** | **0.502** | **0.191** |

That collapse is the real result, and it is why `twinner evaluate` warns you
when a checkpoint is about to be scored against a corpus it was not trained on.
Template coverage, not model architecture, is the binding constraint here — the
fix is more varied training data, which is exactly what `twinner synth` exists
to make easy.

## How it works

```mermaid
flowchart LR
    A["Raw CSV<br/>text, intent, entities"] --> B["Parse entity string<br/>sensorType: motion; ..."]
    B --> C["Locate values as<br/>character spans"]
    C --> D["Project onto tokens<br/>via offset mapping"]
    D --> E["BILOU tags<br/>special tokens masked"]
    E --> F["Fine-tune BERT<br/>tag head + intent head"]
    F --> G["Best checkpoint<br/>by validation joint score"]
    G --> H["Predictor<br/>spans sliced from source text"]
    H --> I["CLI · REST API · Web demo"]
```

**BILOU, not BIO.** Tags are `B-`(egin), `I-`(nside), `L-`(ast), `O-`(utside)
and `U-`(nit). The `U-` tag is what distinguishes a complete one-token entity
from the *start* of a longer one — the difference between reading `motion` as a
finished entity and waiting for a continuation that never comes.

**Character spans, not token joins.** `Cam004` tokenises to `cam`, `##00`, `##4`.
Reconstructing entities by gluing word-pieces back together produces `cam 00 4`
and forces per-entity cleanup hacks. Instead the tokenizer's offset mapping is
used in both directions: entity values are located in the source string first,
then predictions are sliced straight back out of it. Casing, punctuation and
spacing survive untouched, and the API can hand a UI exact offsets to highlight.

## Design notes

Things that are easy to get subtly wrong in a NER pipeline, and how this one
handles them:

| Concern | Approach |
| --- | --- |
| **Padding in the loss** | Padding and `[CLS]`/`[SEP]` carry `-100`, so `CrossEntropyLoss(ignore_index=-100)` skips them. Labelling padding as `O` instead teaches the model to predict `O` on empty positions and inflates every token-level metric. |
| **Label/token alignment** | Tags are produced from the same tokenizer call that produces `input_ids`, so the two can never drift out of sync. |
| **Metric choice** | Roughly 80% of tokens are `O`, so token accuracy is ~0.8 for a model that predicts nothing at all. Scoring is span-level precision/recall/F1, plus an exact-match rate over the whole utterance. |
| **Model selection** | The best checkpoint is chosen by *validation joint score*, not by training loss, and training stops early when validation stops improving. |
| **Checkpoint portability** | Each checkpoint stores its own label scheme, encoder name and hyper-parameters. Inference never re-declares a tag vocabulary that has to "match training" by hand. |
| **Deserialisation safety** | Checkpoints load with `weights_only=True`, so a downloaded `.pt` cannot execute arbitrary code on load. |
| **Absent entities** | `digitalTwinID: not present` is a first-class case in both the data and the generator, so the model learns that an entity may legitimately be missing. |
| **Server defaults** | Binds `127.0.0.1`, no debug mode, CORS closed, request size capped. Exposing the service more widely is an explicit config change, not a default. |
| **Reproducibility** | One seed drives the split, the shuffling and the initialisation; the resolved config and per-epoch history are written next to the checkpoint. |

## CLI

```
twinner synth      generate a synthetic dataset in the expected CSV format
twinner prepare    build and save train/val/test splits
twinner train      fine-tune the joint model, save the best checkpoint
twinner evaluate   score a trained checkpoint on any split
twinner predict    run inference on text, or start a REPL
twinner serve      start the REST API and web demo
```

```bash
twinner train --epochs 6 --batch-size 32 --lr 2e-5 --device cuda
twinner train --encoder distilbert-base-uncased          # any HF encoder
twinner evaluate --split val
twinner evaluate --data data/raw/other.csv    # warns on a corpus mismatch
twinner predict --json "List all smoke sensors in Building003"
twinner predict --interactive
twinner --config configs/experiment.yaml train
```

## REST API

```bash
twinner serve --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Interactive browser demo |
| `GET /docs` | OpenAPI / Swagger UI |
| `GET /health` | Readiness, device, checkpoint provenance |
| `GET /labels` | The label space this checkpoint was trained on |
| `POST /predict` | Analyse one utterance |
| `POST /predict/batch` | Analyse many in one forward pass |

```bash
curl -s localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "Fetch the latest motion data from Cam004"}'
```

```json
{
  "text": "Fetch the latest motion data from Cam004",
  "intent": { "label": "GET_LATEST_DATA", "confidence": 0.9868 },
  "entities": [
    { "type": "SENSOR", "value": "motion", "start": 17, "end": 23, "confidence": 0.8475 },
    { "type": "TWIN",   "value": "Cam004", "start": 34, "end": 40, "confidence": 0.8438 }
  ],
  "entities_by_type": { "SENSOR": ["motion"], "TWIN": ["Cam004"] },
  "latency_ms": 30.02
}
```

Because `start`/`end` are real character offsets, a client can highlight the
original string without re-searching it:

```js
text.slice(entity.start, entity.end) === entity.value   // always true
```

## Web demo

`twinner serve` also serves a single-page demo at `/`: type a query (or pick an
example), and it renders the utterance with entities highlighted inline, the
predicted intent with a confidence bar, a table of spans with offsets, and the
raw JSON response. It is one dependency-free HTML file — no build step.

## Bring your own data

Point the config at any CSV with three columns:

| column | contents |
| --- | --- |
| `text` | the utterance |
| `intent` | one label per row |
| `entities` | `key: value` pairs separated by `;` |

```csv
text,intent,entities
"Fetch the latest motion data from Cam004",GET_LATEST_DATA,"sensorType: motion; digitalTwinID: Cam004"
"List all smoke sensors.",LIST_SENSORS,"sensorType: smoke; digitalTwinID: not present"
```

Entity values must appear verbatim in `text` (matching is case-insensitive);
values listed in `data.null_values` mean "absent from this utterance". Intents
are discovered from the data — no list to maintain.

**Adding a new entity type** is a two-line config change. To teach it locations:

```yaml
data:
  entities:
    SENSOR: sensorType
    TWIN: digitalTwinID
    LOCATION: locationName      # new
```

The BILOU tag set (`B-LOCATION`, `I-LOCATION`, `L-LOCATION`, `U-LOCATION`), the
model's output dimension, the decoder and the API response all follow
automatically. Nothing else needs editing — tagging picks up the new type,
multi-word values included:

```
Show motion sensors in Building001 on the third floor
  motion/U-SENSOR  building/B-TWIN ##00/I-TWIN ##1/L-TWIN  third/B-LOCATION floor/L-LOCATION

List all smoke sensors in the basement
  smoke/U-SENSOR  basement/U-LOCATION
```

(The second utterance names no twin, so no `TWIN` span is produced at all.)

## Configuration

Everything tunable lives in [`configs/default.yaml`](configs/default.yaml) —
data paths and entity map, encoder and sequence length, optimiser and schedule,
serving limits. Copy it, edit it, and pass `--config my.yaml`; the resolved
config is written next to each checkpoint so a run is always reconstructable.
Unknown keys are rejected rather than silently ignored, so a typo fails loudly.

## Project layout

```
src/twinner/
  schema.py     BILOU label scheme, ignore index
  config.py     typed config, YAML loading, CLI overrides
  data.py       parsing, character-span alignment, BILOU tagging, splits
  model.py      joint model, self-describing checkpoints, device resolution
  metrics.py    span decoding, span-level P/R/F1, intent metrics
  train.py      training loop, early stopping, best-checkpoint selection
  evaluate.py   scoring a checkpoint over a split
  predict.py    Predictor, offset-accurate entity extraction, REPL
  api.py        FastAPI service
  synth.py      synthetic dataset generator
  cli.py        command-line entry point
  web/          single-file browser demo
configs/        default.yaml
data/raw/       input datasets (splits are derived, never committed)
tests/          unit tests + API contract tests
```

## Development

```bash
pip install -e ".[dev]"
pytest                      # API tests skip until a checkpoint exists
ruff check src tests
ruff format src tests
```

CI runs lint, format check and tests on Python 3.10/3.11/3.12, plus a smoke job
that trains for one epoch and calls the API — so a change that breaks the
training path cannot pass silently.

## Limitations

- **Synthetic data.** The bundled corpus and everything `twinner synth`
  produces are template-generated. Real deployments need real utterances;
  expect the scores above to drop against unconstrained phrasing.
- **One intent per utterance.** "Show me the cameras and the humidity" gets a
  single label. Multi-intent parsing would need a different head.
- **No entity linking.** `Cam004` is extracted as a string; resolving it against
  a twin registry (and rejecting IDs that don't exist) is the caller's job.
- **English only**, inheriting `bert-base-uncased`. Swapping in a multilingual
  encoder is a one-line config change but needs matching training data.
- **No CRF layer.** A CRF would enforce valid BILOU transitions instead of
  letting the decoder repair malformed sequences after the fact.

## License

MIT — see [LICENSE](LICENSE).
