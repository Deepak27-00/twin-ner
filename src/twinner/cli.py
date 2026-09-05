"""Command-line interface: prepare, train, evaluate, predict, serve, synth.

Heavy imports (torch, transformers) are deliberately deferred into each
sub-command so that `twinner --help` stays instant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from . import __version__
from .config import Config

_EPILOG = """\
examples:
  twinner synth --size 4000            generate a larger synthetic dataset
  twinner prepare                      write train/val/test splits to disk
  twinner train --epochs 5             fine-tune and save the best checkpoint
  twinner evaluate --split test        score the checkpoint on held-out data
  twinner predict "Show me motion sensors in Building003"
  twinner predict --interactive        REPL against the trained model
  twinner serve --port 8000            REST API + browser demo
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="twinner",
        description="Joint intent classification and entity extraction for "
        "digital-twin assistants.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"twinner {__version__}")
    parser.add_argument(
        "--config", default=None, help="path to a YAML config (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging verbosity (default: WARNING)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="build and save train/val/test splits")

    train_parser = subparsers.add_parser("train", help="fine-tune the joint model")
    train_parser.add_argument("--epochs", type=int, help="override train.epochs")
    train_parser.add_argument("--batch-size", type=int, help="override train.batch_size")
    train_parser.add_argument("--lr", type=float, help="override train.learning_rate")
    train_parser.add_argument("--encoder", help="override model.encoder (any HF model id)")
    train_parser.add_argument("--device", help="cpu | cuda | mps | auto")
    train_parser.add_argument("--output-dir", help="override train.output_dir")
    train_parser.add_argument("--data", help="override data.raw_csv")
    train_parser.add_argument("--seed", type=int, help="override train.seed")

    eval_parser = subparsers.add_parser("evaluate", help="score a trained checkpoint")
    eval_parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"], help="split to score"
    )
    eval_parser.add_argument("--device", help="cpu | cuda | mps | auto")
    eval_parser.add_argument("--data", help="override data.raw_csv")

    predict_parser = subparsers.add_parser("predict", help="run inference on text")
    predict_parser.add_argument("text", nargs="*", help="one or more utterances")
    predict_parser.add_argument(
        "-i", "--interactive", action="store_true", help="start a REPL"
    )
    predict_parser.add_argument(
        "--json", action="store_true", help="emit JSON instead of a formatted table"
    )
    predict_parser.add_argument("--device", help="cpu | cuda | mps | auto")

    serve_parser = subparsers.add_parser("serve", help="start the REST API and web demo")
    serve_parser.add_argument("--host", help="override serve.host")
    serve_parser.add_argument("--port", type=int, help="override serve.port")
    serve_parser.add_argument(
        "--reload", action="store_true", help="auto-reload on code changes (development)"
    )

    synth_parser = subparsers.add_parser(
        "synth", help="generate a synthetic dataset in the expected CSV format"
    )
    synth_parser.add_argument("--out", default="data/raw/sensor_queries_large.csv")
    synth_parser.add_argument("--size", type=int, default=4000, help="number of rows")
    synth_parser.add_argument("--seed", type=int, default=42)

    return parser


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config)
    overrides = {
        "train.epochs": getattr(args, "epochs", None),
        "train.batch_size": getattr(args, "batch_size", None),
        "train.learning_rate": getattr(args, "lr", None),
        "train.device": getattr(args, "device", None),
        "train.output_dir": getattr(args, "output_dir", None),
        "train.seed": getattr(args, "seed", None),
        "model.encoder": getattr(args, "encoder", None),
        "data.raw_csv": getattr(args, "data", None),
        "serve.host": getattr(args, "host", None),
        "serve.port": getattr(args, "port", None),
    }
    return config.apply_overrides(overrides)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s"
    )
    config = _load_config(args)

    if args.command == "prepare":
        from .data import prepare_splits

        splits, scheme = prepare_splits(config, save=True)
        print(f"\nWrote splits to {config.data.processed_dir}/")
        for name, rows in splits.items():
            print(f"  {name:<6} {len(rows):>5} examples")
        print(f"\n  intents:      {', '.join(scheme.intents)}")
        print(f"  entity types: {', '.join(scheme.entity_types)}")
        print(f"  BILOU tags:   {scheme.num_tags}\n")
        return 0

    if args.command == "train":
        from .train import train

        train(config)
        return 0

    if args.command == "evaluate":
        from .evaluate import run_evaluation

        run_evaluation(config, split=args.split)
        return 0

    if args.command == "predict":
        from .predict import Predictor, format_prediction, interactive_loop

        predictor = Predictor.from_config(config)
        if args.interactive or not args.text:
            interactive_loop(predictor)
            return 0
        predictions = predictor.predict_batch(list(args.text))
        if args.json:
            print(json.dumps([p.to_dict() for p in predictions], indent=2))
        else:
            for prediction in predictions:
                print(format_prediction(prediction))
        return 0

    if args.command == "serve":
        from .api import serve

        serve(config, reload=args.reload)
        return 0

    if args.command == "synth":
        from .synth import write_dataset

        path = write_dataset(args.out, size=args.size, seed=args.seed)
        print(f"Wrote {args.size} rows to {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
