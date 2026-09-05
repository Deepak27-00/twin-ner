"""Synthetic dataset generator.

The bundled dataset is small and templated, which is fine for a smoke test but
lets the model memorise surface patterns. This generator produces a larger,
more varied corpus in the same CSV format: more phrasings, more sensor types,
more twin naming conventions, and -- importantly -- utterances with no twin ID
at all, so the model learns that an entity may legitimately be absent.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

SENSOR_TYPES = [
    "temperature",
    "humidity",
    "motion",
    "camera",
    "pressure",
    "occupancy",
    "air quality",
    "carbon dioxide",
    "smoke",
    "vibration",
    "energy",
    "water flow",
    "light level",
    "noise",
]

TWIN_PREFIXES = ["Building", "Cam", "Floor", "Zone", "Gateway", "Room", "Plant", "Line"]

# Templates carry {sensor} and, where present, {twin}. Each intent keeps a mix
# of imperative, question and terse phrasings so the model does not key off a
# single leading verb.
TEMPLATES: dict[str, dict[str, list[str]]] = {
    "LIST_SENSORS": {
        "with_twin": [
            "List all {sensor} sensors in {twin}.",
            "Show me every {sensor} sensor on {twin}.",
            "What {sensor} sensors are installed in {twin}?",
            "Give me the {sensor} sensor inventory for {twin}.",
            "{twin} {sensor} sensors, please.",
            "Can you list the {sensor} sensors deployed at {twin}?",
            "I need a list of {sensor} sensors under {twin}.",
            "Enumerate {sensor} sensors for {twin}.",
            "How many {sensor} sensors does {twin} have?",
            "Pull up all {sensor} sensors associated with {twin}.",
        ],
        "without_twin": [
            "List all {sensor} sensors.",
            "Show me every {sensor} sensor we have.",
            "What {sensor} sensors exist?",
            "Give me the full {sensor} sensor inventory.",
            "Enumerate all {sensor} sensors across the estate.",
        ],
    },
    "GET_LATEST_DATA": {
        "with_twin": [
            "Fetch the latest {sensor} data from {twin}.",
            "What is the current {sensor} reading at {twin}?",
            "Get me {twin}'s most recent {sensor} value.",
            "Show the newest {sensor} measurement for {twin}.",
            "Latest {sensor} reading on {twin}?",
            "Pull the last {sensor} datapoint from {twin}.",
            "How is the {sensor} looking in {twin} right now?",
            "Give me real-time {sensor} data for {twin}.",
            "Read the {sensor} sensor at {twin}.",
            "Current {sensor} status of {twin}, please.",
        ],
        "without_twin": [
            "Fetch the latest {sensor} data.",
            "What is the current {sensor} reading?",
            "Show me the newest {sensor} measurement.",
            "Give me real-time {sensor} data.",
        ],
    },
    "GET_AVAILABLE_SENSORS_BY_TYPE": {
        "with_twin": [
            "Retrieve all {sensor} sensors from {twin}.",
            "Which {sensor} sensors are available in {twin}?",
            "Find available {sensor} sensors under {twin}.",
            "Query {twin} for {sensor} sensor availability.",
            "Are there any {sensor} sensors online in {twin}?",
            "Check {twin} for active {sensor} sensors.",
            "Look up available {sensor} devices at {twin}.",
            "Search {twin} for {sensor} sensors that are reporting.",
        ],
        "without_twin": [
            "Which {sensor} sensors are available?",
            "Find all available {sensor} sensors.",
            "Are there any {sensor} sensors online?",
            "Check for active {sensor} sensors.",
        ],
    },
}

NO_TWIN_RATE = 0.18  # share of utterances that mention no digital twin


def _twin_ids(rng: random.Random, count: int = 60) -> list[str]:
    """Build a pool of twin identifiers across several naming conventions."""
    pool: list[str] = []
    while len(pool) < count:
        prefix = rng.choice(TWIN_PREFIXES)
        style = rng.random()
        if style < 0.6:
            pool.append(f"{prefix}{rng.randint(1, 40):03d}")
        elif style < 0.85:
            pool.append(f"{prefix}-{rng.randint(1, 20)}")
        else:
            pool.append(f"{prefix}_{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}")
    return sorted(set(pool))


def generate(size: int = 4000, seed: int = 42) -> list[dict[str, str]]:
    """Generate `size` labelled rows as dicts with text/intent/entities keys."""
    rng = random.Random(seed)
    twins = _twin_ids(rng)
    intents = list(TEMPLATES)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    # Cap the attempts so an over-large `size` degrades to "as many unique rows
    # as the templates can produce" instead of looping forever.
    attempts = 0
    max_attempts = size * 20

    while len(rows) < size and attempts < max_attempts:
        attempts += 1
        intent = rng.choice(intents)
        sensor = rng.choice(SENSOR_TYPES)
        use_twin = rng.random() > NO_TWIN_RATE

        if use_twin:
            twin = rng.choice(twins)
            text = rng.choice(TEMPLATES[intent]["with_twin"]).format(sensor=sensor, twin=twin)
            entities = f"sensorType: {sensor}; digitalTwinID: {twin}"
        else:
            text = rng.choice(TEMPLATES[intent]["without_twin"]).format(sensor=sensor)
            entities = f"sensorType: {sensor}; digitalTwinID: not present"

        if text in seen:
            continue
        seen.add(text)
        rows.append({"text": text, "intent": intent, "entities": entities})

    rng.shuffle(rows)
    return rows


def write_dataset(path: str | Path, size: int = 4000, seed: int = 42) -> Path:
    """Generate and write a dataset CSV, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = generate(size=size, seed=seed)

    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "intent", "entities"])
        writer.writeheader()
        writer.writerows(rows)
    return destination
