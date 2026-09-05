"""API contract tests.

These need a trained checkpoint, so they skip cleanly on a fresh clone (and in
CI) rather than failing. Run `twinner train` first to exercise them.
"""

from __future__ import annotations

import pytest

from twinner.config import Config

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture(scope="module")
def client():
    config = Config.load()
    if not config.checkpoint_path.exists():
        pytest.skip(f"no checkpoint at {config.checkpoint_path}; run 'twinner train' first")

    from twinner.api import create_app

    with fastapi_testclient.TestClient(create_app(config)) as test_client:
        yield test_client


def test_health_reports_ready(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["device"]


def test_labels_lists_the_trained_label_space(client):
    payload = client.get("/labels").json()
    assert payload["intents"]
    assert "O" in payload["tags"]


def test_predict_returns_intent_and_entities(client):
    query = "Fetch the latest motion data from Cam004"
    response = client.post("/predict", json={"text": query})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"]["label"]
    assert 0.0 <= payload["intent"]["confidence"] <= 1.0
    for entity in payload["entities"]:
        # Offsets must slice the original text back to the reported value --
        # the guarantee that makes highlighting possible client-side.
        assert payload["text"][entity["start"] : entity["end"]] == entity["value"]


def test_empty_text_is_rejected(client):
    assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_oversized_text_is_rejected(client):
    long_text = "sensor " * 1000
    assert client.post("/predict", json={"text": long_text}).status_code == 413


def test_batch_prediction_returns_one_result_per_input(client):
    texts = ["List all smoke sensors in Building003", "What is the humidity at Cam001?"]
    payload = client.post("/predict/batch", json={"texts": texts}).json()
    assert len(payload["results"]) == len(texts)


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    config = Config.load()
    texts = ["hello"] * (config.serve.max_batch_size + 1)
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 413


def test_demo_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Twin-NER" in response.text
