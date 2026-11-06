"""API behaviour: which scorer is used, real held-out transactions, input validation.

    pip install -r requirements.txt pytest
    pytest -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as fraud_app  # noqa: E402

SAMPLES = json.loads((ROOT / "samples" / "heldout_transactions.json").read_text(encoding="utf-8"))["transactions"]


@pytest.fixture()
def client():
    return fraud_app.app.test_client()


def evaluate(client, **body):
    res = client.post("/evaluate", json={"card_id": "card-t", "merchant_id": "m-t", "country_code": "US", **body})
    return res.status_code, res.get_json()


def test_without_bank_features_the_rules_score_it(client):
    status, data = evaluate(client, amount_usd=45)
    assert status == 200
    assert data["score_source"] == "rules" and data["ml_score"] is None
    assert data["is_fraud"] is False


def test_rules_flag_a_large_foreign_payment(client):
    status, data = evaluate(client, card_id="card-rules", amount_usd=2400, country_code="RU")
    assert status == 200 and data["score_source"] == "rules"
    assert data["base_score"] == pytest.approx(0.7)  # > $500 (0.4) + international (0.3)


def test_real_heldout_transactions_are_scored_by_the_model(client):
    scores = {"fraud": [], "legit": []}
    for i, s in enumerate(SAMPLES):
        status, data = evaluate(client, card_id=f"card-sample-{i}", amount_usd=s["amount_usd"],
                                time_seconds=s["time_seconds"], bank_features=s["bank_features"])
        assert status == 200 and data["score_source"] == "model"
        scores[s["label"]].append(data["ml_score"])
    # matches the held-out evaluation (recall 0.83, very few false alarms): 9 of these 10 frauds are caught,
    # test-119714 is a genuine miss, and no legitimate payment is flagged
    assert sum(p >= 0.5 for p in scores["fraud"]) == 9
    assert sum(p >= 0.5 for p in scores["legit"]) == 0


@pytest.mark.parametrize("bad", [[0.0] * 27, "not-a-list", [0.0] * 27 + ["x"], [float("nan")] * 28])
def test_malformed_bank_features_are_rejected(client, bad):
    status, data = evaluate(client, amount_usd=10, bank_features=bad)
    assert status == 400 and "error" in data


def test_samples_endpoint_lists_real_transactions(client):
    data = client.get("/samples").get_json()
    labels = [t["label"] for t in data["transactions"]]
    assert labels.count("fraud") == 10 and labels.count("legit") == 10
    assert all(len(t["bank_features"]) == 28 for t in data["transactions"])
