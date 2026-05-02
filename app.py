import os
import time
import uuid
import json
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np

app = Flask(__name__)

# ============================================================
# In-memory stores (replaces Redis for cloud demo)
# ============================================================

# Card transaction history: {card_id: [(timestamp, amount, merchant, country)]}
card_history = defaultdict(list)

# Card profiles: {card_id: {avg_spend, txn_count, countries, hours}}
card_profiles = defaultdict(lambda: {
    "txn_count": 0,
    "total_spend": 0.0,
    "avg_spend": 0.0,
    "max_spend": 0.0,
    "countries": [],
    "hours": [],
    "sum_squares": 0.0,
    "std_dev": 0.0
})

# Merchant recent cards: {merchant_id: [(timestamp, card_id)]}
merchant_cards = defaultdict(list)

# ============================================================
# Load ONNX model
# ============================================================
print("Loading ONNX model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fraud_model.onnx")
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
print(f"Model loaded. Input: {input_name}")

# ============================================================
# Feature engineering
# ============================================================

def get_sliding_window_features(card_id, amount, country):
    now = time.time()
    one_hour_ago = now - 3600
    
    # Clean old entries
    card_history[card_id] = [
        (ts, amt, merch, ctry) 
        for ts, amt, merch, ctry in card_history[card_id]
        if ts > one_hour_ago
    ]
    
    history = card_history[card_id]
    txn_count = len(history)
    total_spend = sum(amt for _, amt, _, _ in history)
    distinct_merchants = len(set(merch for _, _, merch, _ in history))
    is_international = 1.0 if country != "US" else 0.0
    
    return {
        "amount_usd": float(amount),
        "txn_count_1h": float(txn_count),
        "total_spend_1h": float(total_spend),
        "distinct_merch_1h": float(distinct_merchants),
        "is_international": is_international
    }

def get_behavioral_anomaly(card_id, amount, country):
    profile = card_profiles[card_id]
    
    if profile["txn_count"] < 5:
        return 0.0
    
    anomaly = 0.0
    
    # Amount anomaly
    if profile["avg_spend"] > 0:
        ratio = amount / profile["avg_spend"]
        if ratio > 10:
            anomaly += 0.5
        elif ratio > 5:
            anomaly += 0.4
        elif ratio > 3:
            anomaly += 0.25
        elif ratio > 2:
            anomaly += 0.15
    
    # Country anomaly
    if country not in profile["countries"]:
        anomaly += 0.3
    
    return min(1.0, anomaly)

def get_ring_signal(merchant_id, card_id):
    now = time.time()
    thirty_min_ago = now - 1800
    
    # Clean old entries
    merchant_cards[merchant_id] = [
        (ts, cid) 
        for ts, cid in merchant_cards[merchant_id]
        if ts > thirty_min_ago
    ]
    
    distinct_cards = len(set(cid for _, cid in merchant_cards[merchant_id]))
    
    if distinct_cards >= 10:
        return 1.0
    elif distinct_cards >= 7:
        return 0.8
    elif distinct_cards >= 5:
        return 0.6
    elif distinct_cards >= 3:
        return 0.3
    return 0.0

def get_adaptive_threshold(card_id, amount):
    profile = card_profiles[card_id]
    threshold = 0.5
    
    if profile["txn_count"] < 10:
        return threshold
    
    if profile["avg_spend"] > 500:
        threshold += 0.1
    elif profile["avg_spend"] > 200:
        threshold += 0.05
    
    if profile["std_dev"] < 50 and profile["avg_spend"] < 200:
        threshold -= 0.1
    
    if profile["avg_spend"] > 0:
        ratio = amount / profile["avg_spend"]
        if ratio > 5:
            threshold -= 0.15
        elif ratio > 3:
            threshold -= 0.08
    
    return max(0.3, min(0.8, threshold))

def update_profile(card_id, amount, country, merchant, hour):
    profile = card_profiles[card_id]
    
    # Update running mean and std dev (Welford's algorithm)
    profile["txn_count"] += 1
    delta = amount - profile["avg_spend"]
    profile["avg_spend"] += delta / profile["txn_count"]
    delta2 = amount - profile["avg_spend"]
    profile["sum_squares"] += delta * delta2
    
    if profile["txn_count"] > 1:
        profile["std_dev"] = (profile["sum_squares"] / (profile["txn_count"] - 1)) ** 0.5
    
    profile["total_spend"] += amount
    profile["max_spend"] = max(profile["max_spend"], amount)
    
    if country not in profile["countries"]:
        profile["countries"].append(country)
    
    if hour not in profile["hours"]:
        profile["hours"].append(hour)

def build_reasons(card_id, amount, country, merchant, feats, anomaly, ring_score):
    reasons = []
    profile = card_profiles[card_id]
    
    if profile["txn_count"] >= 5 and profile["avg_spend"] > 0:
        ratio = amount / profile["avg_spend"]
        if ratio > 2:
            reasons.append(
                f"Amount ${amount:.0f} is {ratio:.1f}x above card average of ${profile['avg_spend']:.0f}"
            )
    
    if country not in profile["countries"] and profile["txn_count"] >= 5:
        reasons.append(f"{country} is a new country for this card")
    
    if ring_score > 0:
        distinct = len(set(cid for _, cid in merchant_cards[merchant]))
        reasons.append(f"{distinct} cards hit merchant {merchant} in the last 30 minutes")
    
    if feats["txn_count_1h"] > 10:
        reasons.append(f"High velocity: {feats['txn_count_1h']:.0f} transactions in last hour")
    
    if feats["total_spend_1h"] > 1000:
        reasons.append(f"High hourly spend: ${feats['total_spend_1h']:.0f} in last hour")
    
    if not reasons:
        reasons.append("Statistical anomaly detected by ML model")
    
    return reasons

# ============================================================
# API endpoints
# ============================================================

@app.route("/evaluate", methods=["POST"])
def evaluate():
    start = time.perf_counter()
    
    data = request.json
    if not data:
        return jsonify({"error": "invalid request"}), 400
    
    card_id = data.get("card_id", "unknown")
    amount = float(data.get("amount_usd", 0))
    merchant = data.get("merchant_id", "unknown")
    country = data.get("country_code", "US")
    hour = datetime.now().hour
    
    # Get features
    feats = get_sliding_window_features(card_id, amount, country)
    
    # Get scores
    anomaly = get_behavioral_anomaly(card_id, amount, country)
    ring_score = get_ring_signal(merchant, card_id)
    threshold = get_adaptive_threshold(card_id, amount)
    
    # Update stores
    card_history[card_id].append((time.time(), amount, merchant, country))
    merchant_cards[merchant].append((time.time(), card_id))
    update_profile(card_id, amount, country, merchant, hour)
    
    # ML inference
    features = np.array([[
        feats["amount_usd"],
        feats["txn_count_1h"],
        feats["total_spend_1h"],
        feats["distinct_merch_1h"],
        feats["is_international"]
    ]], dtype=np.float32)
    
    result = sess.run(None, {input_name: features})
    ml_score = float(result[1][0][1])
    
    # Combined score
    combined = ml_score * 0.60 + anomaly * 0.25 + ring_score * 0.15
    is_fraud = combined >= threshold
    
    # Build reasons
    reasons = build_reasons(card_id, amount, country, merchant, feats, anomaly, ring_score)
    
    elapsed = (time.perf_counter() - start) * 1000
    
    return jsonify({
        "transaction_id": str(uuid.uuid4()),
        "card_id": card_id,
        "fraud_score": round(combined, 4),
        "is_fraud": is_fraud,
        "threshold": round(threshold, 4),
        "ml_score": round(ml_score, 4),
        "behavioral_anomaly": round(anomaly, 4),
        "ring_signal": round(ring_score, 4),
        "reasons": reasons,
        "latency_ms": round(elapsed, 2)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "cards_tracked": len(card_profiles),
        "model": "XGBoost trained on 284,807 real transactions"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Real-Time Fraud Detection Engine",
        "author": "Adhiswauran V",
        "endpoints": {
            "POST /evaluate": "Evaluate a transaction for fraud",
            "GET /health": "Health check and stats"
        },
        "example": {
            "card_id": "card-42",
            "amount_usd": 1800,
            "merchant_id": "merch-001",
            "country_code": "RU"
        }
    })

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8888))
    print(f"Starting server on port {port}...")
    serve(app, host="0.0.0.0", port=port, threads=4)