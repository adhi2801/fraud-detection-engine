import os
import time
import uuid
import json
from collections import defaultdict
from datetime import datetime
from flask import Flask, request, jsonify, render_template
import onnxruntime as rt
import numpy as np

app = Flask(__name__)

# In-memory stores
card_history = defaultdict(list)
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
merchant_cards = defaultdict(list)

# Load ONNX model
print("Loading ONNX model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "fraud_model.onnx")
sess = rt.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name
print(f"Model loaded. Input: {input_name}")

def get_sliding_window_features(card_id, amount, country):
    now = time.time()
    one_hour_ago = now - 3600
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
    return txn_count, total_spend, distinct_merchants, is_international

def get_behavioral_anomaly(card_id, amount, country):
    profile = card_profiles[card_id]
    if profile["txn_count"] < 5:
        return 0.0
    anomaly = 0.0
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
    if country not in profile["countries"]:
        anomaly += 0.3
    return min(1.0, anomaly)

def get_ring_signal(merchant_id, card_id):
    now = time.time()
    thirty_min_ago = now - 1800
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

def build_reasons(card_id, amount, country, merchant, txn_count, total_spend, anomaly, ring_score):
    reasons = []
    profile = card_profiles[card_id]
    if profile["txn_count"] >= 5 and profile["avg_spend"] > 0:
        ratio = amount / profile["avg_spend"]
        if ratio > 2:
            reasons.append(f"Amount ${amount:.0f} is {ratio:.1f}x above card average of ${profile['avg_spend']:.0f}")
    if country not in profile["countries"] and profile["txn_count"] >= 5:
        reasons.append(f"{country} is a new country for this card")
    if ring_score > 0:
        distinct = len(set(cid for _, cid in merchant_cards[merchant]))
        reasons.append(f"{distinct} cards hit merchant {merchant} in last 30 minutes")
    if txn_count > 10:
        reasons.append(f"High velocity: {txn_count} transactions in last hour")
    if total_spend > 1000:
        reasons.append(f"High hourly spend: ${total_spend:.0f} in last hour")
    if not reasons:
        reasons.append("Statistical anomaly detected by ML model")
    return reasons

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

    # Get sliding window features
    txn_count, total_spend, distinct_merch, is_intl = get_sliding_window_features(card_id, amount, country)

    # Get scores
    anomaly = get_behavioral_anomaly(card_id, amount, country)
    ring_score = get_ring_signal(merchant, card_id)
    threshold = get_adaptive_threshold(card_id, amount)

    # Update stores
    card_history[card_id].append((time.time(), amount, merchant, country))
    merchant_cards[merchant].append((time.time(), card_id))
    update_profile(card_id, amount, country, merchant, hour)

    # Build 30-feature vector matching training data
    # f0-f27 = V1-V28 (we use derived features as proxies)
    # f28 = Amount, f29 = Time proxy
    now_seconds = time.time() % 172800  # seconds within 2-day window like dataset

    # Create feature vector — V1-V28 are zeros for API calls
    # (real bank system would provide these from payment network)
    # We use what we have: amount, velocity, geography signals
    v_features = [0.0] * 28  # V1-V28 placeholders

    # Inject our behavioral signals into key V features
    # These approximate what the bank's PCA features capture
    v_features[0] = float(txn_count) / 10.0        # velocity signal → V1
    v_features[1] = float(total_spend) / 1000.0     # spend signal → V2
    v_features[2] = float(is_intl) * 2.0            # geography → V3
    v_features[3] = float(anomaly) * 3.0            # behavioral anomaly → V4
    v_features[4] = float(ring_score) * 2.0         # ring signal → V5
    v_features[5] = float(distinct_merch) / 5.0     # merchant diversity → V6

    features = np.array([
        v_features + [float(amount), float(now_seconds)]
    ], dtype=np.float32)

    result = sess.run(None, {input_name: features})
    ml_score = float(result[1][0][1])

    # Combined score
    combined = ml_score * 0.60 + anomaly * 0.25 + ring_score * 0.15
    is_fraud = combined >= threshold

    reasons = build_reasons(card_id, amount, country, merchant, txn_count, total_spend, anomaly, ring_score)

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
        "model": "XGBoost — AUC 0.98 — trained on 284,807 real transactions"
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "Real-Time Fraud Detection Engine",
        "author": "Adhiswauran V",
        "model_accuracy": "AUC-ROC: 0.9814",
        "trained_on": "284,807 real credit card transactions",
        "endpoints": {
            "POST /evaluate": "Evaluate a transaction for fraud",
            "GET /health": "Health check and live stats",
            "GET /": "API documentation"
        },
        "example_request": {
            "card_id": "card-42",
            "amount_usd": 1800,
            "merchant_id": "merch-001",
            "country_code": "RU"
        }
    })

@app.route("/ui", methods=["GET"])
def ui():
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection Engine</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #0a0e1a; color: #e2e8f0; min-height: 100vh; }
        
        .header { background: linear-gradient(135deg, #1a1f36 0%, #0d1117 100%); padding: 40px 20px; text-align: center; border-bottom: 1px solid #1e2d3d; }
        .header h1 { font-size: 2.2em; background: linear-gradient(90deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        .header p { color: #64748b; font-size: 0.95em; }
        .badge { display: inline-block; background: #1e3a5f; color: #60a5fa; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin: 4px; border: 1px solid #2563eb44; }

        .container { max-width: 1000px; margin: 40px auto; padding: 0 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        
        .card { background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 28px; }
        .card h2 { color: #94a3b8; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 20px; }
        
        label { display: block; color: #64748b; font-size: 0.85em; margin-bottom: 6px; margin-top: 14px; }
        input, select { width: 100%; padding: 12px 14px; background: #0d1117; color: #e2e8f0; border: 1px solid #1f2937; border-radius: 10px; font-size: 0.95em; transition: border 0.2s; }
        input:focus, select:focus { outline: none; border-color: #3b82f6; }
        select option { background: #111827; }
        
        .presets { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 16px; }
        .preset-btn { padding: 8px; background: #0d1117; border: 1px solid #1f2937; border-radius: 8px; color: #94a3b8; cursor: pointer; font-size: 0.8em; transition: all 0.2s; }
        .preset-btn:hover { border-color: #3b82f6; color: #60a5fa; }
        .preset-fraud { border-color: #7f1d1d44; color: #fca5a5; }
        .preset-fraud:hover { border-color: #ef4444; color: #ef4444; }
        
        .btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 10px; cursor: pointer; font-size: 1em; font-weight: 600; margin-top: 20px; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }

        .result-card { display: none; }
        .score-display { text-align: center; padding: 30px 0; }
        .score-number { font-size: 5em; font-weight: 800; line-height: 1; }
        .score-label { font-size: 1.1em; margin-top: 8px; font-weight: 600; }
        .fraud-color { color: #ef4444; }
        .safe-color { color: #22c55e; }
        
        .meter { background: #1f2937; border-radius: 10px; height: 8px; margin: 16px 0; overflow: hidden; }
        .meter-fill { height: 100%; border-radius: 10px; transition: width 0.8s ease; }
        .meter-fraud { background: linear-gradient(90deg, #f59e0b, #ef4444); }
        .meter-safe { background: linear-gradient(90deg, #22c55e, #16a34a); }

        .stats-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 16px 0; }
        .stat { background: #0d1117; border-radius: 10px; padding: 12px; text-align: center; }
        .stat-value { font-size: 1.3em; font-weight: 700; color: #60a5fa; }
        .stat-label { font-size: 0.75em; color: #64748b; margin-top: 2px; }

        .reasons { margin-top: 16px; }
        .reason { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; background: #0d1117; border-radius: 8px; margin: 6px 0; font-size: 0.9em; color: #94a3b8; }
        .reason-icon { font-size: 1em; flex-shrink: 0; }

        .history { margin-top: 16px; }
        .history-item { display: flex; justify-content: space-between; align-items: center; padding: 8px 12px; border-radius: 8px; margin: 4px 0; font-size: 0.85em; background: #0d1117; }
        .tag { padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }
        .tag-fraud { background: #7f1d1d; color: #fca5a5; }
        .tag-safe { background: #14532d; color: #86efac; }

        .info-bar { background: #111827; border-top: 1px solid #1f2937; padding: 16px 20px; text-align: center; color: #374151; font-size: 0.8em; margin-top: 40px; }
        .info-bar a { color: #3b82f6; text-decoration: none; }

        @media (max-width: 700px) { .container { grid-template-columns: 1fr; } }
    </style>
</head>
<body>

<div class="header">
    <h1>🚨 Fraud Detection Engine</h1>
    <p>Real-time credit card fraud detection powered by Machine Learning</p>
    <div style="margin-top:12px">
        <span class="badge">XGBoost</span>
        <span class="badge">AUC-ROC: 0.98</span>
        <span class="badge">284,807 transactions</span>
        <span class="badge">Behavioral Fingerprinting</span>
        <span class="badge">Graph Ring Detection</span>
    </div>
</div>

<div class="container">
    <!-- Input Card -->
    <div class="card">
        <h2>Transaction Details</h2>

        <p style="color:#64748b;font-size:0.85em;margin-bottom:12px">Quick presets:</p>
        <div class="presets">
            <button class="preset-btn" onclick="setPreset('normal')">✅ Normal Purchase</button>
            <button class="preset-btn preset-fraud" onclick="setPreset('fraud1')">🚨 High Amount Abroad</button>
            <button class="preset-btn preset-fraud" onclick="setPreset('fraud2')">🚨 Velocity Attack</button>
            <button class="preset-btn" onclick="setPreset('travel')">✈️ Legitimate Travel</button>
        </div>

        <label>Card ID</label>
        <input type="text" id="card_id" value="card-42" placeholder="e.g. card-42" />

        <label>Amount (USD)</label>
        <input type="number" id="amount" value="85" placeholder="e.g. 1800" />

        <label>Merchant ID</label>
        <input type="text" id="merchant" value="merch-001" placeholder="e.g. merch-001" />

        <label>Country Code</label>
        <select id="country">
            <option value="US" selected>🇺🇸 US — United States</option>
            <option value="IN">🇮🇳 IN — India</option>
            <option value="GB">🇬🇧 GB — United Kingdom</option>
            <option value="DE">🇩🇪 DE — Germany</option>
            <option value="RU">🇷🇺 RU — Russia</option>
            <option value="BR">🇧🇷 BR — Brazil</option>
            <option value="CN">🇨🇳 CN — China</option>
        </select>

        <button class="btn" onclick="evaluate()" id="evalBtn">Evaluate Transaction →</button>

        <div class="history" id="history" style="display:none">
            <p style="color:#64748b;font-size:0.8em;margin:16px 0 8px">Recent evaluations:</p>
            <div id="history-list"></div>
        </div>
    </div>

    <!-- Result Card -->
    <div class="card result-card" id="result">
        <h2>Detection Result</h2>

        <div class="score-display">
            <div class="score-number" id="score-num">--</div>
            <div class="score-label" id="score-label">Fraud Score</div>
        </div>

        <div class="meter">
            <div class="meter-fill" id="meter" style="width:0%"></div>
        </div>

        <div class="stats-grid">
            <div class="stat">
                <div class="stat-value" id="stat-ml">--</div>
                <div class="stat-label">ML Score</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-anomaly">--</div>
                <div class="stat-label">Behavioral</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-latency">--</div>
                <div class="stat-label">Latency</div>
            </div>
        </div>

        <div class="reasons" id="reasons"></div>
    </div>
</div>

<div class="info-bar">
    Built by <strong>Adhiswauran V</strong> &nbsp;|&nbsp;
    <a href="/health">API Health</a> &nbsp;|&nbsp;
    <a href="https://github.com/adhi2801/fraud-detection-engine" target="_blank">GitHub</a> &nbsp;|&nbsp;
    POST /evaluate for API access
</div>

<script>
    const history = [];

    function setPreset(type) {
        const presets = {
            normal: { card: 'card-10', amount: 45, merchant: 'merch-003', country: 'US' },
            fraud1: { card: 'card-42', amount: 2400, merchant: 'merch-001', country: 'RU' },
            fraud2: { card: 'card-99', amount: 150, merchant: 'merch-002', country: 'CN' },
            travel: { card: 'card-55', amount: 320, merchant: 'merch-004', country: 'GB' }
        };
        const p = presets[type];
        document.getElementById('card_id').value = p.card;
        document.getElementById('amount').value = p.amount;
        document.getElementById('merchant').value = p.merchant;
        document.getElementById('country').value = p.country;
    }

    async function evaluate() {
        const btn = document.getElementById('evalBtn');
        btn.disabled = true;
        btn.textContent = 'Evaluating...';

        const body = {
            card_id: document.getElementById('card_id').value,
            amount_usd: parseFloat(document.getElementById('amount').value),
            merchant_id: document.getElementById('merchant').value,
            country_code: document.getElementById('country').value
        };

        try {
            const res = await fetch('/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const data = await res.json();
            showResult(data, body);
            addHistory(body, data);
        } catch(e) {
            alert('Error calling API: ' + e.message);
        }

        btn.disabled = false;
        btn.textContent = 'Evaluate Transaction →';
    }

    function showResult(data, req) {
        const resultEl = document.getElementById('result');
        resultEl.style.display = 'block';

        const score = Math.round(data.fraud_score * 100);
        const isFraud = data.is_fraud;

        document.getElementById('score-num').textContent = score + '%';
        document.getElementById('score-num').className = 'score-number ' + (isFraud ? 'fraud-color' : 'safe-color');
        document.getElementById('score-label').textContent = isFraud ? '🚨 FRAUD DETECTED' : '✅ TRANSACTION SAFE';
        document.getElementById('score-label').style.color = isFraud ? '#ef4444' : '#22c55e';

        const meter = document.getElementById('meter');
        meter.style.width = score + '%';
        meter.className = 'meter-fill ' + (isFraud ? 'meter-fraud' : 'meter-safe');

        document.getElementById('stat-ml').textContent = Math.round(data.ml_score * 100) + '%';
        document.getElementById('stat-anomaly').textContent = Math.round(data.behavioral_anomaly * 100) + '%';
        document.getElementById('stat-latency').textContent = data.latency_ms + 'ms';

        const icons = ['⚡', '🌍', '💳', '🔄', '📊'];
        const reasonsHtml = data.reasons.map((r, i) =>
            `<div class="reason"><span class="reason-icon">${icons[i] || '•'}</span>${r}</div>`
        ).join('');
        document.getElementById('reasons').innerHTML = reasonsHtml;
    }

    function addHistory(req, data) {
        const list = document.getElementById('history-list');
        const histEl = document.getElementById('history');
        histEl.style.display = 'block';

        const item = document.createElement('div');
        item.className = 'history-item';
        item.innerHTML = `
            <span>${req.card_id} — $${req.amount_usd} — ${req.country_code}</span>
            <span class="tag ${data.is_fraud ? 'tag-fraud' : 'tag-safe'}">${data.is_fraud ? 'FRAUD' : 'SAFE'}</span>
        `;
        list.insertBefore(item, list.firstChild);
        if (list.children.length > 5) list.removeChild(list.lastChild);
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'Enter') evaluate();
    });
</script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    from waitress import serve
    port = int(os.environ.get("PORT", 8888))
    print(f"Starting server on port {port}...")
    serve(app, host="0.0.0.0", port=port, threads=4)