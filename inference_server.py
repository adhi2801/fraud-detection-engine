"""ONNX inference server used by the Go detection service (POST /score on :8888).

The model expects the bank's 28 anonymised PCA features (V1-V28) plus Amount and Time, exactly as in train_model.py.
A request without `bank_features` gets HTTP 422, and the Go scorer then falls back to its behavioural rules. The
synthetic transactions from producer/ carry no bank features, so the streaming demo runs on the rules; replayed real
transactions (samples/heldout_transactions.json) are scored by the model.
"""
import math

from flask import Flask, jsonify, request
import numpy as np
import onnxruntime as rt

N_BANK_FEATURES = 28

app = Flask(__name__)

print("Loading ONNX model...")
sess = rt.InferenceSession("models/fraud_model.onnx")
input_name = sess.get_inputs()[0].name
print(f"Model loaded. Input name: {input_name}")


@app.route("/score", methods=["POST"])
def score():
    data = request.get_json(silent=True) or {}
    bank = data.get("bank_features")
    if not isinstance(bank, list) or len(bank) != N_BANK_FEATURES:
        return jsonify({"error": f"bank_features (V1-V28, {N_BANK_FEATURES} numbers) required"}), 422
    try:
        row = [float(v) for v in bank] + [float(data.get("amount_usd", 0)), float(data.get("time_seconds", 0))]
    except (TypeError, ValueError):
        return jsonify({"error": "bank_features, amount_usd and time_seconds must be numbers"}), 422
    if not all(math.isfinite(v) for v in row):
        return jsonify({"error": "features must be finite"}), 422

    result = sess.run(None, {input_name: np.array([row], dtype=np.float32)})
    fraud_prob = float(result[1][0][1])  # result[1] holds the class probabilities [legit, fraud]
    return jsonify({"fraud_score": fraud_prob, "is_fraud": fraud_prob >= 0.5})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    from waitress import serve
    print("Starting production server with 4 threads...")
    serve(app, host="0.0.0.0", port=8888, threads=4)
