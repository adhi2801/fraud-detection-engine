from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np

app = Flask(__name__)

print("Loading ONNX model...")
sess = rt.InferenceSession("models/fraud_model.onnx")
input_name = sess.get_inputs()[0].name
print(f"Model loaded. Input name: {input_name}")

@app.route("/score", methods=["POST"])
def score():
    data = request.json
    features = np.array([[
        data["amount_usd"],
        data["txn_count_1h"],
        data["total_spend_1h"],
        data["distinct_merch_1h"],
        data["is_international"]
    ]], dtype=np.float32)

    result = sess.run(None, {input_name: features})
    
    # result[1] contains probabilities [[legit_prob, fraud_prob]]
    fraud_prob = float(result[1][0][1])
    
    return jsonify({
        "fraud_score": fraud_prob,
        "is_fraud": fraud_prob >= 0.5
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    from waitress import serve
    print("Starting production server with 4 threads...")
    serve(app, host="0.0.0.0", port=8888, threads=4)