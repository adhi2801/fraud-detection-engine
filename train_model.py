import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

# Use ALL 28 PCA features + Amount + Time
# These are the real anonymized bank features
feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
X = df[feature_cols].astype(np.float32)
# Rename to f0-f29 for ONNX compatibility
X.columns = [f'f{i}' for i in range(len(feature_cols))]
y = df['Class']

print(f"\nUsing {len(feature_cols)} features: V1-V28 + Amount + Time")

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Fraud in test: {y_test.sum()}")

print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42,
    eval_metric='logloss',
    verbosity=0,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC Score: {auc:.4f}")

print("\nExporting to ONNX...")
initial_type = [('features', FloatTensorType([None, 30]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open('models/fraud_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Model saved to models/fraud_model.onnx")
print(f"\nDone! AUC improved to {auc:.4f}")