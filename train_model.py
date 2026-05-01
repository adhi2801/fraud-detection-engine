import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import warnings
warnings.filterwarnings('ignore')

print("Loading dataset...")
df = pd.read_csv('data/creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")

# Build feature matrix
# f0=amount_usd, f1=txn_count_1h, f2=total_spend_1h, f3=distinct_merch_1h, f4=is_international
df['is_international'] = (df['Amount'] > 500).astype(float)
df['txn_count_1h'] = df.groupby(df['Time'] // 3600)['Time'].transform('count')
df['total_spend_1h'] = df.groupby(df['Time'] // 3600)['Amount'].transform('sum')
df['distinct_merch_1h'] = df['txn_count_1h'] * 0.6

# Use f0-f4 naming so ONNX export works
X = pd.DataFrame({
    'f0': df['Amount'],
    'f1': df['txn_count_1h'],
    'f2': df['total_spend_1h'],
    'f3': df['distinct_merch_1h'],
    'f4': df['is_international']
}).astype(np.float32)

y = df['Class']

print("\nSplitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

model.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

print("\nExporting to ONNX...")
initial_type = [('features', FloatTensorType([None, 5]))]
onnx_model = convert_xgboost(model, initial_types=initial_type)

with open('models/fraud_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

print("Model saved to models/fraud_model.onnx")
print("\nDone! Model ready for Go inference.")