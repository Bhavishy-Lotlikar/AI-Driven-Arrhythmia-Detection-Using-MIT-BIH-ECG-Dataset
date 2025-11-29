import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("ecg_cnn_model.keras")

# Load normalization parameters
params = np.load("preprocess_params.npz")
X_min = params["X_min"]
X_max = params["X_max"]

# Load label mappings
with open("class_maps.json", "r") as f:
    mappings = json.load(f)
idx_to_class = {int(k): v for k, v in mappings["idx_to_class"].items()}
class_to_name = mappings["class_to_name"]

# Load input ECG CSV (change file name here if needed)
csv_path = "my_ecg.csv"
df = pd.read_csv(csv_path, header=None)

# Detect if labeled or unlabeled
has_label = (df.shape[1] == 188)

if has_label:
    X_new = df.iloc[:, :-1].values
else:
    X_new = df.values

# Normalize same as training
X_new = (X_new - X_min) / (X_max - X_min + 1e-8)
X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], 1))

# Prediction
probs = model.predict(X_new)
preds = np.argmax(probs, axis=1)

# Report
unique, counts = np.unique(preds, return_counts=True)
total = len(preds)

print("\n===== ECG ARRHYTHMIA REPORT =====")

for u, c in zip(unique, counts):
    code = idx_to_class[u]
    name = class_to_name[code]
    print(f"{name}: {c} beats ({(c/total)*100:.2f}%)")

if np.sum(preds != 0) == 0:
    print("\n🟢 Normal ECG — No arrhythmia detected")
else:
    print("\n🔴 Arrhythmia detected — Please consult a cardiologist")
