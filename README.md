# 🫀 Offline ECG Arrhythmia Detection using Deep Learning (DeepECG-Net)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Model%20Accuracy-98%25-brightgreen)

A lightweight **1D Convolutional Neural Network (CNN)** trained on the **MIT-BIH Arrhythmia Dataset** for beat-level heart arrhythmia detection.

The trained model predicts **5 clinically important ECG beat types**:

| Label | Class Name | Description |
|------|------------|-------------|
| **N** | Normal beat | Healthy cardiac rhythm |
| **S** | Supraventricular ectopic | Abnormal atrial origin |
| **V** | Ventricular ectopic | Ventricular arrhythmia — *potentially dangerous* |
| **F** | Fusion beat | Fusion of ventricular + normal activation |
| **Q** | Unknown / Paced | Pacemaker-related heartbeat |

---

## 📸 Screenshots

| ECG Waveform | Prediction + Confidence Scores |
|-------------|------------------------------|
| ![](docs/waveform.png) | ![](docs/confidence.png) |

*(You can replace these with your own images)*

---

## 🚀 Features
✔ Works **offline** (fully local execution)  
✔ Visual ECG waveform plotting  
✔ Model confidence bar graph  
✔ Dark-mode, modern GUI (CustomTkinter)  
✔ Fast predictions — CPU friendly  
✔ MIT-BIH standard CSV support (187 sample beat format)

---

## 📊 Model Performance
- **98% test accuracy**
- Robust detection of Ventricular (V) arrhythmia
- Class-balanced training for fairness

| Metric | Value |
|--------|------|
| Accuracy | **98%** |
| Weighted F1-score | 0.98 |
| Training time | ~4–6 minutes (Google Colab TPU) |

📌 Evaluation includes confusion matrix, precision, recall, F1-score.

---

## 🧠 Model Architecture — DeepECG-Net

- CNN 1D feature extractors (7×32 and 5×64 filters)
- Batch Normalization
- MaxPool + Dropout
- Dense(64) + Softmax(5 classes)

Optimized to run efficiently on low-resource machines.

---

## 📂 Project Structure
ECG-Arrhythmia-Detection/
│── ecg_gui_app.py # Desktop GUI
│── ecg_cnn_model.keras # Trained model
│── preprocess_params.npz # Normalization parameters
│── class_maps.json # Class name mapping
│── training_notebooks/ # Jupyter/Colab training files
│── DATA/ # (User adds dataset here)
│── README.md # Documentation
│── requirements.txt


---

## 🛠️ Installation & Setup

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Add the Dataset

Download MIT-BIH (CSV format) from Kaggle or PhysioNet and place inside:
```bash
/DATA/
```
- Dataset Reference:
🔗 https://physionet.org/content/mitdb/1.0.0/
- Kaggle Dataset Reference: 
🔗 https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- ⚠️ Due to licensing, the dataset is not distributed with this repo.

---
▶️ Run GUI Application
- Open CMD in the same path as the "ecg_gui_app.py"
  then run
  ```bash
  python ecg_gui_app.py
  ```

Then:

- Click Load ECG CSV
- Move the slider or enter beat index
- View prediction & metrics instantly 🎯

