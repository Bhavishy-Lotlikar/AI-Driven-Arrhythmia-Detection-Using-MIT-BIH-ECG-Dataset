# 🫀 AI-Driven Arrhythmia Detection Using MIT-BIH ECG Dataset

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Model%20Accuracy-98%25-brightgreen)

This repository contains a lightweight **1D Convolutional Neural Network (CNN)** and an offline **desktop GUI** for beat-level ECG arrhythmia detection using the **MIT-BIH Arrhythmia Dataset**.

The trained model classifies each beat into **five ECG classes**:

| Label | Class Name                     | Description                                   |
|-------|--------------------------------|-----------------------------------------------|
| N     | Normal beat                    | Healthy cardiac rhythm                        |
| S     | Supraventricular ectopic beat  | Abnormal atrial-origin beat                   |
| V     | Ventricular ectopic beat       | Ventricular arrhythmia (clinically important) |
| F     | Fusion beat                    | Fusion of normal and ectopic activation       |
| Q     | Unknown / Paced beat           | Pacemaker / unclassified beat                 |

---

## 📊 Model Performance

The final CNN, trained on the MIT-BIH processed beat dataset:

- **Test Accuracy:** **98.0%**
- **Weighted F1-score:** ~0.98
- Strong performance on critical **Ventricular (V)** beats
- Reasonable recall on minority classes (S, F) despite imbalance

Evaluation includes:

- Confusion matrix (raw + normalized)
- Per-class precision, recall, F1-score
- Overall accuracy summary

---

## 🧠 Model Overview – DeepECG-Net

A compact 1D CNN designed to run efficiently on CPU-only systems:

- Conv1D + BatchNorm + MaxPooling blocks
- Dropout regularization
- Dense layer + Softmax (5 outputs)
- Trained with:
  - Adam optimizer
  - Class weights for imbalance
  - EarlyStopping + ReduceLROnPlateau
  - Best-model checkpointing

The full training pipeline is provided in the notebook inside `training_notebooks/`.

---

## 🖥️ Offline GUI Features

The desktop application (`ecg_gui_app.py`) provides:

- CSV loader for ECG beats (187 samples per beat, optional label column)
- Slider + manual input to navigate beats
- Real-time ECG waveform plotting (Matplotlib)
- Confidence bar chart for the 5 classes
- Dark mode UI using **CustomTkinter**
- Works fully **offline** once the model + params are present

---

## 📂 Project Structure

```text
AI-Driven-Arrhythmia-Detection-Using-MIT-BIH-ECG-Dataset/
│
├── ecg_gui_app.py              # Desktop GUI application
├── ecg_cnn_model.keras         # Trained CNN model (Keras format)
├── preprocess_params.npz       # Normalization parameters (X_min, X_max)
├── class_maps.json             # Class label mapping (index -> N/S/V/F/Q)
├── requirements.txt            # Python dependencies
├── README.md                   # This documentation
│
├── training_notebooks/         # Google Colab / Jupyter notebooks
│   └── model_training.ipynb
│
└── DATA/                       # 🔹 Place ECG CSV files here (not tracked)
    ├── mitbih_train.csv        # (user-provided)
    ├── mitbih_test.csv         # (user-provided)
    └── other_ecg_files.csv
```

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Bhavishy-Lotlikar/AI-Driven-Arrhythmia-Detection-Using-MIT-BIH-ECG-Dataset.git
cd AI-Driven-Arrhythmia-Detection-Using-MIT-BIH-ECG-Dataset
```

**or**

**Download as a Zip file and extract to required destination**

### 2️⃣ (Optional) Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset

This project uses the **MIT-BIH Arrhythmia Database** (beat-wise CSV format derived from PhysioNet / Kaggle versions).

Dataset reference (not included in repo):

- PhysioNet MIT-BIH Arrhythmia Database:  
  https://physionet.org/content/mitdb/1.0.0/
- Kaggle DataSet link:
  https://www.kaggle.com/datasets/shayanfazeli/heartbeat  

Place the CSV files (e.g. `mitbih_train.csv`, `mitbih_test.csv`) inside the `DATA/` folder.

Expected formats:

- **Labeled**: 188 columns → 187 samples + 1 label  
- **Unlabeled**: 187 columns → samples only

The GUI and notebook automatically handle both.

---

## ▶️ Running the GUI

Once dependencies and model files are in place:

```bash
python ecg_gui_app.py
```

Steps inside the app:

1. Click **“Load ECG CSV”**
2. Select a file (e.g. `DATA/mitbih_test.csv`)
3. Use the slider or type a beat index
4. View the waveform + prediction + confidence scores

---

## 🧪 Retraining the Model

If you want to retrain or modify the architecture:

1. Open `training_notebooks/model_training.ipynb` in Google Colab / Jupyter.
2. Upload `mitbih_train.csv` and `mitbih_test.csv`.
3. Run cells 1–4 to:
   - Load and preprocess data
   - Train DeepECG-Net
   - Evaluate performance
   - Export:
     - `ecg_cnn_model.keras`
     - `preprocess_params.npz`
     - `class_maps.json`
4. Replace the old files in the repo root with the newly generated ones.

The GUI will automatically use the new model.

---

## 🎯 Future Work

- Improve recall for minority classes (S and F) via:
  - Data augmentation
  - Focal loss / cost-sensitive training
- Real-time continuous ECG stream processing
- TensorFlow Lite conversion for mobile / embedded deployment
- Additional leads and multi-channel models

---

## 👥 Contributors

| Name | Contribution |
|------|--------------|
| **Bhavishy Lotlikar** | Lead development, GUI integration, debugging |
| **Deepam Mhatre** | Training pipeline, preprocessing, class balancing |
| **Aditya Mahale** | GUI layout and interaction design |
| **Parth Mahajan** | Confidence score visualization & slider logic |
| **Purvesh Neve** | Documentation, testing, and result verification |

---

## ⚠️ Disclaimer

This software is intended **for educational and research purposes only**.  
It is **not** a certified medical device and must **not** be used for clinical diagnosis or treatment decisions without proper regulatory approval and expert supervision.

---

## 📜 License

This project is released under the **MIT License**.  
You are free to use, modify and distribute it with attribution.

---

## ⭐ Acknowledgements

- MIT-BIH Arrhythmia Database and PhysioNet for ECG data.
- Open-source contributors of TensorFlow, NumPy, Pandas, Matplotlib, and CustomTkinter.

If you find this project helpful, please consider ⭐ starring the repository on GitHub!
