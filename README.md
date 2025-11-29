# AI-Driven ECG Arrhythmia Detection (MIT-BIH + CNN + GUI)

This project implements an offline ECG arrhythmia detection system using a 1D Convolutional Neural Network (CNN) trained on the MIT-BIH Arrhythmia dataset.  
It also includes a Python desktop GUI to visualize ECG beats, show predicted arrhythmia type, and display confidence scores.

> **Institute:** Sardar Patel Institute of Technology, Department of Electronics and Telecommunication Engineering, Mumbai, India  
> **Guide:** Dr. Sanjuktarani Jena  

---

## ✨ Features

- 1D CNN model trained on **MIT-BIH Arrhythmia** dataset  
- Classifies each beat into 5 classes: **N, S, V, F, Q**  
- Offline **desktop GUI** (Python) built with `customtkinter`  
- ECG waveform plotting inside the app  
- Beat-wise prediction + confidence bar chart  
- Works with CSV files containing segmented beats (187 samples per beat)  
- IEEE-style paper (LaTeX) included in the repo (optional)

---

## 🧠 Model Overview

- Input: 187-sample 1D ECG beat  
- Architecture (simplified):
  - Conv1D (32, kernel=7) + BatchNorm + ReLU + MaxPool + Dropout  
  - Conv1D (64, kernel=5) + BatchNorm + ReLU + MaxPool + Dropout  
  - Dense(64) + ReLU + Dropout  
  - Dense(5) + Softmax  
- Dataset: MIT-BIH Arrhythmia (beat-segmented CSVs)  
- Test accuracy: **~82.5%** on official test split  

Model files (produced in training, stored in repo or downloadable):

- `ecg_cnn_model.keras` – trained Keras model  
- `preprocess_params.npz` – normalization min/max  
- `class_maps.json` – index → class code → class name mapping  

---

## 📁 Repository Structure

Example layout (adapt to your repo):

```text
.
├── models/
│   ├── ecg_cnn_model.keras
│   ├── preprocess_params.npz
│   └── class_maps.json
├── gui/
│   └── ecg_gui_app_dark.py
├── training/
│   ├── train_ecg_cnn.ipynb      # Colab / Jupyter notebook
│   └── utils.py                 # (optional) helper functions
├── data/
│   ├── mitbih_train.csv         # (not committed if large)
│   └── mitbih_test.csv          # (or instructions to download)
├── paper/
│   └── main.tex                 # IEEE LaTeX paper
├── README.md
└── requirements.txt
