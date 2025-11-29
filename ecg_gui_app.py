import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==========================
#  APP THEME / GLOBALS
# ==========================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_BG = "#0b1120"   # main background
CARD_BG = "#020617"  # cards / frames background

MODEL_PATH = "ecg_cnn_model.keras"
PARAMS_PATH = "preprocess_params.npz"
MAP_PATH = "class_maps.json"

# Globals for runtime
beats = None
probs_all = None
preds_all = None
last_index = None
current_file = None

# ==========================
#  LOAD MODEL + PARAMS
# ==========================
if not os.path.exists(MODEL_PATH):
    raise SystemExit(f"Model file not found: {MODEL_PATH}")

try:
    # We only need inference, no training → compile=False avoids custom loss issues
    model = load_model(MODEL_PATH, compile=False)
except Exception as e:
    raise SystemExit(f"Could not load model at '{MODEL_PATH}':\n{e}")

if not os.path.exists(PARAMS_PATH):
    raise SystemExit(f"Preprocessing params file not found: {PARAMS_PATH}")

params = np.load(PARAMS_PATH)
# np.load gives 0-d arrays; convert to Python float
X_min = float(params["X_min"])
X_max = float(params["X_max"])
print("Normalization parameters loaded:", X_min, X_max)

if not os.path.exists(MAP_PATH):
    raise SystemExit(f"Class map file not found: {MAP_PATH}")

with open(MAP_PATH, "r") as f:
    mappings = json.load(f)

# Support both JSON formats:
# 1) {"idx_to_class": {...}, "class_to_name": {...}}
# 2) {"0": "N", "1": "S", "2": "V", "3": "F", "4": "Q"}
if "idx_to_class" in mappings:
    idx_to_class = {int(k): v for k, v in mappings["idx_to_class"].items()}
    class_to_name = mappings["class_to_name"]
else:
    idx_to_class = {int(k): v for k, v in mappings.items()}
    class_to_name = {
        "N": "Normal beat",
        "S": "Supraventricular ectopic beat",
        "V": "Ventricular ectopic beat",
        "F": "Fusion beat",
        "Q": "Unknown / paced beat",
    }

num_classes = len(idx_to_class)
print("Class mapping:", idx_to_class)


# ==========================
#  HELPER FUNCTIONS
# ==========================
def load_ecg_csv(filepath: str):
    """
    Load ECG CSV and return a numpy array of shape (beats, 187).
    - If file has 188 columns: assumes 187 samples + 1 label and drops label.
    - Tries UTF-8 first, then latin1 encoding.
    """
    try:
        df = pd.read_csv(filepath, header=None)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, header=None, encoding="latin1")

    # Drop label if present (187 + 1)
    if df.shape[1] == 188:
        df = df.iloc[:, :-1]

    return df.values


def open_csv():
    """Open CSV, normalize once, and precompute predictions for all beats."""
    global beats, probs_all, preds_all, last_index, current_file, X_min, X_max

    filepath = filedialog.askopenfilename(
        filetypes=[("CSV files", "*.csv")],
        title="Select ECG CSV file"
    )
    if not filepath:
        return

    try:
        data = load_ecg_csv(filepath)
    except Exception as e:
        messagebox.showerror("Error loading CSV", f"Could not read file:\n{e}")
        print("Error loading CSV:", e)
        return

    print("Loaded CSV:", filepath)
    print("Shape:", data.shape)

    if data.ndim != 2 or data.shape[1] != 187:
        messagebox.showerror(
            "Invalid format",
            f"Expected 187 samples per beat.\n"
            f"Got shape: {data.shape}\n\n"
            "Make sure the file is:\n"
            "  • mitbih_train.csv / mitbih_test.csv\n"
            "  • or a CSV with 187 columns (or 187+1 label)."
        )
        return

    beats = data
    current_file = os.path.basename(filepath)

    # Normalize & precompute predictions
    X_norm = (beats - X_min) / (X_max - X_min + 1e-8)
    X_norm = X_norm.reshape((X_norm.shape[0], X_norm.shape[1], 1))

    probs_all_local = model.predict(X_norm, batch_size=256, verbose=0)
    preds_all_local = np.argmax(probs_all_local, axis=1)

    # Store globally
    probs_all = probs_all_local
    preds_all = preds_all_local

    # Configure slider
    beat_slider.configure(
        to=beats.shape[0] - 1,
        state="normal",
        number_of_steps=max(beats.shape[0] - 1, 1)
    )
    beat_entry.delete(0, "end")
    beat_entry.insert(0, "0")
    last_index = None

    status_label.configure(
        text=f"{beats.shape[0]} beats loaded from: {current_file}"
    )
    prediction_label.configure(text="")
    update_display()


def update_display(event=None):
    """Update prediction + plots based on current slider value."""
    global last_index

    if beats is None or probs_all is None or preds_all is None:
        return

    idx = int(round(beat_slider.get()))
    if last_index == idx:
        return
    last_index = idx

    # Sync entry
    beat_entry.delete(0, "end")
    beat_entry.insert(0, str(idx))

    beat = beats[idx]
    probs = probs_all[idx]
    cls = int(preds_all[idx])

    code = idx_to_class[cls]
    name = class_to_name[code]
    conf_percent = probs[cls] * 100

    prediction_label.configure(
        text=f"Beat {idx}  —  {name} ({code})  —  {conf_percent:.1f}%",
        text_color="lime green" if cls == 0 else "red"
    )

    # --- ECG waveform plot ---
    ecg_plot.clear()
    ecg_plot.plot(beat, lw=1.4)
    ecg_plot.set_title("ECG Waveform", color="white", fontsize=11, pad=10)
    ecg_plot.set_xlabel("Sample", color="white")
    ecg_plot.set_ylabel("Amplitude", color="white")
    ecg_plot.set_facecolor("#020617")
    ecg_plot.grid(True, color="#475569", alpha=0.6)
    ecg_plot.tick_params(colors="white")
    for spine in ecg_plot.spines.values():
        spine.set_color("#94a3b8")

    # --- Confidence bar plot ---
    conf_plot.clear()
    label_codes = [idx_to_class[i] for i in range(num_classes)]
    conf_plot.bar(label_codes, probs)
    conf_plot.set_ylim(0, 1)
    conf_plot.set_title("Confidence Scores", color="white", fontsize=11, pad=10)
    conf_plot.set_ylabel("Probability", color="white")
    conf_plot.set_facecolor("#020617")
    conf_plot.tick_params(axis="y", colors="white")
    conf_plot.tick_params(axis="x", colors="white")
    for spine in conf_plot.spines.values():
        spine.set_color("#94a3b8")

    fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.95])
    ecg_canvas.draw()


def manual_set():
    """Jump to beat index typed in the entry box."""
    if beats is None:
        return
    try:
        idx = int(beat_entry.get())
        if 0 <= idx < beats.shape[0]:
            beat_slider.set(idx)
            update_display()
        else:
            messagebox.showwarning(
                "Out of range",
                f"Enter value between 0 and {beats.shape[0] - 1}"
            )
    except ValueError:
        messagebox.showerror("Invalid input", "Enter a valid integer beat index.")


# ==========================
#  GUI SETUP
# ==========================
app = ctk.CTk()
app.title("ECG Arrhythmia Classifier - Dark Mode")
app.configure(fg_color=APP_BG)
app.geometry("1400x800")
app.minsize(1200, 700)

# Try maximize (Windows)
try:
    app.state("zoomed")
except Exception:
    pass

# Top bar
top_frame = ctk.CTkFrame(app, fg_color=CARD_BG)
top_frame.pack(fill="x", pady=(5, 5), padx=10)

title = ctk.CTkLabel(
    top_frame,
    text="🫀 Offline ECG Arrhythmia Detection",
    font=("Segoe UI", 24, "bold")
)
title.pack(side="left", padx=15, pady=10)

load_btn = ctk.CTkButton(
    top_frame,
    text="Load ECG CSV",
    command=open_csv,
    corner_radius=20,
    height=36
)
load_btn.pack(side="right", padx=15, pady=10)

# Status
status_label = ctk.CTkLabel(
    app,
    text="Load a CSV file to begin",
    font=("Segoe UI", 13)
)
status_label.pack(pady=(0, 5))

# Controls frame
controls = ctk.CTkFrame(app, fg_color=CARD_BG)
controls.pack(fill="x", pady=10, padx=20)

slider_label = ctk.CTkLabel(
    controls,
    text="Beat Index:",
    font=("Segoe UI", 12, "bold")
)
slider_label.pack(side="left", padx=(15, 5))

beat_slider = ctk.CTkSlider(
    controls,
    from_=0,
    to=0,
    number_of_steps=1,
    command=update_display
)
beat_slider.pack(side="left", fill="x", expand=True, padx=10, pady=10)
beat_slider.configure(state="disabled")

beat_entry = ctk.CTkEntry(
    controls,
    width=80,
    placeholder_text="Beat #"
)
beat_entry.pack(side="left", padx=(10, 5), pady=10)

apply_btn = ctk.CTkButton(
    controls,
    text="Go",
    width=60,
    command=manual_set
)
apply_btn.pack(side="left", padx=(0, 15), pady=10)

prediction_label = ctk.CTkLabel(
    app,
    text="",
    font=("Segoe UI", 18, "bold")
)
prediction_label.pack(pady=(0, 8))

# Matplotlib figure inside GUI
fig = Figure(figsize=(12, 5), dpi=100)
fig.patch.set_facecolor(APP_BG)

ecg_plot = fig.add_subplot(121)
conf_plot = fig.add_subplot(122)

# Initial placeholder plots
ecg_plot.set_title("ECG Waveform", color="white")
ecg_plot.set_facecolor("#020617")
ecg_plot.text(
    0.5, 0.5,
    "No ECG loaded",
    ha="center",
    va="center",
    color="#64748b",
    transform=ecg_plot.transAxes
)

conf_plot.set_title("Confidence Scores", color="white")
conf_plot.set_facecolor("#020617")
conf_plot.text(
    0.5, 0.5,
    "Prediction will appear here",
    ha="center",
    va="center",
    color="#64748b",
    transform=conf_plot.transAxes
)

for ax in [ecg_plot, conf_plot]:
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")

fig.tight_layout(rect=[0.03, 0.06, 0.97, 0.95])

plot_frame = ctk.CTkFrame(app, fg_color=CARD_BG)
plot_frame.pack(fill="both", expand=True, padx=20, pady=(0, 5))

ecg_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
ecg_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

# --- Class legend under the plots ---
legend_text = (
    "Class codes:\n"
    "N = Normal beat   |   "
    "S = Supraventricular ectopic   |   "
    "V = Ventricular ectopic   |   "
    "F = Fusion beat   |   "
    "Q = Unknown / paced beat"
)

legend_label = ctk.CTkLabel(
    app,
    text=legend_text,
    font=("Segoe UI", 11),
    justify="center"
)
legend_label.pack(pady=(0, 10))

app.mainloop()
