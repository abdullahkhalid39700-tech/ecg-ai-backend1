import os
import numpy as np
import torch
from scipy.signal import find_peaks, resample
from model import CNN_LSTM_Attn

# --- Configuration ---
FS = 360
SAMPLE_LEN = 256
THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Fixed Path Logic ---
# This finds the folder where inference.py is, then looks for the .pt file there
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_model.pt")

# Load Model
model = CNN_LSTM_Attn().to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Successfully loaded: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Critical Error: Could not find {MODEL_PATH}")
    # This prevents the server from crashing on start, allowing you to see the error in the logs
except Exception as e:
    print(f"❌ Error loading model: {e}")

def predict_ecg(ecg_signal: np.ndarray):
    # 1. Check if model is actually loaded
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model file not found on server."}

    # 2. Normalize (Zero mean, unit variance)
    ecg = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-8)

    # 3. R-peak detection
    # 
    peaks, _ = find_peaks(
        ecg,
        distance=int(0.25 * FS),
        prominence=0.6
    )

    if len(peaks) < 2:
        return {"error": "Insufficient R-peaks detected for analysis."}

    beats, rrs = [], []

    # 4. Segmentation
    for i in range(1, len(peaks)):
        p = peaks[i]
        if p - SAMPLE_LEN//2 < 0 or p + SAMPLE_LEN//2 >= len(ecg):
            continue

        beat = ecg[p - SAMPLE_LEN//2:p + SAMPLE_LEN//2]
        beat = resample(beat, SAMPLE_LEN)

        rr = (peaks[i] - peaks[i-1]) / FS
        rr = np.log1p(rr)

        beats.append(beat)
        rrs.append(rr)

    if not beats:
        return {"error": "No valid beats extracted."}

    # 5. Conversion to Tensors
    X = torch.tensor(np.array(beats), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    RR = torch.tensor(np.array(rrs), dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # 6. Prediction
    # 
    with torch.no_grad():
        logits = model(X, RR)
        probs = torch.sigmoid(logits).cpu().numpy()

    avg_prob = float(probs.mean())
    prediction = "ABNORMAL" if avg_prob > THRESHOLD else "NORMAL"

    return {
        "average_probability": round(avg_prob, 4),
        "prediction": prediction,
        "num_beats": len(beats)
    }
