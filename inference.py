import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, resample
from model import CNN_LSTM_Attn

# --- Configuration ---
FS = 360
SAMPLE_LEN = 256
THRESHOLD = 0.6  # If abnormal prob > 0.6, label is ABNORMAL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Path Logic ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_model.pt")

# Initialize model structure
model = CNN_LSTM_Attn().to(DEVICE)

def initialize_model():
    """Loads weights into the model singleton."""
    if os.path.exists(MODEL_PATH):
        try:
            # map_location ensures it loads on CPU if CUDA is unavailable
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"✅ Neural Core Loaded Successfully: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            return False
    else:
        print(f"❌ Critical Error: {MODEL_PATH} not found.")
        return False

# Global flag for model status
MODEL_LOADED = initialize_model()

def predict_ecg(ecg_signal: np.ndarray):
    """
    Processes raw ECG signal, extracts beats, and calculates specific percentages.
    """
    # 1. Verification
    if not MODEL_LOADED:
        return {"error": "AI Engine not initialized. Model file missing on server."}

    if len(ecg_signal) < SAMPLE_LEN:
        return {"error": "Signal length too short for analysis."}

    # 2. Normalization
    std = np.std(ecg_signal)
    if std < 1e-4:
        return {"error": "Invalid ECG signal: Flat line or zero variance detected."}
    
    ecg = (ecg_signal - np.mean(ecg_signal)) / (std + 1e-8)

    # 3. R-peak Detection
    peaks, _ = find_peaks(
        ecg,
        distance=int(0.25 * FS),
        prominence=0.6
    )

    if len(peaks) < 2:
        return {"error": "Insufficient R-peaks detected. Ensure signal quality."}

    beats, rrs = [], []

    # 4. Segmentation & Feature Extraction
    for i in range(1, len(peaks)):
        p = peaks[i]
        half = SAMPLE_LEN // 2
        
        if p - half < 0 or p + half >= len(ecg):
            continue

        beat_segment = ecg[p - half : p + half]
        beat_resampled = resample(beat_segment, SAMPLE_LEN)

        rr_interval = (peaks[i] - peaks[i-1]) / FS
        rr_log = np.log1p(rr_interval)

        beats.append(beat_resampled)
        rrs.append(rr_log)

    if not beats:
        return {"error": "Segmentation failed: No valid beats could be isolated."}

    # 5. Tensor Conversion
    X = torch.tensor(np.array(beats), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    RR = torch.tensor(np.array(rrs), dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # 6. Inference
    with torch.no_grad():
        logits = model(X, RR)
        # We use sigmoid for binary classification probability
        abnormal_probs = torch.sigmoid(logits).cpu().numpy()

    # 7. Aggregation & Percentage Calculation
    # Calculate the average probability of being "Abnormal" across all beats
    avg_abnormal_prob = float(np.mean(abnormal_probs))
    
    # In binary classification: Normal Prob = 100% - Abnormal Prob
    avg_normal_prob = 1.0 - avg_abnormal_prob

    # Determine final label based on threshold
    prediction_label = "ABNORMAL" if avg_abnormal_prob > THRESHOLD else "NORMAL"

    return {
        "prediction": prediction_label,
        "average_probability": round(avg_abnormal_prob if avg_abnormal_prob > THRESHOLD else avg_normal_prob, 4),
        "normal_percentage": round(avg_normal_prob * 100, 2),
        "abnormal_percentage": round(avg_abnormal_prob * 100, 2),
        "num_beats_analyzed": len(beats)
    }
