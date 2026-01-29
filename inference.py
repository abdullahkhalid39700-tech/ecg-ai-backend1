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

# --- Robust Path Logic ---
# Locates ecg_model.pt in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ecg_model.pt")

# --- Singleton Model Initialization ---
# We initialize the model structure outside the function so it loads only once
model = CNN_LSTM_Attn().to(DEVICE)

def initialize_model():
    """Loads weights into the model if the file exists."""
    if os.path.exists(MODEL_PATH):
        try:
            # use weights_only=True for security if using newer torch versions
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

# Attempt to load on startup
MODEL_LOADED = initialize_model()

def predict_ecg(ecg_signal: np.ndarray):
    """
    Processes raw ECG signal, extracts beats, and returns AI prediction.
    """
    # 1. Verification
    if not MODEL_LOADED:
        return {"error": "AI Engine not initialized. Model file missing on server."}

    if len(ecg_signal) < SAMPLE_LEN:
        return {"error": "Signal length too short for analysis."}

    # 2. Normalization (Zero mean, unit variance)
    # Added 1e-8 to prevent division by zero on flat signals
    std = np.std(ecg_signal)
    if std < 1e-4:
        return {"error": "Invalid ECG signal: Flat line or zero variance detected."}
    
    ecg = (ecg_signal - np.mean(ecg_signal)) / (std + 1e-8)

    # 3. R-peak Detection
    # FS * 0.25 assumes a max heart rate of approx 240bpm
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
        
        # Ensure the segment stays within array bounds
        if p - half < 0 or p + half >= len(ecg):
            continue

        # Extract and Resample beat
        beat_segment = ecg[p - half : p + half]
        beat_resampled = resample(beat_segment, SAMPLE_LEN)

        # Calculate Log-RR Interval (Temporal feature)
        rr_interval = (peaks[i] - peaks[i-1]) / FS
        rr_log = np.log1p(rr_interval)

        beats.append(beat_resampled)
        rrs.append(rr_log)

    if not beats:
        return {"error": "Segmentation failed: No valid beats could be isolated."}

    # 5. Tensor Conversion
    # X shape: [Batch, Channel, Length] | RR shape: [Batch, 1]
    X = torch.tensor(np.array(beats), dtype=torch.float32).unsqueeze(1).to(DEVICE)
    RR = torch.tensor(np.array(rrs), dtype=torch.float32).unsqueeze(1).to(DEVICE)

    # 6. Inference
    with torch.no_grad():
        logits = model(X, RR)
        # Apply sigmoid to get probabilities (0.0 to 1.0)
        probs = torch.sigmoid(logits).cpu().numpy()

    # 7. Aggregation
    avg_prob = float(np.mean(probs))
    prediction_label = "ABNORMAL" if avg_prob > THRESHOLD else "NORMAL"

    return {
        "average_probability": round(avg_prob, 4),
        "prediction": prediction_label,
        "num_beats": len(beats)
    }
