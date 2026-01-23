import numpy as np
import torch
from scipy.signal import find_peaks, resample
from model import CNN_LSTM_Attn

FS = 360
SAMPLE_LEN = 256
THRESHOLD = 0.6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM_Attn().to(DEVICE)
model.load_state_dict(torch.load("ecg_model.pt", map_location=DEVICE))
model.eval()

def predict_ecg(ecg_signal: np.ndarray):
    # Normalize like training
    ecg = (ecg_signal - ecg_signal.mean()) / (ecg_signal.std() + 1e-8)

    # R-peak detection
    peaks, _ = find_peaks(
        ecg,
        distance=int(0.25 * FS),
        prominence=0.6
    )

    if len(peaks) < 2:
        return {
            "error": "Not enough R-peaks detected"
        }

    beats, rrs = [], []

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

    if len(beats) == 0:
        return {
            "error": "No valid beats extracted"
        }

    X = torch.tensor(beats, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    RR = torch.tensor(rrs, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logits = model(X, RR)
        probs = torch.sigmoid(logits).cpu().numpy()

    avg_prob = float(probs.mean())
    prediction = "ABNORMAL" if avg_prob > THRESHOLD else "NORMAL"

    return {
        "average_probability": avg_prob,
        "prediction": prediction,
        "num_beats": len(beats)
    }
