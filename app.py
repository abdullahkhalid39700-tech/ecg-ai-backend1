from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
from inference import predict_ecg

app = FastAPI(title="ECG Arrhythmia Detection API")

@app.get("/")
def root():
    return {"status": "ECG model running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # ECG values must be second column
    ecg = df.iloc[:, 1].values.astype(float)

    result = predict_ecg(ecg)
    return result
