from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import io
from inference import predict_ecg

app = FastAPI(title="ECG Arrhythmia Detection API")

# --- CORS SETTINGS ---
# This allows your local index.html to communicate with your Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ECG model running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into memory
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Check if file is empty
        if df.empty:
            return {"error": "The uploaded CSV file is empty."}

        # Get ECG values (assuming they are in the second column per your logic)
        ecg = df.iloc[:, 1].values.astype(float)

        # Run the AI prediction
        result = predict_ecg(ecg)
        
        # Include the waveform in the response so the chart can draw it
        result["waveform"] = ecg.tolist()[:1000] # Sending first 1000 points for the graph
        
        return result

    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}
