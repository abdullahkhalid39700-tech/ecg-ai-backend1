import os
import jwt
import datetime
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import NullPool
from bcrypt import hashpw, gensalt, checkpw
from supabase import create_client, Client

# Import your inference logic
from inference import predict_ecg

app = FastAPI(title="Pulse Prognosis Neural Core API")

# ===================== CONFIGURATION & CLOUD STORAGE =====================
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "pulse_prognosis_secure_key_2026")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
BUCKET_ID = "ecg-storage"

def ensure_storage_exists():
    """Auto-creates the Supabase bucket if missing on startup."""
    try:
        buckets = supabase.storage.list_buckets()
        if not any(b.id == BUCKET_ID for b in buckets):
            supabase.storage.create_bucket(BUCKET_ID, options={"public": True})
            print(f"✅ Created Supabase bucket: {BUCKET_ID}")
    except Exception as e:
        print(f"⚠️ Storage Setup Warning: {e}")

ensure_storage_exists()

# ===================== DATABASE SETUP =====================
engine = create_engine(DATABASE_URL, poolclass=NullPool, connect_args={"sslmode": "require"})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    patients = relationship("Patient", back_populates="owner")
    ecg_files = relationship("ECGFile", back_populates="owner")
    history = relationship("History", back_populates="owner")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="patients")
    ecg_files = relationship("ECGFile", back_populates="patient")

class ECGFile(Base):
    __tablename__ = "ecg_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String) # Stores Supabase URL
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    patient = relationship("Patient", back_populates="ecg_files")
    owner = relationship("User", back_populates="ecg_files")

class History(Base):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    result = Column(String)
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="history")

Base.metadata.create_all(bind=engine)

# ===================== DEPENDENCIES =====================
def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Session expired. Please Login.")
    try:
        payload = jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid Session.")

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== AUTH ROUTES =====================
@app.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == data['email']).first():
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = hashpw(data['password'].encode('utf-8'), gensalt()).decode('utf-8')
    new_user = User(email=data['email'], password=hashed)
    db.add(new_user)
    db.commit()
    return {"message": "Success"}

@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data['email']).first()
    if not user or not checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid Credentials")
    token = jwt.encode({"sub": user.email, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)}, SECRET_KEY, algorithm="HS256")
    return {"token": token}

# ===================== ECG ROUTES =====================
@app.post("/upload-ecg")
async def upload_ecg(
    patient_name: str = Form(...), age: int = Form(...), gender: str = Form(...),
    file: UploadFile = File(...), db: Session = Depends(get_db), email: str = Depends(get_current_user)
):
    user = db.query(User).filter(User.email == email).first()
    
    # Cloud Upload
    file_content = await file.read()
    storage_path = f"{user.id}/{int(datetime.datetime.utcnow().timestamp())}_{file.filename}"
    supabase.storage.from_(BUCKET_ID).upload(storage_path, file_content)
    file_url = supabase.storage.from_(BUCKET_ID).get_public_url(storage_path)

    # Patient Logic
    patient = db.query(Patient).filter(Patient.name == patient_name, Patient.user_id == user.id).first()
    if not patient:
        patient = Patient(name=patient_name, age=age, gender=gender, user_id=user.id)
        db.add(patient)
        db.commit()

    # DB Record
    ecg_entry = ECGFile(filename=file.filename, file_path=file_url, patient_id=patient.id, user_id=user.id)
    db.add(ecg_entry)
    db.commit()
    return {"message": "Stored in Cloud"}

@app.get("/ecg-files")
def list_files(db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    return db.query(ECGFile).filter(ECGFile.user_id == user.id).all()

@app.post("/predict-from-file/{file_id}")
def predict_from_file(file_id: int, db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    ecg_file = db.query(ECGFile).filter(ECGFile.id == file_id, ECGFile.user_id == user.id).first()
    
    if not ecg_file: raise HTTPException(status_code=404, detail="File record missing")

    try:
        df = pd.read_csv(ecg_file.file_path)
        col = 1 if df.shape[1] > 1 else 0
        ecg = df.iloc[:, col].values.astype(float)
        
        prediction = predict_ecg(ecg)
        if "error" in prediction: raise HTTPException(status_code=400, detail=prediction["error"])

        history = History(filename=ecg_file.filename, result=prediction["prediction"], 
                          probability=prediction["average_probability"], user_id=user.id)
        db.add(history)
        db.commit()

        return {**prediction, "waveform": ecg.tolist()[:1000]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis Error: {str(e)}")

@app.get("/history")
def get_history(db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    return db.query(History).filter(History.user_id == user.id).order_by(History.created_at.desc()).all()

@app.get("/")
def root(): return {"status": "Operational", "engine": "Cloud-Native Neural Core"}
