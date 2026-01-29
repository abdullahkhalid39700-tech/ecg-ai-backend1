import os
import io
import shutil
import jwt
import datetime
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import NullPool
from bcrypt import hashpw, gensalt, checkpw
from inference import predict_ecg

app = FastAPI(title="HeartAI Neural Core API")

# ===================== DATABASE SETUP =====================
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "your_fallback_secret")

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    connect_args={"sslmode": "require"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ===================== DATABASE MODELS =====================
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
    file_path = Column(String)
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
    try:
        yield db
    finally:
        db.close()

def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Token")
    try:
        payload = jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid Session")

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== AUTH ROUTES =====================
@app.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    hashed = hashpw(data['password'].encode('utf-8'), gensalt()).decode('utf-8')
    new_user = User(email=data['email'], password=hashed)
    try:
        db.add(new_user)
        db.commit()
        return {"message": "Success"}
    except:
        raise HTTPException(status_code=400, detail="Practitioner already registered")

@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data['email']).first()
    if not user or not checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid Access Code")
    
    token = jwt.encode({
        "sub": user.email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    
    return {"token": token}

# ===================== UPLOAD ECG =====================
@app.post("/upload-ecg")
async def upload_ecg(
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    user = db.query(User).filter(User.email == email).first()

    # Find or create patient
    patient = db.query(Patient).filter(
        Patient.name == patient_name,
        Patient.user_id == user.id
    ).first()
    if not patient:
        patient = Patient(name=patient_name, age=age, gender=gender, user_id=user.id)
        db.add(patient)
        db.commit()

    # Save file locally
    os.makedirs("storage", exist_ok=True)
    timestamp = datetime.datetime.utcnow().timestamp()
    file_location = f"storage/{timestamp}_{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Save DB record
    ecg_entry = ECGFile(
        filename=file.filename,
        file_path=file_location,
        patient_id=patient.id,
        user_id=user.id
    )
    db.add(ecg_entry)
    db.commit()

    return {"message": "ECG stored successfully", "file_path": file_location}

# ===================== LIST ECG FILES =====================
@app.get("/ecg-files")
def list_files(db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    files = db.query(ECGFile).filter(ECGFile.user_id == user.id).all()
    return [
        {
            "file_id": f.id,
            "filename": f.filename,
            "uploaded_at": f.uploaded_at,
            "patient_id": f.patient_id
        }
        for f in files
    ]

# ===================== PREDICT FROM FILE =====================
@app.post("/predict-from-file/{file_id}")
def predict_from_file(file_id: int, db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    ecg_file = db.query(ECGFile).filter(
        ECGFile.id == file_id,
        ECGFile.user_id == user.id
    ).first()
    if not ecg_file:
        raise HTTPException(status_code=404, detail="File not found")

    # Read ECG CSV
    df = pd.read_csv(ecg_file.file_path)
    ecg = df.iloc[:, 1].values.astype(float)
    prediction = predict_ecg(ecg)

    # Save to history
    diag_entry = History(
        filename=ecg_file.filename,
        result=prediction["prediction"],
        probability=float(prediction.get("average_probability", prediction.get("probability", 0))),
        user_id=user.id
    )
    db.add(diag_entry)
    db.commit()

    # Return for frontend
    return {
        "prediction": prediction["prediction"],
        "probability": float(prediction.get("average_probability", prediction.get("probability", 0))),
        "waveform": ecg.tolist()[:1000]
    }

# ===================== HISTORY =====================
@app.get("/history")
def get_history(db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    return db.query(History).filter(History.user_id == user.id).order_by(History.created_at.desc()).all()

@app.get("/")
def root():
    return {"status": "Neural Core Operational", "mode": "Production"}
