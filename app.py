import os
import io
import jwt
import datetime
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
from bcrypt import hashpw, gensalt, checkpw
from inference import predict_ecg

app = FastAPI(title="ECG Arrhythmia Detection API with Auth")

# --- DATABASE SETUP ---
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "your_fallback_secret")

# Essential for Render + Supabase Pooler (Port 6543)
engine = create_engine(
    DATABASE_URL, 
    poolclass=NullPool, 
    connect_args={"sslmode": "require"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

# Create tables in Supabase
Base.metadata.create_all(bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTH LOGIC ---
@app.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    # Hash password
    hashed = hashpw(data['password'].encode('utf-8'), gensalt()).decode('utf-8')
    new_user = User(email=data['email'], password=hashed)
    try:
        db.add(new_user)
        db.commit()
        return {"message": "User created"}
    except:
        raise HTTPException(status_code=400, detail="User already exists")

@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data['email']).first()
    if not user or not checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = jwt.encode({
        "sub": user.email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256")
    
    return {"token": token}

# --- PROTECTED PREDICT ROUTE ---
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    authorization: Annotated[str | None, Header()] = None
):
    # 1. Check for token in Header
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization Header")
    
    try:
        # Verify Token
        jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # 2. Original Prediction Logic
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        if df.empty:
            return {"error": "The uploaded CSV file is empty."}

        ecg = df.iloc[:, 1].values.astype(float)
        result = predict_ecg(ecg)
        result["waveform"] = ecg.tolist()[:1000]
        
        return result
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@app.get("/")
def root():
    return {"status": "ECG model running with security"}
