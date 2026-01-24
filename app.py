import os
import io
import jwt
import datetime
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.pool import NullPool
from bcrypt import hashpw, gensalt, checkpw
from inference import predict_ecg

app = FastAPI(title="HeartAI Neural Core API")

# --- DATABASE SETUP ---
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "your_fallback_secret") #

engine = create_engine(
    DATABASE_URL, 
    poolclass=NullPool, # Essential for Supabase Port 6543
    connect_args={"sslmode": "require"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False) #
    history = relationship("History", back_populates="owner")

class History(Base): # New table for Diagnostic History
    __tablename__ = "history"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    result = Column(String)
    probability = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="history")

Base.metadata.create_all(bind=engine) #

# --- DEPENDENCIES ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper to verify JWT and return email
def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Token")
    try:
        payload = jwt.decode(authorization, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except:
        raise HTTPException(status_code=401, detail="Invalid Session")

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your Render frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTH ROUTES ---
@app.post("/signup")
def signup(data: dict, db: Session = Depends(get_db)):
    hashed = hashpw(data['password'].encode('utf-8'), gensalt()).decode('utf-8') #
    new_user = User(email=data['email'], password=hashed)
    try:
        db.add(new_user)
        db.commit()
        return {"message": "Success"}
    except:
        raise HTTPException(status_code=400, detail="Practitioner already registered")

@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data['email']).first() #
    if not user or not checkpw(data['password'].encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid Access Code")
    
    token = jwt.encode({
        "sub": user.email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }, SECRET_KEY, algorithm="HS256") #
    
    return {"token": token}

# --- DIAGNOSTIC ROUTES ---
@app.post("/predict")
async def predict(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    email: str = Depends(get_current_user)
):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        ecg = df.iloc[:, 1].values.astype(float)
        
        # Run AI Analysis
        prediction = predict_ecg(ecg)
        
        # Get User ID
        user = db.query(User).filter(User.email == email).first()
        
        # Save to History table
        diag_entry = History(
            filename=file.filename,
            result=prediction["prediction"],
            probability=float(prediction["average_probability"]),
            user_id=user.id
        )
        db.add(diag_entry)
        db.commit()

        prediction["waveform"] = ecg.tolist()[:1000]
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history") # New route for frontend to see past tests
def get_history(db: Session = Depends(get_db), email: str = Depends(get_current_user)):
    user = db.query(User).filter(User.email == email).first()
    return db.query(History).filter(History.user_id == user.id).order_by(History.created_at.desc()).all()

@app.get("/")
def root():
    return {"status": "Neural Core Operational", "mode": "Production"}
