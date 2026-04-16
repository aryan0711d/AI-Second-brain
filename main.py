from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo import MongoClient, TEXT
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import os
import requests
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding(text: str) -> list:
    for attempt in range(3):
        response = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": text},
            timeout=30,
        )

        if response.status_code == 503:
            time.sleep(5)
            continue

        if not response.text.strip():
            time.sleep(5)
            continue

        data = response.json()

        if isinstance(data, dict) and "error" in data:
            if "loading" in data["error"].lower():
                time.sleep(5)
                continue
            raise Exception(f"Hugging Face API error: {data['error']}")

        return data[0] if isinstance(data[0], list) else data

    raise Exception("Hugging Face model failed after 3 retries")


groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client_ai = Groq(api_key=groq_key)

MODEL = os.getenv("MODEL_NAME")
if not MODEL:
    raise ValueError("MODEL_NAME not found in .env")  # fixed: was silently None

mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    raise ValueError("MONGO_URL not found in .env")

client = MongoClient(mongo_url)
db = client["ai_second_brain"]
notes_collection = db["notes"]
history_collection = db["chat_history"]
users_collection = db["users"]

notes_collection.create_index([("content", TEXT)])

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not found in .env")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def hash_password(password: str):
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str):
    return pwd_context.verify(plain, hashed)


def create_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


class Note(BaseModel):
    content: str


class Question(BaseModel):
    session_id: str
    question: str


class UserRegister(BaseModel):
    username: str
    password: str


@app.post("/register")
def register(user: UserRegister):
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = hash_password(user.password)
    users_collection.insert_one({"username": user.username, "password": hashed})
    return {"message": "User created successfully"}


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Wrong username or password")
    token = create_token({"sub": str(user["_id"])})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/")
def read_root():
    return {"message": "AI Second Brain is running"}


@app.post("/add-note")
def add_note(note: Note, user_id: str = Depends(get_current_user)):
    try:
        embedding = get_embedding(note.content)
        notes_collection.insert_one(
            {
                "content": note.content,
                "embedding": embedding,
                "user_id": user_id,
            }
        )
        return {"message": "Note saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/ask")
def ask_question(q: Question, user_id: str = Depends(get_current_user)):
    try:
        test_embedding = get_embedding("test")
        return {
            "embedding_length": len(test_embedding),
            "first_value": test_embedding[0],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.delete("/history/{session_id}")
def delete_session_history(session_id: str, user_id: str = Depends(get_current_user)):
    result = history_collection.delete_many(
        {"session_id": session_id, "user_id": user_id}
    )
    return {"message": f"Deleted {result.deleted_count} messages"}


@app.delete("/history")
def delete_all_history(user_id: str = Depends(get_current_user)):
    result = history_collection.delete_many({"user_id": user_id})
    return {"message": f"Deleted {result.deleted_count} messages"}
