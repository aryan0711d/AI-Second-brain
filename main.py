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
import cohere
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY not found in .env")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_embedding(text: str) -> list:
    response = cohere_client.embed(
        texts=[text], model="embed-english-light-v3.0", input_type="search_query"
    )
    return response.embeddings[0]


groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client_ai = Groq(api_key=groq_key)

MODEL = os.getenv("MODEL_NAME")
if not MODEL:
    raise ValueError("MODEL_NAME not found in .env")

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
        query_embedding = get_embedding(q.question)

        vector_results = list(
            notes_collection.aggregate(
                [
                    {
                        "$vectorSearch": {
                            "index": "vector_index",
                            "path": "embedding",
                            "queryVector": query_embedding,
                            "numCandidates": 100,
                            "limit": 5,
                        }
                    },
                    {"$match": {"user_id": user_id}},
                ]
            )
        )

        text_results = list(
            notes_collection.find(
                {"$text": {"$search": q.question}, "user_id": user_id},
                {"score": {"$meta": "textScore"}},
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(5)
        )

        combined = {}
        for note in vector_results:
            combined[note["_id"]] = {"content": note["content"], "score": 1.0}
        for note in text_results:
            if note["_id"] in combined:
                combined[note["_id"]]["score"] += note.get("score", 0)
            else:
                combined[note["_id"]] = {
                    "content": note["content"],
                    "score": note.get("score", 0),
                }

        sorted_notes = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
        top_notes = sorted_notes[:3]
        notes_text = "\n".join([n["content"] for n in top_notes])

        if not notes_text:
            notes_text = (
                "No relevant notes found. Use previous conversation if helpful."
            )

        session_history = list(
            history_collection.find(
                {"session_id": q.session_id, "user_id": user_id},
                {"_id": 0, "role": 1, "content": 1},
            )
        )

        messages = []
        messages.append(
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using the notes provided in the context. If the notes don't contain the answer, say clearly so. Be concise and informative.",
            }
        )
        for msg in session_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"context:\n{notes_text}\n\nQuestion: {q.question}",
            }
        )

        def stream_generator():
            full_answer = ""
            try:
                response = client_ai.chat.completions.create(
                    model=MODEL, messages=messages, stream=True
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_answer += content
                        yield content
            finally:
                if full_answer:
                    history_collection.insert_many(
                        [
                            {
                                "session_id": q.session_id,
                                "user_id": user_id,
                                "role": "user",
                                "content": q.question,
                            },
                            {
                                "session_id": q.session_id,
                                "user_id": user_id,
                                "role": "assistant",
                                "content": full_answer,
                            },
                        ]
                    )

        return StreamingResponse(stream_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")


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
