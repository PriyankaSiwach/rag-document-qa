from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_engine import ask, ingest_text, ingest_url

app = FastAPI(title="RAG Document Q&A")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class TextIngestRequest(BaseModel):
    text: str = Field(..., min_length=1)


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/ingest/text")
def api_ingest_text(body: TextIngestRequest):
    try:
        return ingest_text(body.text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ingest/url")
def api_ingest_url(body: UrlIngestRequest):
    try:
        return ingest_url(body.url)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/ask")
def api_ask(body: AskRequest):
    try:
        return ask(body.question)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
