import os
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME = "rag-demo"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

_pc = None
_embeddings = None
_llm_chain = None
_document_loaded = False


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        if not api_key:
            raise ValueError("PINECONE_API_KEY is missing from .env")
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise ValueError("OPENAI_API_KEY is missing from .env")
        _embeddings = OpenAIEmbeddings()
    return _embeddings


def _get_chain():
    global _llm_chain
    if _llm_chain is None:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer the question using ONLY the context below. "
                    "If the answer is not in the context, say \"I don't know\". "
                    "Do not use outside knowledge.\n\nContext:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )
        _llm_chain = prompt | llm
    return _llm_chain


def ensure_index() -> None:
    pc = _get_pinecone()
    existing = [index["name"] for index in pc.list_indexes()]
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)


def _clear_index() -> None:
    pc = _get_pinecone()
    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    if stats.get("total_vector_count", 0) > 0:
        index.delete(delete_all=True)


def _split_and_store(documents: list[Document]) -> int:
    global _document_loaded
    ensure_index()
    _clear_index()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No text found to index. Try a different paragraph or link.")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        index_name=INDEX_NAME,
    )
    _document_loaded = True
    return len(chunks)


def ingest_text(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Please paste a paragraph first.")
    count = _split_and_store([Document(page_content=text)])
    return {"chunk_count": count, "source": "text"}


def ingest_url(url: str) -> dict:
    url = (url or "").strip()
    if not url:
        raise ValueError("Please enter a URL first.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    os.environ.setdefault(
        "USER_AGENT",
        "RAGDocumentQA/1.0 (+https://localhost; educational project)",
    )
    loader = WebBaseLoader(url)
    documents = loader.load()
    if not documents or not any(doc.page_content.strip() for doc in documents):
        raise ValueError("Could not extract text from that URL.")

    count = _split_and_store(documents)
    return {"chunk_count": count, "source": "url", "url": url}


def ask(question: str) -> dict:
    global _document_loaded
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    if not _document_loaded:
        # Allow asking if index already has vectors from a prior run
        ensure_index()
        stats = _get_pinecone().Index(INDEX_NAME).describe_index_stats()
        if stats.get("total_vector_count", 0) == 0:
            raise ValueError("Load a paragraph or link before asking questions.")
        _document_loaded = True

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=_get_embeddings(),
    )
    relevant_chunks = vectorstore.similarity_search(question, k=3)
    chunk_texts = [chunk.page_content for chunk in relevant_chunks]
    context = "\n\n".join(chunk_texts)

    response = _get_chain().invoke({"context": context, "question": question})
    return {
        "answer": response.content,
        "chunks": chunk_texts,
    }
