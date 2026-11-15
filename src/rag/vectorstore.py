from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import SecretStr
from src.config.settings import settings


def get_embeddings():
    if settings.llm_provider.lower() == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            api_key=SecretStr(settings.gemini_api_key),
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=SecretStr(settings.openai_api_key),
    )

def build_vectorstore(chunks: list[Document]) -> FAISS:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    vs.save_local(str(settings.chroma_persist_dir))
    return vs


def load_vectorstore() -> FAISS:
    return FAISS.load_local(
        str(settings.chroma_persist_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
