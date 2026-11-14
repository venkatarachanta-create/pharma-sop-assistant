from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.config.settings import settings


def get_embeddings():
    if settings.llm_provider.lower() == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.gemini_api_key,
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(settings.chroma_persist_dir),
        collection_name="pharma_sop_docs",
    )


def load_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=get_embeddings(),
        collection_name="pharma_sop_docs",
    )
