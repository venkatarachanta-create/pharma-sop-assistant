from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import settings


def load_and_chunk_pdfs(pdf_dir: Path | None = None) -> list[Document]:
    target_dir = pdf_dir or settings.raw_pdf_dir
    pdf_files = sorted(target_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {target_dir.resolve()}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["doc_name"] = pdf_path.name
            page.metadata["page_num"] = int(page.metadata.get("page", 0)) + 1
        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    return all_chunks
