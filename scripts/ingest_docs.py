import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_and_chunk_pdfs
from src.rag.vectorstore import build_vectorstore


def main():
    print("📄 Loading and chunking PDFs from data/raw_pdfs/...")
    chunks = load_and_chunk_pdfs()
    print(f"✅ Created {len(chunks)} chunks. Building ChromaDB index...")
    build_vectorstore(chunks)
    print("🎉 Vector store persisted to data/chroma_db/ successfully!")


if __name__ == "__main__":
    main()
