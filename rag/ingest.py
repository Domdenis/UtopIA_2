"""
UtopIA — Ingestion RAG
Embeddings locaux via sentence-transformers (pas de clé API supplémentaire).
"""

from pathlib import Path
from typing import List

import fitz
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

CHUNK_CONFIG = {
    "reglementation":      {"chunk_size": 800,  "chunk_overlap": 150},
    "evaluation-clinique": {"chunk_size": 600,  "chunk_overlap": 120},
    "categories-vph":      {"chunk_size": 500,  "chunk_overlap": 100},
    "modeles-conceptuels": {"chunk_size": 700,  "chunk_overlap": 150},
    "argumentaires":       {"chunk_size": 700,  "chunk_overlap": 150},
}
DEFAULT_CHUNK = {"chunk_size": 600, "chunk_overlap": 120}


def extract_pdf(path: Path) -> List[Document]:
    docs = []
    try:
        pdf = fitz.open(str(path))
        for page_num, page in enumerate(pdf):
            text = page.get_text().strip()
            if len(text) < 50:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "category": path.parent.name,
                    "page": page_num + 1,
                }
            ))
        pdf.close()
    except Exception as e:
        print(f"Erreur extraction {path.name}: {e}")
    return docs


def build_vectorstore(api_key: str) -> Chroma:
    """
    Construit le vectorstore en mémoire.
    api_key est gardé en paramètre pour la compatibilité mais non utilisé ici.
    """
    pdf_files = sorted(DOCS_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {DOCS_DIR}")

    # Extraction
    all_raw = []
    for pdf_path in pdf_files:
        all_raw.extend(extract_pdf(pdf_path))

    if not all_raw:
        raise ValueError("Aucun texte extrait des PDFs.")

    # Chunking
    all_chunks = []
    for doc in all_raw:
        category = doc.metadata.get("category", "default")
        config = CHUNK_CONFIG.get(category, DEFAULT_CHUNK)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents([doc])
        for c in chunks:
            c.metadata.update(doc.metadata)
        all_chunks.extend(chunks)

    # Embeddings multilingues locaux — excellent pour le français, aucune clé requise
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="utopia",
    )
    return vectorstore
