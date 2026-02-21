"""
UtopIA — Retriever RAG
Fonctions de recherche sémantique utilisées par les nodes.
Le vectorstore est passé en paramètre (construit par @st.cache_resource).
"""

from typing import List, Optional
from langchain_core.documents import Document


def search(
    query: str,
    k: int = 5,
    vectorstore=None,
    category_filter: Optional[str] = None,
) -> List[Document]:
    """Recherche sémantique dans le vectorstore."""
    if vectorstore is None:
        return []
    try:
        if category_filter:
            results = vectorstore.similarity_search(
                query, k=k, filter={"category": category_filter}
            )
        else:
            results = vectorstore.similarity_search(query, k=k)
        return results
    except Exception:
        return []


def format_context(docs: List[Document], max_chars: int = 3000) -> str:
    """Formate les chunks récupérés en contexte lisible pour le LLM."""
    if not docs:
        return ""
    parts = []
    total = 0
    for doc in docs:
        source = doc.metadata.get("source", "?")
        page = doc.metadata.get("page", "?")
        content = doc.page_content.strip()
        part = f"[{source} p.{page}]\n{content}"
        if total + len(part) > max_chars:
            break
        parts.append(part)
        total += len(part)
    return "\n\n---\n\n".join(parts)


def get_vph_indications(profil_summary: str, vectorstore=None) -> str:
    docs = search(
        f"indications catégorie fauteuil roulant {profil_summary}",
        k=6, vectorstore=vectorstore, category_filter="reglementation"
    )
    return format_context(docs)


def get_positionnement_context(besoins: str, vectorstore=None) -> str:
    docs = search(
        f"positionnement mesures anthropométriques réglages {besoins}",
        k=6, vectorstore=vectorstore, category_filter="evaluation-clinique"
    )
    return format_context(docs)


def get_cpam_context(type_vph: str, vectorstore=None) -> str:
    docs = search(
        f"prise en charge remboursement prescription {type_vph} CPAM réforme",
        k=5, vectorstore=vectorstore, category_filter="reglementation"
    )
    return format_context(docs)
