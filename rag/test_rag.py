"""
UtopIA — Test du RAG
Usage : python rag/test_rag.py
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

from rag.retriever import search, search_with_score, format_context

TEST_QUERIES = [
    {
        "label": "Indications FRE",
        "query": "quelles sont les indications pour prescrire un fauteuil roulant électrique FRE ?",
        "filter": "reglementation",
    },
    {
        "label": "Mesures anthropométriques",
        "query": "comment mesurer la largeur d'assise et la profondeur du fauteuil ?",
        "filter": "evaluation-clinique",
    },
    {
        "label": "Catégories VPH modulaires",
        "query": "différence entre FRMC FRMA FRMP fauteuil manuel modulaire",
        "filter": "categories-vph",
    },
    {
        "label": "Règles remboursement réforme 2025",
        "query": "prise en charge zéro reste à charge achat location fauteuil roulant",
        "filter": "reglementation",
    },
    {
        "label": "Positionnement bassin cyphose",
        "query": "rétroversion bassin cyphose éléments de forme coussin dossier",
        "filter": "evaluation-clinique",
    },
    {
        "label": "MCPAA évaluation posturale",
        "query": "MCPAA mesure contrôle postural assis évaluation",
        "filter": None,  # Tout le corpus
    },
]


def run_tests():
    print("\n🧪 UtopIA RAG — Tests de pertinence\n")
    print("=" * 60)

    for test in TEST_QUERIES:
        print(f"\n📌 {test['label']}")
        print(f"   Query : {test['query'][:70]}...")
        if test["filter"]:
            print(f"   Filtre : {test['filter']}")

        results = search_with_score(
            test["query"],
            k=3,
            category_filter=test["filter"],
        )

        if not results:
            print("   ⚠️  Aucun résultat !")
            continue

        for i, (doc, score) in enumerate(results):
            source = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"\n   [{i+1}] Score: {score:.3f} | {source} p.{page}")
            print(f"       {preview}...")

        print()

    print("=" * 60)
    print("\n✅ Tests terminés\n")


if __name__ == "__main__":
    run_tests()
