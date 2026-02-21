"""
UtopIA — Node 1 : Sélection du modèle conceptuel
"""

import json
import re
from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, un assistant expert en ergothérapie clinique, spécialisé dans
la préconisation de véhicules pour personnes handicapées (VPH) selon la réglementation française.

Tu maîtrises parfaitement les modèles conceptuels en ergothérapie :
- MCREO : centré sur l'occupation, relation Personne-Environnement-Occupation
- PEO : adéquation entre capacités, environnement et tâches
- MOHO : volition, habituation, capacités de performance
- OTIPM : cadre global structurant le processus

Tu réponds en français, de façon structurée et professionnelle."""


def select_model_conceptuel(patient: PatientState, api_key: str, vectorstore=None) -> dict:
    client = Anthropic(api_key=api_key)

    profil = patient.to_context_summary()

    # Contexte RAG si disponible
    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "modele conceptuel ergotherapie evaluation besoins",
                k=3,
                vectorstore=vectorstore
            )
            if docs:
                rag_section = "Contexte documentaire :\n" + format_context(docs)
        except Exception:
            pass

    lines = [
        "Voici le profil d'un patient pour lequel je dois choisir un modèle conceptuel :",
        "",
        profil,
    ]
    if rag_section:
        lines.append("")
        lines.append(rag_section)

    lines += [
        "",
        "Analyse ce profil et réponds en JSON strict avec ce format :",
        '{"modele": "NOM", "justification": "...", "axes_evaluation": ["axe1", "axe2", "axe3", "axe4"]}',
    ]

    user_prompt = "\n".join(lines)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    text = response.content[0].text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {
        "modele": "MCREO",
        "justification": text,
        "axes_evaluation": [
            "Capacités fonctionnelles",
            "Environnement",
            "Occupations prioritaires",
            "Participation sociale"
        ]
    }
