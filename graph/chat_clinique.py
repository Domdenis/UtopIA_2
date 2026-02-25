"""
UtopIA — Node Chat Clinique
Génère des questions ciblées sur le profil patient et collecte les réponses.
"""

import json
import re
from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, un ergothérapeute expert en préconisation de fauteuils roulants (VPH).
Tu mènes un entretien clinique structuré pour compléter l'évaluation d'un patient avant de faire tes préconisations.

Tes questions sont :
- Précises, cliniques, directement liées au profil spécifique du patient
- Organisées par thème : propulsion, contrôle postural, transferts, environnement, activités
- Formulées de façon claire pour un ergothérapeute
- Progressives : chaque réponse peut orienter la question suivante

Tu poses UNE question à la fois, avec éventuellement des sous-points.
Tu utilises les réponses précédentes pour affiner les questions suivantes.
Tu indiques quand tu as suffisamment d'informations pour faire des préconisations."""


def generate_first_question(patient: PatientState, api_key: str, vectorstore=None) -> str:
    """Génère la première question ciblée selon le profil."""
    client = Anthropic(api_key=api_key)

    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "evaluation capacites fonctionnelles fauteuil roulant propulsion transfert",
                k=3, vectorstore=vectorstore
            )
            if docs:
                rag_section = "Références cliniques :\n" + format_context(docs)
        except Exception:
            pass

    lines = [
        "Voici le profil d'un patient pour lequel je dois compléter l'évaluation avant de préconiser un VPH.",
        "",
        patient.to_context_summary(),
        "",
        "Sur la base de ce profil, pose la PREMIÈRE question clinique la plus importante pour orienter",
        "le choix du type de fauteuil (manuel, électrique, avec assistance...).",
        "",
        "La question doit être :",
        "- Très spécifique à CE patient (cite son prénom, sa pathologie, sa situation)",
        "- Centrée sur le point le plus déterminant pour le choix du VPH",
        "- Avec des sous-points si nécessaire (3-4 maximum)",
        "",
        "Commence directement par la question, avec une courte introduction contextuelle.",
        "Utilise des emojis 👉 pour les sous-points.",
    ]
    if rag_section:
        lines.append("")
        lines.append(rag_section)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )
    return response.content[0].text


def generate_next_question(
    patient: PatientState,
    api_key: str,
    conversation_history: list,
    vectorstore=None
) -> dict:
    """
    Génère la question suivante basée sur l'historique de conversation.
    Retourne : {"question": "...", "terminé": bool, "synthese": "..."}
    """
    client = Anthropic(api_key=api_key)

    # Construire l'historique formaté
    history_str = ""
    for msg in conversation_history:
        role = "UtopIA" if msg["role"] == "assistant" else "Ergothérapeute"
        history_str += role + " : " + msg["content"] + "\n\n"

    lines = [
        "Profil patient :",
        patient.to_context_summary(),
        "",
        "Entretien clinique réalisé jusqu'ici :",
        history_str,
        "Sur la base des réponses obtenues :",
        "1. As-tu suffisamment d'informations pour faire des préconisations VPH précises ?",
        "2. Si oui, réponds en JSON : {\"termine\": true, \"synthese\": \"résumé des points clés en 3-4 phrases\"}",
        "3. Si non, pose la prochaine question clinique la plus importante.",
        "   Réponds en JSON : {\"termine\": false, \"question\": \"ta question avec sous-points\"}",
        "",
        "Maximum 5 questions au total. Si on a déjà 4 échanges ou plus, conclure obligatoirement.",
        "Réponds UNIQUEMENT en JSON valide.",
    ]

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )

    text = response.content[0].text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    # Fallback
    nb_exchanges = len([m for m in conversation_history if m["role"] == "assistant"])
    if nb_exchanges >= 4:
        return {"termine": True, "synthese": "Informations suffisantes collectées pour la préconisation."}
    return {"termine": False, "question": text}


def build_chat_synthesis(patient: PatientState, api_key: str, conversation_history: list) -> str:
    """Synthèse finale de l'entretien pour enrichir le PatientState."""
    client = Anthropic(api_key=api_key)

    history_str = ""
    for msg in conversation_history:
        role = "UtopIA" if msg["role"] == "assistant" else "Ergothérapeute"
        history_str += role + " : " + msg["content"] + "\n\n"

    lines = [
        "Sur la base de cet entretien clinique complémentaire :",
        "",
        history_str,
        "Rédige une synthèse clinique structurée en 4-5 phrases qui résume :",
        "- Les capacités de propulsion et d'endurance",
        "- Le contrôle postural",
        "- Les capacités de transfert",
        "- Les contraintes environnementales clés",
        "- Les éléments déterminants pour le choix du VPH",
        "",
        "Cette synthèse sera intégrée directement dans le dossier patient.",
    ]

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )
    return response.content[0].text
