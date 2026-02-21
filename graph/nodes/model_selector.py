"""
UtopIA — Node 1 : Sélection du modèle conceptuel
Analyse le profil patient et choisit le modèle conceptuel ergothérapique adapté.
"""

from anthropic import Anthropic
from graph.state import PatientState
from rag.retriever import search, format_context

SYSTEM_PROMPT = """Tu es UtopIA, un assistant expert en ergothérapie clinique, spécialisé dans
la préconisation de véhicules pour personnes handicapées (VPH) selon la réglementation française.

Tu maîtrises parfaitement les modèles conceptuels en ergothérapie :
- MCREO (Modèle Canadien du Rendement et de l'Engagement Occupationnel) : centré sur l'occupation,
  la relation Personne-Environnement-Occupation, idéal pour explorer le sens des activités
- PEO (Person-Environment-Occupation) : analyse l'adéquation entre capacités, environnement et tâches
- MOHO (Model of Human Occupation) : volition, habituation, capacités de performance
- OTIPM comme cadre global structurant le processus

Tu réponds en français, de façon structurée et professionnelle, en ergothérapeute praticien."""


def select_model_conceptuel(patient: PatientState, api_key: str, vectorstore=None) -> dict:
    """
    Analyse le profil patient et sélectionne le modèle conceptuel le plus adapté.
    Retourne : { modele, justification, axes_evaluation }
    """
    client = Anthropic(api_key=api_key)

    # Contexte RAG
    rag_context = ""
    if vectorstore:
        from rag.retriever import search, format_context
        docs = search(
            f"modèle conceptuel ergothérapie {patient.diagnostic} {patient.situation_sante}",
            k=4, vectorstore=vectorstore
        )
        rag_context = format_context(docs)

    profil = patient.to_context_summary()

    user_prompt = f"""Voici le profil d'un patient pour lequel je dois choisir un modèle conceptuel en ergothérapie :

{profil}

{f"Contexte documentaire :{chr(10)}{rag_context}" if rag_context else ""}

Analyse ce profil et :
1. Choisis le modèle conceptuel le plus adapté (MCREO, PEO, MOHO, ou autre)
2. Justifie ton choix en 3-4 phrases cliniques
3. Identifie les 4-5 axes d'évaluation prioritaires pour ce patient

Réponds en JSON strict avec ce format :
{{
  "modele": "NOM_DU_MODELE",
  "justification": "...",
  "axes_evaluation": ["axe1", "axe2", "axe3", "axe4"]
}}"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    import json, re
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
        "axes_evaluation": ["Capacités fonctionnelles", "Environnement", "Occupations prioritaires", "Participation sociale"]
    }
