"""
UtopIA — Node 2 : Rédaction du diagnostic ergothérapique
Produit un diagnostic structuré selon le modèle conceptuel choisi.
"""

from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert spécialisé en préconisation VPH.
Tu rédiges des diagnostics ergothérapiques professionnels, structurés et argumentés.
Ton style est clinique, précis, orienté sur les besoins fonctionnels et occupationnels.
Tu utilises le vocabulaire professionnel de l'ergothérapie française.
Tu t'appuies sur les données recueillies pour formuler un diagnostic qui servira de base aux préconisations."""


def write_diagnostic(patient: PatientState, api_key: str, vectorstore=None) -> str:
    """
    Rédige le diagnostic ergothérapique basé sur le profil patient
    et le modèle conceptuel sélectionné.
    """
    client = Anthropic(api_key=api_key)

    # Contexte RAG - positionnement et évaluation
    rag_context = ""
    if vectorstore:
        from rag.retriever import search, format_context
        query = f"diagnostic évaluation {patient.diagnostic} {patient.modele_conceptuel_choisi} besoins occupationnels"
        docs = search(query, k=5, vectorstore=vectorstore)
        rag_context = format_context(docs)

    profil = patient.to_context_summary()
    modele = patient.modele_conceptuel_choisi or "MCREO"
    justification_modele = patient.justification_modele or ""

    # Construire les données anthropométriques si disponibles
    mesures_str = ""
    mesures = []
    if patient.largeur_bassin: mesures.append(f"Largeur bassin : {patient.largeur_bassin} cm")
    if patient.longueur_cuisses: mesures.append(f"Longueur cuisses : {patient.longueur_cuisses} cm")
    if patient.longueur_creux_poplite_pied: mesures.append(f"Creux poplité-pied : {patient.longueur_creux_poplite_pied} cm")
    if patient.hauteur_omoplate: mesures.append(f"Hauteur omoplate : {patient.hauteur_omoplate} cm")
    if patient.largeur_tronc: mesures.append(f"Largeur tronc : {patient.largeur_tronc} cm")
    if patient.poids: mesures.append(f"Poids : {patient.poids} kg")
    if mesures:
        mesures_str = "\nMesures anthropométriques :\n" + "\n".join(mesures)

    user_prompt = f"""Rédige un diagnostic ergothérapique complet pour ce patient, en utilisant le modèle {modele}.

PROFIL PATIENT :
{profil}
{mesures_str}

MODÈLE CONCEPTUEL : {modele}
{f"Justification du choix : {justification_modele}" if justification_modele else ""}

{f"RÉFÉRENCES CLINIQUES :{chr(10)}{rag_context}" if rag_context else ""}

Structure ton diagnostic en 4 parties :

## 1. Présentation de la situation
(Synthèse clinique du patient, contexte de la demande)

## 2. Analyse des besoins selon le modèle {modele}
(Besoins fonctionnels, occupationnels, environnementaux identifiés)

## 3. Déficits et ressources
(Points de vigilance cliniques, capacités existantes, facteurs facilitants/limitants)

## 4. Objectifs de la préconisation
(Ce que le VPH doit permettre en termes de participation et d'autonomie)

Sois précis, clinique, et base-toi strictement sur les données fournies. Maximum 500 mots."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text
