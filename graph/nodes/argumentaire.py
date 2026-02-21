"""
UtopIA — Node 4 : Rédaction de l'argumentaire CPAM
Produit l'argumentaire normé post-essais pour la prise en charge.
"""

from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert rédacteur d'argumentaires de prise en charge VPH.
Tu rédiges des argumentaires CPAM professionnels, conformes à la réglementation française (réforme 2025),
structurés pour convaincre la caisse d'assurance maladie de la pertinence de la prescription.

Tes argumentaires sont :
- Cliniquement justifiés, basés sur les déficiences et les besoins objectivés
- Référencés à la nomenclature (codes LPP, catégories VPH)
- Centrés sur la participation sociale et l'autonomie du patient
- Conformes au parcours réglementaire (évaluation → préconisation → essai → prescription)"""


def write_argumentaire(patient: PatientState, api_key: str, vectorstore=None) -> str:
    """
    Rédige l'argumentaire CPAM complet basé sur l'ensemble du parcours.
    """
    client = Anthropic(api_key=api_key)

    # Contexte RAG
    rag_context = ""
    if vectorstore:
        from rag.retriever import search, format_context
        docs = search(
            f"argumentaire prise en charge {patient.at_retenue or patient.categorie_vph_recommandee} CPAM réforme",
            k=5, vectorstore=vectorstore, category_filter="reglementation"
        )
        rag_context = format_context(docs)

    # Construire le contexte complet du parcours
    propositions_str = ""
    if patient.propositions_at:
        lines = []
        for i, prop in enumerate(patient.propositions_at[:4], 1):
            if isinstance(prop, dict):
                lines.append(f"{i}. {prop.get('categorie', '')} — {prop.get('modele', '')} : {prop.get('justification_clinique', '')[:150]}")
        propositions_str = "\n".join(lines)

    at_essayees_str = ", ".join(patient.at_essayees) if patient.at_essayees else "Non renseigné"

    user_prompt = f"""Rédige l'argumentaire complet de prise en charge pour ce dossier CPAM.

═══ DONNÉES PATIENT ═══
{patient.to_context_summary()}

═══ DIAGNOSTIC ERGOTHÉRAPIQUE ═══
{patient.diagnostic_ergo[:600] if patient.diagnostic_ergo else "Non disponible"}

═══ PARCOURS D'ESSAI ═══
VPH essayés : {at_essayees_str}
VPH retenu : {patient.at_retenue or "Non renseigné"}
Observations lors des essais : {patient.observations_essais or "Non renseigné"}
Motifs de rejet des autres VPH : {patient.motifs_rejet or "Non renseigné"}
Réglages définitifs : {patient.reglages_definitifs or "Non renseigné"}

═══ PRÉCONISATIONS ÉTUDIÉES ═══
{propositions_str or "Non disponible"}

{f"═══ RÉFÉRENCES RÉGLEMENTAIRES ═══{chr(10)}{rag_context}" if rag_context else ""}

Rédige l'argumentaire en respectant cette structure :

## Argumentaire de prise en charge — {patient.at_retenue or "VPH préconisé"}

### 1. Présentation de la situation clinique
(Synthèse de la situation du patient, diagnostic, histoire du handicap)

### 2. Justification du besoin de compensation
(En quoi le handicap nécessite un VPH — lien avec les déficiences objectivées)

### 3. Description du VPH préconisé et de ses adjonctions
(Caractéristiques techniques, catégorie, adjonctions nécessaires avec codes LPP)

### 4. Résultats des essais
(Ce qui a été testé, pourquoi le VPH retenu est le plus adapté, pourquoi les autres ont été écartés)

### 5. Impact attendu sur la participation et l'autonomie
(Objectifs fonctionnels, occupationnels, participation sociale attendue)

### 6. Modalités de prise en charge
(Catégorie VPH, mode de prise en charge recommandé : achat/LCD/LLD, justification)

Rédige de façon professionnelle, en 400-600 mots. Utilise des phrases complètes, pas de tirets."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text


def generate_cpam_summary(patient: PatientState, api_key: str) -> dict:
    """
    Génère un résumé structuré pour le pré-remplissage des fiches CPAM.
    """
    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""Pour ce dossier VPH, extrais les informations clés pour les fiches CPAM.

Patient : {patient.prenom} {patient.nom}
Diagnostic : {patient.diagnostic}
VPH retenu : {patient.at_retenue}
Catégorie : {patient.categorie_vph_recommandee}

Réponds en JSON :
{{
  "categorie_vph": "...",
  "mode_prise_en_charge": "achat|LCD|LLD",
  "duree_besoin": "...",
  "equipe_pluri": true/false,
  "medecin_requis": "généraliste|MPR|DU",
  "justification_courte": "...",
  "points_cles_argumentaire": ["...", "...", "..."]
}}"""
        }]
    )

    import json, re
    text = response.content[0].text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
