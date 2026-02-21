"""
UtopIA — Node 4 : Rédaction de l'argumentaire CPAM
"""

import json
import re
from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert rédacteur d'argumentaires de prise en charge VPH.
Tu rédiges des argumentaires CPAM professionnels, conformes à la réglementation française (réforme 2025).
Tes argumentaires sont cliniquement justifiés, référencés à la nomenclature, centrés sur la participation sociale."""


def write_argumentaire(patient: PatientState, api_key: str, vectorstore=None) -> str:
    client = Anthropic(api_key=api_key)

    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "prise en charge remboursement prescription CPAM reforme VPH",
                k=4,
                vectorstore=vectorstore,
                category_filter="reglementation"
            )
            if docs:
                rag_section = "Références réglementaires :\n" + format_context(docs)
        except Exception:
            pass

    at_essayees_str = ", ".join(patient.at_essayees) if patient.at_essayees else "Non renseigné"

    propositions_str = ""
    if patient.propositions_at:
        lines_prop = []
        for i, prop in enumerate(patient.propositions_at[:4], 1):
            if isinstance(prop, dict):
                ligne = str(i) + ". " + prop.get('categorie', '') + " — " + prop.get('modele', '')
                lines_prop.append(ligne)
        propositions_str = "\n".join(lines_prop)

    lines = [
        "Rédige l'argumentaire complet de prise en charge pour ce dossier CPAM.",
        "",
        "DONNÉES PATIENT :",
        patient.to_context_summary(),
        "",
        "DIAGNOSTIC ERGOTHÉRAPIQUE :",
        (patient.diagnostic_ergo[:600] if patient.diagnostic_ergo else "Non disponible"),
        "",
        "PARCOURS D'ESSAI :",
        "VPH essayés : " + at_essayees_str,
        "VPH retenu : " + (patient.at_retenue or "Non renseigné"),
        "Observations : " + (patient.observations_essais or "Non renseigné"),
        "Motifs de rejet : " + (patient.motifs_rejet or "Non renseigné"),
        "Réglages définitifs : " + (patient.reglages_definitifs or "Non renseigné"),
    ]

    if propositions_str:
        lines.append("")
        lines.append("PRÉCONISATIONS ÉTUDIÉES :")
        lines.append(propositions_str)

    if rag_section:
        lines.append("")
        lines.append(rag_section)

    vph_retenu = patient.at_retenue or "VPH préconisé"
    lines += [
        "",
        "Rédige l'argumentaire avec cette structure :",
        "## Argumentaire de prise en charge — " + vph_retenu,
        "### 1. Présentation de la situation clinique",
        "### 2. Justification du besoin de compensation",
        "### 3. Description du VPH préconisé et de ses adjonctions",
        "### 4. Résultats des essais",
        "### 5. Impact attendu sur la participation et l'autonomie",
        "### 6. Modalités de prise en charge",
        "",
        "Rédige en 400-600 mots, phrases complètes, pas de tirets.",
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )

    return response.content[0].text


def generate_cpam_summary(patient: PatientState, api_key: str) -> dict:
    client = Anthropic(api_key=api_key)

    lines = [
        "Pour ce dossier VPH, extrais les informations clés pour les fiches CPAM.",
        "",
        "Patient : " + patient.prenom + " " + patient.nom,
        "Diagnostic : " + patient.diagnostic,
        "VPH retenu : " + (patient.at_retenue or ""),
        "Catégorie : " + (patient.categorie_vph_recommandee or ""),
        "",
        'Réponds en JSON : {"categorie_vph":"...","mode_prise_en_charge":"achat|LCD|LLD","justification_courte":"...","points_cles":["...","...","..."]}',
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=400,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )

    text = response.content[0].text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return {}
