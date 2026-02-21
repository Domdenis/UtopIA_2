"""
UtopIA — Node 3 : Recherche des aides techniques
"""

import json
import re
from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert en préconisation de VPH selon la nomenclature française (réforme décembre 2025).

Tu connais parfaitement :
- Les catégories VPH : FRM, FRMC, FRMA, FRMS, FRMP, FRMV, FRE (A/B/C), FREP (A/B/C), FREV, SCO, CYC, POU_MRE
- Les modèles du marché : Action 3NG/4NG (FRM), Helio C2/A7 (FRMC), Apex C/A Küschall (FRMA),
  Action 5 Rigid (FRMS), Weely/Enzo (FRMP), Levo Summit (FRMV),
  Jazzy Air2/Go Chair (FRE-A), Edge 3/R-Trak (FRE-B), Whill Model C (FRE-C),
  Edge 3 Stretto (FREP-A), 4Front2/Q1450 (FREP-B), Outback (FREP-C), Evo ALTUS (FREV)
- Les règles de remboursement : zéro reste à charge depuis 01/12/2025
- Les critères cliniques de choix"""


def determine_vph_category(patient: PatientState, api_key: str, vectorstore=None) -> str:
    client = Anthropic(api_key=api_key)

    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "indications categorie VPH fauteuil roulant prescription",
                k=4,
                vectorstore=vectorstore,
                category_filter="reglementation"
            )
            if docs:
                rag_section = "Références :\n" + format_context(docs)
        except Exception:
            pass

    lines = [
        "Selon ce profil, quelle est la catégorie VPH la plus adaptée ?",
        "",
        patient.to_context_summary(),
    ]
    if rag_section:
        lines.append("")
        lines.append(rag_section)
    lines += [
        "",
        "Réponds UNIQUEMENT avec le code catégorie suivi d'une courte justification.",
        "Format : CODE | Justification"
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )

    text = response.content[0].text.strip()
    if "|" in text:
        return text.split("|")[0].strip()
    return text.split()[0] if text else "FRMC"


def search_at(patient: PatientState, api_key: str, vectorstore=None, tavily_api_key: str = None) -> list:
    client = Anthropic(api_key=api_key)

    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "fauteuil roulant indication remboursement prescription categorie",
                k=6,
                vectorstore=vectorstore
            )
            if docs:
                rag_section = "Références réglementaires :\n" + format_context(docs)
        except Exception:
            pass

    categorie = patient.categorie_vph_recommandee or "à déterminer"
    diagnostic_ergo = patient.diagnostic_ergo[:400] if patient.diagnostic_ergo else "Non disponible"

    lines = [
        "Propose 3 aides techniques (VPH) pour ce patient.",
        "",
        "PROFIL PATIENT :",
        patient.to_context_summary(),
        "",
        "CATÉGORIE VPH ENVISAGÉE : " + categorie,
        "DIAGNOSTIC ERGO : " + diagnostic_ergo,
    ]
    if rag_section:
        lines.append("")
        lines.append(rag_section)
    lines += [
        "",
        "Pour chaque proposition, fournis un JSON avec ces champs :",
        "- categorie : code VPH",
        "- modele : nom commercial",
        "- justification_clinique : 3-4 phrases",
        "- caracteristiques_cles : liste de 4 points",
        "- code_lpp : code LPP ou vide",
        "- remboursement : achat/LCD/LLD",
        "- avantages : liste de 2-3 avantages",
        "- points_vigilance : liste de 1-2 points",
        "",
        "Réponds UNIQUEMENT avec un tableau JSON valide : [{...}, {...}, {...}]"
    ]

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n".join(lines)}]
    )

    text = response.content[0].text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return [{
        "categorie": categorie,
        "modele": "À déterminer lors de l'essai",
        "justification_clinique": text[:500],
        "caracteristiques_cles": ["Évaluation approfondie nécessaire"],
        "code_lpp": "",
        "remboursement": "Selon durée du besoin",
        "avantages": ["À évaluer"],
        "points_vigilance": ["Essai obligatoire avec l'ergothérapeute"]
    }]
