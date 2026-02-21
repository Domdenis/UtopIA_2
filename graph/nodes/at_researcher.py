"""
UtopIA — Node 3 : Recherche des aides techniques
Agent ReAct qui recherche les AT adaptées au profil patient.
"""

from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert en préconisation de véhicules pour personnes 
handicapées (VPH) selon la nomenclature française (réforme décembre 2025).

Tu connais parfaitement :
- Les catégories VPH : FRM, FRMC, FRMA, FRMS, FRMP, FRMV, FRE (A/B/C), FREP (A/B/C), FREV, SCO, CYC, POU_MRE
- Les modèles du marché : Action 3NG/4NG (FRM), Helio C2/A7 (FRMC), Apex C/A Küschall (FRMA),
  Action 5 Rigid (FRMS), Weely/Enzo (FRMP), Levo Summit (FRMV),
  Jazzy Air2/Go Chair (FRE-A), Edge 3/R-Trak (FRE-B), Whill Model C (FRE-C),
  Edge 3 Stretto (FREP-A), 4Front2/Q1450 (FREP-B), Outback (FREP-C), Evo ALTUS/4Front2 Stand UP (FREV)
- Les règles de remboursement : zéro reste à charge depuis 01/12/2025, essai obligatoire ≥2 modèles
- Les adjonctions LPP et leur facturation
- Les critères cliniques de choix (posture, propulsion, environnement, activités)

Tu proposes des préconisations argumentées, cliniquement justifiées, en lien direct avec le profil patient."""


def search_at(patient: PatientState, api_key: str, vectorstore=None,
              tavily_api_key: str = None) -> list:
    """
    Recherche et propose 3-4 AT adaptées au profil patient.
    Retourne une liste de propositions structurées.
    """
    client = Anthropic(api_key=api_key)

    # Contexte RAG
    rag_context = ""
    if vectorstore:
        from rag.retriever import search, format_context
        queries = [
            f"catégorie VPH indications {patient.diagnostic} {patient.capacites_physiques}",
            f"fauteuil roulant {patient.lieu_vie} {' '.join(patient.deplacements)}",
            f"remboursement prise en charge {patient.diagnostic}",
        ]
        all_docs = []
        for q in queries:
            docs = search(q, k=3, vectorstore=vectorstore)
            all_docs.extend(docs)
        # Dédupliquer
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = d.metadata.get("source", "") + str(d.metadata.get("page", ""))
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)
        rag_context = format_context(unique_docs[:8])

    # Recherche web optionnelle via Tavily
    web_results = ""
    if tavily_api_key:
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=tavily_api_key)
            query = f"fauteuil roulant {patient.categorie_vph_recommandee or patient.diagnostic} France 2025"
            results = tavily.search(query, max_results=3, search_depth="basic")
            if results.get("results"):
                web_parts = []
                for r in results["results"][:3]:
                    web_parts.append(f"• {r['title']}: {r['content'][:200]}")
                web_results = "\n".join(web_parts)
        except Exception:
            pass

    profil = patient.to_context_summary()
    categorie = patient.categorie_vph_recommandee or "à déterminer selon le profil"

    user_prompt = f"""Propose 3 à 4 aides techniques (VPH) pour ce patient.

PROFIL PATIENT :
{profil}

CATÉGORIE VPH ENVISAGÉE : {categorie}
DIAGNOSTIC ERGO : {patient.diagnostic_ergo[:400] if patient.diagnostic_ergo else "Non disponible"}

{f"RÉFÉRENCES RÉGLEMENTAIRES :{chr(10)}{rag_context}" if rag_context else ""}
{f"DONNÉES MARCHÉ :{chr(10)}{web_results}" if web_results else ""}

Pour CHAQUE proposition, fournis un objet JSON avec exactement ces champs :
- "categorie": code VPH (ex: "FRMA")
- "modele": nom commercial (ex: "Küschall K-series")  
- "justification_clinique": pourquoi ce modèle pour CE patient (3-4 phrases)
- "caracteristiques_cles": liste de 4-5 points techniques pertinents
- "code_lpp": code LPP si connu, sinon ""
- "remboursement": modalité de prise en charge (achat/LCD/LLD)
- "avantages": liste de 2-3 avantages pour ce profil
- "points_vigilance": liste de 1-2 points à vérifier lors de l'essai

Réponds UNIQUEMENT avec un tableau JSON valide : [{{...}}, {{...}}, ...]"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    import json, re
    text = response.content[0].text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback structuré
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


def determine_vph_category(patient: PatientState, api_key: str, vectorstore=None) -> str:
    """
    Détermine la catégorie VPH recommandée selon le profil.
    """
    client = Anthropic(api_key=api_key)

    rag_context = ""
    if vectorstore:
        from rag.retriever import search, format_context
        docs = search(
            f"indications catégorie VPH {patient.diagnostic} capacités propulsion",
            k=5, vectorstore=vectorstore, category_filter="reglementation"
        )
        rag_context = format_context(docs)

    profil = patient.to_context_summary()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""Selon ce profil, quelle est la catégorie VPH la plus adaptée ?

{profil}

{f"Références :{chr(10)}{rag_context}" if rag_context else ""}

Réponds UNIQUEMENT avec le code catégorie (ex: FRMA, FRE, FREP-B, etc.) suivi d'une phrase de justification courte.
Format : CODE | Justification"""
        }]
    )

    text = response.content[0].text.strip()
    if "|" in text:
        return text.split("|")[0].strip()
    return text.split()[0] if text else "FRMC"
