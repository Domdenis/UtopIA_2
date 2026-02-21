"""
UtopIA — Node 2 : Rédaction du diagnostic ergothérapique
"""

from anthropic import Anthropic
from graph.state import PatientState

SYSTEM_PROMPT = """Tu es UtopIA, ergothérapeute expert spécialisé en préconisation VPH.
Tu rédiges des diagnostics ergothérapiques professionnels, structurés et argumentés.
Ton style est clinique, précis, orienté sur les besoins fonctionnels et occupationnels.
Tu utilises le vocabulaire professionnel de l'ergothérapie française."""


def write_diagnostic(patient: PatientState, api_key: str, vectorstore=None) -> str:
    client = Anthropic(api_key=api_key)

    rag_section = ""
    if vectorstore:
        try:
            from rag.retriever import search, format_context
            docs = search(
                "diagnostic evaluation besoins positionnement fauteuil roulant",
                k=4,
                vectorstore=vectorstore
            )
            if docs:
                rag_section = "Références cliniques :\n" + format_context(docs)
        except Exception:
            pass

    mesures = []
    if patient.largeur_bassin:
        mesures.append("Largeur bassin : " + str(patient.largeur_bassin) + " cm")
    if patient.longueur_cuisses:
        mesures.append("Longueur cuisses : " + str(patient.longueur_cuisses) + " cm")
    if patient.longueur_creux_poplite_pied:
        mesures.append("Creux poplité-pied : " + str(patient.longueur_creux_poplite_pied) + " cm")
    if patient.hauteur_omoplate:
        mesures.append("Hauteur omoplate : " + str(patient.hauteur_omoplate) + " cm")
    if patient.largeur_tronc:
        mesures.append("Largeur tronc : " + str(patient.largeur_tronc) + " cm")
    if patient.poids:
        mesures.append("Poids : " + str(patient.poids) + " kg")

    modele = patient.modele_conceptuel_choisi or "MCREO"

    lines = [
        "Rédige un diagnostic ergothérapique complet pour ce patient, en utilisant le modèle " + modele + ".",
        "",
        "PROFIL PATIENT :",
        patient.to_context_summary(),
    ]
    if mesures:
        lines.append("")
        lines.append("Mesures anthropométriques :")
        lines.extend(mesures)
    if patient.justification_modele:
        lines.append("")
        lines.append("Justification du modèle : " + patient.justification_modele)
    if rag_section:
        lines.append("")
        lines.append(rag_section)

    lines += [
        "",
        "Structure le diagnostic en 4 parties :",
        "## 1. Présentation de la situation",
        "## 2. Analyse des besoins selon le modèle " + modele,
        "## 3. Déficits et ressources",
        "## 4. Objectifs de la préconisation",
        "",
        "Sois précis, clinique. Maximum 500 mots.",
    ]

    user_prompt = "\n".join(lines)

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}]
    )

    return response.content[0].text
