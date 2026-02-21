"""
UtopIA — Page 4 : Argumentaire CPAM + Export
Génération de l'argumentaire normé et du dossier complet
"""
import streamlit as st
from graph.state import PatientState
from graph.nodes.argumentaire import write_argumentaire
from datetime import datetime

st.set_page_config(page_title="📑 Argumentaire — UtopIA", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.section-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white; padding: 12px 20px; border-radius: 8px;
    font-weight: 600; font-size: 1rem; margin: 1.5rem 0 1rem 0;
}
.stButton > button {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-weight: 600;
}
.dossier-preview {
    background: white; border: 1px solid #dde1e7;
    border-radius: 10px; padding: 2rem;
    font-family: 'Georgia', serif; font-size: 0.92rem;
    line-height: 1.7; color: #1a1a2e;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    max-height: 600px; overflow-y: auto;
}
.export-btn > button {
    background: linear-gradient(135deg, #1a7a4a 0%, #23a165 100%) !important;
}
.checklist-item {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; border-radius: 6px; margin: 4px 0;
}
.check-ok { background: #e6f9f0; }
.check-warn { background: #fff8e1; }
</style>
""", unsafe_allow_html=True)

# ── Vérification prérequis ──────────────────────────────────────────────────
if "patient" not in st.session_state or not st.session_state.patient.at_retenue:
    st.warning("⚠️ Veuillez d'abord compléter les essais et sélectionner le VPH retenu (Page 3).")
    st.stop()

patient: PatientState = st.session_state.patient
api_key = st.session_state.get("api_key", "")
vectorstore = st.session_state.get("vectorstore", None)

# ── En-tête ─────────────────────────────────────────────────────────────────
st.markdown(f"# 📑 Argumentaire CPAM — {patient.prenom} {patient.nom}")
st.caption(f"VPH retenu : **{patient.at_retenue}** · Catégorie : {patient.categorie_vph_recommandee}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECKLIST DOSSIER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">✅ Vérification du dossier</div>', unsafe_allow_html=True)

checks = [
    ("Diagnostic renseigné", bool(patient.diagnostic)),
    ("Évaluation des besoins complète", bool(patient.situation_sante)),
    ("Modèle conceptuel sélectionné", bool(patient.modele_conceptuel_choisi)),
    ("Diagnostic ergothérapique rédigé", bool(patient.diagnostic_ergo)),
    ("Au moins 2 VPH essayés", len(patient.at_essayees) >= 2),
    ("VPH retenu sélectionné", bool(patient.at_retenue)),
    ("Motifs de rejet renseignés", bool(patient.motifs_rejet)),
    ("Réglages définitifs notés", bool(patient.reglages_definitifs)),
]

cols = st.columns(2)
for i, (label, ok) in enumerate(checks):
    with cols[i % 2]:
        icon = "✅" if ok else "⚠️"
        css_class = "check-ok" if ok else "check-warn"
        st.markdown(
            f'<div class="checklist-item {css_class}">{icon} {label}</div>',
            unsafe_allow_html=True
        )

all_ok = all(ok for _, ok in checks)
if not all_ok:
    st.warning("Certains éléments du dossier sont incomplets. L'argumentaire sera généré mais pourra manquer d'informations.")

# ═══════════════════════════════════════════════════════════════════════════
# GÉNÉRATION ARGUMENTAIRE
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📝 Argumentaire de prise en charge</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🤖 Générer l'argumentaire", use_container_width=True):
        with st.spinner("Rédaction de l'argumentaire CPAM..."):
            patient.argumentaire_cpam = write_argumentaire(patient, api_key, vectorstore)
            st.session_state.patient = patient
            st.rerun()

    if patient.argumentaire_cpam:
        if st.button("🔄 Régénérer", use_container_width=True):
            with st.spinner("Nouvelle version..."):
                patient.argumentaire_cpam = write_argumentaire(patient, api_key, vectorstore)
                st.session_state.patient = patient
                st.rerun()

with col1:
    if patient.argumentaire_cpam:
        patient.argumentaire_cpam = st.text_area(
            "Argumentaire (modifiable avant export)",
            value=patient.argumentaire_cpam,
            height=350
        )
        st.session_state.patient = patient
    else:
        st.info("Cliquez sur « Générer l'argumentaire » pour créer automatiquement l'argumentaire CPAM.")

# ═══════════════════════════════════════════════════════════════════════════
# DOSSIER COMPLET
# ═══════════════════════════════════════════════════════════════════════════
if patient.argumentaire_cpam:
    st.markdown('<div class="section-header">📦 Dossier complet</div>', unsafe_allow_html=True)

    # Générer le dossier complet
    def build_dossier(p: PatientState) -> str:
        now = datetime.now().strftime("%d/%m/%Y")
        mesures_lines = []
        if p.largeur_bassin: mesures_lines.append(f"- Largeur bassin : {p.largeur_bassin} cm")
        if p.longueur_cuisses: mesures_lines.append(f"- Longueur cuisses : {p.longueur_cuisses} cm")
        if p.longueur_creux_poplite_pied: mesures_lines.append(f"- Creux poplité-pied : {p.longueur_creux_poplite_pied} cm")
        if p.hauteur_omoplate: mesures_lines.append(f"- Hauteur omoplate : {p.hauteur_omoplate} cm")
        if p.largeur_tronc: mesures_lines.append(f"- Largeur tronc : {p.largeur_tronc} cm")
        if p.poids: mesures_lines.append(f"- Poids : {p.poids} kg")

        at_essayees_str = "\n".join([f"  - {at}" for at in p.at_essayees]) if p.at_essayees else "  Non renseigné"

        dossier = f"""
════════════════════════════════════════════════════════════════
        DOSSIER DE PRÉCONISATION VPH — UTOPIA
        Généré le {now}
════════════════════════════════════════════════════════════════

BÉNÉFICIAIRE
────────────
Nom : {p.nom}
Prénom : {p.prenom}
Date de naissance : {p.date_naissance}
Adresse : {p.adresse}
Âge : {p.age} ans | Sexe : {p.sexe}

SITUATION CLINIQUE
──────────────────
Diagnostic principal : {p.diagnostic}
Situation de santé : {p.situation_sante}
Caractère évolutif : {p.caractere_evolutif}
Capacités physiques : {p.capacites_physiques}
Comorbidités : {p.comorbidites}
{f"Poids : {p.poids} kg | Taille : {p.taille} cm" if p.poids or p.taille else ""}

CONTEXTE DE VIE
───────────────
Lieu de vie : {p.lieu_vie}
Description : {p.description_logement}
Activités : {', '.join(p.activites)}
Déplacements : {', '.join(p.deplacements)}
Activité professionnelle : {'Oui' if p.activite_professionnelle else 'Non'}
Conduite véhicule : {'Oui' if p.conduite_vehicule else 'Non'}

{"MESURES ANTHROPOMÉTRIQUES" + chr(10) + "─" * 24 + chr(10) + chr(10).join(mesures_lines) + chr(10) if mesures_lines else ""}
MODÈLE CONCEPTUEL
─────────────────
Modèle choisi : {p.modele_conceptuel_choisi}
Justification : {p.justification_modele}

DIAGNOSTIC ERGOTHÉRAPIQUE
──────────────────────────
{p.diagnostic_ergo}

ESSAIS RÉALISÉS
───────────────
VPH essayés :
{at_essayees_str}

Observations générales :
{p.observations_essais}

Motifs de rejet :
{p.motifs_rejet}

VPH RETENU
──────────
{p.at_retenue}
Catégorie : {p.categorie_vph_recommandee}
Mode de prise en charge : {p.mode_prise_en_charge}

Réglages et adjonctions définitifs :
{p.reglages_definitifs}

ARGUMENTAIRE DE PRISE EN CHARGE
────────────────────────────────
{p.argumentaire_cpam}

════════════════════════════════════════════════════════════════
Document généré par UtopIA — Agent d'aide à la préconisation VPH
À compléter et signer par l'ergothérapeute prescripteur
════════════════════════════════════════════════════════════════
""".strip()
        return dossier

    patient.dossier_complet = build_dossier(patient)
    st.session_state.patient = patient

    with st.expander("👁️ Prévisualiser le dossier complet", expanded=False):
        st.markdown(
            f'<div class="dossier-preview"><pre style="font-family:inherit;white-space:pre-wrap;">'
            f'{patient.dossier_complet}</pre></div>',
            unsafe_allow_html=True
        )

    # ── Export ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">💾 Export du dossier</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Export TXT
    with col1:
        filename = f"utopia_{patient.nom}_{patient.prenom}_{datetime.now().strftime('%Y%m%d')}.txt"
        st.download_button(
            label="📄 Télécharger le dossier (.txt)",
            data=patient.dossier_complet.encode("utf-8"),
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

    # Export argumentaire seul
    with col2:
        arg_filename = f"argumentaire_cpam_{patient.nom}_{datetime.now().strftime('%Y%m%d')}.txt"
        st.download_button(
            label="📋 Argumentaire seul (.txt)",
            data=patient.argumentaire_cpam.encode("utf-8"),
            file_name=arg_filename,
            mime="text/plain",
            use_container_width=True
        )

    # Export markdown
    with col3:
        md_content = f"# Dossier VPH — {patient.prenom} {patient.nom}\n\n" + patient.dossier_complet
        md_filename = f"utopia_{patient.nom}_{datetime.now().strftime('%Y%m%d')}.md"
        st.download_button(
            label="📝 Exporter en Markdown",
            data=md_content.encode("utf-8"),
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True
        )

    st.success("✅ Dossier prêt à être imprimé, signé par l'ergothérapeute et transmis au distributeur et à la CPAM.")

    # ── Récapitulatif parcours ────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Récapitulatif du parcours</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modèle conceptuel", patient.modele_conceptuel_choisi or "—")
    with col2:
        st.metric("Catégorie VPH", patient.categorie_vph_recommandee or "—")
    with col3:
        st.metric("VPH essayés", len(patient.at_essayees))
    with col4:
        st.metric("Prise en charge", patient.mode_prise_en_charge.split(" ")[0] if patient.mode_prise_en_charge else "—")

    # Nouveau patient
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🆕 Nouveau patient", use_container_width=True):
            st.session_state.patient = PatientState()
            if "obs_par_vph" in st.session_state:
                del st.session_state["obs_par_vph"]
            if "axes_evaluation" in st.session_state:
                del st.session_state["axes_evaluation"]
            st.success("✅ Nouveau patient initialisé. Retournez à la page Évaluation.")
