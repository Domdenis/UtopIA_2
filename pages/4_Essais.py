"""
UtopIA — Page 3 : Suivi des essais VPH
Enregistrement des essais terrain, sélection du VPH retenu
"""
import streamlit as st
from graph.state import PatientState

st.set_page_config(page_title="🧪 Essais — UtopIA", layout="wide")

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
.essai-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 1.2rem; margin-bottom: 0.8rem;
}
.retenu-card {
    background: linear-gradient(135deg, #e6f9f0 0%, #d4f5e5 100%);
    border: 2px solid #1a7a4a; border-radius: 12px; padding: 1.5rem;
}
.regle-box {
    background: #fff8e1; border-left: 4px solid #ffc107;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 1rem 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ── Vérification prérequis ──────────────────────────────────────────────────
if "patient" not in st.session_state or not st.session_state.patient.propositions_at:
    st.warning("⚠️ Veuillez d'abord générer des propositions d'AT (Page 2 — Préconisation).")
    st.stop()

patient: PatientState = st.session_state.patient

# ── En-tête ─────────────────────────────────────────────────────────────────
st.markdown(f"# 🧪 Essais VPH — {patient.prenom} {patient.nom}")
st.caption(f"Catégorie envisagée : {patient.categorie_vph_recommandee} · {patient.mode_prise_en_charge}")

# Rappel réglementaire
st.markdown("""
<div class="regle-box">
    📋 <strong>Réforme décembre 2025 :</strong> L'essai d'au moins <strong>2 modèles</strong>
    avec un ergothérapeute est obligatoire avant toute prescription définitive.
    Le patient n'est pas propriétaire du VPH pendant la période d'essai.
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# RÉCAPITULATIF DES PROPOSITIONS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Propositions à essayer</div>', unsafe_allow_html=True)

propositions_labels = []
for prop in patient.propositions_at:
    if isinstance(prop, dict):
        label = f"{prop.get('categorie', '')} — {prop.get('modele', '')}"
        propositions_labels.append(label)

if propositions_labels:
    patient.at_essayees = st.multiselect(
        "VPH effectivement essayés (cochez ceux qui ont été testés lors des essais terrain)",
        options=propositions_labels,
        default=[a for a in patient.at_essayees if a in propositions_labels]
    )

# Ajouter un modèle non préconisé
with st.expander("➕ Ajouter un VPH non listé (essayé à la demande du patient ou du distributeur)"):
    col1, col2 = st.columns([3, 1])
    with col1:
        nouveau_vph = st.text_input("Modèle testé (ex: Küschall Champion — FRMA)")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Ajouter") and nouveau_vph:
            if nouveau_vph not in patient.at_essayees:
                patient.at_essayees.append(nouveau_vph)
            st.rerun()

st.session_state.patient = patient

# ═══════════════════════════════════════════════════════════════════════════
# OBSERVATIONS PAR VPH ESSAYÉ
# ═══════════════════════════════════════════════════════════════════════════
if patient.at_essayees:
    st.markdown('<div class="section-header">📝 Observations par VPH essayé</div>', unsafe_allow_html=True)

    # Stocker les observations par modèle
    if "obs_par_vph" not in st.session_state:
        st.session_state.obs_par_vph = {}

    for vph in patient.at_essayees:
        with st.expander(f"🦽 {vph}", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.obs_par_vph[vph + "_pos"] = st.text_area(
                    "Points positifs",
                    value=st.session_state.obs_par_vph.get(vph + "_pos", ""),
                    key=f"pos_{vph[:20]}",
                    placeholder="Confort, stabilité, maniabilité, propulsion..."
                )
                st.session_state.obs_par_vph[vph + "_post"] = st.text_area(
                    "Positionnement observé",
                    value=st.session_state.obs_par_vph.get(vph + "_post", ""),
                    key=f"post_{vph[:20]}",
                    placeholder="Alignement bassin, position du tronc, membres..."
                )
            with col2:
                st.session_state.obs_par_vph[vph + "_neg"] = st.text_area(
                    "Points négatifs / inadaptations",
                    value=st.session_state.obs_par_vph.get(vph + "_neg", ""),
                    key=f"neg_{vph[:20]}",
                    placeholder="Difficultés de propulsion, inconfort, incompatibilité..."
                )
                st.session_state.obs_par_vph[vph + "_avq"] = st.text_area(
                    "Réalisation des AVQ avec ce VPH",
                    value=st.session_state.obs_par_vph.get(vph + "_avq", ""),
                    key=f"avq_{vph[:20]}",
                    placeholder="Transferts, déplacements, activités testées..."
                )

    # Synthèse observations
    st.markdown('<div class="section-header">✍️ Synthèse globale des essais</div>', unsafe_allow_html=True)
    patient.observations_essais = st.text_area(
        "Observations générales (résultats WSP si réalisé, points de vigilance, remarques du patient...)",
        value=patient.observations_essais, height=100
    )
    patient.motifs_rejet = st.text_area(
        "Motifs de rejet des VPH non retenus",
        value=patient.motifs_rejet, height=80,
        placeholder="Expliquer pourquoi chaque VPH non retenu a été écarté (obligatoire pour l'argumentaire CPAM)"
    )

# ═══════════════════════════════════════════════════════════════════════════
# VPH RETENU
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">✅ VPH retenu et réglages définitifs</div>', unsafe_allow_html=True)

options_retenu = [""] + (patient.at_essayees if patient.at_essayees else propositions_labels)
idx_ret = options_retenu.index(patient.at_retenue) if patient.at_retenue in options_retenu else 0
patient.at_retenue = st.selectbox("VPH retenu pour prescription", options_retenu, index=idx_ret)

if patient.at_retenue:
    st.markdown(f"""
    <div class="retenu-card">
        <strong>✅ VPH retenu : {patient.at_retenue}</strong><br>
        <small>Ce modèle sera intégré dans la fiche de préconisation et l'argumentaire CPAM.</small>
    </div>
    """, unsafe_allow_html=True)

    patient.reglages_definitifs = st.text_area(
        "Réglages et adjonctions définitifs",
        value=patient.reglages_definitifs, height=150,
        placeholder="""Largeur d'assise : ... cm
Profondeur d'assise : ... cm
Hauteur dossier : ... cm
Angle dossier : ...°
Coussin : modèle, hauteur
Adjonctions : liste des accessoires retenus avec codes LPP
Appuis-tête, repose-pieds, ceinture pelvienne...
Tout autre réglage spécifique"""
    )

st.session_state.patient = patient

# ── Navigation ───────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    ready = bool(patient.at_retenue and len(patient.at_essayees) >= 2)
    if not ready:
        n = len(patient.at_essayees)
        st.caption(f"⚠️ {'Sélectionnez le VPH retenu' if n >= 2 else f'Essayez au moins 2 modèles ({n}/2 renseigné)'}")
    if st.button("📝 Générer l'argumentaire →", use_container_width=True, disabled=not ready):
        patient.statut = "dossier"
        st.session_state.patient = patient
        st.success("✅ Essais enregistrés. Rendez-vous sur la page Argumentaire.")
