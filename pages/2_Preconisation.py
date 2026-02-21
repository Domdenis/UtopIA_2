"""
UtopIA — Page 2 : Préconisation AT
Sélection modèle conceptuel → Diagnostic ergo → Recherche AT
"""
import streamlit as st
from graph.state import PatientState
from graph.nodes.model_selector import select_model_conceptuel
from graph.nodes.diagnostic_writer import write_diagnostic
from graph.nodes.at_researcher import search_at, determine_vph_category

st.set_page_config(page_title="🔬 Préconisation — UtopIA", layout="wide")

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
.at-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
.at-card:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.1); }
.badge {
    display: inline-block; background: #e8f4fd; color: #0f4c75;
    padding: 3px 10px; border-radius: 20px; font-size: 0.8rem;
    font-weight: 600; margin-right: 6px;
}
.badge-green { background: #e6f9f0; color: #1a7a4a; }
.badge-orange { background: #fff3e0; color: #e65c00; }
.stButton > button {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-weight: 600;
}
.modele-box {
    background: linear-gradient(135deg, #f0f7ff 0%, #e8f4fd 100%);
    border: 1px solid #b3d5f5; border-radius: 10px;
    padding: 1rem 1.5rem; margin: 0.5rem 0;
}
.warning-box {
    background: #fff8e1; border-left: 4px solid #ffc107;
    padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Vérification prérequis ──────────────────────────────────────────────────
if "patient" not in st.session_state or not st.session_state.patient.diagnostic:
    st.warning("⚠️ Veuillez d'abord compléter l'évaluation du patient (Page 1).")
    st.stop()

patient: PatientState = st.session_state.patient
api_key = st.session_state.get("api_key", "")

if not api_key:
    st.error("🔑 Clé API Anthropic manquante. Configurez-la sur la page d'accueil.")
    st.stop()

vectorstore = st.session_state.get("vectorstore", None)

# ── En-tête ─────────────────────────────────────────────────────────────────
st.markdown(f"# 🔬 Préconisation — {patient.prenom} {patient.nom}")
st.caption(f"Diagnostic : {patient.diagnostic} · {patient.age} ans · {patient.lieu_vie}")

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : MODÈLE CONCEPTUEL
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧭 Étape 1 — Sélection du modèle conceptuel</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    if patient.modele_conceptuel_choisi:
        st.markdown(f"""
        <div class="modele-box">
            <strong>Modèle sélectionné :</strong>
            <span class="badge">{patient.modele_conceptuel_choisi}</span><br>
            <small>{patient.justification_modele}</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Cliquez sur le bouton pour analyser le profil et sélectionner le modèle conceptuel adapté.")

with col2:
    if st.button("🤖 Analyser le profil", use_container_width=True):
        with st.spinner("Analyse du profil patient..."):
            result = select_model_conceptuel(patient, api_key, vectorstore)
            patient.modele_conceptuel_choisi = result.get("modele", "MCREO")
            patient.justification_modele = result.get("justification", "")
            axes = result.get("axes_evaluation", [])
            st.session_state["axes_evaluation"] = axes
            st.session_state.patient = patient
            st.rerun()

# Override manuel
if patient.modele_conceptuel_choisi:
    with st.expander("✏️ Modifier le modèle conceptuel"):
        modeles = ["MCREO", "PEO", "MOHO", "OTIPM", "Autre"]
        idx = modeles.index(patient.modele_conceptuel_choisi) if patient.modele_conceptuel_choisi in modeles else 0
        patient.modele_conceptuel_choisi = st.selectbox("Modèle", modeles, index=idx)
        patient.justification_modele = st.text_area("Justification", value=patient.justification_modele)
        st.session_state.patient = patient

    # Axes d'évaluation
    axes = st.session_state.get("axes_evaluation", [])
    if axes:
        st.markdown("**Axes d'évaluation prioritaires :**")
        cols = st.columns(len(axes))
        for i, axe in enumerate(axes):
            with cols[i]:
                st.markdown(f'<span class="badge">{axe}</span>', unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : DIAGNOSTIC ERGO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📄 Étape 2 — Diagnostic ergothérapique</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("✍️ Rédiger le diagnostic", use_container_width=True,
                 disabled=not patient.modele_conceptuel_choisi):
        with st.spinner("Rédaction du diagnostic ergothérapique..."):
            patient.diagnostic_ergo = write_diagnostic(patient, api_key, vectorstore)
            st.session_state.patient = patient
            st.rerun()

with col1:
    if patient.diagnostic_ergo:
        patient.diagnostic_ergo = st.text_area(
            "Diagnostic ergothérapique (modifiable)",
            value=patient.diagnostic_ergo,
            height=280
        )
        st.session_state.patient = patient
    else:
        if not patient.modele_conceptuel_choisi:
            st.markdown('<div class="warning-box">⚠️ Sélectionnez d\'abord un modèle conceptuel.</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Cliquez sur « Rédiger le diagnostic » pour générer le diagnostic ergothérapique.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : CATÉGORIE VPH + RECHERCHE AT
# ═══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🦽 Étape 3 — Recherche des aides techniques</div>',
            unsafe_allow_html=True)

# Catégorie VPH
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    categories = [
        "", "FMP", "FMPR", "FRM", "FRMC", "FRMA", "FRMS", "FRMP", "FRMV",
        "FRE-A", "FRE-B", "FRE-C", "FREP-A", "FREP-B", "FREP-C", "FREV",
        "POU_S", "POU_MRE", "BASE", "CYC", "SCO-A", "SCO-B", "SCO-C"
    ]
    idx = categories.index(patient.categorie_vph_recommandee) if patient.categorie_vph_recommandee in categories else 0
    patient.categorie_vph_recommandee = st.selectbox(
        "Catégorie VPH envisagée",
        categories, index=idx,
        help="Laissez vide pour que l'agent détermine automatiquement"
    )
with col2:
    mode_options = ["", "Achat (besoin > 6 mois)", "LCD (besoin ≤ 6 mois)", "LLD (besoin évolutif)"]
    idx_mode = mode_options.index(patient.mode_prise_en_charge) if patient.mode_prise_en_charge in mode_options else 0
    patient.mode_prise_en_charge = st.selectbox("Mode de prise en charge", mode_options, index=idx_mode)

with col3:
    st.markdown("<br>", unsafe_allow_html=True)
    auto_cat = st.button("🎯 Détecter la catégorie", use_container_width=True,
                         disabled=not patient.diagnostic_ergo)

if auto_cat:
    with st.spinner("Détection de la catégorie VPH..."):
        cat = determine_vph_category(patient, api_key, vectorstore)
        patient.categorie_vph_recommandee = cat
        st.session_state.patient = patient
        st.rerun()

st.session_state.patient = patient

# Recherche AT
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔍 Rechercher les AT", use_container_width=True,
                 disabled=not patient.diagnostic_ergo):
        with st.spinner("Recherche des aides techniques adaptées..."):
            propositions = search_at(patient, api_key, vectorstore)
            patient.propositions_at = propositions
            st.session_state.patient = patient
            st.rerun()

with col1:
    if not patient.diagnostic_ergo:
        st.markdown('<div class="warning-box">⚠️ Rédigez d\'abord le diagnostic ergothérapique.</div>',
                    unsafe_allow_html=True)

# Affichage des propositions AT
if patient.propositions_at:
    st.markdown(f"**{len(patient.propositions_at)} proposition(s) générée(s) :**")

    for i, prop in enumerate(patient.propositions_at):
        if not isinstance(prop, dict):
            continue
        with st.container():
            st.markdown(f"""
            <div class="at-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <span class="badge">{prop.get('categorie', 'N/A')}</span>
                        <strong style="font-size:1.1rem;">{prop.get('modele', 'N/A')}</strong>
                    </div>
                    <span class="badge badge-green">{prop.get('remboursement', '')}</span>
                </div>
                <p style="margin: 0.8rem 0; color: #444; font-size:0.9rem;">
                    {prop.get('justification_clinique', '')}
                </p>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**Caractéristiques clés**")
                for carac in prop.get('caracteristiques_cles', []):
                    st.markdown(f"• {carac}")
            with col_b:
                st.markdown("**Avantages pour ce profil**")
                for av in prop.get('avantages', []):
                    st.markdown(f"✓ {av}")
            with col_c:
                st.markdown("**Points de vigilance à l'essai**")
                for pv in prop.get('points_vigilance', []):
                    st.markdown(f"⚠️ {pv}")
                if prop.get('code_lpp'):
                    st.caption(f"Code LPP : `{prop['code_lpp']}`")

    st.divider()
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("➡️ Passer aux essais", use_container_width=True):
            patient.statut = "essais"
            st.session_state.patient = patient
            st.success("✅ Préconisations enregistrées. Rendez-vous sur la page Essais.")
