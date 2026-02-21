"""
UtopIA — Application principale
Point d'entrée Streamlit — Page d'accueil + initialisation RAG
"""

import streamlit as st
import os
from graph.state import PatientState

st.set_page_config(
    page_title="UtopIA — Préconisation VPH",
    page_icon="🦽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display:ital@0;1&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a3455 0%, #0f4c75 60%, #1b6ca8 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: rgba(255,255,255,0.8) !important; }
[data-testid="stSidebarNavLink"] { color: white !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0a3455 0%, #0f4c75 50%, #1b6ca8 100%);
    border-radius: 16px; padding: 3rem 3.5rem;
    color: white; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: "🦽";
    position: absolute; right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 8rem; opacity: 0.1;
}
.hero h1 { font-size: 2.8rem; margin: 0 0 0.5rem 0; color: white; }
.hero p { font-size: 1.1rem; opacity: 0.85; margin: 0; max-width: 600px; }
.hero .version { 
    display: inline-block; background: rgba(255,255,255,0.15);
    padding: 3px 12px; border-radius: 20px; font-size: 0.8rem;
    margin-bottom: 1rem;
}

/* Étapes */
.step-card {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.5rem;
    text-align: center; height: 100%;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.step-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(15,76,117,0.12);
}
.step-number {
    display: inline-block;
    background: linear-gradient(135deg, #0f4c75, #1b6ca8);
    color: white; width: 36px; height: 36px;
    border-radius: 50%; line-height: 36px;
    font-weight: 700; font-size: 1rem;
    margin-bottom: 0.8rem;
}
.step-card h4 { margin: 0.4rem 0; color: #0f4c75; font-family: 'DM Serif Display'; }
.step-card p { font-size: 0.85rem; color: #64748b; margin: 0; }

/* Status */
.status-pill {
    display: inline-block; padding: 4px 14px;
    border-radius: 20px; font-size: 0.82rem; font-weight: 600;
}
.status-recueil { background: #e8f4fd; color: #0f4c75; }
.status-preconisation { background: #fff3e0; color: #e65c00; }
.status-essais { background: #fce4ec; color: #c62828; }
.status-dossier { background: #e6f9f0; color: #1a7a4a; }

/* Boutons */
.stButton > button {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white; border: none; border-radius: 8px;
    padding: 0.6rem 2rem; font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,76,117,0.3);
}

/* RAG status */
.rag-ready { 
    background: #e6f9f0; border: 1px solid #1a7a4a;
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.85rem; color: #1a7a4a; font-weight: 500;
}
.rag-loading { 
    background: #fff8e1; border: 1px solid #ffc107;
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.85rem; color: #856404;
}
.rag-error {
    background: #fdecea; border: 1px solid #e53935;
    border-radius: 8px; padding: 8px 14px;
    font-size: 0.85rem; color: #c62828;
}
</style>
""", unsafe_allow_html=True)


# ── Initialisation session ────────────────────────────────────────────────────
if "patient" not in st.session_state:
    st.session_state.patient = PatientState()
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_status" not in st.session_state:
    st.session_state.rag_status = "idle"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦽 UtopIA")
    st.markdown("*Agent IA — Préconisation VPH*")
    st.divider()

    # Clé API
    st.markdown("**🔑 Configuration API**")

    # Priorité : variable d'environnement (Streamlit Cloud secrets) puis saisie manuelle
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        st.session_state.api_key = env_key
        st.markdown('<div class="rag-ready">✅ Clé API chargée depuis les secrets</div>',
                    unsafe_allow_html=True)
    else:
        api_key_input = st.text_input(
            "Clé Anthropic (sk-ant-...)",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="sk-ant-api...",
        )
        if api_key_input:
            st.session_state.api_key = api_key_input

    st.divider()

    # RAG Status
    st.markdown("**📚 Base de connaissances RAG**")
    rag_status = st.session_state.rag_status

    if rag_status == "ready":
        st.markdown('<div class="rag-ready">✅ Base vectorielle chargée</div>', unsafe_allow_html=True)
    elif rag_status == "loading":
        st.markdown('<div class="rag-loading">⏳ Chargement en cours...</div>', unsafe_allow_html=True)
    elif rag_status == "error":
        st.markdown('<div class="rag-error">❌ Erreur de chargement</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="rag-loading">⚡ Non initialisée</div>', unsafe_allow_html=True)

    if rag_status != "ready" and st.button("🚀 Initialiser le RAG", use_container_width=True):
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            st.error("Renseignez la clé API d'abord.")
        else:
            with st.spinner("Vectorisation des documents..."):
                try:
                    st.session_state.rag_status = "loading"
                    from rag.ingest import build_vectorstore
                    vs = build_vectorstore(api_key)
                    st.session_state.vectorstore = vs
                    st.session_state.rag_status = "ready"
                    st.rerun()
                except Exception as e:
                    st.session_state.rag_status = "error"
                    st.error(f"Erreur : {e}")

    st.divider()

    # Patient en cours
    patient = st.session_state.patient
    if patient.nom or patient.diagnostic:
        st.markdown("**👤 Patient en cours**")
        if patient.nom:
            st.markdown(f"**{patient.prenom} {patient.nom}**")
        if patient.diagnostic:
            st.caption(patient.diagnostic[:60] + ("..." if len(patient.diagnostic) > 60 else ""))

        statut_labels = {
            "recueil": ("Évaluation en cours", "status-recueil"),
            "preconisation": ("Préconisation", "status-preconisation"),
            "essais": ("Essais", "status-essais"),
            "dossier": ("Dossier complet", "status-dossier"),
        }
        label, css = statut_labels.get(patient.statut, ("En cours", "status-recueil"))
        st.markdown(f'<span class="status-pill {css}">{label}</span>', unsafe_allow_html=True)

        st.divider()
        if st.button("🗑️ Réinitialiser le patient", use_container_width=True):
            st.session_state.patient = PatientState()
            for k in ["obs_par_vph", "axes_evaluation"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

    st.divider()
    st.caption("Conforme à la réforme VPH\ndécembre 2025")
    st.caption("OTIPM · MCREO · PEO · WSP-F")


# ── Page principale ───────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="version">Réforme VPH — Décembre 2025</div>
    <h1>UtopIA</h1>
    <p>Agent IA d'aide à la préconisation de Véhicules pour Personnes Handicapées.<br>
    Du recueil des besoins à l'argumentaire CPAM, guidé par l'OTIPM.</p>
</div>
""", unsafe_allow_html=True)

# ── Guide de démarrage ────────────────────────────────────────────────────────
patient = st.session_state.patient
if not st.session_state.get("api_key") or st.session_state.rag_status != "ready":
    st.markdown("## 🚀 Démarrage")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Étape 1 — Clé API**

        Renseignez votre clé Anthropic dans la barre latérale gauche.
        Obtenez une clé sur [console.anthropic.com](https://console.anthropic.com).

        **Étape 2 — Base de connaissances**

        Cliquez sur **Initialiser le RAG** dans la barre latérale.
        L'application vectorise les 6 documents réglementaires (~2-3 min la première fois).
        """)
    with col2:
        st.markdown("""
        **Ce que fait UtopIA :**
        - 📋 Guide le recueil structuré Personne / Environnement / Occupation
        - 🧭 Sélectionne automatiquement le modèle conceptuel adapté (MCREO, PEO, MOHO)
        - 📄 Rédige le diagnostic ergothérapique
        - 🦽 Recherche les AT selon le profil et la nomenclature VPH 2025
        - 📑 Génère l'argumentaire CPAM normé
        """)

    st.info("👈 Commencez par configurer la clé API et initialiser le RAG dans la barre latérale.")
else:
    # App prête — afficher le tableau de bord
    st.markdown("## 📍 Parcours de préconisation")

    cols = st.columns(4)
    steps = [
        ("1", "📋", "Évaluation", "Recueil PEO — Facteurs personnels, environnementaux, occupationnels"),
        ("2", "🔬", "Préconisation", "Modèle conceptuel → Diagnostic ergo → Recherche AT"),
        ("3", "🧪", "Essais", "Enregistrement des essais terrain — sélection du VPH retenu"),
        ("4", "📑", "Argumentaire", "Rédaction normée CPAM — Export du dossier complet"),
    ]

    statut_to_step = {"recueil": 1, "preconisation": 2, "essais": 3, "dossier": 4}
    current_step = statut_to_step.get(patient.statut, 1)

    for i, (num, icon, title, desc) in enumerate(steps):
        with cols[i]:
            is_current = (i + 1 == current_step)
            border_style = "border: 2px solid #1b6ca8;" if is_current else ""
            st.markdown(f"""
            <div class="step-card" style="{border_style}">
                <div class="step-number">{num}</div>
                <div style="font-size:2rem">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
                {"<br><small style='color:#1b6ca8;font-weight:600'>← Étape actuelle</small>" if is_current else ""}
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # Résumé patient si en cours
    if patient.diagnostic:
        st.markdown("## 👤 Patient en cours")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Patient", f"{patient.prenom} {patient.nom}" if patient.nom else "—")
        with col2:
            st.metric("Diagnostic", patient.diagnostic[:30] + "..." if len(patient.diagnostic) > 30 else patient.diagnostic)
        with col3:
            st.metric("VPH proposés", len(patient.propositions_at))
        with col4:
            st.metric("VPH retenu", patient.at_retenue[:20] + "..." if patient.at_retenue and len(patient.at_retenue) > 20 else (patient.at_retenue or "—"))

        st.markdown("👈 Utilisez le menu latéral pour naviguer entre les étapes.")
    else:
        st.markdown("### Nouveau patient")
        st.info("👈 Cliquez sur **📋 Evaluation** dans le menu latéral pour commencer l'évaluation d'un nouveau patient.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📚 Sources : Fiches CPAM officielles, Réforme VPH déc. 2025, Guide WSP-F 5.1, TD Positionnement IFPEK")
with col2:
    st.caption("⚖️ Outil d'aide à la décision — Ne remplace pas le jugement clinique de l'ergothérapeute")
with col3:
    st.caption("🔒 Données traitées localement — Aucun stockage tiers")
