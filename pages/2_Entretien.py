"""
UtopIA — Page 1b : Entretien clinique complémentaire
Questions ciblées générées par UtopIA avant la préconisation
"""
import streamlit as st
from graph.state import PatientState
from graph.nodes.chat_clinique import (
    generate_first_question,
    generate_next_question,
    build_chat_synthesis,
)

st.set_page_config(page_title="Entretien — UtopIA", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.chat-container {
    max-width: 780px;
    margin: 0 auto;
}
.bubble-utopia {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
    color: white;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.8rem 0 0.8rem 0;
    max-width: 85%;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(15,76,117,0.2);
}
.bubble-user {
    background: #f0f7ff;
    border: 1px solid #b3d5f5;
    color: #1a202c;
    padding: 1rem 1.4rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.8rem 0 0.8rem auto;
    max-width: 85%;
    line-height: 1.6;
}
.bubble-user-wrap {
    display: flex;
    justify-content: flex-end;
}
.avatar-utopia {
    font-size: 1.4rem;
    margin-right: 0.5rem;
    vertical-align: middle;
}
.synthese-box {
    background: linear-gradient(135deg, #e6f9f0 0%, #d4f5e5 100%);
    border: 2px solid #1a7a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}
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
.progress-bar {
    background: #e2e8f0; border-radius: 10px; height: 8px; margin: 0.5rem 0;
}
.progress-fill {
    background: linear-gradient(90deg, #0f4c75, #1b6ca8);
    height: 8px; border-radius: 10px; transition: width 0.3s;
}
</style>
""", unsafe_allow_html=True)

# ── Vérification prérequis ─────────────────────────────────────────────────
if "patient" not in st.session_state or not st.session_state.patient.diagnostic:
    st.warning("⚠️ Veuillez d'abord compléter l'évaluation du patient (Page 1).")
    st.stop()

patient: PatientState = st.session_state.patient
api_key = st.session_state.get("api_key", "")
vectorstore = st.session_state.get("vectorstore", None)

if not api_key:
    st.error("🔑 Clé API manquante. Configurez-la sur la page d'accueil.")
    st.stop()

# ── Init session chat ──────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_termine" not in st.session_state:
    st.session_state.chat_termine = False
if "chat_synthese" not in st.session_state:
    st.session_state.chat_synthese = ""

# ── En-tête ────────────────────────────────────────────────────────────────
st.markdown(f"# 💬 Entretien clinique — {patient.prenom} {patient.nom}")
st.caption("UtopIA complète l'évaluation par des questions ciblées avant de générer les préconisations.")

# Progression
nb_questions = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
progress = min(nb_questions / 5, 1.0)
st.markdown(f"""
<div style="display:flex; align-items:center; gap:12px; margin:0.5rem 0 1.5rem 0;">
    <span style="font-size:0.85rem; color:#64748b;">Questions : {nb_questions}/5</span>
    <div class="progress-bar" style="flex:1;">
        <div class="progress-fill" style="width:{int(progress*100)}%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Démarrer l'entretien ───────────────────────────────────────────────────
if not st.session_state.chat_history:
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("🚀 Démarrer l'entretien", use_container_width=True):
            with st.spinner("UtopIA prépare sa première question..."):
                try:
                    question = generate_first_question(patient, api_key, vectorstore)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": question
                    })
                    st.rerun()
                except Exception as e:
                    st.error("Erreur : " + str(e))
    st.stop()

# ── Affichage du chat ──────────────────────────────────────────────────────
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    if msg["role"] == "assistant":
        st.markdown(f"""
        <div style="display:flex; align-items:flex-start; gap:8px; margin:0.8rem 0;">
            <span class="avatar-utopia">🤖</span>
            <div class="bubble-utopia">{msg["content"].replace(chr(10), "<br>")}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bubble-user-wrap">
            <div class="bubble-user">{msg["content"].replace(chr(10), "<br>")}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── Synthèse finale ────────────────────────────────────────────────────────
if st.session_state.chat_termine:
    if st.session_state.chat_synthese:
        st.markdown(f"""
        <div class="synthese-box">
            <strong>✅ Entretien terminé — Synthèse clinique</strong><br><br>
            {st.session_state.chat_synthese.replace(chr(10), "<br>")}
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("➡️ Aller à la Préconisation", use_container_width=True):
            patient.statut = "preconisation"
            st.session_state.patient = patient
            st.success("✅ Rendez-vous sur la page Préconisation dans le menu.")

    with st.expander("🔄 Recommencer l'entretien"):
        if st.button("Effacer et recommencer"):
            st.session_state.chat_history = []
            st.session_state.chat_termine = False
            st.session_state.chat_synthese = ""
            st.rerun()

# ── Zone de réponse ────────────────────────────────────────────────────────
elif st.session_state.chat_history:
    last_msg = st.session_state.chat_history[-1]
    if last_msg["role"] == "assistant":

        st.divider()
        reponse = st.text_area(
            "✍️ Votre réponse",
            placeholder="Répondez à la question d'UtopIA...",
            height=120,
            key="reponse_input"
        )

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Envoyer →", use_container_width=True, disabled=not reponse.strip()):
                # Ajouter la réponse
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": reponse.strip()
                })

                # Générer la prochaine question ou terminer
                with st.spinner("UtopIA analyse votre réponse..."):
                    try:
                        result = generate_next_question(
                            patient, api_key,
                            st.session_state.chat_history,
                            vectorstore
                        )

                        if result.get("termine", False):
                            # Générer la synthèse
                            synthese = build_chat_synthesis(
                                patient, api_key,
                                st.session_state.chat_history
                            )
                            # Sauvegarder dans le patient
                            patient.synthese_demande = (
                                (patient.synthese_demande + "\n\n" if patient.synthese_demande else "") +
                                "Synthèse entretien clinique :\n" + synthese
                            )
                            st.session_state.patient = patient
                            st.session_state.chat_termine = True
                            st.session_state.chat_synthese = result.get("synthese", synthese)
                        else:
                            question = result.get("question", "")
                            if question:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": question
                                })
                        st.rerun()
                    except Exception as e:
                        st.error("Erreur : " + str(e))

        # Option passer une question
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("Passer ⏭️", use_container_width=True):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "*(information non disponible)*"
                })
                with st.spinner("Question suivante..."):
                    try:
                        result = generate_next_question(
                            patient, api_key,
                            st.session_state.chat_history,
                            vectorstore
                        )
                        if result.get("termine", False):
                            synthese = build_chat_synthesis(
                                patient, api_key,
                                st.session_state.chat_history
                            )
                            patient.synthese_demande = (
                                (patient.synthese_demande + "\n\n" if patient.synthese_demande else "") +
                                "Synthèse entretien clinique :\n" + synthese
                            )
                            st.session_state.patient = patient
                            st.session_state.chat_termine = True
                            st.session_state.chat_synthese = result.get("synthese", synthese)
                        else:
                            question = result.get("question", "")
                            if question:
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": question
                                })
                        st.rerun()
                    except Exception as e:
                        st.error("Erreur : " + str(e))
