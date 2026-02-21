"""
UtopIA — Page 1 : Évaluation des besoins (OTIPM Phase 1-2)
Recueil structuré Personne / Environnement / Occupation
"""

import streamlit as st
from graph.state import PatientState

st.set_page_config(page_title="📋 Évaluation — UtopIA", layout="wide")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }
    .section-header {
        background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
        color: white; padding: 12px 20px; border-radius: 8px;
        font-family: 'DM Sans', sans-serif; font-weight: 600;
        font-size: 1rem; margin: 1.5rem 0 1rem 0;
        display: flex; align-items: center; gap: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: 600;
        font-family: 'DM Sans', sans-serif;
        transition: all 0.2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(15,76,117,0.3); }
    .info-box {
        background: #f0f7ff; border-left: 4px solid #1b6ca8;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Init session state ────────────────────────────────────────────────────────
if "patient" not in st.session_state:
    st.session_state.patient = PatientState()

patient = st.session_state.patient

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("# 📋 Évaluation des besoins — VPH")
st.markdown("*Étape 1 du parcours OTIPM — Recueil de données structuré*")

st.markdown("""
<div class="info-box">
    📌 Cette évaluation est conforme à la <strong>Fiche d'évaluation des besoins VPH</strong>
    (Ministère de la Santé, réforme décembre 2025). Elle constitue l'étape préalable obligatoire
    à la fiche de préconisation.
</div>
""", unsafe_allow_html=True)

# ── Identification ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Identification du bénéficiaire</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    patient.nom = st.text_input("Nom", value=patient.nom)
    patient.prenom = st.text_input("Prénom", value=patient.prenom)
with col2:
    patient.date_naissance = st.text_input("Date de naissance (JJ/MM/AAAA)", value=patient.date_naissance)
    patient.adresse = st.text_input("Adresse", value=patient.adresse)

# ── PERSONNE ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🧍 1 — Facteurs personnels</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    patient.age = st.number_input("Âge", min_value=0, max_value=120,
                                   value=patient.age or 0, step=1) or None
    patient.sexe = st.selectbox("Sexe", ["", "Homme", "Femme", "Autre"],
                                  index=["", "Homme", "Femme", "Autre"].index(patient.sexe) if patient.sexe else 0)
with col2:
    patient.lateralite = st.selectbox("Latéralité",
                                       ["", "Droitier(e)", "Gaucher(e)", "Ambidextre"],
                                       index=["", "Droitier(e)", "Gaucher(e)", "Ambidextre"].index(patient.lateralite) if patient.lateralite else 0)
    patient.capacites_cognitives = st.radio(
        "Capacités cognitives pour conduire le VPH",
        [True, False], format_func=lambda x: "Oui" if x else "Non",
        index=0 if patient.capacites_cognitives else 1, horizontal=True
    )
with col3:
    patient.poids = st.number_input("Poids (kg)", min_value=0.0, max_value=300.0,
                                     value=patient.poids or 0.0, step=0.5) or None
    patient.taille = st.number_input("Taille (cm)", min_value=0.0, max_value=250.0,
                                      value=patient.taille or 0.0, step=0.5) or None

patient.diagnostic = st.text_area("Diagnostic / pathologie principale", value=patient.diagnostic,
                                    placeholder="Ex: Paraplégie T6 post-traumatique, SEP rémittente, SLA...")
patient.capacites_physiques = st.text_area("Capacités physiques (force, motricité, équilibre, endurance...)",
                                            value=patient.capacites_physiques, height=80)
patient.comorbidites = st.text_input("Comorbidités éventuelles", value=patient.comorbidites)

# Mesures anthropométriques (optionnel)
with st.expander("📏 Mesures anthropométriques (facultatif — améliore la précision des préconisations)"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        patient.largeur_bassin = st.number_input("Largeur bassin (cm)", 0.0, 80.0,
                                                  patient.largeur_bassin or 0.0, 0.5) or None
        patient.longueur_cuisses = st.number_input("Longueur cuisses (cm)", 0.0, 80.0,
                                                    patient.longueur_cuisses or 0.0, 0.5) or None
    with col2:
        patient.longueur_creux_poplite_pied = st.number_input("Creux poplité-pied (cm)", 0.0, 60.0,
                                                               patient.longueur_creux_poplite_pied or 0.0, 0.5) or None
        patient.hauteur_coude = st.number_input("Hauteur coude/assise (cm)", 0.0, 50.0,
                                                  patient.hauteur_coude or 0.0, 0.5) or None
    with col3:
        patient.hauteur_omoplate = st.number_input("Hauteur pointe omoplate (cm)", 0.0, 80.0,
                                                    patient.hauteur_omoplate or 0.0, 0.5) or None
        patient.hauteur_epaule = st.number_input("Hauteur épaule (cm)", 0.0, 100.0,
                                                   patient.hauteur_epaule or 0.0, 0.5) or None
    with col4:
        patient.largeur_tronc = st.number_input("Largeur tronc (cm)", 0.0, 80.0,
                                                  patient.largeur_tronc or 0.0, 0.5) or None
        patient.epaisseur_tronc = st.number_input("Épaisseur tronc (cm)", 0.0, 60.0,
                                                   patient.epaisseur_tronc or 0.0, 0.5) or None

# ── ENVIRONNEMENT ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏠 2 — Facteurs environnementaux</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    patient.lieu_vie = st.selectbox("Lieu de vie",
                                     ["", "Appartement", "Maison", "Établissement médico-social (dont EHPAD)"],
                                     index=["", "Appartement", "Maison", "Établissement médico-social (dont EHPAD)"].index(patient.lieu_vie) if patient.lieu_vie else 0)
    patient.escaliers = st.checkbox("Présence d'escaliers", value=patient.escaliers)
    patient.ascenseur = st.checkbox("Présence d'ascenseur", value=patient.ascenseur)
with col2:
    patient.terrain_accidente = st.checkbox("Terrains accidentés / côtes", value=patient.terrain_accidente)
    patient.activite_professionnelle = st.checkbox("Activité professionnelle", value=patient.activite_professionnelle)
    patient.conduite_vehicule = st.checkbox("Conduit un véhicule motorisé", value=patient.conduite_vehicule)
with col3:
    patient.transports_commun = st.checkbox("Utilise les transports en commun", value=patient.transports_commun)
    patient.environnement_humain = st.selectbox("Environnement humain",
                                                  ["", "Personne seule", "En couple", "En famille", "Avec auxiliaires de vie"],
                                                  index=["", "Personne seule", "En couple", "En famille", "Avec auxiliaires de vie"].index(patient.environnement_humain) if patient.environnement_humain else 0)

patient.description_logement = st.text_input("Description du lieu de vie (accessibilité, configuration...)",
                                               value=patient.description_logement)

# ── OCCUPATION ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 3 — Usage et activités (Occupation)</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    patient.premiere_acquisition = st.radio("Type de demande",
                                             [True, False],
                                             format_func=lambda x: "Première acquisition" if x else "Renouvellement / cumul",
                                             horizontal=True,
                                             index=0 if patient.premiere_acquisition else 1)
    if not patient.premiere_acquisition:
        patient.ancien_vph_categorie = st.text_input("Catégorie et modèle du VPH actuel",
                                                       value=patient.ancien_vph_categorie)

    st.markdown("**Types d'activités envisagées avec le fauteuil :**")
    activites_options = ["Vie quotidienne", "Travail", "Loisirs"]
    patient.activites = st.multiselect("", activites_options,
                                        default=[a for a in patient.activites if a in activites_options])

with col2:
    patient.temps_utilisation = st.text_input("Temps d'utilisation prévu (par jour/semaine)",
                                               value=patient.temps_utilisation,
                                               placeholder="Ex: 8h/jour, 5 jours/semaine")
    st.markdown("**Types de déplacements :**")
    deplacement_options = ["Intérieur", "Extérieur", "Tout-terrain"]
    patient.deplacements = st.multiselect("", deplacement_options,
                                           default=[d for d in patient.deplacements if d in deplacement_options])

# ── SITUATION DE SANTÉ ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🩺 4 — Situation de santé</div>', unsafe_allow_html=True)

patient.situation_sante = st.text_area(
    "Description (pathologie, anamnèse, caractère évolutif, comorbidités, problématique respiratoire...)",
    value=patient.situation_sante, height=120,
    placeholder="Décrivez l'histoire de la maladie, l'évolution prévisible, les complications eventuelles..."
)

patient.caractere_evolutif = st.selectbox(
    "Caractère évolutif de la pathologie",
    ["", "Stable", "Progressif", "Fluctuant/rémittent", "En amélioration"],
    index=["", "Stable", "Progressif", "Fluctuant/rémittent", "En amélioration"].index(patient.caractere_evolutif) if patient.caractere_evolutif else 0
)

# ── Synthèse ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📝 Synthèse de la demande</div>', unsafe_allow_html=True)

patient.synthese_demande = st.text_area(
    "Synthèse libre (projet de vie, objectifs prioritaires, contexte particulier...)",
    value=patient.synthese_demande, height=100
)

st.session_state.patient = patient

# ── Bouton d'analyse ──────────────────────────────────────────────────────────
st.divider()
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🔍 Analyser le profil →", use_container_width=True):
        if not patient.diagnostic:
            st.error("⚠️ Veuillez renseigner au minimum le diagnostic du patient.")
        else:
            patient.statut = "preconisation"
            st.session_state.patient = patient
            st.success("✅ Profil enregistré. Rendez-vous sur la page Préconisation.")
            st.balloons()
