# 🦽 UtopIA — Agent IA Préconisation VPH

Agent d'aide à la préconisation de Véhicules pour Personnes Handicapées (VPH),
conforme à la réforme française de décembre 2025, structuré selon le cadre OTIPM.

## Déploiement Streamlit Cloud

### 1. Structure du dépôt GitHub (tout committer, y compris les PDFs)
```
utopia/
├── app.py
├── requirements.txt
├── .streamlit/config.toml
├── graph/state.py + nodes/
├── rag/ingest.py + retriever.py
├── pages/ (4 pages)
└── docs/ (PDFs — obligatoires dans le repo)
```

### 2. Secrets Streamlit Cloud
Dans "Advanced settings > Secrets" :
```toml
ANTHROPIC_API_KEY = "sk-ant-api03-..."
```

### 3. Déployer
1. Push sur GitHub
2. share.streamlit.io → New app → sélectionner app.py
3. Ajouter le secret ANTHROPIC_API_KEY
4. Deploy

### Premier lancement
- L'app démarre en ~2 min
- Cliquer "Initialiser le RAG" dans la sidebar (~2-3 min, vectorise les PDFs)
- Le RAG est mis en cache pour toute la session

## Installation locale
```bash
pip install -r requirements.txt
# Créer .env avec : ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Ajouter des documents
Déposer les PDFs dans docs/[categorie]/ et re-déployer.
Les catégories : reglementation, evaluation-clinique, categories-vph, modeles-conceptuels, argumentaires

## Stack
Streamlit · Claude Opus 4 · Voyage-3 embeddings · ChromaDB · LangChain · PyMuPDF

## Avertissement
Outil d'aide à la décision — ne remplace pas le jugement de l'ergothérapeute.
Tous les documents générés doivent être validés et signés par un ergothérapeute habilité.
