# League of Legends : ML Draft & Runes Assistant

*(English version below)*

Un outil analytique de niveau professionnel pour League of Legends, conçu pour recommander les meilleurs choix de champions et les meilleures pages de runes en phase de draft. Ce projet s'appuie sur des modèles de Machine Learning (XGBoost) entraînés sur des données de niveau Challenger et utilise la théorie des jeux (Valeurs de Shapley / SHAP) pour rendre chaque décision 100% interprétable.

## Fonctionnalités Principales

- **Draft Optimizer par Rôle :** Recommande les meilleurs champions à sélectionner en fonction de votre équipe, de l'équipe ennemie, des bans, et du **rôle visé** (Top, Jungle, Mid, Bot, Support).
- **Interprétabilité SHAP (Pros & Cons) :** Le modèle n'est pas une boîte noire. Chaque recommandation est justifiée par une analyse mathématique des synergies alliées (avantages) et des vulnérabilités face aux ennemis (inconvénients), visualisée via des graphiques *Waterfall* (Plotly).
- **Matchup & Runes Engine (Arbre Complet) :** Prédit la page de runes la plus optimale (Arbre principal, secondaire et fragments de statistiques) pour un matchup spécifique grâce à un modèle de classification multi-classes et une technique de *Class Hashing*.
- **Interface UI Streamlit :** Un dashboard analytique interactif intégré avec l'API Riot Data Dragon pour afficher en temps réel les portraits des champions et les icônes officielles des runes.

## Stack Technique

- **Data Extraction :** `riotwatcher` (Riot API Match-V5)
- **Data Processing :** `pandas`, `numpy`, `pyarrow` (Parquet)
- **Machine Learning :** `scikit-learn` (LabelEncoder), `xgboost` (Binary & Multi-class Softprob)
- **Explicabilité (XAI) :** `shap` (TreeExplainer)
- **Frontend & Visualisation :** `streamlit`, `plotly`
- **CI/CD & Automatisation :** GitHub Actions, Playwright (Wake-up cron job)

## Architecture du Pipeline (V3)

Le projet est divisé en scripts modulaires exécutables séquentiellement :
1. `01_data_extraction.py` : Aspiration de la meta Challenger (Gestion des Rate Limits, Checkpoints).
2. `02_preprocessing.py` : Encodage Zéro-Somme (+1/-1) et cartographie dynamique des rôles.
3. `03_model_training.py` : Entraînement XGBoost pour la prédiction de victoire.
4. `04_inference_engine.py` : Filtrage par rôle et dichotomie SHAP (Avantages/Inconvénients).
5. `05_runes_preprocessing.py` : *Class Hashing* des arbres de runes complets sur les matchups directs.
6. `06_runes_training.py` : Filtrage de la meta (Anti-Bruit) et entraînement Multi-classes haute vitesse (`tree_method='hist'`).
7. `07_streamlit_app.py` : L'interface utilisateur finale combinant les deux modèles.

## Installation & Utilisation

1. Clonez ce dépôt :  
```bash
   git clone [https://github.com/matteo-dev/lol-ml-draft-assistant]
   cd lol-ml-draft-assistant
```

2. Installer les dépendances :
```bash
  pip install - r requirements.txt
```

3. Lancer le frontend :
```bash
  stramlit run 07_streamlit_app.py
```


# League of Legends: ML Draft & Runes Assistant

A professional-grade analytical tool for League of Legends designed to recommend the best champion picks and rune pages during the draft phase. This project leverages Machine Learning models (XGBoost) trained on Challenger-tier data and uses game theory (Shapley Values / SHAP) to make every decision 100% interpretable.

## Core Features

- **Role-Specific Draft Optimizer:** Recommends the best champions to pick based on your team, the enemy team, bans, and your target role (Top, Jungle, Mid, Bot, Support).
- **SHAP Interpretability (Pros & Cons):** The model is not a black box. Every recommendation is justified by a mathematical analysis of allied synergies (Pros) and enemy vulnerabilities (Cons), visualized through Plotly Waterfall charts.
- **Matchup & Runes Engine (Full Tree):** Predicts the most optimal full rune page (Primary, Secondary, and Stat Shards) for a specific matchup using a multi-class classification model and Class Hashing.
- **Streamlit UI Dashboard:** An interactive analytical dashboard fully integrated with the Riot Data Dragon API to fetch real-time champion portraits and official rune icons.

## Tech Stack

- **Data Extraction:** riotwatcher (Riot API Match-V5)
- **Data Processing:** pandas, numpy, pyarrow (Parquet)
- **Machine Learning:** scikit-learn (LabelEncoder), xgboost (Binary & Multi-class Softprob)
- **Explainable AI (XAI):** shap (TreeExplainer)
- **Frontend & DataViz:** streamlit, plotly
- **CI/CD & Automation:** GitHub Actions, Playwright (Wake-up cron job)

## Pipeline Architecture (V3)

The project is divided into modular scripts executed sequentially:

1. `01_data_extraction.py`: Challenger meta extraction (Rate Limit handling, Checkpointing).
2. `02_preprocessing.py`: Zero-Sum Encoding (+1/-1) and dynamic role mapping.
3. `03_model_training.py`: XGBoost training for win probability prediction.
4. `04_inference_engine.py`: Role filtering and SHAP dichotomy parsing (Pros/Cons).
5. `05_runes_preprocessing.py`: Class Hashing for full rune trees on direct matchups.
6. `06_runes_training.py`: Meta filtering (Noise reduction) and high-speed Multi-class training (`tree_method='hist'`).
7. `07_streamlit_app.py`: The final user interface combining both models.

## Installation & Usage

1. Clonez the repository :  
```bash
   git clone [https://github.com/matteo-dev/lol-ml-draft-assistant]
   cd lol-ml-draft-assistant
```

2. Install requirements :
```bash
  pip install - r requirements.txt
```

3. Run Frontend :
```bash
  stramlit run 07_streamlit_app.py
```
