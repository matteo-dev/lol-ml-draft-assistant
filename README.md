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
