import os
import logging
from typing import Tuple
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DraftModelTrainerV3:
    """
    Gère le chargement des données vectorisées V3, l'entraînement du modèle XGBoost,
    l'évaluation des performances et la sauvegarde de l'artefact.
    """

    def __init__(self, data_filepath: str, model_output_filepath: str):
        self.data_filepath = data_filepath
        self.model_output_filepath = model_output_filepath
        self.model = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Charge la matrice d'entraînement et sépare les features (X) de la target (y).
        """
        logger.info(f"Chargement des données depuis {self.data_filepath}...")
        df = pd.read_parquet(self.data_filepath)
        
        if 'target_win' not in df.columns:
            raise ValueError("La colonne 'target_win' est introuvable dans le dataset.")

        y = df['target_win']
        X = df.drop(columns=['target_win'])
        
        logger.info(f"Données chargées : {X.shape[0]} échantillons, {X.shape[1]} features.")
        return X, y

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Sépare les données en train/test, entraîne XGBoost avec early stopping, 
        et affiche les métriques de performance.
        """
        # Sécurisation pour éviter le crash sur les tout petits datasets de test locaux
        test_size = 0.2 if len(X) > 20 else 0.4
        is_stratify = y if len(X) > 20 else None

        logger.info("Séparation des données en Train / Test...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=is_stratify
            )
        except ValueError:
            logger.warning("Stratification impossible sur ce mini dataset. Fallback standard.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        logger.info(f"Taille du set d'entraînement : {X_train.shape[0]}")
        logger.info(f"Taille du set de validation : {X_test.shape[0]}")

        # Configuration des hyperparamètres XGBoost optimisés
        # - max_depth limité pour forcer le modèle à chercher des synergies globales et éviter l'overfitting
        clf = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc',
            early_stopping_rounds=25,
            random_state=42,
            n_jobs=-1
        )

        logger.info("Début de l'entraînement du modèle XGBoost (Draft V3)...")
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        self.model = clf
        logger.info(f"Entraînement terminé (Meilleure itération : {clf.best_iteration})")

        # --- Évaluation ---
        logger.info("Évaluation du modèle sur le set de test...")
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, y_proba)
            logger.info(f"✅ ROC-AUC : {auc:.4f}")
        except ValueError:
            logger.warning("ROC-AUC ignoré (une seule classe présente dans le set de test actuel).")

        logger.info(f"✅ Précision (Accuracy) : {acc:.2%}\n")
        
        print("\n--- Rapport de Classification ---")
        print(classification_report(y_test, y_pred, zero_division=0))

    def save_model(self) -> None:
        """
        Sauvegarde le modèle entraîné au format JSON natif XGBoost.
        """
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder. Lancer l'entraînement d'abord.")
            
        try:
            self.model.save_model(self.model_output_filepath)
            logger.info(f"Modèle sauvegardé avec succès sous : {self.model_output_filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")
            raise

    def run_pipeline(self) -> None:
        """Orchestrateur de la phase d'entraînement."""
        try:
            X, y = self.load_data()
            self.train_and_evaluate(X, y)
            self.save_model()
            logger.info("🚀 Phase 3 terminée avec succès.")
        except Exception as e:
            logger.error(f"Erreur fatale dans le pipeline d'entraînement : {e}")
            raise

if __name__ == "__main__":
    # --- TEST RUN ---
    INPUT_FILE = "processed_drafts_v3.parquet"
    MODEL_OUTPUT = "xgboost_draft_model_v3.json"

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fichier introuvable : {INPUT_FILE}. Veuillez d'abord exécuter 02_preprocessing_draft.py.")
    else:
        trainer = DraftModelTrainerV3(
            data_filepath=INPUT_FILE,
            model_output_filepath=MODEL_OUTPUT
        )
        trainer.run_pipeline()