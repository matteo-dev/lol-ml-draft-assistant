import os
import json
import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RunesModelTrainerV3:
    """
    Gère l'entraînement XGBoost Multi-Classes (Runes V3).
    Intègre une optimisation massive : filtrage de la meta et tree_method='hist'.
    """

    def __init__(self, data_filepath: str, config_filepath: str, model_output_filepath: str, encoder_filepath: str):
        self.data_filepath = data_filepath
        self.config_filepath = config_filepath
        self.model_output_filepath = model_output_filepath
        self.encoder_filepath = encoder_filepath
        self.model = None
        self.config: Dict = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(f"Chargement des données depuis {self.data_filepath}...")
        df = pd.read_parquet(self.data_filepath)
        
        if 'target_rune_page' not in df.columns:
            raise ValueError("La colonne target est introuvable.")

        with open(self.config_filepath, 'r') as f:
            self.config = json.load(f)

        y = df['target_rune_page']
        X = df.drop(columns=['target_rune_page'])
        return X, y

    def _optimize_and_filter(self, X: pd.DataFrame, y: pd.Series, min_samples: int = 10) -> Tuple[pd.DataFrame, pd.Series, int]:
        """
        Filtre les builds 'troll' ou uniques pour ne garder que la vraie Meta.
        Ré-encode les classes proprement de 0 à K-1 pour XGBoost.
        """
        counts = y.value_counts()
        valid_classes = counts[counts >= min_samples].index
        
        logger.info(f"Filtrage Anti-Bruit : Réduction de {len(counts)} à {len(valid_classes)} pages de runes ultra-meta.")
        
        # Filtrage
        mask = y.isin(valid_classes)
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()
        
        logger.info(f"Données conservées : {len(y_filtered)} matchups sur {len(y)} ({len(y_filtered)/len(y):.1%}).")
        
        # Ré-encodage strict sans trous
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_filtered)
        y_series = pd.Series(y_encoded, index=y_filtered.index)
        
        # Création du dictionnaire de traduction : Nouvel ID XGBoost -> Ancien ID JSON
        mapping = {int(new_id): str(old_id) for new_id, old_id in enumerate(le.classes_)}
        
        with open(self.encoder_filepath, 'w') as f:
            json.dump(mapping, f, indent=4)
        logger.info(f"Dictionnaire de traduction sauvegardé sous {self.encoder_filepath}")
            
        return X_filtered, y_series, len(valid_classes)

    def train_and_evaluate(self, X_raw: pd.DataFrame, y_raw: pd.Series) -> None:
        # 1. Optimisation Massive
        X, y, num_classes = self._optimize_and_filter(X_raw, y_raw, min_samples=10)

        # 2. Split (La stratification fonctionne maintenant car toutes les classes ont au moins 10 samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Set d'entraînement : {X_train.shape[0]} échantillons | Validation : {X_test.shape[0]}")

        # 3. Modélisation Haute Vitesse (tree_method='hist')
        clf = xgb.XGBClassifier(
            n_estimators=100,       # Réduit, 100 itérations suffisent amplement
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_classes,
            tree_method='hist',     # OPTIMISATION MAJEURE : calcul par histogrammes
            early_stopping_rounds=15,
            random_state=42,
            n_jobs=-1
        )

        logger.info("Début de l'entraînement XGBoost (Flash Speed)...")
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )

        self.model = clf
        logger.info(f"Entraînement terminé avec succès (Itération optimale : {clf.best_iteration})")

        # 4. Évaluation
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"✅ Précision globale (Accuracy) : {acc:.2%}")

    def save_model(self) -> None:
        if self.model is None:
            raise RuntimeError("Aucun modèle à sauvegarder.")
        try:
            self.model.save_model(self.model_output_filepath)
            logger.info(f"Modèle sauvegardé : {self.model_output_filepath}")
        except Exception as e:
            logger.error(f"Erreur de sauvegarde : {e}")

    def run_pipeline(self) -> None:
        try:
            X, y = self.load_data()
            self.train_and_evaluate(X, y)
            self.save_model()
            logger.info("🚀 Phase 6 (V3 Rapide) terminée.")
        except Exception as e:
            logger.error(f"Erreur fatale : {e}")
            raise

if __name__ == "__main__":
    # --- RUN PRODUCTION ---
    INPUT_FILE = "processed_runes_v3.parquet"
    CONFIG_FILE = "runes_features_v3.json"
    MODEL_OUTPUT = "xgboost_runes_model_v3.json"
    ENCODER_FILE = "runes_encoder.json"

    if not os.path.exists(INPUT_FILE) or not os.path.exists(CONFIG_FILE):
        print("❌ Fichiers introuvables. Exécutez le preprocessing des runes d'abord.")
    else:
        trainer = RunesModelTrainerV3(
            data_filepath=INPUT_FILE,
            config_filepath=CONFIG_FILE,
            model_output_filepath=MODEL_OUTPUT,
            encoder_filepath=ENCODER_FILE
        )
        trainer.run_pipeline()