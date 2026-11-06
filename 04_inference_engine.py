import os
import json
import logging
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DraftInferenceEngineV3:
    """
    Moteur de recommandation de draft basé sur XGBoost et SHAP.
    Filtre par rôle et sépare les explications en Avantages / Inconvénients.
    """

    def __init__(self, model_path: str, config_path: str, roles_path: str):
        self.model_path = model_path
        self.config_path = config_path
        self.roles_path = roles_path
        
        self.model = None
        self.feature_columns: List[str] = []
        self.champion_ids: List[int] = []
        self.champion_roles: Dict[str, List[str]] = {}
        self.explainer = None
        
        self._load_assets()

    def _load_assets(self) -> None:
        """Charge le modèle, les métadonnées des features, les rôles et initialise SHAP."""
        try:
            logger.info("Chargement des configurations (Features & Rôles)...")
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                self.feature_columns = config['feature_columns']
                self.champion_ids = config['champion_ids']
                
            with open(self.roles_path, 'r') as f:
                # json keys are strings, convert back to int for internal logic
                roles_data = json.load(f)
                self.champion_roles = {int(k): v for k, v in roles_data.items()}

            logger.info(f"Chargement du modèle XGBoost depuis {self.model_path}...")
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            
            logger.info("Initialisation du TreeExplainer SHAP (cela peut prendre quelques secondes)...")
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info(" Moteur d'inférence V3 prêt.")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du moteur : {e}")
            raise

    def _generate_candidate_matrix(
        self, 
        current_allies: List[int], 
        current_enemies: List[int], 
        candidates: List[int]
    ) -> pd.DataFrame:
        """Crée une matrice dense contenant toutes les drafts potentielles à évaluer."""
        num_candidates = len(candidates)
        num_features = len(self.champion_ids)
        
        matrix = np.zeros((num_candidates, num_features), dtype=np.int8)
        champ_to_idx = {champ_id: idx for idx, champ_id in enumerate(self.champion_ids)}

        for row_idx, candidate in enumerate(candidates):
            # Alliés existants
            for ally in current_allies:
                if ally in champ_to_idx:
                    matrix[row_idx, champ_to_idx[ally]] = 1
            
            # Le candidat est un allié
            if candidate in champ_to_idx:
                matrix[row_idx, champ_to_idx[candidate]] = 1
                
            # Ennemis
            for enemy in current_enemies:
                if enemy in champ_to_idx:
                    matrix[row_idx, champ_to_idx[enemy]] = -1

        return pd.DataFrame(matrix, columns=self.feature_columns)

    def _analyze_shap_impact(
        self, 
        candidate: int, 
        feature_vector: pd.Series, 
        shap_values: np.ndarray
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Analyse les valeurs de Shapley et les trie en Avantages (Pros) et Inconvénients (Cons).
        """
        avantages = []
        inconvenients = []
        
        for idx, feature_name in enumerate(self.feature_columns):
            feature_val = feature_vector.iloc[idx]
            shap_val = shap_values[idx]
            
            # Ignorer les features inactives ou à impact microscopique
            if feature_val == 0 or abs(shap_val) < 0.001:  
                continue
                
            champ_id = int(feature_name.replace("champ_", ""))
            
            # On n'explique pas le champion par lui-même, mais par ses interactions
            if champ_id == candidate:
                continue

            impact_data = {
                "champion_id": champ_id,
                "shap_value": float(shap_val)
            }

            if feature_val == 1:
                if shap_val > 0:
                    impact_data["reason"] = "Excellente synergie avec"
                    avantages.append(impact_data)
                else:
                    impact_data["reason"] = "Manque de synergie avec"
                    inconvenients.append(impact_data)
            elif feature_val == -1:
                if shap_val > 0:
                    impact_data["reason"] = "Fort contre"
                    avantages.append(impact_data)
                else:
                    impact_data["reason"] = "Vulnérable face à"
                    inconvenients.append(impact_data)

        # Tri par force d'impact (valeur absolue)
        avantages.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        inconvenients.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        
        return {
            "pros": avantages[:3],      # Top 3 avantages
            "cons": inconvenients[:2]   # Top 2 inconvénients
        }

    def get_recommendations(
        self, 
        current_allies: List[int], 
        current_enemies: List[int], 
        bans: List[int], 
        target_role: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Filtre par rôle, prédit les winrates, extrait les SHAP bidirectionnels 
        et renvoie les meilleures recommandations.
        """
        logger.info(f"Recherche de recommandations pour le rôle {target_role}...")
        
        unavailable = set(current_allies + current_enemies + bans)
        
        # Filtrage strict par disponibilité ET par rôle valide
        candidates = []
        for c in self.champion_ids:
            if c not in unavailable and target_role in self.champion_roles.get(c, []):
                candidates.append(c)
        
        if not candidates:
            return {"error": f"Aucun champion disponible ou viable pour le rôle {target_role}."}

        # 1. Génération de la matrice et Prédiction vectorisée
        X_candidates = self._generate_candidate_matrix(current_allies, current_enemies, candidates)
        probabilities = self.model.predict_proba(X_candidates)[:, 1]

        # 2. Identification du Top K
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_candidates = [candidates[i] for i in top_indices]
        top_probs = [float(probabilities[i]) for i in top_indices]

        # 3. Explicabilité SHAP sur le Top K
        X_top = X_candidates.iloc[top_indices]
        shap_values = self.explainer.shap_values(X_top)
        
        recommendations = []
        for i, candidate_id in enumerate(top_candidates):
            candidate_shap_vals = shap_values[i]
            candidate_vector = X_top.iloc[i]
            
            explanations = self._analyze_shap_impact(candidate_id, candidate_vector, candidate_shap_vals)
            
            recommendations.append({
                "champion_id": candidate_id,
                "win_probability": round(top_probs[i], 4),
                "analysis": explanations
            })

        return {
            "status": "success",
            "target_role": target_role,
            "top_picks": recommendations
        }

if __name__ == "__main__":
    # --- TEST RUN ---
    MODEL_FILE = "xgboost_draft_model_v3.json"
    CONFIG_FILE = "model_features_v3.json"
    ROLES_FILE = "champion_roles.json"

    if not all(os.path.exists(f) for f in [MODEL_FILE, CONFIG_FILE, ROLES_FILE]):
        print(" Fichiers introuvables. Assurez-vous d'avoir exécuté les Phases 2 et 3 (V3).")
    else:
        engine = DraftInferenceEngineV3(
            model_path=MODEL_FILE, 
            config_path=CONFIG_FILE,
            roles_path=ROLES_FILE
        )

        # Simulation d'une draft
        # Prenons quelques IDs qui existent forcément dans le modèle
        sample_allies = engine.champion_ids[0:2]
        sample_enemies = engine.champion_ids[2:4]
        sample_bans = engine.champion_ids[4:6]
        
        # Test sur un rôle standard
        target = "MIDDLE"

        print("\n--- Simulation de Draft V3 ---")
        print(f"Alliés : {sample_allies} | Ennemis : {sample_enemies} | Rôle visé : {target}\n")

        result = engine.get_recommendations(
            current_allies=sample_allies,
            current_enemies=sample_enemies,
            bans=sample_bans,
            target_role=target,
            top_k=2
        )

        print(json.dumps(result, indent=4, ensure_ascii=False))