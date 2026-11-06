import os
import json
import logging
from typing import Tuple, List, Dict, Set
import pandas as pd
import numpy as np
from collections import defaultdict

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DraftPreprocessorV3:
    """
    Gère le nettoyage, l'encodage vectoriel et la cartographie des rôles
    à partir du dataset V3 pour l'entraînement du modèle XGBoost.
    """

    def __init__(self, raw_filepath: str, output_matrix_filepath: str, config_filepath: str, roles_filepath: str):
        self.raw_filepath = raw_filepath
        self.output_matrix_filepath = output_matrix_filepath
        self.config_filepath = config_filepath
        self.roles_filepath = roles_filepath
        self.unique_champions: List[int] = []
        
    def load_data(self) -> pd.DataFrame:
        """Charge le dataset brut V3."""
        logger.info(f"Chargement des données depuis {self.raw_filepath}...")
        if self.raw_filepath.endswith('.parquet'):
            return pd.read_parquet(self.raw_filepath)
        return pd.read_csv(self.raw_filepath)

    def _parse_and_extract_roles(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
        """
        Extrait les compositions d'équipes et cartographie dynamiquement 
        les rôles viables de chaque champion en fonction de la meta Challenger.
        """
        logger.info("Parsing des compositions et analyse dynamique des rôles...")
        
        parsed_data = []
        role_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for row in df.itertuples(index=False):
            match_id = getattr(row, 'match_id')
            blue_win = getattr(row, 'blue_win')
            participants = getattr(row, 'participants')
            
            blue_picks = []
            red_picks = []
            
            for p in participants:
                champ_id = p.get('champion_id')
                team_id = p.get('team_id')
                position = p.get('team_position')
                
                # Assignation aux équipes
                if team_id == 100:
                    blue_picks.append(champ_id)
                elif team_id == 200:
                    red_picks.append(champ_id)
                    
                # Comptage des rôles pour la cartographie
                if position and position != "Invalid":
                    role_counts[champ_id][position] += 1
                    
            parsed_data.append({
                'match_id': match_id,
                'blue_picks': blue_picks,
                'red_picks': red_picks,
                'blue_win': blue_win
            })

        # Détermination des rôles viables (seuil de tolérance : joué > 2% du temps à ce poste)
        champion_roles = {}
        for champ_id, positions in role_counts.items():
            total_games = sum(positions.values())
            valid_roles = [pos for pos, count in positions.items() if (count / total_games) >= 0.02]
            champion_roles[champ_id] = valid_roles

        return pd.DataFrame(parsed_data), champion_roles

    def _extract_perspectives(self, df_parsed: pd.DataFrame) -> pd.DataFrame:
        """Dédouble le dataset pour créer une ligne par point de vue (Bleu et Rouge)."""
        logger.info("Dédoublement du dataset (Perspectives Bleue et Rouge)...")
        
        blue_perspective = pd.DataFrame({
            'match_id': df_parsed['match_id'],
            'allies': df_parsed['blue_picks'],
            'enemies': df_parsed['red_picks'],
            'win': df_parsed['blue_win']
        })
        
        red_perspective = pd.DataFrame({
            'match_id': df_parsed['match_id'],
            'allies': df_parsed['red_picks'],
            'enemies': df_parsed['blue_picks'],
            'win': 1 - df_parsed['blue_win']
        })
        
        combined_df = pd.concat([blue_perspective, red_perspective], ignore_index=True)
        return combined_df

    def _build_feature_matrix(self, df_perspectives: pd.DataFrame) -> pd.DataFrame:
        """Vectorisation mathématique (+1 Allié, -1 Ennemi, 0 Absent)."""
        logger.info("Création de l'espace vectoriel strict...")
        
        all_picks = [champ for sublist in df_perspectives['allies'] for champ in sublist] + \
                    [champ for sublist in df_perspectives['enemies'] for champ in sublist]
        self.unique_champions = sorted(list(set(all_picks)))
        
        logger.info(f"{len(self.unique_champions)} champions uniques identifiés.")
        
        champ_to_idx = {champ_id: idx for idx, champ_id in enumerate(self.unique_champions)}
        
        num_rows = len(df_perspectives)
        num_cols = len(self.unique_champions)
        X_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
        
        for row_idx, (allies, enemies) in enumerate(zip(df_perspectives['allies'], df_perspectives['enemies'])):
            for ally in allies:
                if ally in champ_to_idx:
                    X_matrix[row_idx, champ_to_idx[ally]] = 1
            for enemy in enemies:
                if enemy in champ_to_idx:
                    X_matrix[row_idx, champ_to_idx[enemy]] = -1
                    
        feature_columns = [f"champ_{int(champ_id)}" for champ_id in self.unique_champions]
        features_df = pd.DataFrame(X_matrix, columns=feature_columns)
        features_df['target_win'] = df_perspectives['win'].astype(np.int8)
        
        return features_df

    def process(self) -> None:
        """Orchestrateur principal du pipeline de preprocessing de la Draft."""
        try:
            df_raw = self.load_data()
            if df_raw.empty:
                logger.error("Le dataset brut est vide.")
                return

            df_parsed, champion_roles = self._parse_and_extract_roles(df_raw)
            df_perspectives = self._extract_perspectives(df_parsed)
            df_processed = self._build_feature_matrix(df_perspectives)
            
            # Sauvegarde de la matrice
            logger.info(f"Sauvegarde du dataset processé dans {self.output_matrix_filepath}...")
            df_processed.to_parquet(self.output_matrix_filepath, index=False)
            
            # Sauvegarde de la config des features (pour l'inférence)
            config = {
                "feature_columns": [f"champ_{int(c)}" for c in self.unique_champions],
                "champion_ids": [int(c) for c in self.unique_champions]
            }
            with open(self.config_filepath, 'w') as f:
                json.dump(config, f, indent=4)
                
            # Sauvegarde du dictionnaire des rôles (pour le filtrage UI)
            with open(self.roles_filepath, 'w') as f:
                json.dump(champion_roles, f, indent=4)
                
            logger.info(f"Fichiers JSON de configuration et de rôles sauvegardés.")
            logger.info("✅ Preprocessing Draft V3 terminé avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur fatale lors du preprocessing : {e}")
            raise

if __name__ == "__main__":
    # --- TEST RUN ---
    INPUT_FILE = "raw_matches_v3.parquet"
    OUTPUT_FILE = "processed_drafts_v3.parquet"
    CONFIG_FILE = "model_features_v3.json"
    ROLES_FILE = "champion_roles.json"

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fichier introuvable : {INPUT_FILE}. Veuillez d'abord exécuter 01_data_extraction.py.")
    else:
        preprocessor = DraftPreprocessorV3(
            raw_filepath=INPUT_FILE,
            output_matrix_filepath=OUTPUT_FILE,
            config_filepath=CONFIG_FILE,
            roles_filepath=ROLES_FILE
        )
        preprocessor.process()

        if os.path.exists(OUTPUT_FILE) and os.path.exists(ROLES_FILE):
            print("\n--- Aperçu de la Matrice d'Entraînement ---")
            df_check = pd.read_parquet(OUTPUT_FILE)
            print(df_check.iloc[:, :8].head())
            
            print("\n--- Aperçu des Rôles Extraits (Top 3 champions) ---")
            with open(ROLES_FILE, 'r') as f:
                roles = json.load(f)
                for i, (champ, poss) in enumerate(roles.items()):
                    if i < 3: print(f"Champion {champ}: {poss}")