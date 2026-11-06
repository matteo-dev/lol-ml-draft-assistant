import os
import json
import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Ajout d'un Encodeur JSON sur-mesure pour gérer les types NumPy ---
class NumpyEncoder(json.JSONEncoder):
    """
    Encodeur personnalisé pour le module JSON.
    Convertit automatiquement les types NumPy (issus de Pandas/Parquet) en types natifs Python.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class RunesPreprocessorV3:
    """
    Transforme les données brutes V3 en une matrice mathématique focalisée sur le matchup.
    Utilise le 'Class Hashing' pour transformer une page de rune entière en une seule 
    classe cible pour la classification multi-classes XGBoost.
    """

    def __init__(self, raw_filepath: str, output_filepath: str, config_filepath: str):
        self.raw_filepath = raw_filepath
        self.output_filepath = output_filepath
        self.config_filepath = config_filepath
        # Stockage de l'objet complet de rune pour chaque hash
        self.hash_to_runes_dict: Dict[str, Dict[str, Any]] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Charge le dataset brut V3."""
        logger.info(f"Chargement des données depuis {self.raw_filepath}...")
        if self.raw_filepath.endswith('.parquet'):
            return pd.read_parquet(self.raw_filepath)
        return pd.read_csv(self.raw_filepath)

    def _hash_rune_page(self, runes: Dict[str, Any]) -> str:
        """
        Crée une chaîne de caractères unique (Hash) représentant la page entière.
        Sauvegarde en parallèle le dictionnaire complet pour la traduction finale.
        """
        if not runes or runes.get('primary_style') is None:
            return "INVALID"
            
        p_style = runes.get('primary_style')
        # S'assurer de convertir en string même si c'est un ndarray
        p_sels = "_".join(map(str, runes.get('primary_selections', [])))
        s_style = runes.get('sub_style')
        s_sels = "_".join(map(str, runes.get('sub_selections', [])))
        stats = f"{runes.get('stat_offense')}_{runes.get('stat_flex')}_{runes.get('stat_defense')}"
        
        rune_hash = f"P:{p_style}[{p_sels}]|S:{s_style}[{s_sels}]|Stats:[{stats}]"
        
        # Mémorisation du dictionnaire lié à ce hash
        if rune_hash not in self.hash_to_runes_dict:
            self.hash_to_runes_dict[rune_hash] = runes
            
        return rune_hash

    def _extract_matchups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplatit le JSON des participants et aligne les adversaires directs (Matchup).
        """
        logger.info("Extraction des matchups directs (Lane vs Lane) et hashing des pages...")
        
        flat_data = []
        for row in df.itertuples(index=False):
            match_id = getattr(row, 'match_id')
            for p in getattr(row, 'participants'):
                pos = p.get('team_position')
                if not pos or pos == "Invalid":
                    continue
                    
                rune_hash = self._hash_rune_page(p.get('runes', {}))
                if rune_hash == "INVALID":
                    continue
                    
                flat_data.append({
                    'match_id': match_id,
                    'team_position': pos,
                    'win': p.get('win'),
                    'champion_id': p.get('champion_id'),
                    'rune_hash': rune_hash
                })

        df_flat = pd.DataFrame(flat_data)
        
        # Séparation Gagnants / Perdants pour le croisement
        df_winners = df_flat[df_flat['win'] == 1]
        df_losers = df_flat[df_flat['win'] == 0]
        
        # Double fusion (Perspective Gagnant ET Perspective Perdant)
        matchups_w = pd.merge(df_winners, df_losers, on=['match_id', 'team_position'], suffixes=('', '_opp'))
        matchups_l = pd.merge(df_losers, df_winners, on=['match_id', 'team_position'], suffixes=('', '_opp'))
        
        df_matchups = pd.concat([matchups_w, matchups_l], ignore_index=True)
        
        # Sélection des colonnes utiles
        df_final = df_matchups[['champion_id', 'champion_id_opp', 'team_position', 'rune_hash']].copy()
        
        logger.info(f"{len(df_final)} situations de matchups extraites.")
        return df_final

    def _build_feature_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Vectorise les features (Champion, Adversaire, Position) et encode la Target (Page Hash).
        """
        logger.info("Création de l'espace vectoriel strict...")
        
        unique_champs = sorted(list(set(df['champion_id'].unique()) | set(df['champion_id_opp'].unique())))
        unique_positions = sorted(list(df['team_position'].unique()))
        unique_hashes = sorted(list(df['rune_hash'].unique()))
        
        logger.info(f"{len(unique_hashes)} pages de runes complètes uniques identifiées.")
        
        champ_to_idx = {champ_id: idx for idx, champ_id in enumerate(unique_champs)}
        pos_to_idx = {pos: idx for idx, pos in enumerate(unique_positions)}
        hash_to_idx = {r_hash: idx for idx, r_hash in enumerate(unique_hashes)}
        
        # Le dictionnaire vital pour Streamlit : idx -> Objet JSON Runes Complet
        idx_to_runes = {idx: self.hash_to_runes_dict[r_hash] for r_hash, idx in hash_to_idx.items()}
        
        champ_cols = [f"is_champ_{int(c)}" for c in unique_champs]
        opp_cols = [f"is_opp_{int(c)}" for c in unique_champs]
        pos_cols = [f"is_pos_{p}" for p in unique_positions]
        
        feature_columns = champ_cols + opp_cols + pos_cols
        
        num_rows = len(df)
        num_cols = len(feature_columns)
        X_matrix = np.zeros((num_rows, num_cols), dtype=np.int8)
        y_target = np.zeros(num_rows, dtype=np.int32)
        
        logger.info("Encodage de la matrice en cours...")
        offset_opp = len(champ_cols)
        offset_pos = offset_opp + len(opp_cols)
        
        for i, row in enumerate(df.itertuples(index=False)):
            champ_val = getattr(row, 'champion_id')
            opp_val = getattr(row, 'champion_id_opp')
            pos_val = getattr(row, 'team_position')
            hash_val = getattr(row, 'rune_hash')
            
            if champ_val in champ_to_idx:
                X_matrix[i, champ_to_idx[champ_val]] = 1
            if opp_val in champ_to_idx:
                X_matrix[i, offset_opp + champ_to_idx[opp_val]] = 1
            if pos_val in pos_to_idx:
                X_matrix[i, offset_pos + pos_to_idx[pos_val]] = 1
                
            if hash_val in hash_to_idx:
                y_target[i] = hash_to_idx[hash_val]
                
        features_df = pd.DataFrame(X_matrix, columns=feature_columns)
        features_df['target_rune_page'] = y_target
        
        config = {
            "champion_ids": [int(c) for c in unique_champs],
            "positions": unique_positions,
            "feature_columns": feature_columns,
            "idx_to_runes": idx_to_runes # Embarque l'arbre complet !
        }
        
        return features_df, config

    def process(self) -> None:
        """Orchestrateur du pipeline."""
        try:
            df_raw = self.load_data()
            if df_raw.empty:
                logger.error("Le dataset brut est vide.")
                return

            df_matchups = self._extract_matchups(df_raw)
            df_processed, config = self._build_feature_matrix(df_matchups)
            
            logger.info(f"Sauvegarde du dataset processé ({df_processed.shape[0]} lignes) dans {self.output_filepath}...")
            df_processed.to_parquet(self.output_filepath, index=False)
            
            # Application du NumpyEncoder pour sérialiser sereinement les ndarray
            with open(self.config_filepath, 'w') as f:
                json.dump(config, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Configuration (incluant les arbres complets) sauvegardée dans {self.config_filepath}.")
            
            logger.info("✅ Preprocessing des Runes V3 terminé avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur fatale lors du preprocessing des Runes : {e}")
            raise

if __name__ == "__main__":
    # --- TEST RUN ---
    INPUT_FILE = "raw_matches_v3.parquet"
    OUTPUT_FILE = "processed_runes_v3.parquet"
    CONFIG_FILE = "runes_features_v3.json"

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Fichier introuvable : {INPUT_FILE}. Veuillez exécuter la Phase 1 (V3) d'abord.")
    else:
        preprocessor = RunesPreprocessorV3(
            raw_filepath=INPUT_FILE,
            output_filepath=OUTPUT_FILE,
            config_filepath=CONFIG_FILE
        )
        preprocessor.process()

        if os.path.exists(OUTPUT_FILE) and os.path.exists(CONFIG_FILE):
            df_check = pd.read_parquet(OUTPUT_FILE)
            print("\n--- Aperçu de la Matrice de Runes ---")
            print(f"Dimensions : {df_check.shape} (Lignes x Colonnes)")
            
            # Vérification de l'export JSON des arbres
            with open(CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
                sample_idx = str(df_check['target_rune_page'].iloc[0])
                print("\n--- Aperçu de l'Arbre de Runes cible (pour le Dashboard) ---")
                print(json.dumps(cfg['idx_to_runes'][sample_idx], indent=2))