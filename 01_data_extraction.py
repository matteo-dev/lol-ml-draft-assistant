import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from riotwatcher import LolWatcher, ApiError

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LeagueDataExtractorV3:
    """
    Gère l'extraction des données Riot (Match-V5) avec une granularité maximale :
    - Positions (Rôles) des joueurs
    - Arbre complet des Runes (Styles, Perks, Stat Shards)
    Intègre une sauvegarde par Checkpoint pour les longues sessions d'extraction.
    """

    def __init__(self, api_key: str, region: str = 'euw1', routing_region: str = 'europe'):
        self.watcher = LolWatcher(api_key)
        self.region = region
        self.routing_region = routing_region

    def get_high_elo_puuids(self, limit: int = 300) -> List[str]:
        """Récupère les PUUIDs des joueurs du ladder Challenger."""
        puuids: List[str] = []
        try:
            logger.info(f"Récupération du ladder Challenger sur {self.region}...")
            challenger_league = self.watcher.league.challenger_by_queue(self.region, 'RANKED_SOLO_5x5')
            entries = challenger_league['entries'][:limit]

            for entry in entries:
                if 'puuid' in entry:
                    puuids.append(entry['puuid'])
                    
            logger.info(f"{len(puuids)} PUUIDs récupérés avec succès.")
        except ApiError as err:
            logger.error(f"Erreur lors de la récupération du ladder: {err}")

        return puuids

    def _parse_runes(self, perks: Dict[str, Any]) -> Dict[str, Any]:
        """Extrait et structure l'arbre complet des runes d'un joueur."""
        stat_perks = perks.get('statPerks', {})
        styles = perks.get('styles', [])
        
        parsed_runes = {
            "stat_offense": stat_perks.get('offense'),
            "stat_flex": stat_perks.get('flex'),
            "stat_defense": stat_perks.get('defense'),
            "primary_style": None,
            "primary_selections": [],
            "sub_style": None,
            "sub_selections": []
        }

        for style in styles:
            description = style.get('description')
            selections = [s.get('perk') for s in style.get('selections', [])]
            
            if description == 'primaryStyle':
                parsed_runes["primary_style"] = style.get('style')
                parsed_runes["primary_selections"] = selections
            elif description == 'subStyle':
                parsed_runes["sub_style"] = style.get('style')
                parsed_runes["sub_selections"] = selections

        return parsed_runes

    def get_match_data(self, match_id: str) -> Optional[Dict[str, Any]]:
        """Extrait les données de la draft, des rôles et des runes du match."""
        try:
            match = self.watcher.match.by_id(self.routing_region, match_id)
            info = match.get('info', {})

            # Filtrer uniquement les drafts classées Solo/Duo 5v5 (queueId 420)
            if info.get('queueId') != 420:
                return None

            teams = info.get('teams', [])
            participants_data = info.get('participants', [])

            if len(teams) != 2 or len(participants_data) != 10:
                return None

            blue_team = next((t for t in teams if t['teamId'] == 100), None)
            red_team = next((t for t in teams if t['teamId'] == 200), None)

            if not blue_team or not red_team:
                return None

            # Nettoyage des bans (-1 signifie "aucun ban")
            blue_bans = [b['championId'] for b in blue_team.get('bans', []) if b['championId'] != -1]
            red_bans = [b['championId'] for b in red_team.get('bans', []) if b['championId'] != -1]

            # Parsing détaillé des participants (Rôle + Runes complètes)
            participants = []
            for p in participants_data:
                # Filtrer les positions "Invalid" (ex: AFK remake)
                position = p.get('teamPosition')
                if not position or position == "Invalid":
                    return None
                    
                participants.append({
                    "team_id": p.get('teamId'),
                    "champion_id": p.get('championId'),
                    "team_position": position,
                    "win": int(p.get('win', False)),
                    "runes": self._parse_runes(p.get('perks', {}))
                })

            return {
                'match_id': match_id,
                'game_version': info.get('gameVersion'),
                'blue_win': int(blue_team.get('win', False)),
                'blue_bans': blue_bans,
                'red_bans': red_bans,
                'participants': participants
            }

        except ApiError as err:
            if err.response.status_code == 404:
                pass  # Silencieux pour les 404 (fréquent sur les vieux matchs)
            else:
                logger.error(f"Erreur API sur le match {match_id}: {err}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue lors du parsing du match {match_id} : {e}")
            return None

    def build_dataset(
        self, 
        output_filepath: str,
        num_players: int = 5, 
        matches_per_player: int = 5,
        checkpoint_step: int = 500
    ) -> pd.DataFrame:
        """Construit le dataset complet avec checkpoints de sécurité."""
        puuids = self.get_high_elo_puuids(limit=num_players)
        all_match_ids = set()

        logger.info(f"Récupération des listes de matchs pour {len(puuids)} joueurs...")
        for i, puuid in enumerate(puuids):
            try:
                m_ids = self.watcher.match.matchlist_by_puuid(
                    self.routing_region, puuid, queue=420, count=matches_per_player
                )
                all_match_ids.update(m_ids)
                if (i + 1) % 50 == 0:
                    logger.info(f"... {i + 1}/{len(puuids)} joueurs scannés.")
            except ApiError as err:
                logger.error(f"Erreur matchlist pour {puuid}: {err}")

        logger.info(f"{len(all_match_ids)} matchs uniques identifiés. Début du Parsing détaillé...")

        dataset = []
        match_list = list(all_match_ids)
        
        for i, mid in enumerate(match_list):
            data = self.get_match_data(mid)
            if data:
                dataset.append(data)
                
            if (i + 1) % 100 == 0:
                logger.info(f"Parsing en cours : {i + 1}/{len(match_list)} matchs analysés...")
                
            if (i + 1) % checkpoint_step == 0 and len(dataset) > 0:
                chkpt_file = output_filepath.replace(".parquet", "_checkpoint.parquet")
                pd.DataFrame(dataset).to_parquet(chkpt_file, index=False)
                logger.info(f"💾 Checkpoint sauvegardé ({len(dataset)} matchs).")

        df = pd.DataFrame(dataset)
        logger.info(f"🎉 Extraction terminée. {len(df)} matchs valides récupérés.")
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str) -> None:
        try:
            df.to_parquet(filepath, index=False)
            logger.info(f"✅ Dataset final sauvegardé sous : {filepath}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}")

if __name__ == "__main__":
    # --- TEST RUN ---
    API_KEY = os.getenv("RIOT_API_KEY", "RGAPI-9bea1c34-01ae-4b16-a046-b57b6080850e")
    OUTPUT_FILE = "raw_matches_v3.parquet"

    extractor = LeagueDataExtractorV3(api_key=API_KEY, region="euw1", routing_region="europe")

    # Paramétrage réduit pour tester le parsing des nouveaux objets sans griller le rate-limit
    logger.info("Démarrage du test de validation (300 joueurs, 20 matchs)...")
    df_raw = extractor.build_dataset(
        output_filepath=OUTPUT_FILE,
        num_players=300, 
        matches_per_player=20,
        checkpoint_step=500
    )

    if not df_raw.empty:
        # Vérification visuelle de la nouvelle structure des participants
        print("\n--- Aperçu de la structure d'un participant (Rôles & Runes) ---")
        sample_participant = df_raw.iloc[0]['participants'][0]
        import json
        print(json.dumps(sample_participant, indent=2))
        
        extractor.save_dataset(df_raw, OUTPUT_FILE)
    else:
        print("\nAucun match récupéré. Vérifie la clé API ou l'état des serveurs.")