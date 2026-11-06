import os
import sys
import json
import gzip
import importlib.util
import requests
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from typing import Dict, Any, List

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="LoL ML Assistant V3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fragments de Stats (Stat Shards) ---
STAT_SHARDS_MAPPING = {
    5001: "Santé Scaling", 5002: "Armure", 5003: "Résistance Magique",
    5005: "Vitesse d'Attaque", 5007: "Accélération de Compétence",
    5008: "Force Adaptative", 5010: "Vitesse de Déplacement",
    5011: "Santé Brute", 5013: "Ténacité"
}

# --- Import dynamique sécurisé de la Phase 4 (Draft V3) ---
@st.cache_resource
def load_draft_inference_engine():
    file_path = "04_inference_engine.py"
    module_name = "inference_engine_module"
    
    if not os.path.exists(file_path):
        st.error(f"Fichier manquant : {file_path}. Le backend d'inférence draft est requis.")
        st.stop()
        
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.DraftInferenceEngineV3
    except Exception as e:
        st.error(f"Erreur lors du chargement de la Draft Inference V3 : {e}")
        st.stop()

# --- Moteur d'Inférence des Runes (Arbre Complet, Encodeur & Support GZIP) ---
class RunesInferenceEngineV3:
    def __init__(self, model_path: str, config_path: str, encoder_path: str):
        self.model = xgb.XGBClassifier()
        
        # Chargement intelligent du modèle XGBoost (supporte le .json.gz en mémoire)
        if model_path.endswith('.gz'):
            with gzip.open(model_path, 'rb') as f:
                self.model.load_model(bytearray(f.read()))
        else:
            self.model.load_model(model_path)
        
        # Lecture transparente du JSON compressé (GZIP) pour la config
        if config_path.endswith('.gz'):
            with gzip.open(config_path, 'rt', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
        with open(encoder_path, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)

    def get_top_rune_pages(self, champ_id: int, opp_id: int, position: str, top_k: int = 2) -> List[Dict[str, Any]]:
        features = self.config['feature_columns']
        X_matrix = np.zeros((1, len(features)), dtype=np.int8)
        
        champ_col = f"is_champ_{champ_id}"
        opp_col = f"is_opp_{opp_id}"
        pos_col = f"is_pos_{position}"

        if champ_col in features: X_matrix[0, features.index(champ_col)] = 1
        if opp_col in features: X_matrix[0, features.index(opp_col)] = 1
        if pos_col in features: X_matrix[0, features.index(pos_col)] = 1

        probs = self.model.predict_proba(X_matrix)[0]
        top_encoded_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for enc_idx in top_encoded_indices:
            old_idx = self.encoder.get(str(enc_idx))
            page_data = self.config['idx_to_runes'].get(str(old_idx), {})
            results.append({
                "page": page_data,
                "probability": float(probs[enc_idx])
            })
        return results

# --- API Data Dragon Étendue ---
class DataDragonAPI:
    def __init__(self):
        self.version = self._get_latest_version()
        self.champ_data = self._get_champion_data()
        self.id_to_name = self._build_champ_mapping()
        self.runes_mapping = self._build_runes_mapping()

    @st.cache_data
    def _get_latest_version(_self) -> str:
        return requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]

    @st.cache_data
    def _get_champion_data(_self) -> Dict[str, Any]:
        url = f"http://ddragon.leagueoflegends.com/cdn/{_self.version}/data/fr_FR/champion.json"
        return requests.get(url).json()['data']

    def _build_champ_mapping(self) -> Dict[int, Dict[str, str]]:
        mapping = {}
        for _, data in self.champ_data.items():
            champ_id = int(data['key'])
            mapping[champ_id] = {
                "name": data['name'],
                "image_url": f"http://ddragon.leagueoflegends.com/cdn/{self.version}/img/champion/{data['image']['full']}"
            }
        return mapping

    @st.cache_data
    def _build_runes_mapping(_self) -> Dict[int, Dict[str, str]]:
        mapping = {}
        url = f"https://ddragon.leagueoflegends.com/cdn/{_self.version}/data/fr_FR/runesReforged.json"
        try:
            trees = requests.get(url).json()
            for tree in trees:
                mapping[tree['id']] = {"name": tree['name'], "image_url": f"https://ddragon.leagueoflegends.com/cdn/img/{tree['icon']}"}
                for slot in tree['slots']:
                    for rune in slot['runes']:
                        mapping[rune['id']] = {"name": rune['name'], "image_url": f"https://ddragon.leagueoflegends.com/cdn/img/{rune['icon']}"}
        except Exception as e:
            st.warning(f"Erreur de chargement des icônes de Runes : {e}")
        return mapping

    def get_champion_info(self, champ_id: int) -> Dict[str, str]:
        return self.id_to_name.get(champ_id, {"name": f"Inconnu ({champ_id})", "image_url": ""})

    def get_rune_info(self, rune_id: int) -> Dict[str, str]:
        return self.runes_mapping.get(rune_id, {"name": f"Rune ({rune_id})", "image_url": ""})

# --- Composants UI ---
def plot_shap_waterfall(pros: List[Dict], cons: List[Dict], ddragon: DataDragonAPI):
    all_reasons = pros + cons
    if not all_reasons:
        return go.Figure().add_annotation(text="Aucune donnée SHAP significative", showarrow=False)

    x_labels, y_values = [], []
    for r in all_reasons:
        c_info = ddragon.get_champion_info(r['champion_id'])
        x_labels.append(f"{r['reason']}<br>{c_info['name']}")
        y_values.append(r['shap_value'])

    fig = go.Figure(go.Waterfall(
        name="Impact SHAP", orientation="v", measure=["relative"] * len(all_reasons),
        x=x_labels, text=[f"{v:+.3f}" for v in y_values], textposition="outside", y=y_values,
        connector={"line": {"color": "rgba(200,200,200,0.5)"}},
        decreasing={"marker": {"color": "#ef553b"}}, increasing={"marker": {"color": "#00cc96"}}
    ))
    fig.update_layout(
        title="Impact Mathématique sur la Victoire (Log-Odds)", title_font_size=12, height=300,
        margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def display_rune_tree(page: Dict[str, Any], ddragon: DataDragonAPI):
    col_p, col_s, col_stat = st.columns([2, 1.5, 1.5])
    
    with col_p:
        st.markdown("**Arbre Principal**")
        p_style = page.get('primary_style')
        if p_style:
            info = ddragon.get_rune_info(p_style)
            st.image(info['image_url'], width=30, caption=info['name'])
            for r_id in page.get('primary_selections', []):
                r_info = ddragon.get_rune_info(r_id)
                st.image(r_info['image_url'], width=40, caption=r_info['name'])
                
    with col_s:
        st.markdown("**Arbre Secondaire**")
        s_style = page.get('sub_style')
        if s_style:
            info = ddragon.get_rune_info(s_style)
            st.image(info['image_url'], width=30, caption=info['name'])
            for r_id in page.get('sub_selections', []):
                r_info = ddragon.get_rune_info(r_id)
                st.image(r_info['image_url'], width=30, caption=r_info['name'])

    with col_stat:
        st.markdown("**Statistiques**")
        st.write(f"🗡️ {STAT_SHARDS_MAPPING.get(page.get('stat_offense'), 'Offensif')}")
        st.write(f"⚖️ {STAT_SHARDS_MAPPING.get(page.get('stat_flex'), 'Flexible')}")
        st.write(f"🛡️ {STAT_SHARDS_MAPPING.get(page.get('stat_defense'), 'Défensif')}")

def get_existing_file(candidates: List[str]) -> str:
    """Retourne le premier fichier existant parmi une liste de candidats, ou une chaîne vide."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

# --- Fonction Principale ---
def main():
    st.title("🧠 League of Legends : Analyste Draft & Runes V3")

    with st.spinner("Chargement de l'API Data Dragon..."):
        ddragon = DataDragonAPI()
    
    DraftInferenceEngineV3 = load_draft_inference_engine()
    
    # Recherche souple des fichiers (supporte les versions compressées .gz)
    DRAFT_MODEL = get_existing_file(["xgboost_draft_model_v3.json.gz", "xgboost_draft_model_v3.json"])
    DRAFT_CONFIG = get_existing_file(["model_features_v3.json.gz", "model_features_v3.json"])
    ROLES_FILE = get_existing_file(["champion_roles.json.gz", "champion_roles.json"])
    RUNES_MODEL = get_existing_file(["xgboost_runes_model_v3.json.gz"])
    RUNES_CONFIG = get_existing_file(["runes_features_v3.json.gz", "runes_features_v3.json"])
    RUNES_ENCODER = get_existing_file(["runes_encoder.json.gz", "runes_encoder.json"])
    
    missing_files = []
    if not DRAFT_MODEL: missing_files.append("Modèle Draft")
    if not DRAFT_CONFIG: missing_files.append("Config Draft")
    if not ROLES_FILE: missing_files.append("Fichier Rôles")
    if not RUNES_MODEL: missing_files.append("Modèle Runes")
    if not RUNES_CONFIG: missing_files.append("Config Runes")
    if not RUNES_ENCODER: missing_files.append("Encodeur Runes")

    if missing_files:
        st.error(f"Fichiers manquants : {', '.join(missing_files)}. Veuillez vérifier vos exports.")
        st.stop()

    engine_draft = DraftInferenceEngineV3(
        model_path=DRAFT_MODEL, 
        config_path=DRAFT_CONFIG, 
        roles_path=ROLES_FILE
    )
    engine_runes = RunesInferenceEngineV3(
        model_path=RUNES_MODEL, 
        config_path=RUNES_CONFIG,
        encoder_path=RUNES_ENCODER
    )

    available_champs = engine_draft.champion_ids
    champ_options = {c_id: ddragon.get_champion_info(c_id)['name'] for c_id in available_champs}
    def format_champ(c_id): return champ_options.get(c_id, str(c_id))

    tab_draft, tab_runes = st.tabs(["🔮 Draft Optimizer (Rôles & SHAP)", "⚔️ Matchups (Arbres de Runes)"])

    # -------------------------------------
    # ONGLET 1 : DRAFT & SHAP PROS/CONS
    # -------------------------------------
    with tab_draft:
        st.sidebar.header("Configuration de la Draft")
        target_role = st.sidebar.selectbox("🎯 Rôle recherché", options=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"])
        st.sidebar.markdown("---")
        bans = st.sidebar.multiselect("Bans (Globaux)", options=available_champs, format_func=format_champ, max_selections=10)
        allies = st.sidebar.multiselect("Notre Équipe", options=available_champs, format_func=format_champ, max_selections=4)
        enemies = st.sidebar.multiselect("Équipe Ennemie", options=available_champs, format_func=format_champ, max_selections=5)

        overlap = set(bans) & set(allies) | set(bans) & set(enemies) | set(allies) & set(enemies)
        if overlap:
            st.sidebar.error("Erreur : Un champion ne peut avoir qu'un seul statut (Ban/Allié/Ennemi).")
        else:
            if st.sidebar.button("🧠 Analyser la Draft", type="primary", use_container_width=True):
                if len(allies) >= 5:
                    st.info("Composition complète.")
                else:
                    with st.spinner(f"Recherche du meilleur {target_role}..."):
                        results = engine_draft.get_recommendations(allies, enemies, bans, target_role, top_k=2)
                        
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.subheader(f"Top 2 Recommandations pour : {target_role}")
                        cols = st.columns(2)
                        
                        for i, rec in enumerate(results["top_picks"]):
                            c_info = ddragon.get_champion_info(rec["champion_id"])
                            analysis = rec.get("analysis", {"pros": [], "cons": []})
                            
                            with cols[i]:
                                st.markdown(f"### #{i+1} - {c_info['name']}")
                                
                                img_col, metric_col = st.columns([1, 2])
                                with img_col:
                                    if c_info["image_url"]: st.image(c_info["image_url"], width=100)
                                with metric_col:
                                    st.metric(label="Win Probabilité", value=f"{rec['win_probability']*100:.1f}%")

                                st.markdown("✅ **Avantages de ce pick :**")
                                for pro in analysis['pros']:
                                    t_name = ddragon.get_champion_info(pro['champion_id'])['name']
                                    st.write(f"- {pro['reason']} **{t_name}** *(+{pro['shap_value']:.2f})*")
                                if not analysis['pros']: st.write("- Aucun avantage saillant.")

                                st.markdown("❌ **Inconvénients / Risques :**")
                                for con in analysis['cons']:
                                    t_name = ddragon.get_champion_info(con['champion_id'])['name']
                                    st.write(f"- {con['reason']} **{t_name}** *({con['shap_value']:.2f})*")
                                if not analysis['cons']: st.write("- Aucun risque majeur détecté.")

                                fig = plot_shap_waterfall(analysis['pros'], analysis['cons'], ddragon)
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"draft_{i}_{rec['champion_id']}")

    # -------------------------------------
    # ONGLET 2 : RUNES MULTI-CLASSES V3
    # -------------------------------------
    with tab_runes:
        st.markdown("### Optimisation de l'Arbre de Runes Complet")
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1: position = st.selectbox("Rôle (Lane)", options=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], key="r_pos")
        with rc2: our_champ = st.selectbox("Notre Champion", options=available_champs, format_func=format_champ, key="r_our")
        with rc3: opp_champ = st.selectbox("Adversaire Direct", options=available_champs, format_func=format_champ, key="r_opp")

        if st.button("⚡ Générer les Pages de Runes", use_container_width=True):
            if our_champ == opp_champ:
                st.warning("Matchup miroir impossible en draft classique.")
            else:
                rune_results = engine_runes.get_top_rune_pages(our_champ, opp_champ, position, top_k=2)
                st.markdown("---")
                st.subheader(f"Meilleures Pages pour {format_champ(our_champ)} vs {format_champ(opp_champ)}")
                
                rune_cols = st.columns(2)
                for i, rec in enumerate(rune_results):
                    with rune_cols[i]:
                        st.markdown(f"#### Option #{i+1} (Prob: {rec['probability']*100:.1f}%)")
                        if rec['page']:
                            display_rune_tree(rec['page'], ddragon)
                        else:
                            st.error("Données de runes invalides ou corrompues pour cette classe.")

if __name__ == "__main__":
    main()
