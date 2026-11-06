import os
import sys
import json
import importlib.util
import requests
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from typing import Dict, Any, List

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LoL ML Assistant V3",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Stat Shards Mapping ---
STAT_SHARDS_MAPPING = {
    5001: "Scaling Health", 5002: "Armor", 5003: "Magic Resist",
    5005: "Attack Speed", 5007: "Ability Haste",
    5008: "Adaptive Force", 5010: "Movement Speed",
    5011: "Flat Health", 5013: "Tenacity"
}

# --- Secure Dynamic Import of Phase 4 (Draft V3) ---
@st.cache_resource
def load_draft_inference_engine():
    file_path = "04_inference_engine.py"
    module_name = "inference_engine_module"
    
    if not os.path.exists(file_path):
        st.error(f"Missing file: {file_path}. The draft inference backend is required.")
        st.stop()
        
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.DraftInferenceEngineV3
    except Exception as e:
        st.error(f"Error loading Draft Inference V3: {e}")
        st.stop()

# --- Runes Inference Engine (Full Tree & Encoder) ---
class RunesInferenceEngineV3:
    def __init__(self, model_path: str, config_path: str, encoder_path: str):
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(encoder_path, 'r') as f:
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
            # 1. XGBoost returns a new ID (enc_idx)
            # 2. The encoder translates it to the old ID (old_idx)
            old_idx = self.encoder.get(str(enc_idx))
            
            # 3. Retrieve the full page with this old ID
            page_data = self.config['idx_to_runes'].get(str(old_idx), {})
            
            results.append({
                "page": page_data,
                "probability": float(probs[enc_idx])
            })
        return results

# --- Extended Data Dragon API ---
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
        url = f"http://ddragon.leagueoflegends.com/cdn/{_self.version}/data/en_US/champion.json"
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
        url = f"https://ddragon.leagueoflegends.com/cdn/{_self.version}/data/en_US/runesReforged.json"
        try:
            trees = requests.get(url).json()
            for tree in trees:
                mapping[tree['id']] = {"name": tree['name'], "image_url": f"https://ddragon.leagueoflegends.com/cdn/img/{tree['icon']}"}
                for slot in tree['slots']:
                    for rune in slot['runes']:
                        mapping[rune['id']] = {"name": rune['name'], "image_url": f"https://ddragon.leagueoflegends.com/cdn/img/{rune['icon']}"}
        except Exception as e:
            st.warning(f"Error loading Rune icons: {e}")
        return mapping

    def get_champion_info(self, champ_id: int) -> Dict[str, str]:
        return self.id_to_name.get(champ_id, {"name": f"Unknown ({champ_id})", "image_url": ""})

    def get_rune_info(self, rune_id: int) -> Dict[str, str]:
        return self.runes_mapping.get(rune_id, {"name": f"Rune ({rune_id})", "image_url": ""})

# --- UI Components ---
def plot_shap_waterfall(pros: List[Dict], cons: List[Dict], ddragon: DataDragonAPI):
    all_reasons = pros + cons
    if not all_reasons:
        return go.Figure().add_annotation(text="No significant SHAP data", showarrow=False)

    x_labels, y_values = [], []
    for r in all_reasons:
        c_info = ddragon.get_champion_info(r['champion_id'])
        x_labels.append(f"{r['reason']}<br>{c_info['name']}")
        y_values.append(r['shap_value'])

    fig = go.Figure(go.Waterfall(
        name="SHAP Impact", orientation="v", measure=["relative"] * len(all_reasons),
        x=x_labels, text=[f"{v:+.3f}" for v in y_values], textposition="outside", y=y_values,
        connector={"line": {"color": "rgba(200,200,200,0.5)"}},
        decreasing={"marker": {"color": "#ef553b"}}, increasing={"marker": {"color": "#00cc96"}}
    ))
    fig.update_layout(
        title="Mathematical Impact on Victory (Log-Odds)", title_font_size=12, height=300,
        margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def display_rune_tree(page: Dict[str, Any], ddragon: DataDragonAPI):
    col_p, col_s, col_stat = st.columns([2, 1.5, 1.5])
    
    with col_p:
        st.markdown("**Primary Tree**")
        p_style = page.get('primary_style')
        if p_style:
            info = ddragon.get_rune_info(p_style)
            st.image(info['image_url'], width=30, caption=info['name'])
            for r_id in page.get('primary_selections', []):
                r_info = ddragon.get_rune_info(r_id)
                st.image(r_info['image_url'], width=40, caption=r_info['name'])
                
    with col_s:
        st.markdown("**Secondary Tree**")
        s_style = page.get('sub_style')
        if s_style:
            info = ddragon.get_rune_info(s_style)
            st.image(info['image_url'], width=30, caption=info['name'])
            for r_id in page.get('sub_selections', []):
                r_info = ddragon.get_rune_info(r_id)
                st.image(r_info['image_url'], width=30, caption=r_info['name'])

    with col_stat:
        st.markdown("**Statistics**")
        st.write(f"🗡️ {STAT_SHARDS_MAPPING.get(page.get('stat_offense'), 'Offense')}")
        st.write(f"⚖️ {STAT_SHARDS_MAPPING.get(page.get('stat_flex'), 'Flex')}")
        st.write(f"🛡️ {STAT_SHARDS_MAPPING.get(page.get('stat_defense'), 'Defense')}")

# --- Main Function ---
def main():
    st.title("🧠 League of Legends: Draft & Runes Analyst V3")

    with st.spinner("Loading Data Dragon API..."):
        ddragon = DataDragonAPI()
    
    DraftInferenceEngineV3 = load_draft_inference_engine()
    
    # Strict validation of the 6 required files
    REQUIRED_FILES = [
        "xgboost_draft_model_v3.json", "model_features_v3.json", "champion_roles.json",
        "xgboost_runes_model_v3.json", "runes_features_v3.json", "runes_encoder.json"
    ]
    
    missing_files = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing_files:
        st.error(f"Missing files: {', '.join(missing_files)}. Please rerun the training.")
        st.stop()

    engine_draft = DraftInferenceEngineV3(
        model_path=REQUIRED_FILES[0], 
        config_path=REQUIRED_FILES[1], 
        roles_path=REQUIRED_FILES[2]
    )
    engine_runes = RunesInferenceEngineV3(
        model_path=REQUIRED_FILES[3], 
        config_path=REQUIRED_FILES[4],
        encoder_path=REQUIRED_FILES[5]
    )

    available_champs = engine_draft.champion_ids
    champ_options = {c_id: ddragon.get_champion_info(c_id)['name'] for c_id in available_champs}
    def format_champ(c_id): return champ_options.get(c_id, str(c_id))

    tab_draft, tab_runes = st.tabs(["🔮 Draft Optimizer (Roles & SHAP)", "⚔️ Matchups (Rune Trees)"])

    # -------------------------------------
    # TAB 1: DRAFT & SHAP PROS/CONS
    # -------------------------------------
    with tab_draft:
        st.sidebar.header("Draft Configuration")
        target_role = st.sidebar.selectbox("🎯 Target Role", options=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"])
        st.sidebar.markdown("---")
        bans = st.sidebar.multiselect("Bans (Global)", options=available_champs, format_func=format_champ, max_selections=10)
        allies = st.sidebar.multiselect("Our Team", options=available_champs, format_func=format_champ, max_selections=4)
        enemies = st.sidebar.multiselect("Enemy Team", options=available_champs, format_func=format_champ, max_selections=5)

        overlap = set(bans) & set(allies) | set(bans) & set(enemies) | set(allies) & set(enemies)
        if overlap:
            st.sidebar.error("Error: A champion can only have one status (Ban/Ally/Enemy).")
        else:
            if st.sidebar.button("🧠 Analyze Draft", type="primary", use_container_width=True):
                if len(allies) >= 5:
                    st.info("Full composition.")
                else:
                    with st.spinner(f"Searching for the best {target_role}..."):
                        results = engine_draft.get_recommendations(allies, enemies, bans, target_role, top_k=2)
                        
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.subheader(f"Top 2 Recommendations for: {target_role}")
                        cols = st.columns(2)
                        
                        for i, rec in enumerate(results["top_picks"]):
                            c_info = ddragon.get_champion_info(rec["champion_id"])
                            analysis = rec.get("analysis", {"pros": [], "cons": []})
                            
                            with cols[i]:
                                st.markdown(f"### #{i+1}\n{c_info['name']}")
                                
                                img_col, metric_col = st.columns([1, 2])
                                with img_col:
                                    if c_info["image_url"]: st.image(c_info["image_url"], width=100)
                                with metric_col:
                                    st.metric(label="Win Probability", value=f"{rec['win_probability']*100:.1f}%")

                                st.markdown("✅ **Pros of this pick:**")
                                for pro in analysis['pros']:
                                    t_name = ddragon.get_champion_info(pro['champion_id'])['name']
                                    st.write(f"- {pro['reason']} **{t_name}** *(+{pro['shap_value']:.2f})*")
                                if not analysis['pros']: st.write("- No notable advantages.")

                                st.markdown("❌ **Cons / Risks:**")
                                for con in analysis['cons']:
                                    t_name = ddragon.get_champion_info(con['champion_id'])['name']
                                    st.write(f"- {con['reason']} **{t_name}** *({con['shap_value']:.2f})*")
                                if not analysis['cons']: st.write("- No major risks detected.")

                                fig = plot_shap_waterfall(analysis['pros'], analysis['cons'], ddragon)
                                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"draft_{i}_{rec['champion_id']}")

    # -------------------------------------
    # TAB 2: MULTI-CLASS RUNES V3
    # -------------------------------------
    with tab_runes:
        st.markdown("### Full Rune Tree Optimization")
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1: position = st.selectbox("Role (Lane)", options=["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"], key="r_pos")
        with rc2: our_champ = st.selectbox("Our Champion", options=available_champs, format_func=format_champ, key="r_our")
        with rc3: opp_champ = st.selectbox("Direct Matchup (Opponent)", options=available_champs, format_func=format_champ, key="r_opp")

        if st.button("⚡ Generate Rune Pages", use_container_width=True):
            if our_champ == opp_champ:
                st.warning("Mirror matchups are not possible in a classic draft.")
            else:
                rune_results = engine_runes.get_top_rune_pages(our_champ, opp_champ, position, top_k=2)
                st.markdown("---")
                st.subheader(f"Best Pages for {format_champ(our_champ)} vs {format_champ(opp_champ)}")
                
                rune_cols = st.columns(2)
                for i, rec in enumerate(rune_results):
                    with rune_cols[i]:
                        st.markdown(f"#### Option #{i+1} (Prob: {rec['probability']*100:.1f}%)")
                        if rec['page']:
                            display_rune_tree(rec['page'], ddragon)
                        else:
                            st.error("Invalid or corrupted rune data for this class.")

if __name__ == "__main__":
    main()