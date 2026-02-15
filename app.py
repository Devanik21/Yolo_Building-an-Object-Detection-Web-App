import streamlit as st
import numpy as np
from atlas_engine import AethericEngine
import time

# --- Page Config & Theming ---
st.set_page_config(page_title="The Aetheric Atlas 🌌", layout="wide", page_icon="🌌")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .hud-label {
        font-family: 'Monospace';
        color: #00ff88;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    h1, h2, h3 { color: #fff; font-family: 'Monospace'; }
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if 'engine' not in st.session_state:
    st.session_state.engine = AethericEngine()
if 'observed' not in st.session_state:
    st.session_state.observed = {}

engine = st.session_state.engine

# --- Sidebar Navigation ---
st.sidebar.title("🌌 AETHERIC ATLAS")
st.sidebar.caption("21-Vector STEM Multiverse")
st.sidebar.divider()

selection = st.sidebar.selectbox(
    "📡 SELECT VECTOR",
    range(len(engine.subjects)),
    format_func=lambda i: f"V-{i+1:02d}: {engine.subjects[i]}"
)

# --- Main Layout ---
st.title(f"Vector {selection+1:02d}: {engine.subjects[selection]}")

col_meta, col_obs = st.columns([1, 2])

with col_meta:
    st.markdown(f"""
    <div class="glass-card">
        <div class="hud-label">Observational Status</div>
        <p style='font-size: 1.2em;'>Status: {"Observed" if selection in st.session_state.observed else "Latent"}</p>
        <div class="hud-label">Vector Coordinates</div>
        <p>θ: {np.random.rand():.4f} | φ: {np.random.rand():.4f}</p>
        <div class="hud-label">Technical Maturity</div>
        <p>Level: Nobel Tier (Phase 4)</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("✨ OBSERVE MANIFOLD ✨", type="primary", use_container_width=True):
        st.session_state.observed[selection] = True
    
    if st.button("🔄 RESET WAVEFUNCTION", use_container_width=True):
        if selection in st.session_state.observed:
            del st.session_state.observed[selection]
            st.rerun()

with col_obs:
    if selection in st.session_state.observed:
        with st.spinner("Collapsing wavefunction..."):
            fig = engine.get_simulation(selection, seed=int(time.time()) // 60)
            st.pyplot(fig)
            st.caption(f"Real-time latent projection of {engine.subjects[selection]}. 0-cheat simulation.")
    else:
        st.markdown(f"""
        <div style='text-align: center; padding: 100px 20px; border: 1px dashed rgba(255,255,255,0.1); border-radius: 12px;'>
            <p style='color: #555; font-family: monospace;'>[ SUBJECT {selection+1:02d} IN LATENT STATE ]</p>
            <p style='color: #444;'>Wavefunction not yet observed. Click 'Observe' to render manifold.</p>
        </div>
        """, unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("Aesthetic Engine: Stigmergy 1.0")
st.sidebar.caption("Built for Top 1% STEM Showcase")
