"""Streamlit app for chaotic S-box construction and analysis."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import (  # noqa: E402
    METRIC_TARGETS,
    count_passing_metrics,
    evaluate_aes_sbox,
    evaluate_sbox,
    metric_passes,
)
from src.sbox_builder import TUNED_R, TUNED_X0, build_sbox, sbox_to_table  # noqa: E402

st.set_page_config(
    page_title="Chaotic S-box GF(2^8)",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 45%, #0f172a 100%);
        border: 1px solid rgba(129, 140, 248, 0.35);
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.75rem; color: #f8fafc; }
    .hero p { margin: 0.5rem 0 0; color: #c7d2fe; font-size: 0.95rem; }
    .pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.35rem;
        margin-top: 0.5rem;
    }
    .pill-good {
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .pill-warn {
        background: rgba(245, 158, 11, 0.18);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sbox_result(r: float, x0: float) -> tuple[list[int], dict, object, list[list[float]]]:
    """Build S-box and evaluate metrics (cached by parameter pair)."""
    sbox, meta = build_sbox(r=r, x0=x0)
    metrics = evaluate_sbox(sbox)
    return sbox, meta, metrics, metrics.sac_matrix.tolist()


@st.cache_data(show_spinner=False)
def load_aes_benchmark() -> dict[str, float]:
    return evaluate_aes_sbox().as_dict()


def resolve_params(use_default: bool, r_manual: float, x0_manual: float) -> tuple[float, float]:
    if use_default:
        return TUNED_R, TUNED_X0
    return r_manual, x0_manual


def render_hero(passed: int, total: int) -> None:
    badge = "pill-good" if passed == total else "pill-warn"
    label = "Semua metrik memenuhi target" if passed == total else f"{passed}/{total} metrik memenuhi target"
    st.markdown(
        f"""
        <div class="hero">
            <h1>🔐 Chaotic S-box GF(2⁸)</h1>
            <p>Konstruksi S-box 8×8 via Logistic Map + GF(2⁸) inverse + affine transform.
            Pengujian NL, SAC, BIC-NL, BIC-SAC, LAP, dan DAP.</p>
            <span class="pill {badge}">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metrics_dataframe(
    metrics_dict: dict[str, float],
    aes_metrics: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for key, val in metrics_dict.items():
        op, target = METRIC_TARGETS[key]
        target_label = f"{op} {target}" if op != "~" else f"≈ {target}"
        rows.append(
            {
                "Metrik": key,
                "Nilai": round(val, 6),
                "AES": round(aes_metrics[key], 6),
                "Target": target_label,
                "Status": "✅ Baik" if metric_passes(key, val) else "⚠️ Kurang",
                "Δ vs AES": round(val - aes_metrics[key], 6),
            }
        )
    return pd.DataFrame(rows)


def render_sac_heatmap(sac_matrix: list[list[float]]) -> None:
    rows = []
    for out_bit, row in enumerate(sac_matrix):
        for in_bit, value in enumerate(row):
            rows.append({"Output bit": f"Out {out_bit}", "Input bit": f"In {in_bit}", "Flip prob": value})

    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_rect(cornerRadius=2)
        .encode(
            x=alt.X("Input bit:N", title="Input bit"),
            y=alt.Y("Output bit:N", title="Output bit", sort="-x"),
            color=alt.Color(
                "Flip prob:Q",
                scale=alt.Scale(scheme="redblue", domain=[0, 1], reverse=True),
                title="P(flip)",
            ),
            tooltip=[
                alt.Tooltip("Output bit:N"),
                alt.Tooltip("Input bit:N"),
                alt.Tooltip("Flip prob:Q", format=".4f"),
            ],
        )
        .properties(height=360, title="Strict Avalanche Criterion (SAC) — target ≈ 0.5")
    )
    st.altair_chart(chart, width="stretch")


def sbox_to_csv(sbox: list[int]) -> str:
    buffer = io.StringIO()
    buffer.write("row,col,value_hex,value_dec\n")
    for i, val in enumerate(sbox):
        buffer.write(f"{i // 16},{i % 16},{val:02X},{val}\n")
    return buffer.getvalue()


# --- Sidebar ---
with st.sidebar:
    st.header("Parameter")
    use_default = st.toggle("Parameter default", value=True)

    if use_default:
        r_manual, x0_manual = TUNED_R, TUNED_X0
    else:
        r_manual = st.slider("Parameter r", min_value=3.57, max_value=4.0, value=TUNED_R, step=0.01)
        x0_manual = st.number_input(
            "Seed x₀",
            min_value=0.001,
            max_value=0.999,
            value=TUNED_X0,
            step=0.001,
            format="%.9f",
        )

    regenerate = st.button("Hitung S-box", type="primary", width="stretch")

r, x0 = resolve_params(use_default, r_manual, x0_manual)
param_key = (round(r, 9), round(x0, 9))

needs_build = regenerate or "sbox" not in st.session_state or st.session_state.get("param_key") != param_key

if needs_build:
    with st.spinner("Menghitung S-box dan metrik kriptografi..."):
        sbox, meta, metrics, sac_matrix = load_sbox_result(r, x0)
        st.session_state.update(
            param_key=param_key,
            sbox=sbox,
            meta=meta,
            metrics=metrics,
            sac_matrix=sac_matrix,
            r=r,
            x0=x0,
        )

sbox = st.session_state["sbox"]
meta = st.session_state["meta"]
metrics = st.session_state["metrics"]
sac_matrix = st.session_state["sac_matrix"]
aes_metrics = load_aes_benchmark()
passed, total = count_passing_metrics(metrics)

render_hero(passed, total)

summary1, summary2, summary3, summary4 = st.columns(4)
summary1.metric("Parameter r", f"{st.session_state['r']:.4f}")
summary2.metric("Seed x₀", f"{st.session_state['x0']:.6f}")
summary3.metric("Bijective", "✅ Ya" if meta["bijective"] else "❌ Tidak")
summary4.metric("Skor metrik", f"{passed}/{total}")

tab_overview, tab_sbox, tab_analysis, tab_method = st.tabs(
    ["📊 Ringkasan", "🧮 S-box", "📈 Analisis", "📖 Metodologi"]
)

with tab_overview:
    st.subheader("Perbandingan Metrik vs AES")
    metrics_df = build_metrics_dataframe(metrics.as_dict(), aes_metrics)
    st.dataframe(
        metrics_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Nilai": st.column_config.NumberColumn(format="%.6f"),
            "AES": st.column_config.NumberColumn(format="%.6f"),
            "Δ vs AES": st.column_config.NumberColumn(format="%+.6f"),
        },
    )

    st.progress(passed / total, text=f"{passed} dari {total} metrik memenuhi target")

with tab_sbox:
    st.subheader("Tabel S-box (16×16, hexadecimal)")
    table = sbox_to_table(sbox)
    hex_df = pd.DataFrame(
        table,
        columns=[f"{c:X}" for c in range(16)],
        index=[f"{row:X}" for row in range(16)],
    )
    st.dataframe(hex_df, width="stretch", height=440)
    st.download_button(
        label="⬇️ Download S-box (CSV)",
        data=sbox_to_csv(sbox),
        file_name=f"sbox_r{st.session_state['r']}_x0{st.session_state['x0']}.csv",
        mime="text/csv",
        width="stretch",
    )

with tab_analysis:
    st.subheader("Heatmap SAC Matrix")
    render_sac_heatmap(sac_matrix)

    st.subheader("Distribusi SAC")
    sac_flat = np.array(sac_matrix).flatten()
    hist_df = pd.DataFrame({"SAC": sac_flat})
    hist = (
        alt.Chart(hist_df)
        .mark_bar(color="#818cf8")
        .encode(
            x=alt.X("SAC:Q", bin=alt.Bin(maxbins=20), title="Flip probability"),
            y=alt.Y("count()", title="Jumlah sel"),
        )
        .properties(height=260, title="Histogram nilai SAC (target pusat ≈ 0.5)")
    )
    rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(color="#fbbf24", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(hist + rule, width="stretch")

with tab_method:
    st.markdown(
        """
        ### Pipeline konstruksi
        1. **Logistic Map** — `x_{n+1} = r · x_n · (1 - x_n)` menghasilkan urutan kaotik
        2. **Permutasi bijektif** — rank-order sorting dari 256 nilai kaotik
        3. **Multiplicative inverse** — operasi di GF(2⁸) dengan polinom AES `0x11B`
        4. **Affine transform** — `S(x) = M·x ⊕ c` di GF(2)

        ### Parameter
        **r** = 4.0 · **x₀** = 0.5

        ### Target kualitas (8-bit S-box)
        | Metrik | Target |
        |--------|--------|
        | NL (min) | ≥ 104 (ideal 112) |
        | SAC | ≈ 0.5 |
        | BIC-SAC | Joint prob ≈ 0.25 |
        | LAP / DAP | Semakin kecil semakin baik |
        """
    )

    st.subheader("Affine transform")
    st.dataframe(pd.DataFrame(meta["matrix"]), width="stretch")
    st.write(f"Konstanta affine: `0x{meta['constant']:02X}` ({meta['constant']})")
