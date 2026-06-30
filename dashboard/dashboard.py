"""
dashboard.py  —  PHAST Pipeline Dashboard
Run with: .venv\Scripts\streamlit.exe run dashboard/dashboard.py
"""

import os
import sys
import pickle
import subprocess
import threading
import time
import glob
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc, FancyArrowPatch
import matplotlib.patheffects as pe

# ── repo root so relative imports work ───────────────────────────────────────
# Auto-detect repo root by walking up from this file until we find scripts/run_pipeline.py
def _find_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "scripts" / "run_pipeline.py").exists():
            return candidate
    # Fallback: assume same folder as this file
    return here

REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))

def _venv_python() -> str:
    for p in [
        REPO_ROOT / ".venv" / "Scripts" / "python.exe",
        REPO_ROOT / ".venv" / "bin" / "python",
    ]:
        if p.is_file():
            return str(p)
    return sys.executable

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PHAST · Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── design tokens ─────────────────────────────────────────────────────────────
# Palette: deep space navy base, cyan instrument accent, amber alert, green confirm
# Signature: Ω gauge rendered as a spacecraft instrument dial
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
    --bg-base:    #080d14;
    --bg-panel:   #0d1520;
    --bg-card:    #111d2e;
    --bg-hover:   #16253a;
    --border:     #1e3352;
    --border-lit: #2a4a73;
    --cyan:       #38bdf8;
    --cyan-dim:   #0369a1;
    --amber:      #f59e0b;
    --green:      #22c55e;
    --red:        #ef4444;
    --text-hi:    #e2eaf4;
    --text-mid:   #7fa5c8;
    --text-lo:    #3d5a7a;
    --mono:       'Space Mono', monospace;
    --sans:       'Space Grotesk', sans-serif;
}

html, body, [class*="css"] {
    background-color: var(--bg-base) !important;
    color: var(--text-hi) !important;
    font-family: var(--sans) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-panel) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: var(--sans) !important; }

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-lit) !important;
    border-radius: 4px !important;
    color: var(--text-hi) !important;
    font-family: var(--mono) !important;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--cyan) !important;
    color: var(--cyan) !important;
    font-family: var(--mono) !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.1em !important;
    border-radius: 3px !important;
    padding: 0.45rem 1.2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: var(--cyan) !important;
    color: var(--bg-base) !important;
}
.stButton > button[kind="primary"] {
    background: var(--cyan) !important;
    color: var(--bg-base) !important;
    font-weight: 700 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-mid) !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 0 !important;
    padding: 0.5rem 1.4rem !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--text-mid) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.4rem !important;
    color: var(--text-hi) !important;
    font-weight: 700 !important;
}

/* Progress bars */
.stProgress > div > div > div {
    background: var(--cyan) !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.06em !important;
    color: var(--text-mid) !important;
}

/* Hide default streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Custom card */
.phast-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.phast-card-title {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    color: var(--text-mid);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}

/* Verdict banner */
.verdict-banner {
    border-radius: 6px;
    padding: 1.4rem 2rem;
    margin: 0.8rem 0;
    border-left: 4px solid;
}
.verdict-planet  { background: #0f2d1a; border-color: #22c55e; }
.verdict-maybe   { background: #2d1f0a; border-color: #f59e0b; }
.verdict-reject  { background: #2d0a0a; border-color: #ef4444; }
.verdict-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-mid);
    margin-bottom: 0.3rem;
}
.verdict-title {
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.verdict-sub {
    font-family: var(--sans);
    font-size: 0.88rem;
    color: var(--text-mid);
    margin-top: 0.4rem;
}

/* Pipeline status steps */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.45rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.25rem;
    font-family: var(--mono);
    font-size: 0.75rem;
}
.step-done    { background: #0f2d1a; color: #22c55e; }
.step-running { background: #1a1a0a; color: #f59e0b; }
.step-pending { background: var(--bg-card); color: var(--text-lo); }
.step-failed  { background: #2d0a0a; color: #ef4444; }

/* Tag pills */
.tag-green  { display:inline-block; background:#0f2d1a; color:#22c55e; border:1px solid #22c55e44; border-radius:3px; padding:0.15rem 0.55rem; font-family:var(--mono); font-size:0.72rem; margin:0.1rem; }
.tag-red    { display:inline-block; background:#2d0a0a; color:#ef4444; border:1px solid #ef444444; border-radius:3px; padding:0.15rem 0.55rem; font-family:var(--mono); font-size:0.72rem; margin:0.1rem; }
.tag-amber  { display:inline-block; background:#2d1f0a; color:#f59e0b; border:1px solid #f59e0b44; border-radius:3px; padding:0.15rem 0.55rem; font-family:var(--mono); font-size:0.72rem; margin:0.1rem; }
.tag-cyan   { display:inline-block; background:#04213a; color:#38bdf8; border:1px solid #38bdf844; border-radius:3px; padding:0.15rem 0.55rem; font-family:var(--mono); font-size:0.72rem; margin:0.1rem; }

/* Top header bar */
.phast-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 1rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.phast-logo {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    color: var(--cyan);
}
.phast-sub {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-lo);
    letter-spacing: 0.12em;
    margin-top: 0.1rem;
}
.phast-timestamp {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-lo);
    letter-spacing: 0.1em;
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

def data_dir(run_tag: str) -> Path:
    if run_tag == "_flat_run":
        return REPO_ROOT / "data"
    return REPO_ROOT / "data" / run_tag

def load_pkl(run_tag: str, stage: int):
    p = data_dir(run_tag) / f"stage{stage}_output.pkl"
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

def available_runs() -> list[str]:
    """Return all TIC_xxx_Sn run folders that have at least stage 1 output."""
    runs = []
    data_root = REPO_ROOT / "data"
    if not data_root.exists():
        return runs
    for d in sorted(data_root.iterdir()):
        if d.is_dir() and d.name.startswith("TIC_") and (d / "stage1_output.pkl").exists():
            runs.append(d.name)
    return runs

def stages_complete(run_tag: str) -> dict[int, bool]:
    return {s: (data_dir(run_tag) / f"stage{s}_output.pkl").exists() for s in range(1, 9)}

def score_color(v: float) -> str:
    if v >= 0.80: return "#22c55e"
    if v >= 0.50: return "#f59e0b"
    return "#ef4444"

def verdict_class(classification: str) -> str:
    cl = classification.lower()
    if "planet" in cl or "exoplanet" in cl: return "planet"
    if "binary" in cl or "artifact" in cl or "false" in cl: return "reject"
    return "maybe"


# ── matplotlib theme ──────────────────────────────────────────────────────────

def apply_mpl_theme():
    plt.rcParams.update({
        "figure.facecolor":  "#0d1520",
        "axes.facecolor":    "#111d2e",
        "axes.edgecolor":    "#1e3352",
        "axes.labelcolor":   "#7fa5c8",
        "axes.titlecolor":   "#e2eaf4",
        "text.color":        "#e2eaf4",
        "xtick.color":       "#7fa5c8",
        "ytick.color":       "#7fa5c8",
        "grid.color":        "#1e3352",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "figure.dpi":        130,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

apply_mpl_theme()


# ── gauge plot ────────────────────────────────────────────────────────────────

def draw_omega_gauge(score: float, label: str = "Ω") -> plt.Figure:
    """Draw a spacecraft-instrument style semicircular gauge for the Oddity Score."""
    fig, ax = plt.subplots(figsize=(3.2, 1.9), facecolor="#0d1520")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.35, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # Track arc
    arc_bg = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180,
                 color="#1e3352", lw=12, zorder=1)
    ax.add_patch(arc_bg)

    # Coloured fill arc proportional to score
    theta_end = score * 180
    col = score_color(score)
    arc_fill = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=theta_end,
                   color=col, lw=12, zorder=2)
    ax.add_patch(arc_fill)

    # Threshold ticks
    for thresh, tc in [(0.5, "#f59e0b"), (0.8, "#22c55e")]:
        ang = np.radians(thresh * 180)
        x0, y0 = 0.88 * np.cos(ang), 0.88 * np.sin(ang)
        x1, y1 = 1.08 * np.cos(ang), 1.08 * np.sin(ang)
        ax.plot([x0, x1], [y0, y1], color=tc, lw=2, zorder=3)
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang),
                f"{thresh:.1f}", color=tc, fontsize=5.5, ha="center", va="center",
                fontfamily="monospace")

    # Scale labels
    for val, txt in [(0, "0"), (1, "1")]:
        ang = np.radians(val * 180)
        ax.text(1.18 * np.cos(ang), 1.18 * np.sin(ang),
                txt, color="#3d5a7a", fontsize=5.5, ha="center", va="center",
                fontfamily="monospace")

    # Central score value
    ax.text(0, 0.18, f"{score:.3f}", ha="center", va="center",
            fontsize=17, fontweight="bold", color=col, fontfamily="monospace",
            zorder=5)
    ax.text(0, -0.05, label, ha="center", va="center",
            fontsize=7.5, color="#7fa5c8", fontfamily="monospace", zorder=5)

    # Needle
    ang = np.radians(score * 180)
    nx, ny = 0.72 * np.cos(ang), 0.72 * np.sin(ang)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8))
    ax.plot(0, 0, "o", color=col, markersize=5, zorder=6)

    fig.tight_layout(pad=0.2)
    return fig


# ── light curve plot ─────────────────────────────────────────────────────────

def plot_lightcurve(s1: dict, s2: dict) -> plt.Figure:
    time  = np.array(s1["time"])
    _flux_raw = np.asarray(s1["flux"], dtype=float)
    flux  = np.where(np.isfinite(_flux_raw), _flux_raw, np.nan)
    period = float(s2["period"])
    t0     = float(s2["t0"])
    dur    = float(s2["duration"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), facecolor="#0d1520",
                                    gridspec_kw={"height_ratios": [2, 1.2], "hspace": 0.05})

    # Raw light curve
    ax1.plot(time, flux, lw=0.5, color="#1e3a5f", alpha=0.7, rasterized=True)
    ax1.plot(time, flux, ".", ms=0.8, color="#38bdf8", alpha=0.4, rasterized=True)

    # Mark transit times
    phase0 = t0 % period
    tmin, tmax = time.min(), time.max()
    n_min = int((tmin - phase0) / period)
    n_max = int((tmax - phase0) / period) + 1
    for n in range(n_min, n_max + 1):
        tc = phase0 + n * period
        if tmin <= tc <= tmax:
            ax1.axvspan(tc - dur / 2, tc + dur / 2, color="#38bdf8", alpha=0.12, zorder=0)
            ax1.axvline(tc, color="#38bdf8", lw=0.8, alpha=0.5, ls="--", zorder=1)

    ax1.set_ylabel("Norm. Flux", fontsize=8)
    ax1.set_title(f"TIC {s1['tic_id']}  |  Sector {s1['sector']}  |  TESS Light Curve",
                  fontsize=9, color="#e2eaf4", pad=6)
    ax1.tick_params(labelbottom=False)
    ax1.grid(True, alpha=0.2)

    # Phase-folded
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1
    sort_idx = np.argsort(phase)
    ph_s, fl_s = phase[sort_idx], flux[sort_idx]

    ax2.plot(ph_s, fl_s, ".", ms=1.2, color="#38bdf8", alpha=0.5, rasterized=True)

    # Bin the phase curve
    bins = np.linspace(-0.5, 0.5, 80)
    bin_idx = np.digitize(ph_s, bins)
    bin_med = [np.nanmedian(fl_s[bin_idx == i]) for i in range(1, len(bins))]
    bin_ctr = 0.5 * (bins[:-1] + bins[1:])
    valid = [not np.isnan(b) for b in bin_med]
    ax2.plot(np.array(bin_ctr)[valid], np.array(bin_med)[valid],
             lw=1.8, color="#f59e0b", zorder=4)

    ax2.axvspan(-dur / period / 2, dur / period / 2, color="#38bdf8", alpha=0.12)
    ax2.axhline(1.0, color="#3d5a7a", lw=0.7, ls="--")
    ax2.set_xlabel(f"Phase  (P = {period:.5f} d)", fontsize=8)
    ax2.set_ylabel("Norm. Flux", fontsize=8)
    ax2.set_xlim(-0.5, 0.5)
    ax2.grid(True, alpha=0.2)

    for ax in (ax1, ax2):
        ax.set_facecolor("#111d2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3352")

    return fig


# ── physics radar ─────────────────────────────────────────────────────────────

def plot_physics_radar(metrics: dict, weights: dict) -> plt.Figure:
    labels = list(metrics.keys())
    scores = [float(np.clip(metrics[k], 0, 1)) for k in labels]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    scores_c = scores + [scores[0]]
    angles_c  = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw={"polar": True},
                           facecolor="#0d1520")
    ax.set_facecolor("#111d2e")

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_c, [r] * (N + 1), color="#1e3352", lw=0.7, zorder=1)
        ax.text(0, r, f"{r:.2f}", color="#3d5a7a", fontsize=5.5, ha="center")

    # Fill
    fill_colors = [score_color(s) for s in scores]
    ax.fill(angles, scores, alpha=0.15, color="#38bdf8", zorder=2)
    ax.plot(angles_c, scores_c, color="#38bdf8", lw=1.5, zorder=3)

    # Dots colored by pass/fail
    for ang, s in zip(angles, scores):
        col = score_color(s)
        ax.plot(ang, s, "o", color=col, ms=6, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(
        [l.replace("_", "\n") for l in labels],
        fontsize=6.5, color="#7fa5c8"
    )
    ax.set_yticks([])
    ax.spines["polar"].set_edgecolor("#1e3352")
    ax.set_title("Physics Metrics", fontsize=9, color="#e2eaf4", pad=18)
    fig.patch.set_facecolor("#0d1520")
    fig.tight_layout()
    return fig


# ── training loss plot ────────────────────────────────────────────────────────

def plot_training_loss(s4: dict) -> plt.Figure:
    train_loss = s4["training"]["train_loss"]
    val_loss   = s4["training"]["validation_loss"]
    epochs     = np.arange(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0d1520")
    ax.set_facecolor("#111d2e")
    ax.plot(epochs, train_loss, color="#38bdf8", lw=1.5, label="Train")
    ax.plot(epochs, val_loss,   color="#f59e0b", lw=1.5, label="Validation", ls="--")
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.set_title("Autoencoder Training Curve", fontsize=9, color="#e2eaf4")
    ax.legend(fontsize=8, facecolor="#111d2e", edgecolor="#1e3352", labelcolor="#e2eaf4")
    ax.grid(True, alpha=0.25)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3352")
    fig.tight_layout()
    return fig


# ── anomaly score plot ────────────────────────────────────────────────────────

def plot_anomaly_scores(s4: dict) -> plt.Figure:
    ad     = s4["anomaly_detection"]
    times  = np.array(ad["window_center_times"])
    errors = np.array(ad["window_errors"])
    is_ano = np.array(ad["window_is_anomaly"])
    thresh = float(ad["threshold"])

    fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0d1520")
    ax.set_facecolor("#111d2e")

    normal_mask = ~is_ano
    ax.scatter(times[normal_mask],  errors[normal_mask],
               s=8, color="#38bdf8", alpha=0.6, label="Normal", zorder=3)
    ax.scatter(times[~normal_mask], errors[~normal_mask],
               s=14, color="#ef4444", alpha=0.9, label="Anomaly", zorder=4)
    ax.axhline(thresh, color="#f59e0b", lw=1.5, ls="--",
               label=f"Threshold ({thresh:.3f})", zorder=2)

    ax.set_xlabel("Time (BTJD)", fontsize=8)
    ax.set_ylabel("Reconstruction Error", fontsize=8)
    ax.set_title("Autoencoder Anomaly Detection", fontsize=9, color="#e2eaf4")
    ax.legend(fontsize=8, facecolor="#111d2e", edgecolor="#1e3352", labelcolor="#e2eaf4")
    ax.grid(True, alpha=0.25)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e3352")
    fig.tight_layout()
    return fig


# ── run pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(tic_id: int, sector: int, start_at: int, end_at: int,
                 log_placeholder):
    """Launch run_pipeline.py as a subprocess, stream output, return (rc, log)."""
    cmd = [
        str(_venv_python()),
        str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--tic", str(tic_id),
        "--sector", str(sector),
        "--start-at", str(start_at),
        "--end-at", str(end_at),
    ]
    log_lines = []
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(REPO_ROOT)
    )
    for line in process.stdout:
        log_lines.append(line.rstrip())
        log_placeholder.code("\n".join(log_lines[-60:]), language=None)
    process.wait()
    return process.returncode, log_lines


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.6rem 0 1rem 0; border-bottom: 1px solid #1e3352; margin-bottom: 1rem;">
        <div style="font-family:'Space Mono',monospace; font-size:0.95rem; font-weight:700;
                    letter-spacing:0.2em; color:#38bdf8;">PHAST</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.58rem; color:#3d5a7a;
                    letter-spacing:0.1em; margin-top:0.15rem;">
            PHYSICS-HARDCODED ANOMALY &amp;<br>SIGNAL TAXONOMY
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.65rem; '
                'color:#7fa5c8; letter-spacing:0.12em; text-transform:uppercase; '
                'margin-bottom:0.5rem;">Target Configuration</div>',
                unsafe_allow_html=True)

    tic_input = st.text_input("TIC ID", value="261136679", placeholder="e.g. 261136679",
                               help="TESS Input Catalog identifier")
    sector    = st.number_input("Sector", min_value=1, max_value=96, value=1, step=1)

    col_a, col_b = st.columns(2)
    with col_a:
        start_at = st.number_input("From stage", 1, 8, 1, help="Resume from stage N")
    with col_b:
        end_at = st.number_input("To stage", 1, 8, 8)

    run_btn = st.button("▶  RUN PIPELINE", use_container_width=True)

    st.markdown("---")
    st.markdown('<div style="font-family:\'Space Mono\',monospace; font-size:0.65rem; '
                'color:#7fa5c8; letter-spacing:0.12em; text-transform:uppercase; '
                'margin-bottom:0.5rem;">Past Runs</div>',
                unsafe_allow_html=True)

    runs = available_runs()
    if runs:
        # Prepend a blank sentinel so the dashboard starts empty
        options = ["— select a run —"] + runs

        active_run_tag_state = st.session_state.get("active_run_tag")
        if active_run_tag_state and active_run_tag_state != st.session_state.get("last_synced_run_tag"):
            st.session_state["run_selectbox"] = active_run_tag_state
            st.session_state["last_synced_run_tag"] = active_run_tag_state

        active_run_tag = st.session_state.get("active_run_tag")
        if active_run_tag and active_run_tag in runs:
            default_idx = options.index(active_run_tag)
        else:
            # No pipeline run yet this session — show blank
            default_idx = 0

        selected_run_raw = st.selectbox("Load run", options,
                                        index=default_idx,
                                        key="run_selectbox",
                                        label_visibility="collapsed")
        selected_run = None if selected_run_raw == "— select a run —" else selected_run_raw
    else:
        selected_run = None
        st.caption("No completed runs yet.")

    st.markdown("---")
    st.markdown(f'<div style="font-family:\'Space Mono\',monospace; font-size:0.6rem; '
                f'color:#3d5a7a; letter-spacing:0.08em;">'
                f'ISRO BAH 2026 · Team ExoScan<br>'
                f'PICT Pune · v2.0</div>', unsafe_allow_html=True)


# ── main area ─────────────────────────────────────────────────────────────────

# Header bar
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M:%S  UTC")
st.markdown(f"""
<div class="phast-header">
    <div>
        <div class="phast-logo">PHAST · EXOPLANET DETECTION PIPELINE</div>
        <div class="phast-sub">Physics-Hardcoded Anomaly & Signal Taxonomy · TESS Light Curve Analysis</div>
    </div>
    <div class="phast-timestamp">{now_str}</div>
</div>
""", unsafe_allow_html=True)


# ── pipeline runner section ───────────────────────────────────────────────────

# Show result from the last completed run (persists across reruns via session state)
if "last_pipeline" in st.session_state:
    r = st.session_state["last_pipeline"]
    if r["rc"] == 0:
        st.success(f"Pipeline complete — TIC {r['tic_id']}  Sector {r['sector']}")
    else:
        st.error(f"Pipeline failed (exit code {r['rc']}) — TIC {r['tic_id']}  Sector {r['sector']}")
        with st.expander("Pipeline log", expanded=True):
            st.code(r["log"], language=None)

if run_btn:
    try:
        tic_id = int(tic_input.strip())
    except ValueError:
        st.error("TIC ID must be an integer.")
        st.stop()

    st.markdown(f"#### Running pipeline for **TIC {tic_id}**  ·  Sector {sector}")
    log_ph = st.empty()

    with st.spinner("Pipeline executing — this may take several minutes per stage…"):
        rc, log_lines = run_pipeline(tic_id, sector, int(start_at), int(end_at), log_ph)

    # Store result and the run_tag so sidebar auto-switches to the new run
    run_tag_completed = f"TIC_{tic_id}_S{int(sector)}"
    st.session_state["last_pipeline"] = {
        "rc": rc,
        "tic_id": tic_id,
        "sector": sector,
        "log": "\n".join(log_lines[-100:]),
    }
    if rc == 0:
        st.session_state["active_run_tag"] = run_tag_completed
        st.session_state["last_synced_run_tag"] = None
    st.rerun()


# ── results section ───────────────────────────────────────────────────────────

if selected_run is None:
    st.markdown("""
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:55vh;gap:1.2rem;text-align:center;">
    <div style="font-family:'Space Mono',monospace;font-size:2.5rem;color:#1e3352;
                letter-spacing:0.1em;">◎</div>
    <div style="font-family:'Space Mono',monospace;font-size:1rem;color:#38bdf8;
                letter-spacing:0.18em;text-transform:uppercase;">PHAST · READY</div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:0.9rem;color:#7fa5c8;
                max-width:400px;line-height:1.6;">
        Enter a TIC ID and sector in the sidebar, then click
        <span style="color:#38bdf8;font-family:'Space Mono',monospace;">▶ RUN PIPELINE</span>
        to begin analysis.
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

run_tag = selected_run
done    = stages_complete(run_tag)
tic_id_display = run_tag.replace("TIC_", "TIC ").replace("_S", "  ·  Sector ")

# Parse TIC and sector from run_tag for display
parts = run_tag.split("_")
tic_num = parts[1] if len(parts) > 1 else "?"
sector_num = parts[2].replace("S","") if len(parts) > 2 else "?"

# Load all available stages
s = {i: load_pkl(run_tag, i) for i in range(1, 9)}

# ── pipeline progress bar ─────────────────────────────────────────────────────
stage_labels = {
    1: "Preprocessing",
    2: "TLS Period Search",
    3: "Physics Pre-filter",
    4: "Autoencoder",
    5: "Physics Validator",
    6: "Oddity Score Ω",
    7: "Parameter Estimation",
    8: "Classification & Report",
}

st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\',monospace;'
            'font-size:0.65rem; letter-spacing:0.14em; color:#7fa5c8; '
            'text-transform:uppercase; margin-bottom:0.6rem;">Pipeline Status</div>',
            unsafe_allow_html=True)

cols = st.columns(8)
for i, col in enumerate(cols, 1):
    with col:
        is_done = done.get(i, False)
        icon  = "✓" if is_done else "○"
        color = "#22c55e" if is_done else "#3d5a7a"
        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem 0.2rem;
                    background:{'#0f2d1a' if is_done else '#0d1520'};
                    border:1px solid {'#22c55e44' if is_done else '#1e3352'};
                    border-radius:4px;">
            <div style="font-family:'Space Mono',monospace; font-size:1rem;
                        color:{color}; font-weight:700;">{icon}</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.55rem;
                        color:{'#22c55e' if is_done else '#3d5a7a'};
                        letter-spacing:0.06em; margin-top:0.15rem;">S{i}</div>
            <div style="font-family:'Space Mono',monospace; font-size:0.52rem;
                        color:#3d5a7a; margin-top:0.1rem;">{stage_labels[i][:9]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Require at least stage 1
if not s[1]:
    st.warning(f"No data found for **{run_tag}**. Run the pipeline first.")
    st.stop()


# ── TABS ──────────────────────────────────────────────────────────────────────
tab_verdict, tab_lc, tab_autoenc, tab_physics, tab_orbit, tab_report, tab_raw = st.tabs([
    "VERDICT",
    "LIGHT CURVE",
    "AUTOENCODER",
    "PHYSICS",
    "ORBITAL PARAMS",
    "REPORT",
    "RAW DATA",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VERDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_verdict:
    if not s[6]:
        st.info("Run stages 1–6 to see the verdict.")
    else:
        s6 = s[6]
        oddity   = float(s6.get("priority_score", s6.get("oddity_score", 0.0)))
        classif  = s6["classification"]
        rec      = s6["recommendation"]
        priority = s6["priority"]
        vc       = verdict_class(classif)

        # — Gauge + verdict side by side —
        col_gauge, col_verdict = st.columns([1, 2])

        with col_gauge:
            fig_gauge = draw_omega_gauge(oddity)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close(fig_gauge)

        with col_verdict:
            vc_color = {"planet": "#22c55e", "maybe": "#f59e0b", "reject": "#ef4444"}[vc]
            st.markdown(f"""
            <div class="verdict-banner verdict-{vc}">
                <div class="verdict-label">Final Classification</div>
                <div class="verdict-title" style="color:{vc_color};">{classif}</div>
                <div class="verdict-sub">{rec}</div>
            </div>
            """, unsafe_allow_html=True)

            col_p, col_q, col_r = st.columns(3)
            col_p.metric("Oddity Score Ω",  f"{oddity:.3f}")
            col_q.metric("Priority",         priority)
            if s[5]:
                col_r.metric("Planet Prob.",
                             f"{float(s[5]['planet_probability']):.4f}")

        st.markdown("---")

        # — Score breakdown —
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        ae_score = float(s6["input_scores"]["autoencoder_score"])
        ph_score = float(s6["input_scores"]["physics_only_score"])
        ae_wt    = float(s6["weighting"]["autoencoder_weight"])
        ph_wt    = float(s6["weighting"]["physics_weight"])

        with col_sc1:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Autoencoder Score Â</div>
                <div style="font-family:'Space Mono',monospace; font-size:2rem;
                            font-weight:700; color:{score_color(ae_score)};">
                    {ae_score:.3f}
                </div>
                <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.3rem;">
                    Weight: {ae_wt:.2f}
                </div>
                <div style="background:#1e3352; border-radius:2px; height:4px; margin-top:0.6rem;">
                    <div style="width:{ae_score*100:.0f}%; height:4px;
                                background:{score_color(ae_score)}; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sc2:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Physics Score P̂</div>
                <div style="font-family:'Space Mono',monospace; font-size:2rem;
                            font-weight:700; color:{score_color(ph_score)};">
                    {ph_score:.3f}
                </div>
                <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.3rem;">
                    Weight: {ph_wt:.2f}
                </div>
                <div style="background:#1e3352; border-radius:2px; height:4px; margin-top:0.6rem;">
                    <div style="width:{ph_score*100:.0f}%; height:4px;
                                background:{score_color(ph_score)}; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_sc3:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Oddity Score Ω</div>
                <div style="font-family:'Space Mono',monospace; font-size:2rem;
                            font-weight:700; color:{score_color(oddity)};">
                    {oddity:.3f}
                </div>
                <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.3rem;">
                    Mode: {s6['weighting']['mode'].replace('_',' ')}
                </div>
                <div style="background:#1e3352; border-radius:2px; height:4px; margin-top:0.6rem;">
                    <div style="width:{oddity*100:.0f}%; height:4px;
                                background:{score_color(oddity)}; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # — Target info + blend/centroid row —
        if s[5]:
            st.markdown("---")
            col_t, col_b, col_c = st.columns(3)
            with col_t:
                st.markdown(f"""
                <div class="phast-card">
                    <div class="phast-card-title">Target</div>
                    <div style="font-family:'Space Mono',monospace; color:#38bdf8;
                                font-size:1.1rem; font-weight:700;">TIC {s[5].get('tic_id','?')}</div>
                    <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.4rem;">
                        RA  {s[5].get('ra', 0):.5f}°<br>
                        Dec {s[5].get('dec', 0):.5f}°<br>
                        CROWDSAP {s[1].get('crowdsap', 0):.5f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                br  = s[5].get("blend_risk", "—")
                brf = s[5].get("blend_risk_flag", "—")
                bc  = "#22c55e" if brf == "LOW" else "#ef4444"
                st.markdown(f"""
                <div class="phast-card">
                    <div class="phast-card-title">Blend Risk</div>
                    <div style="font-family:'Space Mono',monospace; color:{bc};
                                font-size:1.1rem; font-weight:700;">{brf}</div>
                    <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.4rem;">
                        Level: {br}<br>
                        Contamination: {(1-float(s[1].get('crowdsap',1)))*100:.3f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_c:
                csh = float(s[5].get("centroid_shift", 0))
                cscore = float(s[5].get("centroid_score", 0))
                cflag = s[5].get("centroid_flag", "—")
                cc = "#22c55e" if cflag == "CLEAR" else "#ef4444"
                st.markdown(f"""
                <div class="phast-card">
                    <div class="phast-card-title">Centroid Check</div>
                    <div style="font-family:'Space Mono',monospace; color:{cc};
                                font-size:1.1rem; font-weight:700;">{cflag}</div>
                    <div style="font-size:0.78rem; color:#7fa5c8; margin-top:0.4rem;">
                        Shift: {csh:.4f} px<br>
                        Score: {cscore:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # — Strengths & concerns —
        if s[5]:
            strengths = s[5].get("strengths", [])
            concerns  = s[5].get("concerns", [])
            st.markdown("---")
            col_str, col_con = st.columns(2)
            with col_str:
                st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\','
                            'monospace;font-size:0.65rem;letter-spacing:0.14em;color:#22c55e;'
                            'text-transform:uppercase;margin-bottom:0.4rem;">Strengths</div>',
                            unsafe_allow_html=True)
                tags = "".join(f'<span class="tag-green">{x.replace("_"," ")}</span>'
                               for x in strengths)
                st.markdown(tags, unsafe_allow_html=True)
            with col_con:
                st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\','
                            'monospace;font-size:0.65rem;letter-spacing:0.14em;color:#ef4444;'
                            'text-transform:uppercase;margin-bottom:0.4rem;">Concerns</div>',
                            unsafe_allow_html=True)
                tags = "".join(f'<span class="tag-red">{x.replace("_"," ")}</span>'
                               for x in concerns)
                st.markdown(tags or '<span class="tag-green">None</span>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIGHT CURVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_lc:
    if not s[1] or not s[2]:
        st.info("Requires stages 1 and 2.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Period",    f"{float(s[2]['period']):.5f} d")
        col_b.metric("TLS SDE",   f"{float(s[2]['sde']):.2f}")
        col_c.metric("TLS SNR",   f"{float(s[2]['snr_tls']):.2f}")
        col_d.metric("Transits",  str(s[2]["distinct_transit_count"]))
        st.markdown("<br>", unsafe_allow_html=True)
        fig_lc = plot_lightcurve(s[1], s[2])
        st.pyplot(fig_lc, use_container_width=True)
        plt.close(fig_lc)

        st.markdown("---")
        col_m, col_n = st.columns(2)
        with col_m:
            st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\','
                        'monospace;font-size:0.65rem;letter-spacing:0.14em;color:#7fa5c8;'
                        'text-transform:uppercase;margin-bottom:0.4rem;">TLS Metrics</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="phast-card" style="font-family:'Space Mono',monospace; font-size:0.8rem;">
                Period &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {float(s[2]['period']):.6f} days<br>
                Duration &nbsp;&nbsp;&nbsp;&nbsp; {float(s[2]['duration'])*24:.4f} hours<br>
                T0 &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {float(s[2]['t0']):.5f} BTJD<br>
                Depth (TLS) &nbsp; {(1-float(s[2]['depth_tls']))*100:.4f}%<br>
                Odd/Even σ &nbsp;&nbsp; {float(s[2]['odd_even_mismatch_tls']):.3f}
            </div>
            """, unsafe_allow_html=True)

        if s[3]:
            with col_n:
                st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\','
                            'monospace;font-size:0.65rem;letter-spacing:0.14em;color:#7fa5c8;'
                            'text-transform:uppercase;margin-bottom:0.4rem;">Pre-filter</div>',
                            unsafe_allow_html=True)
                flags = s[3]["flags"]
                dec   = s[3]["decision"]
                dec_color = "#22c55e" if dec == "PASSED" else "#ef4444"
                flag_html = ""
                for fname, fval in flags.items():
                    fc = "#ef4444" if fval else "#22c55e"
                    ftext = "FLAGGED" if fval else "CLEAR"
                    flag_html += (f'<div style="display:flex;justify-content:space-between;'
                                  f'margin-bottom:0.2rem;">'
                                  f'<span style="color:#7fa5c8;">{fname.replace("_"," ")}</span>'
                                  f'<span style="color:{fc};font-weight:700;">{ftext}</span></div>')
                st.markdown(f"""
                <div class="phast-card" style="font-family:'Space Mono',monospace; font-size:0.78rem;">
                    <div style="font-size:1rem;font-weight:700;color:{dec_color};
                                margin-bottom:0.6rem;">{dec}</div>
                    {flag_html}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — AUTOENCODER
# ══════════════════════════════════════════════════════════════════════════════
with tab_autoenc:
    if not s[4]:
        st.info("Requires stage 4.")
    else:
        s4 = s[4]
        ad = s4["anomaly_detection"]
        tr = s4["training"]
        rl = s4["reliability"]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Anomaly Score",  f"{float(ad['candidate_anomaly_score']):.4f}")
        col_b.metric("Detected",       "Yes" if ad["candidate_detected"] else "No")
        col_c.metric("Epochs trained", str(tr["completed_epochs"]))
        col_d.metric("Best val loss",  f"{float(tr['best_validation_loss']):.5f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_loss, col_ano = st.columns([1, 1.6])
        with col_loss:
            fig_loss = plot_training_loss(s4)
            st.pyplot(fig_loss, use_container_width=True)
            plt.close(fig_loss)
        with col_ano:
            fig_ano = plot_anomaly_scores(s4)
            st.pyplot(fig_ano, use_container_width=True)
            plt.close(fig_ano)

        st.markdown("---")
        col_w, col_x = st.columns(2)
        with col_w:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Windowing</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.78rem;">
                    Window size &nbsp;&nbsp;&nbsp; {s4['windowing']['window_size']} pts<br>
                    Total windows &nbsp;&nbsp; {s4['windowing']['total_windows']}<br>
                    Train windows &nbsp;&nbsp; {s4['windowing']['training_windows']}<br>
                    Candidate windows {s4['windowing']['candidate_windows']}<br>
                    Cadence &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {s4['windowing']['cadence_minutes']:.2f} min
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_x:
            rel_ok = rl.get("autoencoder_reliable", False)
            rel_c  = "#22c55e" if rel_ok else "#ef4444"
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Reliability</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.78rem;">
                    <span style="color:{rel_c};font-weight:700;">
                        {'RELIABLE' if rel_ok else 'UNRELIABLE'}
                    </span><br><br>
                    Injection detected &nbsp; {'Yes' if rl.get('injection_detected') else 'No'}<br>
                    High variability &nbsp;&nbsp; {'Yes' if rl.get('high_variability_warning') else 'No'}<br>
                    Normal scatter &nbsp;&nbsp;&nbsp; {float(rl.get('normal_scatter',0)):.6f}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PHYSICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_physics:
    if not s[5]:
        st.info("Requires stage 5.")
    else:
        s5 = s[5]
        metrics = s5["physics_metrics"]
        weights = s5.get("weights", {})

        col_r, col_t = st.columns([1.2, 1])
        with col_r:
            fig_rad = plot_physics_radar(metrics, weights)
            st.pyplot(fig_rad, use_container_width=True)
            plt.close(fig_rad)

        with col_t:
            st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\','
                        'monospace;font-size:0.65rem;letter-spacing:0.14em;color:#7fa5c8;'
                        'text-transform:uppercase;margin-bottom:0.4rem;">Rule Scorecard</div>',
                        unsafe_allow_html=True)
            PASS_THRESHOLD = 0.5
            for metric, score in sorted(metrics.items(),
                                        key=lambda x: float(x[1]), reverse=True):
                sc = float(np.clip(score, 0, 1))
                is_pass = sc >= PASS_THRESHOLD
                bar_col = score_color(sc)
                wt = float(weights.get(metric, 1))
                label = metric.replace("_", " ").title()
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin-bottom:0.35rem;gap:0.5rem;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                                color:#7fa5c8;width:160px;flex-shrink:0;">{label}</div>
                    <div style="flex:1;background:#1e3352;border-radius:2px;height:6px;">
                        <div style="width:{sc*100:.0f}%;height:6px;background:{bar_col};
                                    border-radius:2px;"></div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                                color:{bar_col};width:42px;text-align:right;">{sc:.3f}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                color:{'#22c55e' if is_pass else '#ef4444'};width:32px;">
                        {'✓' if is_pass else '✗'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        col_pp, col_fpp, col_bfpp = st.columns(3)
        col_pp.metric("Planet Probability",  f"{float(s5['planet_probability']):.6f}")
        col_fpp.metric("Approx FPP",         f"{float(s5['approx_fpp']):.5f}")
        col_bfpp.metric("Bayesian FPP",      f"{float(s5['bayesian_fpp']):.2e}")

        if s5.get("interpretation"):
            st.markdown("---")
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Interpretation</div>
                <div style="font-size:0.9rem; color:#e2eaf4; line-height:1.6;">
                    {s5['interpretation']}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ORBITAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with tab_orbit:
    if not s[7]:
        st.info("Requires stage 7 (BATMAN + MCMC parameter estimation).")
    else:
        s7 = s[7]

        # Key metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Period",         f"{float(s7['period']):.6f} d")
        c2.metric("Transit Depth",  f"{float(s7['transit_depth'])*100:.4f}%")
        c3.metric("Duration",       f"{float(s7['duration'])*24:.3f} h")
        c4.metric("Reduced χ²",     f"{float(s7['reduced_chi2']):.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        col_params, col_mcmc = st.columns(2)

        with col_params:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">BATMAN Fitted Parameters</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.82rem;
                            line-height:2.0;">
                    <div style="display:flex;justify-content:space-between;
                                border-bottom:1px solid #1e3352;padding-bottom:0.3rem;
                                margin-bottom:0.3rem;">
                        <span style="color:#7fa5c8;">Parameter</span>
                        <span style="color:#7fa5c8;">Value</span>
                        <span style="color:#7fa5c8;">Unit</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">Orbital Period</span>
                        <span style="color:#38bdf8;">{float(s7['period']):.6f}</span>
                        <span style="color:#7fa5c8;">days</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">Transit Epoch T₀</span>
                        <span style="color:#38bdf8;">{float(s7['t0']):.5f}</span>
                        <span style="color:#7fa5c8;">BTJD</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">Rp/Rs</span>
                        <span style="color:#38bdf8;">
                            {float(s7['rp_rs']):.5f}
                            <span style="color:#3d5a7a;font-size:0.7rem;">
                                +{float(s7['rp_rs_upper']):.5f}/−{float(s7['rp_rs_lower']):.5f}
                            </span>
                        </span>
                        <span style="color:#7fa5c8;">—</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">a/Rs</span>
                        <span style="color:#38bdf8;">
                            {float(s7['a_rs']):.4f}
                            <span style="color:#3d5a7a;font-size:0.7rem;">
                                +{float(s7['a_rs_upper']):.4f}/−{float(s7['a_rs_lower']):.4f}
                            </span>
                        </span>
                        <span style="color:#7fa5c8;">—</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">Inclination</span>
                        <span style="color:#38bdf8;">
                            {float(s7['inclination']):.3f}
                            <span style="color:#3d5a7a;font-size:0.7rem;">
                                +{float(s7['inclination_upper']):.3f}/−{float(s7['inclination_lower']):.3f}
                            </span>
                        </span>
                        <span style="color:#7fa5c8;">deg</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#e2eaf4;">Impact Parameter b</span>
                        <span style="color:#38bdf8;">{float(s7['impact_parameter']):.4f}</span>
                        <span style="color:#7fa5c8;">—</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_mcmc:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">MCMC Diagnostics</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.82rem;
                            line-height:2.0;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">RMSE</span>
                        <span style="color:#38bdf8;">{float(s7['rmse']):.6f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Reduced χ²</span>
                        <span style="color:#38bdf8;">{float(s7['reduced_chi2']):.4f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Acceptance Fraction</span>
                        <span style="color:#38bdf8;">{float(s7['acceptance_fraction']):.4f}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">MCMC Samples</span>
                        <span style="color:#38bdf8;">{len(s7['samples'])}</span>
                    </div>
                </div>
                <div style="margin-top:1rem;">
                    <div class="phast-card-title">Classification</div>
                    <span class="tag-{'green' if 'PLANET' in s7['classification'] else 'amber'}">
                        {s7['classification']}
                    </span>
                    <span class="tag-{'green' if s7['priority']=='HIGH' else 'amber'}">
                        PRIORITY: {s7['priority']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # MCMC posterior scatter (rp_rs samples)
        if s7.get("samples") is not None:
            samples = np.array(s7["samples"])
            if samples.ndim == 2 and samples.shape[1] >= 2:
                fig_post, ax_post = plt.subplots(figsize=(8, 3), facecolor="#0d1520")
                ax_post.set_facecolor("#111d2e")
                ax_post.hist(samples[:, 0], bins=60, color="#38bdf8",
                             alpha=0.75, edgecolor="none")
                ax_post.axvline(float(s7["rp_rs"]), color="#f59e0b",
                                lw=1.8, ls="--", label=f"Rp/Rs = {float(s7['rp_rs']):.5f}")
                ax_post.set_xlabel("Rp/Rs", fontsize=8)
                ax_post.set_ylabel("Count", fontsize=8)
                ax_post.set_title("MCMC Posterior — Radius Ratio Rp/Rs",
                                  fontsize=9, color="#e2eaf4")
                ax_post.legend(fontsize=8, facecolor="#111d2e",
                               edgecolor="#1e3352", labelcolor="#e2eaf4")
                for sp in ax_post.spines.values(): sp.set_edgecolor("#1e3352")
                st.pyplot(fig_post, use_container_width=True)
                plt.close(fig_post)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    s8  = s[8]
    s1r = s[1] or {}; s2r = s[2] or {}; s3r = s[3] or {}
    s5r = s[5] or {}; s6r = s[6] or {}; s7r = s[7] or {}

    if not s8 and not s[6]:
        st.info("Run all 8 stages to generate the full report.")
    else:
        tic_id_r      = s8.get("tic_id") if s8 else s1r.get("tic_id", "?")
        classif_r     = s8.get("classification") if s8 else s6r.get("classification", "—")
        reason_r      = s8.get("classification_reason") if s8 else "—"
        action_r      = s8.get("recommended_action") if s8 else s6r.get("recommendation", "—")
        oddity_r      = float(s8.get("oddity_score", 0) if s8 else s6r.get("priority_score", s6r.get("oddity_score", 0)))
        ae_score_r    = float(s8.get("autoencoder_score", 0) if s8 else s6r.get("input_scores", {}).get("autoencoder_score", 0))
        ph_score_r    = float(s8.get("physics_only_score", 0) if s8 else s6r.get("input_scores", {}).get("physics_only_score", 0))
        planet_prob_r = float(s8.get("planet_probability", 0) if s8 else s5r.get("planet_probability", 0))
        bayesian_fpp_r= float(s8.get("bayesian_fpp", 0) if s8 else s5r.get("bayesian_fpp", 0))
        period_r      = float(s8.get("period", 0) if s8 else s2r.get("period", 0))
        depth_pct_r   = float(s8.get("depth_pct", 0) if s8 else (1 - float(s2r.get("depth_tls", 1))) * 100)
        dur_h_r       = float(s8.get("duration_hours", 0) if s8 else float(s2r.get("duration", 0)) * 24)
        rp_rs_r       = float(s8.get("rp_rs", 0) if s8 else s7r.get("rp_rs", 0))
        t0_r          = float(s8.get("t0", 0) if s8 else s2r.get("t0", 0))
        incl_r        = float(s8.get("inclination", 0) if s8 else s7r.get("inclination", 0))
        a_rs_r        = float(s8.get("a_rs", 0) if s8 else s7r.get("a_rs", 0))
        impact_r      = float(s8.get("impact_parameter", 0) if s8 else s7r.get("impact_parameter", 0))
        chi2_r        = float(s8.get("reduced_chi2", 0) if s8 else s7r.get("reduced_chi2", 0))
        mcmc_acc_r    = float(s8.get("acceptance_fraction", 0) if s8 else s7r.get("acceptance_fraction", 0))
        snr_r         = float(s2r.get("snr_tls", 0))
        transit_ct_r  = int(s2r.get("distinct_transit_count", 0))
        stage3_dec_r  = s8.get("stage3_decision") if s8 else s3r.get("decision", "—")
        n_passed_r    = int(s8.get("n_rules_passed", 0) if s8 else 0)
        n_total_r     = int(s8.get("n_rules_total", 0) if s8 else 0)
        blend_r       = s8.get("blend_risk_level") if s8 else s5r.get("blend_risk", "—")
        centroid_r    = float(s8.get("centroid_shift", 0) if s8 else s5r.get("centroid_shift", 0))
        priority_r    = s8.get("priority") if s8 else s6r.get("priority", "—")
        weighting_r   = s8.get("weighting_mode") if s8 else s6r.get("weighting", {}).get("mode", "—")
        timestamp_r   = s8.get("timestamp") if s8 else datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        strengths_r   = s5r.get("strengths", [])
        concerns_r    = s5r.get("concerns", [])
        physics_metrics_r = s8.get("physics_metrics") if s8 else s5r.get("physics_metrics", {})

        vc_r = verdict_class(classif_r)
        vc_color_r = {"planet": "#22c55e", "maybe": "#f59e0b", "reject": "#ef4444"}[vc_r]

        # Report header
        st.markdown(f"""
        <div style="background:#0d1520;border:1px solid #1e3352;border-radius:8px;
                    padding:1.6rem 2rem;margin-bottom:1.2rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                letter-spacing:0.18em;color:#3d5a7a;text-transform:uppercase;
                                margin-bottom:0.3rem;">PHAST PIPELINE v2.0 — CANDIDATE REPORT</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                color:#3d5a7a;margin-bottom:0.8rem;">
                        Physics-Hardcoded Anomaly & Signal Taxonomy · ISRO BAH 2026
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:1.6rem;
                                font-weight:700;color:#38bdf8;letter-spacing:0.06em;">
                        TIC {tic_id_r}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                                color:#3d5a7a;letter-spacing:0.1em;">Generated</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                                color:#7fa5c8;margin-top:0.2rem;">{timestamp_r}</div>
                    <div style="margin-top:0.8rem;font-family:'Space Mono',monospace;
                                font-size:0.72rem;color:#3d5a7a;">Priority</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.9rem;
                                color:{'#22c55e' if priority_r=='HIGH' else '#f59e0b'};
                                font-weight:700;">{priority_r}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Classification banner
        st.markdown(f"""
        <div class="verdict-banner verdict-{vc_r}">
            <div class="verdict-label">Final Classification</div>
            <div class="verdict-title" style="color:{vc_color_r};">{classif_r}</div>
            <div class="verdict-sub">{reason_r}</div>
            <div style="margin-top:0.6rem;font-family:'Space Mono',monospace;
                        font-size:0.82rem;color:{vc_color_r};font-weight:600;">
                ▶ {action_r}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Score row
        c1r,c2r,c3r,c4r,c5r = st.columns(5)
        c1r.metric("Oddity Score Ω", f"{oddity_r:.3f}")
        c2r.metric("Autoencoder Â",  f"{ae_score_r:.3f}")
        c3r.metric("Physics P̂",      f"{ph_score_r:.3f}")
        c4r.metric("Planet Prob.",    f"{planet_prob_r:.5f}")
        c5r.metric("Bayesian FPP",   f"{bayesian_fpp_r:.2e}")

        st.markdown("---")

        col_rl, col_rr = st.columns(2)
        with col_rl:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Orbital Parameters (BATMAN + MCMC)</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.8rem;line-height:2.1;">
                    <div style="display:flex;justify-content:space-between;border-bottom:1px solid #1e3352;padding-bottom:0.2rem;margin-bottom:0.2rem;">
                        <span style="color:#7fa5c8;">Parameter</span><span style="color:#7fa5c8;">Value</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Period</span><span style="color:#38bdf8;">{period_r:.5f} days</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">T₀</span><span style="color:#38bdf8;">{t0_r:.5f} BTJD</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Transit Depth</span><span style="color:#38bdf8;">{depth_pct_r:.4f}%</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Duration</span><span style="color:#38bdf8;">{dur_h_r:.3f} h</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Rp/Rs</span><span style="color:#38bdf8;">{rp_rs_r:.5f}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">a/Rs</span><span style="color:#38bdf8;">{a_rs_r:.4f}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Inclination</span><span style="color:#38bdf8;">{incl_r:.3f}°</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Impact Param b</span><span style="color:#38bdf8;">{impact_r:.4f}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">TLS SNR</span><span style="color:#38bdf8;">{snr_r:.2f}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Transit Count</span><span style="color:#38bdf8;">{transit_ct_r}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">Reduced χ²</span><span style="color:#38bdf8;">{chi2_r:.3f}</span></div>
                    <div style="display:flex;justify-content:space-between;"><span style="color:#e2eaf4;">MCMC Acceptance</span><span style="color:#38bdf8;">{mcmc_acc_r:.3f}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_rr:
            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Physics Validation</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.8rem;line-height:2.1;">
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Pre-filter</span>
                        <span style="color:{'#22c55e' if stage3_dec_r=='PASSED' else '#ef4444'};font-weight:700;">{stage3_dec_r}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Physics Rules</span>
                        <span style="color:#38bdf8;">{n_passed_r}/{n_total_r} passed</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Centroid Shift</span>
                        <span style="color:#38bdf8;">{centroid_r:.4f} px</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Blend Risk</span>
                        <span style="color:{'#22c55e' if str(blend_r).upper() in ('LOW',) else '#ef4444'};">{blend_r}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;">
                        <span style="color:#7fa5c8;">Weighting Mode</span>
                        <span style="color:#7fa5c8;font-size:0.72rem;">{str(weighting_r).replace("_"," ")}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="phast-card">
                <div class="phast-card-title">Evidence Summary</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                            color:#22c55e;letter-spacing:0.1em;margin-bottom:0.3rem;">STRENGTHS</div>
                {"".join(f'<span class="tag-green">{x.replace("_"," ")}</span>' for x in strengths_r) or '<span style="color:#3d5a7a;font-size:0.75rem;">None recorded</span>'}
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                            color:#ef4444;letter-spacing:0.1em;margin-top:0.6rem;margin-bottom:0.3rem;">CONCERNS</div>
                {"".join(f'<span class="tag-red">{x.replace("_"," ")}</span>' for x in concerns_r) or '<span class="tag-green">None</span>'}
            </div>
            """, unsafe_allow_html=True)

        # Physics scorecard
        if physics_metrics_r:
            st.markdown("---")
            st.markdown(
                '<div class="phast-card-title" style="font-family:&quot;Space Mono&quot;,monospace;'
                'font-size:0.65rem;letter-spacing:0.14em;color:#7fa5c8;text-transform:uppercase;'
                'margin-bottom:0.6rem;">Physics Rule Scorecard</div>',
                unsafe_allow_html=True)
            rows = sorted(physics_metrics_r.items(), key=lambda x: float(x[1]), reverse=True)
            mid = (len(rows) + 1) // 2
            colA, colB = st.columns(2)
            for col_mc, chunk in [(colA, rows[:mid]), (colB, rows[mid:])]:
                with col_mc:
                    for metric, score in chunk:
                        sc = float(np.clip(score, 0, 1))
                        bar_col = score_color(sc)
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;margin-bottom:0.3rem;gap:0.4rem;">
                            <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                                        color:#7fa5c8;width:150px;flex-shrink:0;">{metric.replace("_"," ").title()}</div>
                            <div style="flex:1;background:#1e3352;border-radius:2px;height:5px;">
                                <div style="width:{sc*100:.0f}%;height:5px;background:{bar_col};border-radius:2px;"></div>
                            </div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.67rem;
                                        color:{bar_col};width:38px;text-align:right;">{sc:.3f}</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.67rem;
                                        color:{'#22c55e' if sc>=0.5 else '#ef4444'};width:16px;">
                                {'✓' if sc>=0.5 else '✗'}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Download button
        st.markdown("---")
        report_txt = f"""PHAST PIPELINE v2.0 — CANDIDATE REPORT
Physics-Hardcoded Anomaly and Signal Taxonomy · ISRO BAH 2026
Generated : {timestamp_r}
{'='*72}
TIC ID                    : {tic_id_r}
Classification            : {classif_r}
Reason                    : {reason_r}
Recommended Action        : {action_r}
Priority                  : {priority_r}
{'-'*72}
SCORES
{'-'*72}
Oddity Score Omega        : {oddity_r:.4f}
Autoencoder Score         : {ae_score_r:.4f}
Physics Score             : {ph_score_r:.4f}
Planet Probability        : {planet_prob_r:.6f}
Bayesian FPP              : {bayesian_fpp_r:.4e}
{'-'*72}
ORBITAL PARAMETERS
{'-'*72}
Period                    : {period_r:.5f} days
T0                        : {t0_r:.5f} BTJD
Transit Depth             : {depth_pct_r:.4f}%
Duration                  : {dur_h_r:.3f} hours
Rp/Rs                     : {rp_rs_r:.5f}
a/Rs                      : {a_rs_r:.4f}
Inclination               : {incl_r:.3f} deg
Impact Parameter b        : {impact_r:.4f}
TLS SNR                   : {snr_r:.2f}
Transit Count             : {transit_ct_r}
Reduced Chi2              : {chi2_r:.3f}
MCMC Acceptance           : {mcmc_acc_r:.3f}
{'-'*72}
PHYSICS
{'-'*72}
Pre-filter                : {stage3_dec_r}
Rules Passed              : {n_passed_r}/{n_total_r}
Centroid Shift            : {centroid_r:.4f} px
Blend Risk                : {blend_r}
Weighting Mode            : {weighting_r}
{'-'*72}
EVIDENCE
{'-'*72}
Strengths  : {', '.join(strengths_r) or 'None'}
Concerns   : {', '.join(concerns_r) or 'None'}
{'='*72}
PHAST · Team ExoScan · PICT Pune
"""
        st.download_button(
            label="⬇  Download Report (.txt)",
            data=report_txt,
            file_name=f"TIC_{tic_id_r}_PHAST_report.txt",
            mime="text/plain",
        )

        # Show saved report file from stage 8 if it exists
        for candidate in [
            REPO_ROOT / "reports" / f"TIC_{tic_id_r}_candidate_report.txt",
            REPO_ROOT / "reports" / f"{run_tag}_report.txt",
        ]:
            if candidate.exists():
                with st.expander(f"Stage 8 saved report: {candidate.name}", expanded=False):
                    st.code(candidate.read_text(), language=None)
                break


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RAW DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab_raw:
    st.markdown('<div class="phast-card-title" style="font-family:\'Space Mono\',monospace;'
                'font-size:0.65rem;letter-spacing:0.14em;color:#7fa5c8;text-transform:uppercase;'
                'margin-bottom:0.8rem;">Stage Output Summary</div>',
                unsafe_allow_html=True)

    for stage_num in range(1, 9):
        data = s[stage_num]
        is_done = data is not None
        with st.expander(
            f"{'✓' if is_done else '○'}  Stage {stage_num} — {stage_labels[stage_num]}",
            expanded=False
        ):
            if not is_done:
                st.caption("Not yet run.")
            else:
                if isinstance(data, dict):
                    for k, v in data.items():
                        vtype = type(v).__name__
                        if isinstance(v, dict):
                            st.markdown(f'`{k}` → **{vtype}** with keys: '
                                        f'`{list(v.keys())}`')
                        elif hasattr(v, "__len__") and not isinstance(v, str):
                            st.markdown(f'`{k}` → **{vtype}**[{len(v)}]')
                        else:
                            st.markdown(f'`{k}` → `{repr(v)[:120]}`')
                else:
                    st.write(data)