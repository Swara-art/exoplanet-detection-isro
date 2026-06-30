"""
run_pipeline.py  —  PHAST per-star pipeline runner
====================================================
Run the full 8-stage PHAST pipeline for any TIC target.

Usage
-----
    python scripts/run_pipeline.py --tic 261136679
    python scripts/run_pipeline.py --tic 261136679 --sector 1
    python scripts/run_pipeline.py --tic 261136679 --sector 1 --start-at 3
    python scripts/run_pipeline.py --tic 261136679 --stage 4

Each run gets its own output folder:
    data/TIC_261136679_S1/stage1_output.pkl  ...stage8_output.pkl
    reports/TIC_261136679_S1_report.txt
    plots/TIC_261136679_S1/  ...

Re-running the same TIC from a later stage reads that star's existing pkl files,
so you never re-download TESS data you already have.
"""

import asyncio
import argparse
import json
import os
import subprocess as _sp
import sys
import time
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# Fix Windows asyncio/zmq warning
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Prefer the repo's .venv Python so pipeline always uses installed packages,
# regardless of which Python launched the dashboard or this script.
def _venv_python() -> str:
    candidates = [
        os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),  # Windows
        os.path.join(REPO_ROOT, ".venv", "bin", "python"),          # macOS/Linux
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return sys.executable  # fallback: whatever is currently running

PYTHON_EXE = _venv_python()


def ensure_venv_kernel() -> str:
    """Register the venv Python as a Jupyter kernel and return its name.
    Falls back to 'python3' if registration fails."""
    kernel_name = "phast_venv"
    try:
        r = _sp.run(
            [PYTHON_EXE, "-m", "jupyter", "kernelspec", "list", "--json"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            specs = json.loads(r.stdout).get("kernelspecs", {})
            if kernel_name in specs:
                return kernel_name
        _sp.run(
            [PYTHON_EXE, "-m", "ipykernel", "install",
             "--user", "--name", kernel_name, "--display-name", "PHAST (.venv)"],
            check=True, capture_output=True, timeout=60
        )
        print(f"[pipeline] registered kernel '{kernel_name}' → {PYTHON_EXE}", flush=True)
        return kernel_name
    except Exception as exc:
        print(f"[pipeline] kernel registration failed ({exc}), falling back to 'python3'", flush=True)
        return "python3"


STAGE_NOTEBOOKS = {
    1: "notebooks/stage1_preprocessing.ipynb",
    2: "notebooks/stage2_TLS.ipynb",
    3: "notebooks/stage3_physics_filters.ipynb",
    4: "notebooks/stage4_autoencoder.ipynb",
    5: "notebooks/stage5_Full_Physics_Validator.ipynb",
    6: "notebooks/stage6_Adaptive_Candidate_Priority.ipynb",
    7: "notebooks/stage7_Parameter_Estimation.ipynb",
    8: "notebooks/stage8_ClassificationandReporting.ipynb",
}


def make_run_tag(tic_id: int, sector: int) -> str:
    """Unique tag for this star+sector run, used in folder names."""
    return f"TIC_{tic_id}_S{sector}"


def ensure_folders(run_tag: str):
    """Create per-star output folders so runs never collide."""
    for folder in [
        f"data/{run_tag}",
        f"plots/{run_tag}",
        f"checkpoints/{run_tag}",
        "reports",
        "models",
    ]:
        os.makedirs(os.path.join(REPO_ROOT, folder), exist_ok=True)


def _load_pkl(run_tag: str, stage: int):
    p = os.path.join(REPO_ROOT, "data", run_tag, f"stage{stage}_output.pkl")
    if not os.path.exists(p):
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _write_fallback_pkl(run_tag: str, stage: int, tic_id: int, sector: int, reason: str):
    """Write a stage-specific fallback pkl with safe defaults so Stage 8 can still run."""
    import pickle
    import math
    p = os.path.join(REPO_ROOT, "data", run_tag, f"stage{stage}_output.pkl")

    base = {"status": "insufficient_data", "tic_id": tic_id, "sector": sector, "reason": reason}

    stage_defaults = {
        3: {
            "decision": "PASS",
            "flags": {},
            "metrics": {},
            "rejection_reasons": [],
            "warning_reasons": [f"Insufficient data: {reason}"],
        },
        4: {
            "autoencoder_score": 0.0,
            "anomaly_score": 0.0,
            "reconstruction_error": 0.0,
            "threshold": 0.0,
            "is_anomaly": False,
            "reliable": False,
        },
        5: {
            "physics_score": 0.0,
            "planet_probability": 0.0,
            "bayesian_fpp": 1.0,
            "approx_fpp": 1.0,
            "confidence": 0.0,
            "physics_metrics": {},
            "weights": {},
            "strengths": [],
            "concerns": [f"Insufficient data: {reason}"],
            "blend_risk_flag": "UNKNOWN",
            "centroid_shift": math.nan,
            "centroid_score": 0.5,
            "pixel_score": 0.5,
            "tpf_available": False,
        },
        6: {
            "oddity_score": 0.0,
            "classification": "INSUFFICIENT DATA",
            "recommendation": "Acquire additional sectors before analysis.",
            "priority": "LOW",
            "priority_score": 0.0,
            "weighting": {"mode": "default", "autoencoder_weight": 0.5, "physics_weight": 0.5},
            "input_scores": {"autoencoder_score": 0.0, "physics_only_score": 0.0},
            "adaptive_context": {
                "blend_risk_level": "LOW",
                "variability_class": "QUIET",
                "autoencoder_reliable": False,
            },
            "warnings": [f"Insufficient data: {reason}"],
        },
        7: {
            "rp_rs": 0.0, "rp_rs_lower": 0.0, "rp_rs_upper": 0.0,
            "a_rs": 0.0, "a_rs_lower": 0.0, "a_rs_upper": 0.0,
            "inclination": 90.0, "inclination_lower": 0.0, "inclination_upper": 0.0,
            "impact_parameter": 0.0,
            "transit_depth": 0.0,
            "reduced_chi2": 0.0,
            "rmse": 0.0,
            "acceptance_fraction": 0.0,
            "priority_score": 0.0,
        },
    }

    payload = {**base, **stage_defaults.get(stage, {})}
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    print(f"  [fallback] Wrote insufficient-data placeholder for stage {stage}.", flush=True)


def build_bootstrap_cell(tic_id: int, sector: int, run_tag: str) -> str:
    """
    Returns Python source that is injected as the very first cell of every
    notebook before execution. It does three things:
      1. Mocks google.colab and remaps /content/drive/... paths to local repo.
      2. Overrides PIPELINE_CONFIG so the notebook targets the requested star.
      3. Redirects DATA_DIR / CHECKPOINT_DIR / PLOT_DIR to the per-star subfolder
         so outputs are isolated and never overwrite a previous run.
    """
    # Use forward slashes even on Windows (Python's open() handles it fine)
    repo = REPO_ROOT.replace("\\", "/")
    return f"""\
# ── injected by run_pipeline.py ─────────────────────────────────────────────
import asyncio, sys, os, types, builtins

# Fix Windows async warning inside the kernel too
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ── 1. mock google.colab ────────────────────────────────────────────────────
_g = types.ModuleType("google")
_c = types.ModuleType("google.colab")
_d = types.ModuleType("google.colab.drive")
_d.mount = lambda *a, **k: print("[pipeline] mock drive.mount — using local repo")
_c.drive = _d; _g.colab = _c
sys.modules.update({{"google": _g, "google.colab": _c, "google.colab.drive": _d}})

# ── 2. remap hardcoded Colab paths → local repo / per-star subfolder ────────
COLAB_PREFIX = "/content/drive/MyDrive/exoplanet_pipeline"
RUN_ROOT     = "{repo}/data/{run_tag}"   # per-star data folder
REPO_ROOT    = "{repo}"

def _remap(p):
    try: p = os.fspath(p)
    except TypeError: return p
    if not isinstance(p, str): return p
    # Normalize to forward slashes — Windows os.path.join produces backslashes
    # which break startswith comparisons against the forward-slash COLAB_PREFIX.
    p = p.replace("\\\\", "/")
    if p.startswith(COLAB_PREFIX + "/data"):
        return p.replace(COLAB_PREFIX + "/data", RUN_ROOT)
    if p.startswith(COLAB_PREFIX + "/checkpoints"):
        return p.replace(COLAB_PREFIX + "/checkpoints", REPO_ROOT + "/checkpoints/{run_tag}")
    if p.startswith(COLAB_PREFIX + "/plots"):
        return p.replace(COLAB_PREFIX + "/plots", REPO_ROOT + "/plots/{run_tag}")
    if p.startswith(COLAB_PREFIX + "/reports"):
        return p.replace(COLAB_PREFIX + "/reports", REPO_ROOT + "/reports")
    if p.startswith(COLAB_PREFIX + "/models"):
        return p.replace(COLAB_PREFIX + "/models", REPO_ROOT + "/models")
    if p.startswith(COLAB_PREFIX + "/code"):
        return p.replace(COLAB_PREFIX + "/code", REPO_ROOT + "/code")
    if p.startswith(COLAB_PREFIX):
        return p.replace(COLAB_PREFIX, REPO_ROOT)
    return p

if not getattr(builtins, "_phast_patched", False):
    import io
    _open, _io_open, _mkdirs = builtins.open, io.open, os.makedirs
    _exists, _isdir, _isfile, _listdir = os.path.exists, os.path.isdir, os.path.isfile, os.listdir
    _stat = os.stat
    builtins.open      = lambda f, *a, **k: _open(_remap(f), *a, **k)
    io.open            = lambda f, *a, **k: _io_open(_remap(f), *a, **k)
    os.makedirs        = lambda n, *a, **k: _mkdirs(_remap(n), *a, **k)
    os.path.exists     = lambda p, *a, **k: _exists(_remap(p), *a, **k)
    os.path.isdir      = lambda p, *a, **k: _isdir(_remap(p), *a, **k)
    os.path.isfile     = lambda p, *a, **k: _isfile(_remap(p), *a, **k)
    os.listdir         = lambda p=".", *a, **k: _listdir(_remap(p), *a, **k)
    os.stat            = lambda p, *a, **k: _stat(_remap(p), *a, **k)
    try:
        import IPython.core.interactiveshell
        IPython.core.interactiveshell.io_open = io.open
    except Exception:
        pass
    try:
        import torch
        _torch_save = torch.save
        _torch_load = torch.load
        torch.save = lambda obj, f, *a, **k: _torch_save(obj, _remap(f), *a, **k)
        torch.load = lambda f, *a, **k: _torch_load(_remap(f), *a, **k)
    except Exception:
        pass
    builtins._phast_patched = True

# Put code/ on the path so `import stage1_preprocessing` works
_code = os.path.join(REPO_ROOT, "code")
if _code not in sys.path: sys.path.insert(0, _code)

# ── 3. override PIPELINE_CONFIG with the user's requested star ──────────────
PIPELINE_CONFIG = {{
    "target":                   "TIC {tic_id}",
    "sector":                   {sector},
    "author":                   "SPOC",
    "cadence":                  "short",
    "flux_type":                "pdcsap_flux",
    "quality_bitmask":          "default",
    "remove_positive_outliers": True,
    "outlier_sigma_upper":      6,
    "gap_threshold_multiplier": 5,
    "use_gp_detrending":        True,
    "minimum_gp_timescale_days": 0.75,
    "save_intermediate_outputs": True,
}}

# Also expose the per-star paths so notebook cells that re-declare DATA_DIR
# pick up our override automatically.
DATA_DIR       = RUN_ROOT
CHECKPOINT_DIR = REPO_ROOT + "/checkpoints/{run_tag}"
PLOT_DIR       = REPO_ROOT + "/plots/{run_tag}"
REPORT_DIR     = REPO_ROOT + "/reports"
MODEL_DIR      = REPO_ROOT + "/models"

print(f"[pipeline] target : TIC {tic_id}  sector {sector}")
print(f"[pipeline] data   : {{DATA_DIR}}")
# ── end injection ────────────────────────────────────────────────────────────
"""


def run_notebook(stage_num: int, nb_path: str, bootstrap_src: str, timeout: int,
                 kernel_name: str = "python3"):
    abs_path = os.path.join(REPO_ROOT, nb_path)
    print("=" * 80, flush=True)
    print(f"RUNNING STAGE {stage_num}: {os.path.basename(nb_path)}", flush=True)
    print("=" * 80, flush=True)

    if not os.path.exists(abs_path):
        print(f"[FAILED] Notebook not found: {abs_path}")
        sys.exit(1)

    nb = nbformat.read(abs_path, as_version=4)

    # Neutralize any bare PIPELINE_CONFIG = { ... } cells in the notebook.
    # The bootstrap cell (injected below) already provides the correct config;
    # if we let the notebook's own cell run it overwrites the target with the
    # hardcoded TIC ID that was used during development.
    for cell in nb.cells:
        src = "".join(cell.get("source", []))
        stripped = src.lstrip()
        if stripped.startswith("PIPELINE_CONFIG") and "=" in stripped.split("\n")[0]:
            cell["source"] = "# PIPELINE_CONFIG defined by run_pipeline.py bootstrap — skipped here\n"

    # Inject the bootstrap as the first cell (in memory only — never saved back)
    nb.cells.insert(0, nbformat.v4.new_code_cell(source=bootstrap_src))

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    notebook_dir = os.path.dirname(abs_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{notebook_dir}{os.pathsep}" + env.get("PYTHONPATH", "")

    t0 = time.time()
    try:
        ep.preprocess(nb, {"metadata": {"path": notebook_dir}, "env": env})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAILED] Stage {stage_num} after {elapsed:.1f}s", flush=True)
        print(f"  Error: {e}", flush=True)
        # Save the executed notebook (with traceback) for debugging — minus the injected cell
        nb.cells.pop(0)
        error_path = abs_path.replace(".ipynb", f".ERROR.ipynb")
        nbformat.write(nb, error_path)
        print(f"  Error notebook saved to: {error_path}", flush=True)
        raise

    elapsed = time.time() - t0
    # Save executed outputs back (minus injected bootstrap cell)
    nb.cells.pop(0)
    nbformat.write(nb, abs_path)
    print(f"[SUCCESS] Stage {stage_num} done in {elapsed:.1f}s\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="PHAST — run the exoplanet detection pipeline for a single TIC target."
    )
    parser.add_argument(
        "--tic", type=int, required=True,
        help="TIC ID of the target star, e.g. --tic 261136679"
    )
    parser.add_argument(
        "--sector", type=int, default=1,
        help="TESS sector number (default: 1)"
    )
    parser.add_argument(
        "--stage", type=int, choices=range(1, 9),
        help="Run only this one stage."
    )
    parser.add_argument(
        "--start-at", type=int, choices=range(1, 9), default=1,
        help="Start from this stage (default: 1). Useful when resuming a run."
    )
    parser.add_argument(
        "--end-at", type=int, choices=range(1, 9), default=8,
        help="Stop after this stage (default: 8)."
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-notebook timeout in seconds (default: 3600)."
    )
    args = parser.parse_args()

    run_tag = make_run_tag(args.tic, args.sector)
    ensure_folders(run_tag)

    kernel_name = ensure_venv_kernel()
    bootstrap = build_bootstrap_cell(args.tic, args.sector, run_tag)

    if args.stage:
        stages = [args.stage]
    else:
        stages = list(range(args.start_at, args.end_at + 1))

    print(f"\nPHAST pipeline — TIC {args.tic}  sector {args.sector}")
    print(f"Output folder  : data/{run_tag}/")
    print(f"Stages to run  : {stages}\n")

    insufficient_data_reason = None

    for s in stages:
        # After stage 2 completes, check whether there are enough transits to
        # proceed.  If not, write fallback pkls for all remaining stages and
        # jump straight to stage 8 so it can emit an "insufficient data" report.
        if s == 3 and insufficient_data_reason is None:
            s2 = _load_pkl(run_tag, 2)
            if s2 is not None:
                dtc = s2.get("distinct_transit_count", 99)
                sde = s2.get("sde", 99)
                if dtc < 2 or sde < 5:
                    insufficient_data_reason = (
                        f"Only {dtc} distinct transit(s) detected "
                        f"(SDE={sde:.2f}). Insufficient data for full analysis."
                    )
                    print(f"\n[pipeline] {insufficient_data_reason}", flush=True)

        if insufficient_data_reason and s in (3, 4, 5, 6, 7):
            _write_fallback_pkl(run_tag, s, args.tic, args.sector, insufficient_data_reason)
            continue

        try:
            run_notebook(s, STAGE_NOTEBOOKS[s], bootstrap, args.timeout, kernel_name)
        except Exception as exc:
            # Stages 1-2 are critical — abort if they fail.
            if s <= 2:
                print("Pipeline aborted.", flush=True)
                sys.exit(1)
            # Stages 3-7: write a fallback and keep going so stage 8 can report.
            reason = str(exc).split("\n")[0][:200]
            _write_fallback_pkl(run_tag, s, args.tic, args.sector, reason)
            if insufficient_data_reason is None:
                insufficient_data_reason = reason

    print("=" * 80)
    print(f"ALL STAGES COMPLETE — TIC {args.tic}  sector {args.sector}")
    print(f"Report : reports/TIC_{args.tic}_candidate_report.txt")
    print(f"Plots  : reports/TIC_{args.tic}_stage8_dashboard.png")
    print(f"         reports/TIC_{args.tic}_orbital_parameters.png")
    print(f"         reports/TIC_{args.tic}_physics_table.png")
    print("=" * 80)


if __name__ == "__main__":
    main()