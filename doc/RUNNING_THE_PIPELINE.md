# PHAST Pipeline — How to Run

**Physics-Hardcoded Anomaly and Signal Taxonomy**
Exoplanet detection pipeline for TESS light curves.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 (recommended) |
| pip | latest |
| Git | any |
| Internet | required (TESS data is downloaded from MAST) |

> **GPU is optional.** Stage 4 (autoencoder) uses PyTorch. It runs fine on CPU — a GPU only speeds it up. MCMC in Stage 7 is CPU-bound and is not accelerated by a GPU at all.

---

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd exoplanet-detection-isro
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux**
  ```bash
  source .venv/bin/activate
  ```

---

## 3. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs every package used across all 8 stages. It may take a few minutes the first time.

### GPU (optional)

The `requirements.txt` installs the CPU build of PyTorch by default. If you have an NVIDIA GPU with CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

---

## 4. Running the Pipeline

There are two ways to run PHAST: the **automated script** (recommended) or **notebook by notebook** manually.

---

### Method A — Automated Script (Recommended)

The script `scripts/run_pipeline.py` executes all 8 notebooks in order for a given star, handles path remapping, and isolates outputs per star so runs never overwrite each other.

**Run the full pipeline for a star:**

```bash
python scripts/run_pipeline.py --tic 261136679
```

**Specify a TESS sector:**

```bash
python scripts/run_pipeline.py --tic 261136679 --sector 4
```

**Resume from a specific stage (e.g. Stage 3 onwards):**

```bash
python scripts/run_pipeline.py --tic 261136679 --sector 1 --start-at 3
```

**Run only one specific stage:**

```bash
python scripts/run_pipeline.py --tic 261136679 --stage 4
```

**Run a range of stages:**

```bash
python scripts/run_pipeline.py --tic 261136679 --start-at 2 --end-at 6
```

**All available flags:**

| Flag | Default | Description |
|---|---|---|
| `--tic` | *(required)* | TIC ID of the target star |
| `--sector` | `1` | TESS sector number |
| `--stage` | — | Run only this one stage (1–8) |
| `--start-at` | `1` | Start from this stage |
| `--end-at` | `8` | Stop after this stage |
| `--timeout` | `3600` | Per-notebook timeout in seconds |

**Outputs** are written to:

```
data/TIC_261136679_S1/      ← stage pkl files (stage1_output.pkl … stage8_output.pkl)
plots/TIC_261136679_S1/     ← diagnostic plots
checkpoints/TIC_261136679_S1/
reports/                    ← final candidate report (.txt) and dashboard (.png)
models/                     ← saved autoencoder weights
```

---

### Method B — Notebooks Manually (VSCode / JupyterLab)

Each notebook is self-contained. Open them in order in VSCode or JupyterLab and run all cells.

**Important:** The first cell of every notebook must be:

```python
import sys, os
sys.path.insert(0, os.path.abspath(".."))
import phast_bootstrap
```

This is already present in every notebook. `phast_bootstrap.py` mocks `google.colab` and remaps the hardcoded Colab paths to your local repo, so notebooks written for Colab run unchanged locally.

**Stage order:**

| Stage | Notebook | What it does |
|---|---|---|
| 1 | `stage1_preprocessing.ipynb` | Downloads TESS light curve from MAST, cleans, normalises, GP-detrends |
| 2 | `stage2_TLS.ipynb` | Runs Transit Least Squares period search; keeps signals with SNR > 7 |
| 3 | `stage3_physics_filters.ipynb` | Fast physics pre-filter; rejects obvious false positives (eclipsing binaries, V-shapes) |
| 4 | `stage4_autoencoder.ipynb` | Trains a per-star autoencoder; computes anomaly score Â |
| 5 | `stage5_Full_Physics_Validator.ipynb` | Runs 17 deterministic physics checks; computes planet probability P̂ |
| 6 | `stage6_Adaptive_Candidate_Priority.ipynb` | Fuses Â and P̂ into final oddity score Ω; assigns priority (HIGH/MEDIUM/LOW) |
| 7 | `stage7_Parameter_Estimation.ipynb` | BATMAN transit model fit + MCMC for orbital parameters |
| 8 | `stage8_ClassificationandReporting.ipynb` | Final classification, dashboard plots, and candidate report |

Each stage saves its results as a `.pkl` file in `data/`. The next stage loads that file — so they must be run in order from the beginning, or resumed using `--start-at` if earlier stages have already been run.

---

## 5. Output Files

After a complete run on TIC 261136679 sector 1:

```
data/TIC_261136679_S1/
    stage1_output.pkl
    stage2_output.pkl
    stage3_output.pkl
    stage4_output.pkl  (+ autoencoder checkpoint in checkpoints/)
    stage5_output.pkl
    stage6_output.pkl
    stage7_output.pkl
    stage8_output.pkl

reports/
    TIC_261136679_candidate_report.txt   ← human-readable verdict
    TIC_261136679_stage8_dashboard.png   ← score summary plot
    TIC_261136679_physics_table.png      ← all 17 physics metrics
    TIC_261136679_orbital_parameters.png ← BATMAN/MCMC results

models/
    autoencoder_TIC_261136679.pt         ← trained PyTorch model
```

---

## 6. Common Issues

**`ModuleNotFoundError: No module named 'google.colab'`**
You forgot to run `import phast_bootstrap` at the top of the notebook. It is already in the first cell — make sure you ran it before any other cell.

**`KeyError: 'priority_score'` in Stage 7**
Your Stage 6 pkl was saved before the key was renamed. Re-run Stage 6, then Stage 7.

**`UnicodeEncodeError` when saving the report (Stage 8)**
The report contains Unicode symbols (Ω, ✓, ⚠). The `open()` call must use `encoding="utf-8"`. This is already fixed in the current notebook.

**Stage 1 fails to download data**
Check your internet connection. TESS data is fetched live from MAST via `lightkurve`. If a specific sector has no short-cadence data for your target, try `--sector 2` (or whichever sector observed the star).

**Stage 4 is very slow**
The autoencoder trains on CPU by default. Install the CUDA build of PyTorch (see Section 3) or reduce epochs inside the notebook.

**Stage 7 MCMC takes over 10 minutes**
This is normal for `nsteps=8000` with 64 walkers. Reduce `nsteps` in the notebook cell if you need a faster test run (e.g. `nsteps=2000`). Convergence may be lower.

---

## 7. Project Structure

```
exoplanet-detection-isro/
├── notebooks/          ← the 8 pipeline notebooks (source of truth)
├── code/               ← importable Python modules used by the notebooks
├── scripts/
│   └── run_pipeline.py ← automated runner
├── phast_bootstrap.py  ← Colab mock + path remapping (required)
├── requirements.txt    ← all dependencies
├── data/               ← stage outputs (created automatically)
├── plots/              ← diagnostic plots (created automatically)
├── reports/            ← final reports (created automatically)
├── models/             ← saved model weights (created automatically)
├── checkpoints/        ← training checkpoints (created automatically)
└── doc/
    └── RUNNING_THE_PIPELINE.md   ← this file
```
