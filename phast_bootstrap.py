"""
phast_bootstrap.py  —  makes the Colab notebooks run anywhere (VSCode, terminal, CI).

Your 8 notebooks were written for Google Colab, so they all:
  1. do `from google.colab import drive` and `drive.mount(...)`
  2. hardcode paths like "/content/drive/MyDrive/exoplanet_pipeline/data/stageN_output.pkl"

This module fixes BOTH at runtime, with no notebook edits, no admin rights, and no
symlink/junction. Just make it the very first thing that runs:

    import phast_bootstrap          # in a notebook, put this in the first cell
    # ... rest of the notebook runs unchanged ...

How it works:
  - Registers a fake `google.colab.drive` module so the import + mount() become no-ops.
  - Monkeypatches open / os.makedirs / os.path.exists / os.listdir so any path that
    starts with the Colab prefix is transparently rewritten to this repo's real folder.

The repo root is wherever THIS file lives, so it travels with the project.
"""
import builtins
import os
import sys
import types

# --- repo root = the folder containing this file -----------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# The literal prefix every notebook hardcodes. Everything under it maps into the repo.
COLAB_PREFIX = "/content/drive/MyDrive/exoplanet_pipeline"

# Standard subfolders the notebooks expect to exist.
for _sub in ("data", "checkpoints", "plots", "reports", "models", "code"):
    os.makedirs(os.path.join(PROJECT_DIR, _sub), exist_ok=True)


def _remap(path):
    """Rewrite a Colab-style path to its local equivalent; leave everything else alone."""
    try:
        path = os.fspath(path)            # handle pathlib.Path, etc.
    except TypeError:
        return path
    if isinstance(path, str) and path.startswith(COLAB_PREFIX):
        return PROJECT_DIR + path[len(COLAB_PREFIX):]
    return path


# --- patch the file/dir functions the notebooks actually use -----------------
# Guard with a flag so re-running the bootstrap cell doesn't double-wrap.
if not getattr(builtins, "_phast_patched", False):
    import io
    _open, _io_open, _makedirs = builtins.open, io.open, os.makedirs
    _exists, _listdir, _isdir, _isfile = os.path.exists, os.listdir, os.path.isdir, os.path.isfile
    _stat = os.stat

    builtins.open      = lambda file, *a, **k: _open(_remap(file), *a, **k)
    io.open            = lambda file, *a, **k: _io_open(_remap(file), *a, **k)
    os.makedirs        = lambda name, *a, **k: _makedirs(_remap(name), *a, **k)
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


# --- fake out google.colab ---------------------------------------------------
if "google.colab" not in sys.modules:
    _google = sys.modules.get("google") or types.ModuleType("google")
    _colab  = types.ModuleType("google.colab")
    _drive  = types.ModuleType("google.colab.drive")

    def _mount(mountpoint="/content/drive", force_remount=False):
        print(f"[phast_bootstrap] mock drive.mount('{mountpoint}') -> using local repo "
              f"at {PROJECT_DIR}")
        os.makedirs(mountpoint, exist_ok=True)

    _drive.mount = _mount
    _colab.drive = _drive
    _google.colab = _colab
    sys.modules["google"] = _google
    sys.modules["google.colab"] = _colab
    sys.modules["google.colab.drive"] = _drive

# Put code/ on the import path so `import stage1_preprocessing` works locally too.
_code = os.path.join(PROJECT_DIR, "code")
if _code not in sys.path:
    sys.path.insert(0, _code)

print(f"[phast_bootstrap] ready. Colab paths now resolve under: {PROJECT_DIR}")