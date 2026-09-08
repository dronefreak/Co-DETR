#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Recreate a known-working `codetr` conda environment for training/evaluating/
# running inference with Co-DETR on this repo
# (MMDetection 2.25.3 + MMCV-full 1.5.0, the OpenMMLab 1.x stack).
#
# This is the environment combination that has actually been validated for
# end-to-end training + evaluation + inference on this codebase. CUDA 12.x /
# PyTorch 2.x are NOT yet validated for this repo (mmcv-full 1.x has no
# official prebuilt wheels for CUDA 12) -- see docs/en/get_started.md for
# details and status.
#
# Assumptions on the target machine:
#   * conda / miniconda installed
#   * an NVIDIA GPU whose driver supports the CUDA 11.3 runtime
#     (driver >= 465; newer drivers are fine -- tested with driver 580 on an
#     RTX A5000). Ampere (sm_86), Turing and Volta GPUs all work with the
#     cu113 wheels used below.
#   * outbound access to download.pytorch.org and download.openmmlab.com
#   * run from the repo root:  bash tools/setup_codetr_env.sh
#
# Result: conda env `codetr`, Python 3.8, torch 1.11.0+cu113, mmcv-full 1.5.0,
# mmdet 2.25.3 (this repo, editable), numpy 1.23.5.
# ---------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-codetr}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- locate conda ----------------------------------------------------------
if [ -z "${CONDA_EXE:-}" ]; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda"; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
else
  source "$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
fi

# --- create env ----------------------------------------------------------
conda create -y -n "$ENV_NAME" python=3.8
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
PIP="$PY -m pip"
echo ">>> using python: $PY"

# numpy first, pinned below 1.24 (mmcv-full 1.x / mmdet 2.x use numpy type
# aliases such as np.float/np.int/np.bool that were removed in numpy 1.24)
$PIP install --upgrade pip
$PIP install "numpy==1.23.5"

# torch / torchvision -- must come from the pytorch cu113 wheel index
$PIP install --index-url https://download.pytorch.org/whl/cu113 \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113

# mmcv-full 1.5.0 -- prebuilt wheel with compiled CUDA ops (no local nvcc needed)
$PIP install mmcv-full==1.5.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html

# remaining runtime deps used by Co-DETR (see requirements/codetr.txt)
$PIP install -r "$REPO_DIR/requirements/codetr.txt"

# this repo's bundled mmdet 2.25.3, editable, without pulling its requirements
( cd "$REPO_DIR" && $PIP install -e . --no-deps --no-build-isolation )

# --- verify ------------------------------------------------------------
$PY - <<'PYEOF'
import torch, mmcv, mmdet, numpy
from mmcv.ops import RoIAlign          # forces the compiled ops to load
print("python  :", __import__("sys").version.split()[0])
print("torch   :", torch.__version__, "| cuda", torch.version.cuda,
      "| gpu ok", torch.cuda.is_available())
print("mmcv    :", mmcv.__version__, "(ops loaded)")
print("mmdet   :", mmdet.__version__)
print("numpy   :", numpy.__version__)
from projects import *                 # registers CoDETR / ViT / SFP / CoDINOHead
from mmdet.models import DETECTORS
print("CoDETR registered:", "CoDETR" in DETECTORS.module_dict)
PYEOF

echo
echo ">>> done. activate with:  conda activate $ENV_NAME"
