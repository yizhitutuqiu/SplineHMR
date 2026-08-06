# Deployment Guide

This guide describes the currently validated `splinehmr` conda environment and the required model assets for running both Spline-Opt and Spline-Diff.

SplineHMR demo inference starts from precomputed GVHMR-style files. The default input is `inputs/climbing_3mb`, and additional prepared sequences such as `inputs/climbing_2_3mb` can be selected with `--input_dir`/`--input`. Full GVHMR source/checkpoints are not required for Spline-Opt.

## Quick summary

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR

conda create -n splinehmr --clone gvhmr -y
conda activate splinehmr
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

bash scripts/setup_scorehmr.sh
bash scripts/install_scorehmr_deps.sh
bash scripts/sync_scorehmr_assets.sh
```

Then place/download the licensed body models and large ScoreHMR checkpoints described below, and check:

```bash
python scripts/check_assets.py --method spline-opt
python scripts/check_assets.py --method spline-diff
```

## 1. Environment

### Recommended: clone the working GVHMR environment

This is the route we have validated in the current container:

```bash
conda deactivate
conda create -n splinehmr --clone gvhmr -y
conda activate splinehmr
python -m pip install --upgrade pip setuptools wheel
```

Install the local SplineHMR package in editable mode:

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR
python -m pip install -e .
```

This installs `splinehmr-demo` as a console script, while keeping `python -m splinehmr.demo` available.

### Alternative: build a new environment

If you are not cloning `gvhmr`, install CUDA-compatible PyTorch and PyTorch3D first. We do not pin those packages in `requirements.txt` because their wheels depend on your CUDA, driver, Python, and PyTorch ABI.

After PyTorch/PyTorch3D are working, install SplineHMR:

```bash
conda create -n splinehmr python=3.10 -y
conda activate splinehmr
python -m pip install --upgrade pip setuptools wheel

# Install CUDA-compatible torch/torchvision/pytorch3d for your machine first.
# Then:
python -m pip install -r requirements.txt
python -m pip install -e .
```

The lightweight dependency files are:

```text
requirements.txt             # SplineHMR demo + patched ScoreHMR Python deps, excluding torch/pytorch3d
requirements-scorehmr.txt    # ScoreHMR-side extra deps used by scripts/install_scorehmr_deps.sh
environment.yml              # Minimal conda skeleton; still requires torch/pytorch3d handling
pyproject.toml               # Editable/package install metadata
```

## 2. Spline-Opt assets

Spline-Opt requires the licensed neutral SMPL and SMPL-X body models inside this repository:

```text
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

Download sources:

- SMPL neutral model: register at the SMPLify/SMPL website and download the neutral SMPL model. The commonly used file is renamed to `SMPL_NEUTRAL.pkl`.
- SMPL-X neutral model: register at the SMPL-X website and download the neutral SMPL-X NPZ model, then place `SMPLX_NEUTRAL.npz` as shown above.

For local validation only, if these files already exist in the old GVHMR workspace:

```bash
mkdir -p assets/body_models/smpl assets/body_models/smplx

cp /root/autodl-tmp/work/SplineHMR/third_party/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
   assets/body_models/smpl/SMPL_NEUTRAL.pkl

cp /root/autodl-tmp/work/SplineHMR/third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
   assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

The sparse regressors needed by Spline-Opt/rendering are already included under:

```text
multi_view_smpl_optimizer/utils/body_model/
```

## 3. Spline-Diff / ScoreHMR source and assets

Prepare the patched ScoreHMR tree:

```bash
bash scripts/setup_scorehmr.sh
bash scripts/install_scorehmr_deps.sh
```

If the network is unavailable and the modified ScoreHMR tree already exists in the parent workspace:

```bash
mkdir -p third_party
cp -a /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR third_party/ScoreHMR
python -m pip install -e third_party/ScoreHMR
```

Sync small ScoreHMR runtime assets tracked by this repository:

```bash
bash scripts/sync_scorehmr_assets.sh
```

This copies files such as `SMPL_to_J19.pkl`, `smpl_mean_params.npz`, `stats/*.npz`, and `pare_config.yaml` from `assets/scorehmr/data/` into `third_party/ScoreHMR/data/`.

ScoreHMR also needs the licensed SMPL neutral model:

```bash
mkdir -p third_party/ScoreHMR/data/smpl
cp assets/body_models/smpl/SMPL_NEUTRAL.pkl third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl
```

Large ScoreHMR weights are not tracked. Download them with ScoreHMR's `download_data.sh` or copy them from an existing local checkout:

```bash
mkdir -p third_party/ScoreHMR/data/model_weights/score_hmr
mkdir -p third_party/ScoreHMR/data/model_weights/pare

cp /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR/data/model_weights/score_hmr/model-100.pt \
   third_party/ScoreHMR/data/model_weights/score_hmr/model-100.pt

cp /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt \
   third_party/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt
```

Required ScoreHMR runtime paths are checked by:

```bash
python scripts/check_assets.py --method spline-diff
```

## 4. Smoke tests

Asset checks:

```bash
python scripts/check_assets.py --method spline-opt
python scripts/check_assets.py --method spline-diff
```

Import check:

```bash
python - <<'PY'
import torch, smplx, pytorch3d
import cv2, yacs, skimage, einops, loguru
import score_hmr
print('torch:', torch.__version__)
print('cuda:', torch.cuda.is_available())
print('ScoreHMR import OK')
PY
```

Spline-Opt:

```bash
python -m splinehmr.demo --method spline-opt --max_frames 3 --max_iter 1 --skip_render
```

Spline-Diff:

```bash
python -m splinehmr.demo --method spline-diff --max_frames 3 --skip_render
```

Spline-Diff runtime defaults are self-contained in `configs/spline_diff.yaml`. This file controls the ScoreHMR checkpoint selector, sampling/guidance settings, and B-spline projection parameters such as `m_per_t`, `order`, `blend_weight`, `use_tanh`, and `tanh_amp`. The demo reads it by default; use `--spline_diff_config <path>` or the short CLI overrides (`--m_per_t`, `--scorehmr_blend_weight`, `--scorehmr_optim_iters`, `--scorehmr_use_tanh`, `--scorehmr_tanh_amp`) for quick experiments.

Rendering smoke tests:

```bash
python -m splinehmr.demo --method spline-opt --max_frames 2 --max_iter 1
python -m splinehmr.demo --method spline-diff --max_frames 2
```

## 5. Inputs and outputs

The demo input argument may be either an input directory or its video file:

```bash
python -m splinehmr.demo --method spline-opt --input_dir inputs/climbing_2_3mb
python -m splinehmr.demo --method spline-opt --input inputs/climbing_2_3mb/0_input_video.mp4
```

Each input directory must contain:

```text
0_input_video.mp4
hmr4d_results.pt
preprocess/bbx.pt
preprocess/vitpose.pt
```

Outputs are written to:

```text
outputs/<input_name>/<method>/
```

For example, `--input_dir inputs/climbing_2_3mb --method spline-opt` writes to:

```text
outputs/climbing_2_3mb/spline-opt/
```

Each rendered run produces:

```text
hmr4d_results_before.pt
hmr4d_results.pt
render_before.mp4
render_after.mp4
render_compare.mp4
render_overlay_compare.mp4
report.json
```

`render_compare.mp4` is the side-by-side comparison. `render_overlay_compare.mp4` overlays both meshes in one frame: red is before optimization and green is after optimization, with an in-frame legend.

## 6. Upstream references

- SMPL-X model/code: https://github.com/vchoutas/smplx and https://smpl-x.is.tue.mpg.de/
- SMPL neutral model: https://smplify.is.tue.mpg.de/ ; SMPL family website: https://smpl.is.tue.mpg.de/
- ScoreHMR: https://github.com/statho/ScoreHMR
