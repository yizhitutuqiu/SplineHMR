# Deployment: validated `splinehmr` conda environment

This guide records the currently validated deployment flow used in our development container. It starts from a working `gvhmr` conda environment and creates a separate `splinehmr` environment that runs both Spline-Opt and Spline-Diff.

The demo starts from precomputed GVHMR-style input files under `inputs/climbing_3mb`; full GVHMR source/checkpoints are not required for Spline-Opt inference.

## 1. Create the conda environment

```bash
conda deactivate
conda create -n splinehmr --clone gvhmr -y
conda activate splinehmr
python -m pip install --upgrade pip setuptools wheel
```

## 2. Install ScoreHMR dependencies

Keep the inherited `gvhmr` GPU stack, especially PyTorch and PyTorch3D. Install only the additional ScoreHMR-side dependencies:

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR

python -m pip install \
  smplx==0.1.28 \
  pyrender \
  opencv-python \
  yacs \
  scikit-image \
  einops \
  ema_pytorch \
  loguru
```

## 3. Prepare ScoreHMR source

Preferred path:

```bash
bash scripts/setup_scorehmr.sh
python -m pip install -e third_party/ScoreHMR
```

If the network is unavailable and the modified ScoreHMR tree already exists in the parent workspace, use the local copy:

```bash
mkdir -p third_party
cp -a /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR third_party/ScoreHMR
python -m pip install -e third_party/ScoreHMR
```

## 4. Prepare assets

Licensed SMPL/SMPLX body models must be placed inside the SplineHMR repo:

```bash
mkdir -p assets/body_models/smpl assets/body_models/smplx
```

Required paths:

```text
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

For local validation, if these files already exist in the old GVHMR workspace:

```bash
cp /root/autodl-tmp/work/SplineHMR/third_party/GVHMR/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
   assets/body_models/smpl/SMPL_NEUTRAL.pkl

cp /root/autodl-tmp/work/SplineHMR/third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
   assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

ScoreHMR also needs the SMPL neutral file:

```bash
mkdir -p third_party/ScoreHMR/data/smpl
cp assets/body_models/smpl/SMPL_NEUTRAL.pkl third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl
```

Sync small ScoreHMR runtime assets tracked by this repository:

```bash
bash scripts/sync_scorehmr_assets.sh
```

Large ScoreHMR weights are not tracked. Download them via ScoreHMR or copy them from the existing workspace:

```bash
mkdir -p third_party/ScoreHMR/data/model_weights/score_hmr
mkdir -p third_party/ScoreHMR/data/model_weights/pare

cp /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR/data/model_weights/score_hmr/model-100.pt \
   third_party/ScoreHMR/data/model_weights/score_hmr/model-100.pt

cp /root/autodl-tmp/work/SplineHMR/bss-smplify/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt \
   third_party/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt
```

## 5. Check the installation

```bash
python scripts/check_assets.py --method spline-opt
python scripts/check_assets.py --method spline-diff
```

Optional import check:

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

## 6. Run smoke tests

Spline-Opt:

```bash
python -m splinehmr.demo --method spline-opt --max_frames 3 --max_iter 1 --skip_render
```

Spline-Diff:

```bash
python -m splinehmr.demo --method spline-diff --max_frames 3 --skip_render
```

Rendering smoke tests:

```bash
python -m splinehmr.demo --method spline-opt --max_frames 2 --max_iter 1
python -m splinehmr.demo --method spline-diff --max_frames 2
```

## 7. Outputs

Outputs are written to:

```text
outputs/climbing_3mb/<method>/
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
