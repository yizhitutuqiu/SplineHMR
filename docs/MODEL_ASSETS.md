# Model Assets

SplineHMR demo inference is self-contained inside this repository. It does not fall back to absolute paths outside the SplineHMR repository.

The bundled demo input is already a precomputed GVHMR-style result:

```text
inputs/climbing_3mb/hmr4d_results.pt
```

This means Spline-Opt does **not** require a full GVHMR checkout or GVHMR checkpoints.

## Required for Spline-Opt

SMPL/SMPL-X body models are license-gated and cannot be redistributed in this repository. Download them from the official websites after registration:

- SMPL neutral model: https://smplify.is.tue.mpg.de/ ; SMPL family website: https://smpl.is.tue.mpg.de/
- SMPL-X neutral model: https://smpl-x.is.tue.mpg.de/

Place the files exactly here:

```text
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

The SplineHMR demo already includes the small sparse regressor files needed by Spline-Opt/rendering under:

```text
multi_view_smpl_optimizer/utils/body_model/
```

Check Spline-Opt assets:

```bash
python scripts/check_assets.py --method spline-opt
```

## Required for Spline-Diff / ScoreHMR

Prepare patched ScoreHMR source:

```bash
bash scripts/setup_scorehmr.sh
bash scripts/install_scorehmr_deps.sh
```

ScoreHMR's large weights are not tracked. Download them using ScoreHMR's `download_data.sh` or copy them from an existing local checkout. The required paths are:

```text
third_party/ScoreHMR/data/model_weights/score_hmr/model-100.pt
third_party/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt
```

ScoreHMR also needs the licensed neutral SMPL model:

```text
third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl
```

If you already placed `assets/body_models/smpl/SMPL_NEUTRAL.pkl`, you can copy it:

```bash
mkdir -p third_party/ScoreHMR/data/smpl
cp assets/body_models/smpl/SMPL_NEUTRAL.pkl third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl
```

Small ScoreHMR runtime assets are tracked under `assets/scorehmr/data/`. Sync them into the ScoreHMR runtime tree with:

```bash
bash scripts/sync_scorehmr_assets.sh
```

This covers small files such as:

```text
third_party/ScoreHMR/data/SMPL_to_J19.pkl
third_party/ScoreHMR/data/smpl_mean_params.npz
third_party/ScoreHMR/data/stats/*.npz
third_party/ScoreHMR/data/model_weights/pare/pare_config.yaml
```

Check Spline-Diff assets:

```bash
python scripts/check_assets.py --method spline-diff
```

## Optional GVHMR source

GVHMR is useful only if users want to reproduce preprocessing from a raw video to `hmr4d_results.pt`. It is not required for the bundled Spline-Opt/Spline-Diff demo inputs.

If needed, place it under:

```text
third_party/GVHMR/
```

## More deployment details

See:

```text
docs/DEPLOYMENT.md
```
