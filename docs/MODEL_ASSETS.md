# Model Assets

SplineHMR demo inference is self-contained inside this repository. It does not fall back to absolute paths outside `SplineHMR/`.

For the first open-source demo slice, the input is assumed to be a precomputed GVHMR-style result such as:

```text
inputs/climbing_3mb/hmr4d_results.pt
```

That means Spline-Opt does **not** require a full GVHMR checkout or GVHMR checkpoints.

## Spline-Opt required assets

SMPL/SMPLX body models are license-gated. Register and download them from the official SMPL/SMPL-X websites, then place the neutral models here:

```text
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

The demo uses local sparse regressor files under:

```text
multi_view_smpl_optimizer/utils/body_model/
```

These regressor files are included with the SplineHMR demo code because they are needed by Spline-Opt and rendering.

## Spline-Diff / ScoreHMR required assets

Prepare ScoreHMR source under this repository and apply the Spline-Diff patch:

```bash
bash scripts/setup_scorehmr.sh
```

Download ScoreHMR's released data/model weights:

```bash
bash scripts/download_scorehmr_data.sh
```

ScoreHMR also requires the licensed neutral SMPL model. Download it from the official SMPL/SMPLify site, rename it if necessary, and place it here:

```text
third_party/ScoreHMR/data/smpl/SMPL_NEUTRAL.pkl
```

The original ScoreHMR README asks users to rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`.

Small ScoreHMR runtime assets are kept in this repository under `assets/scorehmr/data/` and can be synchronized into the runtime ScoreHMR tree with:

```bash
bash scripts/sync_scorehmr_assets.sh
```

Large ScoreHMR weights are not tracked and must be downloaded or copied locally:

```text
third_party/ScoreHMR/data/model_weights/score_hmr/model-100.pt
third_party/ScoreHMR/data/model_weights/pare/pare_checkpoint.ckpt
```

## Optional GVHMR source

GVHMR is useful if users want to reproduce the preprocessing step from raw video to `hmr4d_results.pt`, but it is not required for the bundled Spline-Opt demo inference.

If needed, place it here:

```text
third_party/GVHMR/
```

## Check

Check only Spline-Opt assets:

```bash
python scripts/check_assets.py --method spline-opt
```

Check Spline-Diff assets as well:

```bash
python scripts/check_assets.py --method spline-diff
```
