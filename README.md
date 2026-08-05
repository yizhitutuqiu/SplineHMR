# SplineHMR

Demo code for SplineHMR. This first open-source slice focuses on inference from precomputed GVHMR outputs:

- Spline-Opt: B-spline control-space LBFGS refinement.
- Spline-Diff: ScoreHMR sampling with B-spline temporal projection.

Runtime code and assets are kept inside this repository. Spline-Opt starts from precomputed `hmr4d_results.pt` and does not require a full GVHMR checkout; only local SMPL/SMPLX body-model assets are required for optimization/rendering.

## Demo Input

The bundled demo input is `inputs/climbing_3mb`:

```text
inputs/climbing_3mb/
├── 0_input_video.mp4
├── hmr4d_results.pt
└── preprocess/
    ├── bbx.pt
    └── vitpose.pt
```

## Prepare Code And Assets

Spline-Opt does not need the full GVHMR repository. Prepare the licensed neutral SMPL/SMPLX models here:

```text
assets/body_models/smpl/SMPL_NEUTRAL.pkl
assets/body_models/smplx/SMPLX_NEUTRAL.npz
```

Then check the minimal Spline-Opt assets:

```bash
python scripts/check_assets.py --method spline-opt
```

For Spline-Diff, prepare ScoreHMR source, apply the Spline-Diff patch, and install dependencies:

```bash
bash scripts/setup_scorehmr.sh
bash scripts/install_scorehmr_deps.sh
bash scripts/download_scorehmr_data.sh
```

For Spline-Diff, also verify ScoreHMR assets with:

```bash
python scripts/check_assets.py --method spline-diff
```

SMPL/SMPLX body models are license-gated and must be downloaded by the user. More details are in `docs/MODEL_ASSETS.md`. The validated conda deployment flow is in `docs/DEPLOYMENT.md`. GVHMR source is optional only if users want to reproduce preprocessing from raw videos.

## Run Spline-Opt

Use the validated `splinehmr` conda environment:

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR
conda run -n splinehmr python -m splinehmr.demo --method spline-opt
```

Spline-Opt defaults are copied from GVHMR's demo postprocess config used by `--enable-bspline-refine`, now stored at `configs/spline_opt.yaml`. Important defaults include `m_per_t: fps_div_2`, `max_iter: 60`, `amp_*: 1.0`, `prior_w_body_pose: 0.1`, `prior_w_global_orient/transl: 0.2`, `smooth_w: 0.02`, `static_motion_w: 2`, and `optimize_pose_in_rot6d: true`.

Optional quick test:

```bash
conda run -n splinehmr python -m splinehmr.demo --method spline-opt --max_frames 60 --max_iter 20
```

## Run Spline-Diff

```bash
cd /root/autodl-tmp/work/SplineHMR/SplineHMR
conda run -n splinehmr python -m splinehmr.demo --method spline-diff
```

Spline-Diff reads the patched ScoreHMR config from `third_party/ScoreHMR/custom/configs/bspline.yaml`. By default this mirrors the current modified ScoreHMR tree: `m_per_t: fps_div_3`, `blend_weight: 0.6`, `use_tanh: true`, `tanh_amp: 10.0`.

ScoreHMR outputs are treated as SMPL, not SMPLX. For Spline-Diff rendering, the demo uses SMPL directly, padding/cropping body pose to 69D, matching the special handling used in the original bss-smplify pipeline.

## Outputs

For either method, outputs are written to `outputs/<input_name>/<method>/`:

```text
hmr4d_results_before.pt
hmr4d_results.pt
render_before.mp4
render_after.mp4
render_compare.mp4
render_overlay_compare.mp4
report.json
```

`render_compare.mp4` is the side-by-side comparison. `render_overlay_compare.mp4` overlays both meshes in one frame: red is before optimization and green is after optimization, with a legend.

The `hmr4d_results.pt` file keeps the original GVHMR-style dict layout, with refined `smpl_params_incam` values.
