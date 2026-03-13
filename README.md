# Height-Map Fall Detection

This repository extracts the paper-specific contribution from a heavily modified `Grounded-SAM-2` workspace: generating per-person height maps from RGB videos and training a weakly supervised fall detector on those height-map sequences.

The code is organized as an extension layer on top of existing upstream projects instead of republishing third-party source code:

- `Grounded-SAM-2` for grounding, segmentation, and tracking
- `ml-depth-pro` for monocular depth estimation

## Repository layout

```text
heightmap-fall-detection/
├── configs/
├── scripts/
│   ├── generate_heightmaps.py
│   └── train.py
└── src/heightmap_fall_detection/
    ├── configs.py
    ├── data.py
    ├── losses.py
    ├── metrics.py
    ├── model.py
    ├── preprocessing.py
    └── train.py
```

## What is included

- Height-map generation pipeline with floor-plane estimation
- Per-track `64x64` height-map export
- Transformer-based fall detector for video-level supervision
- RFDS, URFD, and Le2i training presets

## What is intentionally excluded

- Large datasets
- Model checkpoints
- Paper figures and previews
- The full upstream `Grounded-SAM-2` codebase

## Environment

Install this repository's Python dependencies first:

```bash
pip install -r requirements.txt
```

You also need local checkouts of:

- `Grounded-SAM-2`
- `ml-depth-pro`

The preprocessing script expects paths to both repositories and to the required checkpoints.

## Height-map generation

The generator writes output in this structure:

```text
<output-root>/
├── Fall/
│   └── <video-name>/
│       └── track_<id>/
│           ├── 00000.npy
│           └── ...
└── NonFall/
    └── <video-name>/
        └── track_<id>/
```

Example:

```bash
python scripts/generate_heightmaps.py \
  --input /path/to/RFDS/fall \
  --output-root ./datasets/rfds \
  --preview-root ./previews/rfds \
  --label Fall \
  --grounded-sam2-root /path/to/Grounded-SAM-2 \
  --depth-pro-root /path/to/ml-depth-pro \
  --depth-pro-checkpoint /path/to/ml-depth-pro/checkpoints/depth_pro.pt \
  --sam2-checkpoint /path/to/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  --tracking-device cuda:0 \
  --depth-device cuda:1
```

Run the same command with `--label NonFall` for negative videos.

## Training

Train with a dataset preset:

```bash
python scripts/train.py --config rfds --root-dir ./datasets/rfds
python scripts/train.py --config urfd --root-dir ./datasets/urfd
python scripts/train.py --config le2i --root-dir ./datasets/le2i
```

Each run writes:

- fold checkpoints
- `summary.json` with fold-wise and mean metrics

## Notes for GitHub release

- Keep this repository separate from your experimental `Grounded-SAM-2` tree.
- Do not commit raw datasets, previews, or `.npy` outputs.
- If you want a cleaner public history, initialize a fresh Git repo from this folder and commit only these files.
