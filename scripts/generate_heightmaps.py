from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-track height maps from RGB videos.")
    parser.add_argument("--input", required=True, help="Input video file or directory.")
    parser.add_argument("--output-root", required=True, help="Root directory where .npy height maps are saved.")
    parser.add_argument("--label", required=True, choices=["Fall", "NonFall"], help="Output category directory.")
    parser.add_argument("--preview-root", help="Optional root directory for JPEG preview maps.")
    parser.add_argument("--grounded-sam2-root", required=True, help="Path to the Grounded-SAM-2 checkout.")
    parser.add_argument("--depth-pro-root", required=True, help="Path to the ml-depth-pro checkout.")
    parser.add_argument("--depth-pro-checkpoint", required=True, help="Path to the depth_pro checkpoint.")
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to the SAM2 checkpoint.")
    parser.add_argument(
        "--sam2-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config path relative to the Grounded-SAM-2 root or absolute path.",
    )
    parser.add_argument(
        "--grounding-dino-model",
        default="IDEA-Research/grounding-dino-base",
        help="Hugging Face Grounding DINO model id.",
    )
    parser.add_argument("--tracking-device", default="cuda:0", help="Device for Grounded-SAM-2 tracking.")
    parser.add_argument("--depth-device", default="cuda:0", help="Device for depth estimation.")
    parser.add_argument("--detection-interval", type=int, default=20, help="Detection interval for tracker refresh.")
    parser.add_argument("--person-prompt", default="person . chair", help="Tracking prompt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from heightmap_fall_detection.preprocessing import discover_videos, generate_heightmaps_for_inputs

    sam2_config = args.sam2_config
    if not Path(sam2_config).is_absolute():
        sam2_config = str(Path(args.grounded_sam2_root) / sam2_config)

    video_paths = discover_videos(args.input)
    generate_heightmaps_for_inputs(
        video_paths=video_paths,
        output_root=args.output_root,
        preview_root=args.preview_root,
        label_dir=args.label,
        grounded_sam2_root=args.grounded_sam2_root,
        depth_pro_root=args.depth_pro_root,
        depth_pro_checkpoint=args.depth_pro_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=sam2_config,
        grounding_dino_model=args.grounding_dino_model,
        detection_interval=args.detection_interval,
        person_prompt=args.person_prompt,
        tracking_device=args.tracking_device,
        depth_device=args.depth_device,
    )


if __name__ == "__main__":
    main()
