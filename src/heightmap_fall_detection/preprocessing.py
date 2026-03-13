from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


FLOOR_TERMS = [
    "floor",
    "carpet",
    "wooden floor",
    "tiled floor",
    "stone floor",
    "marble floor",
    "ground surface",
    "floor surface",
    "walkway",
    "pathway",
    "aisle",
    "corridor",
    "concrete floor",
    "ground",
    "bottom surface",
    "ground plane",
    "floor plane",
    "road",
    "asphalt",
    "pavement",
    "sidewalk",
    "crosswalk",
    "dirt path",
    "gravel path",
    "grass",
    "dirt",
    "sand",
    "beach",
    "snowfield",
    "snowy ground",
    "ice",
    "icy patch",
    "plaza",
    "deck",
    "sports field",
    "track",
    "wood floor",
    "hardwood floor",
    "tile floor",
    "linoleum",
    "hallway",
    "living room floor",
    "lobby",
    "kitchen floor",
    "gym floor",
    "stage",
    "cement",
    "forest road",
]

EXCLUSION_TERMS = [
    "stairs",
    "steps",
    "chair",
    "table",
    "person",
    "bed",
    "sofa",
    "railing",
    "wall",
    "ceiling",
    "furniture",
    "object",
    "handrail",
    "vertical surface",
    "door",
    "column",
]


def prepare_external_imports(grounded_sam2_root: str, depth_pro_root: str) -> None:
    for root in [grounded_sam2_root, os.path.join(depth_pro_root, "src")]:
        if root not in sys.path:
            sys.path.insert(0, root)


@dataclass
class RuntimeModules:
    build_sam2: object
    build_sam2_video_predictor: object
    sam2_image_predictor_cls: object
    mask_dictionary_model_cls: object
    object_info_cls: object
    depth_pro: object
    vit_preset: object


def load_runtime_modules(grounded_sam2_root: str, depth_pro_root: str) -> RuntimeModules:
    prepare_external_imports(grounded_sam2_root, depth_pro_root)
    from depth_pro.network.vit_factory import ViTPreset
    import depth_pro
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo

    return RuntimeModules(
        build_sam2=build_sam2,
        build_sam2_video_predictor=build_sam2_video_predictor,
        sam2_image_predictor_cls=SAM2ImagePredictor,
        mask_dictionary_model_cls=MaskDictionaryModel,
        object_info_cls=ObjectInfo,
        depth_pro=depth_pro,
        vit_preset=ViTPreset,
    )


class GroundingDinoPredictor:
    def __init__(self, model_id: str, device: str) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    def predict(
        self,
        image: Image.Image,
        text_prompts: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> tuple[torch.Tensor, list[str]]:
        inputs = self.processor(images=image, text=text_prompts, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )
        return results[0]["boxes"], results[0]["labels"]


class SAM2ImageSegmentor:
    def __init__(self, runtime: RuntimeModules, sam_model_cfg: str, sam_model_ckpt: str, device: str) -> None:
        sam_model = runtime.build_sam2(sam_model_cfg, sam_model_ckpt, device=device)
        self.predictor = runtime.sam2_image_predictor_cls(sam_model)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)

    def predict_masks_from_boxes(self, boxes: torch.Tensor) -> np.ndarray | None:
        masks, _, _ = self.predictor.predict(box=boxes, multimask_output=False)
        if masks is None:
            return None
        if masks.ndim == 2:
            return masks[None]
        if masks.ndim == 4:
            return masks.squeeze(1)
        return masks


class IncrementalObjectTracker:
    def __init__(
        self,
        runtime: RuntimeModules,
        grounding_model_id: str,
        sam2_model_cfg: str,
        sam2_ckpt_path: str,
        device: str,
        prompt_text: str = "person.",
        detection_interval: int = 20,
        frame_buffer_size: int = 150,
    ) -> None:
        self.runtime = runtime
        self.device = torch.device(device)
        self.prompt_text = prompt_text
        self.detection_interval = detection_interval
        self.grounding_predictor = GroundingDinoPredictor(model_id=grounding_model_id, device=device)
        self.sam2_segmentor = SAM2ImageSegmentor(runtime, sam2_model_cfg, sam2_ckpt_path, device=device)
        self.video_predictor = runtime.build_sam2_video_predictor(sam2_model_cfg, sam2_ckpt_path, device=device)
        self.MaskDictionaryModel = runtime.mask_dictionary_model_cls
        self.ObjectInfo = runtime.object_info_cls

        self.inference_state = self.video_predictor.init_state()
        self.inference_state["frame_buffer_size"] = frame_buffer_size
        self.inference_state["images"] = torch.empty((0, 3, 1024, 1024), device=self.device)
        self.total_frames = 0
        self.objects_count = 0
        self.last_mask_dict = self.MaskDictionaryModel()
        self.track_dict = self.MaskDictionaryModel()
        self.plane_coeffs = None
        self.focallength_px = None
        self.image_dims = None

    def set_prompt(self, new_prompt: str) -> None:
        self.prompt_text = new_prompt

    def set_camera_params(self, plane_coeffs: np.ndarray | None, focallength_px: float, image_dims: tuple[int, int]) -> None:
        self.plane_coeffs = plane_coeffs
        self.focallength_px = focallength_px
        self.image_dims = image_dims

    @staticmethod
    def _get_box_from_mask(mask: np.ndarray) -> np.ndarray:
        if mask.sum() == 0:
            return np.array([0, 0, 0, 0])
        rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
        if rows.sum() == 0 or cols.sum() == 0:
            return np.array([0, 0, 0, 0])
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return np.array([x_min, y_min, x_max, y_max])

    def _create_height_map(self, obj_info, full_depth_map: np.ndarray, output_size: tuple[int, int] = (64, 64)) -> np.ndarray:
        if self.plane_coeffs is None or self.focallength_px is None or self.image_dims is None:
            return np.zeros(output_size, dtype=np.float32)

        a, b, c = self.plane_coeffs
        image_h, image_w = self.image_dims
        mask_np = obj_info.mask.cpu().numpy().squeeze()
        if not mask_np.any():
            return np.zeros(output_size, dtype=np.float32)

        x1, y1, x2, y2 = self._get_box_from_mask(mask_np).astype(int)
        if x1 >= x2 or y1 >= y2:
            return np.zeros(output_size, dtype=np.float32)

        depth_resized_full = cv2.resize(full_depth_map, (image_w, image_h))
        person_mask_roi = mask_np[y1:y2, x1:x2]
        person_depth_roi = depth_resized_full[y1:y2, x1:x2]
        jj, ii = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))

        z_values = person_depth_roi[person_mask_roi]
        ii_values = ii[person_mask_roi]
        jj_values = jj[person_mask_roi]
        if z_values.size == 0:
            return np.zeros(output_size, dtype=np.float32)

        x_3d = (jj_values - image_w / 2) * z_values / self.focallength_px
        y_3d = (image_h / 2 - ii_values) * z_values / self.focallength_px

        signed_distance_numerator = (a * x_3d + b * y_3d + c) - z_values
        denominator = np.sqrt(a**2 + b**2 + 1)
        if denominator == 0:
            return np.zeros(output_size, dtype=np.float32)

        signed_distance = signed_distance_numerator / denominator
        camera_side_sign = np.sign(c) if c != 0 else 1
        point_side_sign = np.sign(signed_distance)
        height_values = np.abs(signed_distance)
        height_values[point_side_sign != camera_side_sign] = 0

        cropped_height_map = np.zeros(person_mask_roi.shape, dtype=np.float32)
        cropped_height_map[person_mask_roi] = height_values
        roi_h, roi_w = cropped_height_map.shape
        max_dim = max(roi_h, roi_w)
        padded_map = np.zeros((max_dim, max_dim), dtype=np.float32)
        pad_x, pad_y = (max_dim - roi_w) // 2, (max_dim - roi_h) // 2
        padded_map[pad_y : pad_y + roi_h, pad_x : pad_x + roi_w] = cropped_height_map
        return cv2.resize(padded_map, output_size, interpolation=cv2.INTER_AREA)

    def add_image(self, image_np: np.ndarray, depth_map: np.ndarray | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        img_pil = Image.fromarray(image_np)
        frame_idx = -1

        if self.total_frames % self.detection_interval == 0:
            if self.inference_state["video_height"] is None:
                self.inference_state["video_height"], self.inference_state["video_width"] = image_np.shape[:2]
            boxes, labels = self.grounding_predictor.predict(img_pil, self.prompt_text)
            if boxes.shape[0] > 0:
                primary_target = self.prompt_text.split(".")[0].strip()
                target_indices = [i for i, label in enumerate(labels) if primary_target in label]
                if target_indices:
                    filtered_boxes = boxes[target_indices]
                    filtered_labels = [labels[i] for i in target_indices]
                    self.sam2_segmentor.set_image(image_np)
                    masks = self.sam2_segmentor.predict_masks_from_boxes(filtered_boxes)
                    mask_dict = self.MaskDictionaryModel()
                    mask_dict.add_new_frame_annotation(
                        torch.from_numpy(masks).to(self.device),
                        filtered_boxes.clone(),
                        filtered_labels,
                    )
                    self.objects_count = mask_dict.update_masks(self.last_mask_dict, 0.3, self.objects_count)
                    frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)
                    self.video_predictor.reset_state(self.inference_state)
                    for obj_id, obj_info in mask_dict.labels.items():
                        self.video_predictor.add_new_mask(self.inference_state, frame_idx, obj_id, obj_info.mask)
                    self.track_dict = copy.deepcopy(mask_dict)
                    self.last_mask_dict = copy.deepcopy(mask_dict)
        elif self.track_dict.labels:
            frame_idx = self.video_predictor.add_new_frame(self.inference_state, image_np)

        height_map_dict: dict[int, np.ndarray] = {}
        if not self.track_dict.labels:
            self.total_frames += 1
            return image_np, height_map_dict

        if frame_idx != -1:
            _, obj_ids, video_res_masks = self.video_predictor.infer_single_frame(self.inference_state, frame_idx)
            frame_masks = self.MaskDictionaryModel()
            for i, obj_id in enumerate(obj_ids):
                out_mask = video_res_masks[i] > 0.0
                object_info = self.ObjectInfo(
                    instance_id=obj_id,
                    mask=out_mask[0],
                    class_name=self.track_dict.get_target_class_name(obj_id),
                )
                if depth_map is not None:
                    height_map_dict[int(object_info.instance_id)] = self._create_height_map(object_info, depth_map)
                frame_masks.labels[obj_id] = object_info
            self.last_mask_dict = copy.deepcopy(frame_masks)

        annotated_frame = self.visualize_frame(image_np, self.last_mask_dict)
        self.total_frames += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return annotated_frame, height_map_dict

    def visualize_frame(self, image_np: np.ndarray, mask_dict_obj) -> np.ndarray:
        annotated_frame = image_np.copy()
        if not mask_dict_obj or not mask_dict_obj.labels:
            return annotated_frame

        all_boxes = []
        all_masks = []
        all_tracker_ids = []
        labels = []
        for obj_info in mask_dict_obj.labels.values():
            if getattr(obj_info, "mask", None) is None:
                continue
            mask_np = obj_info.mask.cpu().numpy()
            while mask_np.ndim > 2:
                mask_np = mask_np.squeeze(0)
            if mask_np.ndim != 2 or not mask_np.any():
                continue
            all_boxes.append(self._get_box_from_mask(mask_np))
            all_masks.append(mask_np)
            all_tracker_ids.append(int(obj_info.instance_id))
            labels.append(f"ID:{obj_info.instance_id} {obj_info.class_name}")

        if not all_tracker_ids:
            return annotated_frame

        detections = sv.Detections(
            xyxy=np.array(all_boxes),
            mask=np.stack(all_masks, axis=0),
            class_id=np.zeros(len(all_tracker_ids), dtype=int),
            tracker_id=np.array(all_tracker_ids, dtype=int),
        )
        annotated_frame = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(annotated_frame, detections)
        annotated_frame = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(annotated_frame, detections)
        annotated_frame = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK).annotate(
            annotated_frame, detections, labels
        )
        return annotated_frame


def build_depth_model(runtime: RuntimeModules, checkpoint_path: str, device: str) -> tuple[object, object]:
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class DepthProConfig:
        patch_encoder_preset: runtime.vit_preset
        image_encoder_preset: runtime.vit_preset
        decoder_features: int
        checkpoint_uri: Optional[str] = None
        fov_encoder_preset: Optional[runtime.vit_preset] = None
        use_fov_head: bool = True

    config = DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri=checkpoint_path,
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    model, transform = runtime.depth_pro.create_model_and_transforms(
        config=config,
        device=torch.device(device),
        precision=torch.half if "cuda" in device else torch.float32,
    )
    model.eval()
    return model, transform


def segment_floor_mask(
    frame_rgb: np.ndarray,
    grounding_predictor: GroundingDinoPredictor,
    sam2_segmentor: SAM2ImageSegmentor,
) -> np.ndarray:
    prompt = " . ".join(FLOOR_TERMS + EXCLUSION_TERMS)
    image = Image.fromarray(frame_rgb)
    boxes, labels = grounding_predictor.predict(image, prompt)
    if boxes.shape[0] == 0:
        return np.zeros(frame_rgb.shape[:2], dtype=bool)

    sam2_segmentor.set_image(frame_rgb)
    masks = sam2_segmentor.predict_masks_from_boxes(boxes)
    if masks is None:
        return np.zeros(frame_rgb.shape[:2], dtype=bool)

    masks = masks.astype(bool)
    floor_masks = [masks[i] for i, label in enumerate(labels) if any(term in label for term in FLOOR_TERMS)]
    exclusion_masks = [masks[i] for i, label in enumerate(labels) if any(term in label for term in EXCLUSION_TERMS)]
    if not floor_masks:
        return np.zeros(frame_rgb.shape[:2], dtype=bool)

    combined_floor_mask = np.logical_or.reduce(floor_masks)
    if exclusion_masks:
        combined_exclusion_mask = np.logical_or.reduce(exclusion_masks)
        combined_floor_mask = np.logical_and(combined_floor_mask, np.logical_not(combined_exclusion_mask))
    return combined_floor_mask


def estimate_floor_plane(
    frame_rgb: np.ndarray,
    depth_map: np.ndarray,
    focal_px: float,
    grounding_predictor: GroundingDinoPredictor,
    sam2_segmentor: SAM2ImageSegmentor,
    erosion_kernel_size: int = 7,
    crop_ratio: float = 0.05,
) -> np.ndarray | None:
    raw_floor_mask = segment_floor_mask(frame_rgb, grounding_predictor, sam2_segmentor)
    if not raw_floor_mask.any() or focal_px is None:
        return None

    kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded_floor_mask = cv2.erode(raw_floor_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    height, width = eroded_floor_mask.shape
    y_margin, x_margin = int(height * crop_ratio), int(width * crop_ratio)
    center_crop_mask = np.zeros_like(eroded_floor_mask, dtype=bool)
    center_crop_mask[y_margin : height - y_margin, x_margin : width - x_margin] = True
    final_floor_mask = np.logical_and(eroded_floor_mask, center_crop_mask)
    if not final_floor_mask.any():
        return None

    depth_map_resized = cv2.resize(depth_map, (width, height))
    jj, ii = np.meshgrid(np.arange(width), np.arange(height))
    z_values = depth_map_resized[final_floor_mask]
    ii_values = ii[final_floor_mask]
    jj_values = jj[final_floor_mask]

    x_values = (jj_values - width / 2) * z_values / focal_px
    y_values = (height / 2 - ii_values) * z_values / focal_px
    points_3d = np.vstack((x_values, y_values, z_values)).T
    if points_3d.shape[0] < 100:
        return None

    pca = PCA(n_components=3)
    pca.fit(points_3d)
    normal = pca.components_[2]
    centroid = pca.mean_
    if normal[2] < 0:
        normal = -normal
    nx, ny, nz = normal
    if nz == 0:
        return None

    d = -np.dot(normal, centroid)
    return np.array([-nx / nz, -ny / nz, -d / nz], dtype=np.float32)


def discover_videos(input_path: str) -> list[str]:
    path = Path(input_path)
    if path.is_file():
        return [str(path)]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    video_files = []
    for root, _, files in os.walk(path):
        for file_name in files:
            if file_name.lower().endswith((".avi", ".mp4", ".mov", ".mkv")):
                video_files.append(str(Path(root) / file_name))
    return sorted(video_files)


def save_preview_map(height_map: np.ndarray, save_path: str, height_range: tuple[float, float] = (0.0, 3.0)) -> None:
    np.nan_to_num(height_map, copy=False, nan=0.0, posinf=height_range[1], neginf=0.0)
    clipped_map = np.clip(height_map, height_range[0], height_range[1])
    scale = 255.0 / (height_range[1] - height_range[0])
    normalized = ((clipped_map - height_range[0]) * scale).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, color_map)


def generate_heightmaps_for_video(
    video_path: str,
    output_root: str,
    preview_root: str | None,
    label_dir: str,
    grounded_sam2_root: str,
    depth_pro_root: str,
    depth_pro_checkpoint: str,
    sam2_checkpoint: str,
    sam2_config: str,
    grounding_dino_model: str,
    detection_interval: int = 20,
    person_prompt: str = "person . chair",
    tracking_device: str = "cuda:0",
    depth_device: str = "cuda:0",
) -> None:
    runtime = load_runtime_modules(grounded_sam2_root, depth_pro_root)

    if "cuda" in tracking_device:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(torch.device(tracking_device)).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    depth_model, depth_transform = build_depth_model(runtime, checkpoint_path=depth_pro_checkpoint, device=depth_device)
    tracker = IncrementalObjectTracker(
        runtime=runtime,
        grounding_model_id=grounding_dino_model,
        sam2_model_cfg=sam2_config,
        sam2_ckpt_path=sam2_checkpoint,
        device=tracking_device,
        detection_interval=detection_interval,
    )
    tracker.set_prompt(person_prompt)

    output_video_dir = Path(output_root) / label_dir / Path(video_path).name
    preview_video_dir = Path(preview_root) / label_dir / Path(video_path).name if preview_root else None
    output_video_dir.mkdir(parents=True, exist_ok=True)
    if preview_video_dir:
        preview_video_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    focal_px_tensor = None
    focal_px_float = None
    image_dims = None
    last_valid_plane = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_idx == 0:
                image_dims = frame_rgb.shape[:2]
                img_depth_init, _, f_px_init = runtime.depth_pro.load_img(Image.fromarray(frame_rgb))
                img_tensor_init = depth_transform(img_depth_init).unsqueeze(0).to(torch.device(depth_device))
                with torch.no_grad():
                    prediction_init = depth_model.infer(img_tensor_init, f_px=f_px_init)
                focal_px_tensor = prediction_init["focallength_px"]
                focal_px_float = float(focal_px_tensor.item())

            img_depth, _, _ = runtime.depth_pro.load_img(Image.fromarray(frame_rgb))
            img_tensor = depth_transform(img_depth).unsqueeze(0).to(torch.device(depth_device))
            with torch.no_grad():
                prediction = depth_model.infer(img_tensor, f_px=focal_px_tensor)
            depth_map = prediction["depth"].cpu().numpy()

            current_plane = estimate_floor_plane(
                frame_rgb,
                depth_map,
                focal_px_float,
                tracker.grounding_predictor,
                tracker.sam2_segmentor,
            )
            if current_plane is not None:
                last_valid_plane = current_plane
            if last_valid_plane is not None and image_dims is not None:
                tracker.set_camera_params(last_valid_plane, focal_px_float, image_dims)

            _, height_map_dict = tracker.add_image(frame_rgb, depth_map)
            for track_id, height_map in height_map_dict.items():
                track_data_dir = output_video_dir / f"track_{track_id}"
                track_data_dir.mkdir(parents=True, exist_ok=True)
                np.save(track_data_dir / f"{frame_idx:05d}.npy", height_map)

                if preview_video_dir:
                    track_preview_dir = preview_video_dir / f"track_{track_id}"
                    track_preview_dir.mkdir(parents=True, exist_ok=True)
                    save_preview_map(height_map, str(track_preview_dir / f"{frame_idx:05d}.jpg"))

            frame_idx += 1
    finally:
        cap.release()


def generate_heightmaps_for_inputs(
    video_paths: Iterable[str],
    output_root: str,
    preview_root: str | None,
    label_dir: str,
    grounded_sam2_root: str,
    depth_pro_root: str,
    depth_pro_checkpoint: str,
    sam2_checkpoint: str,
    sam2_config: str,
    grounding_dino_model: str,
    detection_interval: int,
    person_prompt: str,
    tracking_device: str,
    depth_device: str,
) -> None:
    for video_path in video_paths:
        print(f"[Preprocess] {video_path}")
        generate_heightmaps_for_video(
            video_path=video_path,
            output_root=output_root,
            preview_root=preview_root,
            label_dir=label_dir,
            grounded_sam2_root=grounded_sam2_root,
            depth_pro_root=depth_pro_root,
            depth_pro_checkpoint=depth_pro_checkpoint,
            sam2_checkpoint=sam2_checkpoint,
            sam2_config=sam2_config,
            grounding_dino_model=grounding_dino_model,
            detection_interval=detection_interval,
            person_prompt=person_prompt,
            tracking_device=tracking_device,
            depth_device=depth_device,
        )

