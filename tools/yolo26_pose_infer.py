#!/usr/bin/env python3
"""Run YOLO26 pose inference on an image and export human keypoints as JSON.

Example:
    python tools/yolo26_pose_infer.py \
        --image path/to/person.jpg \
        --output outputs/person_keypoints.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an Ultralytics YOLO26 pose model to infer human keypoints "
            "from one image and save JSON output."
        )
    )
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to the output JSON file. "
            "Parent directories are created automatically."
        ),
    )
    parser.add_argument(
        "--model",
        default="yolo26n-pose.pt",
        help=(
            "YOLO26 pose checkpoint name/path, e.g. yolo26n-pose.pt, "
            "yolo26s-pose.pt, or a custom .pt file."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Inference device passed to Ultralytics, e.g. cpu, 0, 0,1. "
            "Defaults to Ultralytics auto-selection."
        ),
    )
    return parser.parse_args()


def tensor_rows_to_list(tensor: Any) -> list[list[float]]:
    """Convert a torch tensor-like object with rows into a plain nested float list."""
    return tensor.detach().cpu().numpy().astype(float).tolist()


def scalar_tensor_to_list(tensor: Any) -> list[float]:
    """Convert a 1-D torch tensor-like object into a plain float list."""
    return tensor.detach().cpu().numpy().astype(float).tolist()


def build_prediction(
    result: Any,
    image_path: Path,
    model_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image_height, image_width = result.orig_shape
    boxes_xyxy = (
        tensor_rows_to_list(result.boxes.xyxy) if result.boxes is not None else []
    )
    box_confidences = (
        scalar_tensor_to_list(result.boxes.conf) if result.boxes is not None else []
    )
    box_classes = (
        scalar_tensor_to_list(result.boxes.cls) if result.boxes is not None else []
    )

    keypoints_xy = (
        tensor_rows_to_list(result.keypoints.xy) if result.keypoints is not None else []
    )
    keypoints_conf = (
        tensor_rows_to_list(result.keypoints.conf)
        if result.keypoints is not None and result.keypoints.conf is not None
        else []
    )

    predictions: list[dict[str, Any]] = []
    names = result.names or {}

    for person_index, xy_points in enumerate(keypoints_xy):
        confidence_points = (
            keypoints_conf[person_index] if person_index < len(keypoints_conf) else []
        )
        keypoints = []
        for keypoint_index, xy in enumerate(xy_points):
            confidence = (
                confidence_points[keypoint_index]
                if keypoint_index < len(confidence_points)
                else None
            )
            keypoints.append(
                {
                    "index": keypoint_index,
                    "name": COCO_KEYPOINT_NAMES[keypoint_index]
                    if keypoint_index < len(COCO_KEYPOINT_NAMES)
                    else f"keypoint_{keypoint_index}",
                    "x": xy[0],
                    "y": xy[1],
                    "confidence": confidence,
                    "visible": confidence is None or confidence >= args.conf,
                }
            )

        class_id = (
            int(box_classes[person_index])
            if person_index < len(box_classes)
            else None
        )
        predictions.append(
            {
                "person_id": person_index,
                "class_id": class_id,
                "class_name": (
                    names.get(class_id, "person")
                    if class_id is not None
                    else "person"
                ),
                "box": {
                    "xyxy": (
                        boxes_xyxy[person_index]
                        if person_index < len(boxes_xyxy)
                        else None
                    ),
                    "confidence": (
                        box_confidences[person_index]
                        if person_index < len(box_confidences)
                        else None
                    ),
                },
                "keypoints": keypoints,
            }
        )

    return {
        "image": {
            "path": str(image_path),
            "width": image_width,
            "height": image_height,
        },
        "model": model_name,
        "inference": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "device": args.device,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "keypoint_format": "COCO-17 [x, y, confidence] in input-image pixel coordinates",
        "num_persons": len(predictions),
        "predictions": predictions,
    }


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    predict_kwargs = {
        "source": str(image_path),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
    }
    if args.device is not None:
        predict_kwargs["device"] = args.device

    results = model.predict(**predict_kwargs)

    if len(results) != 1:
        raise RuntimeError(
            f"Expected one inference result for one image, got {len(results)} results."
        )

    payload = build_prediction(results[0], image_path, args.model, args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {payload['num_persons']} person prediction(s) to {output_path}")


if __name__ == "__main__":
    main()
