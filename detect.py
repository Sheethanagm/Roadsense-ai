from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np


HAZARD_CLASSES = [
    "Pothole",
    "Road Crack",
    "Waterlogging",
    "Fallen Tree",
    "Road Debris",
    "Animal",
    "Damaged Vehicle",
    "Accident Zone",
]

PROJECT_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = PROJECT_DIR / "data" / "ultralytics_config"
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "road_hazard_mpl"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

HIGH_SEVERITY = {"Pothole", "Fallen Tree", "Accident Zone", "Animal"}
MEDIUM_SEVERITY = {"Waterlogging", "Damaged Vehicle", "Road Debris", "Road Crack"}

SEVERITY_COLORS = {
    "Low": (0, 200, 0),
    "Medium": (0, 165, 255),
    "High": (0, 0, 255),
}

ALIAS_MAP = {
    "pothole": "Pothole",
    "road crack": "Road Crack",
    "crack": "Road Crack",
    "waterlogging": "Waterlogging",
    "water puddle": "Waterlogging",
    "flood": "Waterlogging",
    "fallen tree": "Fallen Tree",
    "tree": "Fallen Tree",
    "road debris": "Road Debris",
    "debris": "Road Debris",
    "animal": "Animal",
    "animals on road": "Animal",
    "damaged vehicle": "Damaged Vehicle",
    "accident zone": "Accident Zone",
    "accident scene": "Accident Zone",
}


def normalize_label(label: str) -> str:
    cleaned = label.strip().lower()
    return ALIAS_MAP.get(cleaned, label.strip().title())


def classify_severity(hazard_type: str, confidence: float) -> str:
    normalized = normalize_label(hazard_type)
    if normalized in HIGH_SEVERITY:
        return "High" if confidence >= 0.25 else "Medium"
    if normalized in MEDIUM_SEVERITY:
        return "Medium"
    return "Low"


def recommendation_for(hazard_type: str, severity: str) -> str:
    normalized = normalize_label(hazard_type)
    base = {
        "Pothole": "Reduce speed and avoid abrupt lane changes.",
        "Road Crack": "Slow down and maintain lane discipline.",
        "Waterlogging": "Drive cautiously and avoid deep water patches.",
        "Fallen Tree": "Stop and reroute immediately.",
        "Road Debris": "Steer around the obstacle with caution.",
        "Animal": "Slow down and keep a safe distance.",
        "Damaged Vehicle": "Reduce speed and prepare to stop if needed.",
        "Accident Zone": "Proceed carefully and follow diversion guidance.",
    }.get(normalized, "Proceed with caution.")
    if severity == "High":
        return base
    if severity == "Medium":
        return base + " Monitor the road ahead closely."
    return base + " Maintain awareness."


@dataclass
class DetectorStatus:
    is_ready: bool
    status_message: str


class DetectionEngine:
    def __init__(self, model_path: Path, imgsz: int = 640) -> None:
        self.model_path = Path(model_path)
        self.imgsz = imgsz
        self.model = None
        self.names = {}
        self.status = DetectorStatus(False, "Model not loaded")
        self._load_model()

    @property
    def is_ready(self) -> bool:
        return self.status.is_ready

    @property
    def status_message(self) -> str:
        return self.status.status_message

    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.status = DetectorStatus(
                False,
                f"Missing model file: {self.model_path.name}. Add a trained YOLOv8 model to model/best.pt.",
            )
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.names = getattr(self.model, "names", {})
            self.status = DetectorStatus(True, f"Loaded {self.model_path.name}")
        except Exception as exc:
            self.model = None
            self.status = DetectorStatus(False, f"Failed to load YOLO model: {exc}")

    def _predict(self, frame_bgr: np.ndarray, confidence: float) -> list[dict]:
        if self.model is None:
            return []

        results = self.model.predict(source=frame_bgr, conf=confidence, imgsz=self.imgsz, verbose=False)
        detections: list[dict] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0]) if box.cls is not None else -1
                raw_label = self.names.get(class_id, str(class_id)) if isinstance(self.names, dict) else str(class_id)
                hazard_type = normalize_label(raw_label)
                score = float(box.conf[0]) if box.conf is not None else 0.0
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                severity = classify_severity(hazard_type, score)
                detections.append(
                    {
                        "hazard_type": hazard_type,
                        "confidence": score,
                        "severity": severity,
                        "bbox": [x1, y1, x2, y2],
                        "recommendation": recommendation_for(hazard_type, severity),
                    }
                )
        return detections

    def detect_image(self, frame_bgr: np.ndarray, confidence: float = 0.25) -> tuple[np.ndarray, list[dict]]:
        annotated = frame_bgr.copy()
        detections = self._predict(frame_bgr, confidence)
        annotate_frame(annotated, detections)
        return annotated, detections


def annotate_frame(frame_bgr: np.ndarray, detections: Iterable[dict]) -> np.ndarray:
    height = frame_bgr.shape[0]
    overlay = frame_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        severity = detection["severity"]
        color = SEVERITY_COLORS.get(severity, (255, 255, 255))
        label = f"{detection['hazard_type']} {detection['confidence']:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(0, y1 - label_size[1] - baseline - 6)
        cv2.rectangle(
            overlay,
            (x1, label_y),
            (x1 + label_size[0] + 10, label_y + label_size[1] + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            label,
            (x1 + 5, label_y + label_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    if detections:
        cv2.putText(
            overlay,
            f"Hazards detected: {len(detections)}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    frame_bgr[:] = overlay
    return frame_bgr


def load_image_file(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")
    return image


def save_image(output_path: Path, image_bgr: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)


def process_video_file(
    detector: DetectionEngine,
    input_path: Path,
    output_path: Path,
    confidence: float,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], int]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detection_log: list[dict] = []
    processed_frames = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        annotated, detections = detector.detect_image(frame, confidence)
        detection_log.extend(detections)
        writer.write(annotated)
        processed_frames += 1
        if progress_callback and total_frames:
            progress_callback(
                min(processed_frames / total_frames, 1.0),
                f"Processing frame {processed_frames}/{total_frames}",
            )

    capture.release()
    writer.release()
    if progress_callback:
        progress_callback(1.0, "Video processing complete")
    return detection_log, processed_frames
