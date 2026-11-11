"""Détection des véhicules avec YOLO (Ultralytics).

Par défaut : YOLOv8n pré-entraîné COCO (classes voiture / bus / camion).
Pour un site industriel, déposer un modèle entraîné dans models/vehicle_detector.pt.
"""
from __future__ import annotations

import threading

import numpy as np

import config


class VehicleDetector:
    def __init__(self) -> None:
        self.model = None
        self.name = "non chargé"
        self.custom = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loaded = False

    def weights_path(self) -> str:
        if config.VEHICLE_MODEL_CUSTOM.exists():
            return str(config.VEHICLE_MODEL_CUSTOM)
        return str(config.VEHICLE_MODEL_DEFAULT)

    def load(self) -> bool:
        with self._lock:
            if self._loaded:
                return self.model is not None
            self._loaded = True
            try:
                from ultralytics import YOLO

                self.custom = config.VEHICLE_MODEL_CUSTOM.exists()
                self.model = YOLO(self.weights_path())
                self.name = "YOLO site (vehicle_detector.pt)" if self.custom else "YOLOv8n COCO (généraliste)"
            except Exception as exc:  # modèle absent, hors-ligne, etc.
                self.error = str(exc)
                self.model = None
            return self.model is not None

    @property
    def available(self) -> bool:
        return self.load()

    def _classes(self):
        return None if self.custom else list(config.VEHICLE_CLASSES.keys())

    def label(self, cls_id: int) -> str:
        if self.custom and self.model is not None:
            return str(self.model.names.get(cls_id, "Véhicule"))
        return config.VEHICLE_CLASSES.get(cls_id, "Véhicule")

    def detect(self, img: np.ndarray, conf: float = 0.3) -> list[dict]:
        """Retourne les véhicules détectés, du plus grand au plus petit."""
        if not self.load():
            return []
        with self._lock:
            res = self.model.predict(img, conf=conf, classes=self._classes(), verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(b.conf[0]),
                "label": self.label(int(b.cls[0])),
                "area": (x2 - x1) * (y2 - y1),
            })
        out.sort(key=lambda d: d["area"], reverse=True)
        return out

    def info(self) -> dict:
        self.load()
        return {
            "role": "Détection des véhicules",
            "name": self.name,
            "loaded": self.model is not None,
            "specialized": self.custom,
            "error": self.error,
        }
