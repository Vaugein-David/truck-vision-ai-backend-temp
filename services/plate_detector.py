"""Localisation des plaques d'immatriculation.

Deux stratégies :
1. Modèle YOLO dédié (models/plate_detector.pt) s'il est fourni — recommandé en production.
2. Sinon : détection des zones de texte (détecteur PP-OCR) puis filtrage par
   géométrie et par format d'immatriculation.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

import config
from services.plate_recognizer import PlateRecognizer, normalize, plate_format


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PlateDetector:
    def __init__(self, recognizer: PlateRecognizer) -> None:
        self.recognizer = recognizer
        self.model = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if config.PLATE_MODEL_CUSTOM.exists():
                try:
                    from ultralytics import YOLO

                    self.model = YOLO(str(config.PLATE_MODEL_CUSTOM))
                except Exception as exc:
                    self.error = str(exc)

    @property
    def available(self) -> bool:
        self.load()
        return self.model is not None or self.recognizer.available

    def detect(self, img: np.ndarray, vehicle_bbox=None) -> list[dict]:
        """Retourne des candidats {bbox, text, score, format} triés du plus au moins probable."""
        self.load()
        if self.model is not None:
            return self._detect_yolo(img)
        return self._detect_text(img, vehicle_bbox)

    def _detect_yolo(self, img: np.ndarray) -> list[dict]:
        with self._lock:
            res = self.model.predict(img, conf=0.25, verbose=False)[0]
        out = []
        for b in res.boxes:
            out.append({"bbox": [float(v) for v in b.xyxy[0]], "text": "", "score": float(b.conf[0]), "format": None})
        out.sort(key=lambda c: c["score"], reverse=True)
        return out

    def _regions(self, img: np.ndarray, offset=(0, 0), min_side: int = 0) -> list[dict]:
        scale = 1.0
        h, w = img.shape[:2]
        if min_side and max(h, w) < min_side:
            scale = min_side / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        regions = self.recognizer.read_regions(img)
        for r in regions:
            r["box"] = r["box"] / scale + np.array(offset, dtype=float)
        return regions

    def _detect_text(self, img: np.ndarray, vehicle_bbox) -> list[dict]:
        h, w = img.shape[:2]
        passes = []
        vb = None
        if vehicle_bbox:
            x1, y1, x2, y2 = (int(max(0, v)) for v in vehicle_bbox)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 40 and y2 - y1 > 40:
                vb = (x1, y1, x2, y2)
                passes.append((img[y1:y2, x1:x2], (x1, y1), 1280))
                ym = y1 + (y2 - y1) // 2  # moitié basse : zone habituelle de la plaque
                passes.append((img[ym:y2, x1:x2], (x1, ym), 1600))
        passes.append((img, (0, 0), 960))

        found: list[dict] = []
        for sub, off, min_side in passes:
            found += self._candidates(self._regions(sub, off, min_side), img, vb)
            if any(c["format"] in ("SIV", "FNI") and c["plate_like"] for c in found):
                break
        # Une même zone peut être vue par plusieurs passes : on garde la meilleure lecture.
        found.sort(key=lambda c: c["rank"], reverse=True)
        kept: list[dict] = []
        for c in found:
            if all(_iou(c["bbox"], k["bbox"]) < 0.4 for k in kept):
                kept.append(c)
        return kept

    @staticmethod
    def _light_background(img: np.ndarray, bbox) -> bool:
        """Une plaque porte des caractères sombres sur fond clair (blanc, jaune…).

        Écarte les marquages publicitaires clairs sur fond sombre (pare-soleil, bâche, portière).
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.shape[0] < 6 or crop.shape[1] < 12:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        band = max(1, gray.shape[0] // 8)
        background = np.concatenate([gray[:band].ravel(), gray[-band:].ravel()])
        return float(np.median(background)) > thr

    def _candidates(self, regions: list[dict], img: np.ndarray, vb) -> list[dict]:
        items = []
        for r in regions:
            xs, ys = r["box"][:, 0], r["box"][:, 1]
            items.append({
                "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
                "text": r["text"], "score": r["score"],
            })
        # Fusion de fragments voisins sur une même ligne (ex. « AB-123 » + « CD »).
        merged = list(items)
        srt = sorted(items, key=lambda i: i["bbox"][0])
        for i, a in enumerate(srt):
            for b in srt[i + 1:]:
                ha, hb = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
                same_line = abs((a["bbox"][1] + a["bbox"][3]) - (b["bbox"][1] + b["bbox"][3])) / 2 < 0.4 * max(ha, hb)
                gap = b["bbox"][0] - a["bbox"][2]
                if same_line and -0.2 * ha < gap < 1.2 * max(ha, hb) and 0.6 < ha / max(hb, 1e-6) < 1.6:
                    merged.append({
                        "bbox": [a["bbox"][0], min(a["bbox"][1], b["bbox"][1]), b["bbox"][2], max(a["bbox"][3], b["bbox"][3])],
                        "text": a["text"] + b["text"], "score": min(a["score"], b["score"]),
                    })
        out = []
        for c in merged:
            norm = normalize(c["text"])
            fmt = plate_format(norm)
            bw, bh = c["bbox"][2] - c["bbox"][0], c["bbox"][3] - c["bbox"][1]
            ratio = bw / max(bh, 1e-6)
            if not (1.5 <= ratio <= 8.0):
                continue
            light = self._light_background(img, c["bbox"])
            cy = (c["bbox"][1] + c["bbox"][3]) / 2
            # Sans véhicule détecté, la position est mesurée sur toute l'image.
            rel_y = (cy - vb[1]) / (vb[3] - vb[1]) if vb else cy / max(img.shape[0], 1)
            if fmt in ("SIV", "FNI"):
                # Format officiel : indice fort. Un fond sombre reste possible (photo de nuit) mais pénalisé.
                rank = c["score"] + (0.5 if fmt == "SIV" else 0.3) + (0.2 if light else -0.2)
            elif fmt == "AUTRE":
                # Texte alphanumérique quelconque : n'est retenu que s'il ressemble physiquement à une plaque.
                if rel_y is not None and rel_y < 0.3:
                    continue
                rank = c["score"] + (0.0 if light else -0.2)
            else:
                # Lecture partielle d'une zone qui a l'aspect d'une plaque, sur le véhicule :
                # la plaque est localisée mais sera déclarée illisible (aucun texte retourné).
                partial = len(norm) >= 4 and any(ch.isdigit() for ch in norm) and len(norm) <= 12
                if not (partial and rel_y is not None and 0.3 <= rel_y <= 1.0 and 2.0 <= ratio <= 7.0):
                    continue
                rank = c["score"] * 0.5
            out.append({**c, "format": fmt, "rank": rank, "plate_like": light})
        return out

    def info(self) -> dict:
        self.load()
        return {
            "role": "Localisation des plaques",
            "name": "YOLO site (plate_detector.pt)" if self.model is not None else "Détecteur de texte PP-OCR + filtrage par format",
            "loaded": self.available,
            "specialized": self.model is not None,
            "error": self.error,
        }
