"""Lecture OCR des plaques d'immatriculation.

Moteurs pris en charge :
- PaddleOCR (si le paquet `paddleocr` est installé, voir requirements-paddle.txt) ;
- RapidOCR : les mêmes modèles PP-OCR de PaddleOCR exécutés via ONNX Runtime
  (léger, CPU, adapté à une petite machine locale). Moteur par défaut.

Règle absolue : on ne retourne jamais de texte inventé. Une lecture incertaine
est signalée comme telle, une lecture trop faible est rejetée.
"""
from __future__ import annotations

import re
import threading

import cv2
import numpy as np

import config

SIV = re.compile(r"^[A-HJ-NP-TV-Z]{2}\d{3}[A-HJ-NP-TV-Z]{2}$")   # AA-123-AA (depuis 2009)
FNI = re.compile(r"^\d{2,4}[A-Z]{2,3}\d{2,3}$")                   # 1234 AB 56 (avant 2009)
GENERIC = re.compile(r"^(?=.*\d)(?=.*[A-Z])[A-Z0-9]{5,9}$")


def normalize(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def plate_format(norm: str) -> str | None:
    if SIV.match(norm):
        return "SIV"
    if FNI.match(norm):
        return "FNI"
    if GENERIC.match(norm):
        return "AUTRE"
    return None


def pretty(norm: str, fmt: str | None) -> str:
    if fmt == "SIV":
        return f"{norm[:2]}-{norm[2:5]}-{norm[5:]}"
    return norm


class PlateRecognizer:
    def __init__(self) -> None:
        self.engine = None
        self.kind: str | None = None
        self.name = "non chargé"
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> bool:
        with self._lock:
            if self._loaded:
                return self.engine is not None
            self._loaded = True
            wanted = config.OCR_ENGINE.lower()
            if wanted in ("auto", "paddleocr"):
                try:
                    from paddleocr import PaddleOCR

                    try:
                        self.engine = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
                    except Exception:
                        self.engine = PaddleOCR(lang="en")
                    self.kind, self.name = "paddleocr", "PaddleOCR (PP-OCR)"
                    return True
                except Exception as exc:
                    if wanted == "paddleocr":
                        self.error = f"PaddleOCR indisponible : {exc}"
            if wanted in ("auto", "rapidocr"):
                try:
                    from rapidocr_onnxruntime import RapidOCR

                    self.engine = RapidOCR()
                    self.kind, self.name = "rapidocr", "PP-OCR (PaddleOCR) via ONNX Runtime"
                    self.error = None
                    return True
                except Exception as exc:
                    self.error = f"Moteur OCR indisponible : {exc}"
            return False

    @property
    def available(self) -> bool:
        return self.load()

    # -- lecture de zones de texte dans une image (détection + reconnaissance) --
    def read_regions(self, img: np.ndarray) -> list[dict]:
        if not self.load() or img is None or img.size == 0:
            return []
        out: list[dict] = []
        with self._lock:
            if self.kind == "rapidocr":
                result, _ = self.engine(img)
                for box, text, score in result or []:
                    out.append({"box": np.array(box, dtype=float), "text": str(text), "score": float(score)})
            else:
                out = self._paddle_regions(img)
        return out

    def _paddle_regions(self, img: np.ndarray) -> list[dict]:
        out: list[dict] = []
        try:  # API PaddleOCR 2.x
            res = self.engine.ocr(img, cls=False)
            for line in (res[0] if res and res[0] else []):
                box, (text, score) = line
                out.append({"box": np.array(box, dtype=float), "text": str(text), "score": float(score)})
        except TypeError:  # API PaddleOCR 3.x
            for page in self.engine.predict(img):
                for box, text, score in zip(page["rec_polys"], page["rec_texts"], page["rec_scores"]):
                    out.append({"box": np.array(box, dtype=float), "text": str(text), "score": float(score)})
        return out

    # -- lecture d'un recadrage de plaque (reconnaissance seule) --
    def read_plate_variants(self, crop: np.ndarray) -> list[tuple[str, float]]:
        """Lectures complètes (détection + reconnaissance) du recadrage brut et agrandi x2."""
        if not self.load() or crop is None or crop.size == 0:
            return []
        out: list[tuple[str, float]] = []
        big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        for variant in (crop, big):
            for r in self.read_regions(variant):
                if r["text"]:
                    out.append((r["text"], r["score"]))
        return out

    def read_crop(self, crop: np.ndarray) -> tuple[str, float]:
        if not self.load() or crop is None or crop.size == 0:
            return "", 0.0
        with self._lock:
            if self.kind == "rapidocr":
                result, _ = self.engine(crop, use_det=False, use_cls=False, use_rec=True)
                if result:
                    return str(result[0][0]), float(result[0][1])
                return "", 0.0
        regions = self.read_regions(crop)
        if not regions:
            return "", 0.0
        best = max(regions, key=lambda r: r["score"])
        return best["text"], best["score"]

    def info(self) -> dict:
        self.load()
        return {
            "role": "Lecture des plaques (OCR)",
            "name": self.name,
            "loaded": self.engine is not None,
            "specialized": True,
            "error": self.error,
        }
