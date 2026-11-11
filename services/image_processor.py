"""Validation, chargement, stockage et amélioration des images."""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import config

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm"}


class ValidationError(Exception):
    """Fichier refusé (format, taille ou contenu)."""


def sniff_kind(head: bytes) -> str | None:
    """Identifie le type réel du fichier d'après ses octets de signature."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return "avi"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip"):
        return "mp4"
    return None


def validate_upload(filename: str, head: bytes, size: int, kind: str, max_mb: float) -> str:
    """Vérifie extension, signature et taille. Retourne l'extension normalisée."""
    ext = Path(filename or "").suffix.lower()
    allowed = IMAGE_EXT if kind == "image" else VIDEO_EXT
    if ext not in allowed:
        raise ValidationError(
            f"Format « {ext or 'inconnu'} » non accepté. Formats autorisés : "
            + ", ".join(sorted(e[1:].upper() for e in allowed)) + "."
        )
    if size <= 0:
        raise ValidationError("Le fichier est vide.")
    if size > max_mb * 1024 * 1024:
        raise ValidationError(
            f"Fichier trop volumineux ({size / 1048576:.1f} Mo). Limite : {max_mb:g} Mo."
        )
    real = sniff_kind(head)
    expected = {"jpeg", "png", "webp"} if kind == "image" else {"mp4", "avi", "webm"}
    if real not in expected:
        raise ValidationError(
            "Le contenu du fichier ne correspond pas à son extension (fichier corrompu ou non autorisé)."
        )
    return ext


def new_storage_dir(category: str = "analyses") -> tuple[str, Path]:
    """Crée storage/<category>/<AAAA-MM>/<uuid>/ et retourne (id, chemin)."""
    sid = uuid.uuid4().hex
    d = config.STORAGE_DIR / category / datetime.now().strftime("%Y-%m") / sid
    d.mkdir(parents=True, exist_ok=True)
    return sid, d


def rel(path: Path) -> str:
    """Chemin relatif au stockage (utilisé dans les réponses API)."""
    return path.resolve().relative_to(config.STORAGE_DIR).as_posix()


def safe_path(relative: str) -> Path:
    """Résout un chemin relatif en interdisant toute sortie du stockage."""
    p = (config.STORAGE_DIR / relative).resolve()
    if p != config.STORAGE_DIR and config.STORAGE_DIR not in p.parents:
        raise ValidationError("Chemin non autorisé.")
    return p


def delete_dir(relative: str) -> bool:
    p = safe_path(relative)
    if p == config.STORAGE_DIR or not p.is_dir():
        return False
    shutil.rmtree(p, ignore_errors=True)
    return True


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValidationError("Image illisible : le fichier ne contient pas une image valide.")
    return img


def save_jpeg(path: Path, img: np.ndarray, quality: int = 88) -> Path:
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def make_thumbnail(img: np.ndarray, width: int = 480) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)


def crop(img: np.ndarray, bbox, margin: float = 0.0) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return img[y1:y2, x1:x2]


def enhance_plate(plate: np.ndarray) -> np.ndarray:
    """Améliore la lisibilité d'un recadrage de plaque (agrandissement, contraste, netteté)."""
    if plate.size == 0:
        return plate
    h, w = plate.shape[:2]
    if h < 96:
        s = 96 / h
        plate = cv2.resize(plate, (int(w * s), 96), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(plate, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 2)
    return cv2.addWeighted(out, 1.5, blur, -0.5, 0)


def quality_metrics(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return {
        "brightness": float(gray.mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
    }


def draw_annotations(img: np.ndarray, vehicle_bbox=None, plate_bbox=None, plate_text: str | None = None) -> np.ndarray:
    out = img.copy()
    t = max(2, int(min(out.shape[:2]) / 300))
    if vehicle_bbox:
        x1, y1, x2, y2 = map(int, vehicle_bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (150, 70, 20), t)
    if plate_bbox:
        x1, y1, x2, y2 = map(int, plate_bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (60, 180, 30), t)
        if plate_text:
            scale = max(0.6, out.shape[1] / 1400)
            cv2.putText(out, plate_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, scale, (60, 180, 30), t, cv2.LINE_AA)
    return out
