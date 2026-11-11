"""Chaîne d'analyse d'une image : véhicule -> plaque -> propreté."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from services import image_processor as ip
from services.engines import cleanliness_classifier, plate_detector, plate_recognizer, vehicle_detector
from services.plate_recognizer import normalize, plate_format, pretty

MODES = ("PLATE", "CLEANLINESS", "FULL")
MIN_READABLE = 0.35  # en dessous : plaque déclarée illisible, aucun texte retourné


def read_plate(img: np.ndarray, vehicle_bbox, threshold: float) -> dict:
    """Localise, recadre, améliore et lit la plaque. Ne retourne jamais de texte inventé."""
    base = {"status": "NON_DETECTEE", "text": None, "confidence": 0.0, "bbox": None, "format": None,
            "engine": plate_recognizer.name, "message": "Aucune plaque d'immatriculation détectée dans l'image.",
            "crop": None}
    if not plate_detector.available:
        return {**base, "status": "INDISPONIBLE",
                "message": "Moteur OCR non installé : reconnaissance de plaque indisponible."}

    best = None      # meilleure lecture conforme à un format d'immatriculation
    located = None   # plaque localisée mais sans lecture exploitable
    for cand in plate_detector.detect(img, vehicle_bbox)[:4]:
        crop = ip.crop(img, cand["bbox"], margin=0.12)
        if crop.size == 0:
            continue
        readings = [(cand["text"], cand["score"])] if cand["text"] else []
        readings.append(plate_recognizer.read_crop(ip.enhance_plate(crop)))
        scored = []
        for text, score in readings:
            norm = normalize(text)
            fmt = plate_format(norm)
            # Un texte quelconque (format « AUTRE ») n'est accepté que si le détecteur l'a lui-même retenu.
            if fmt in ("SIV", "FNI") or (fmt == "AUTRE" and cand["format"] == "AUTRE"):
                scored.append((score + {"SIV": 0.5, "FNI": 0.3, "AUTRE": 0.0}[fmt], norm, fmt, score))
        if not scored:
            if located is None:
                located = {"bbox": cand["bbox"], "crop": crop, "score": cand["score"]}
            continue
        scored.sort(reverse=True)
        rank, norm, fmt, score = scored[0]
        if best is None or rank > best["rank"]:
            best = {"rank": rank, "norm": norm, "format": fmt, "score": score, "bbox": cand["bbox"], "crop": crop}

    if best is None and located is not None:
        return {**base, "status": "ILLISIBLE", "bbox": [round(v, 1) for v in located["bbox"]], "crop": located["crop"],
                "message": "Une plaque a été localisée mais sa lecture est incomplète (trop petite, floue ou masquée). "
                           "Aucune immatriculation n'est proposée."}
    if best is None:
        return base
    result = {**base, "bbox": [round(v, 1) for v in best["bbox"]], "crop": best["crop"],
              "confidence": round(best["score"], 4), "format": best["format"]}
    if best["score"] < MIN_READABLE:
        return {**result, "status": "ILLISIBLE", "format": None,
                "message": "Une plaque est présente mais elle est illisible (confiance trop faible)."}
    text = pretty(best["norm"], best["format"])
    if best["score"] >= threshold and best["format"] in ("SIV", "FNI"):
        return {**result, "status": "RECONNUE", "text": text, "message": "Plaque reconnue."}
    reason = ("format non standard ou lecture incomplète" if best["format"] == "AUTRE"
              else f"confiance inférieure au seuil ({threshold:.0%})")
    return {**result, "status": "A_VERIFIER", "text": text,
            "message": f"Lecture incertaine ({reason}) : vérification humaine nécessaire."}


def analyze_image(path: Path, out_dir: Path, mode: str, plate_threshold: float = 0.8,
                  clean_threshold: float = 0.7) -> dict:
    t0 = time.time()
    img = ip.load_image(path)
    ip.save_jpeg(out_dir / "thumb.jpg", ip.make_thumbnail(img), 80)

    vehicles = vehicle_detector.detect(img)
    vehicle = vehicles[0] if vehicles else None
    vbbox = vehicle["bbox"] if vehicle else None

    plate = cleanliness = None
    files = {"original": ip.rel(path), "thumbnail": ip.rel(out_dir / "thumb.jpg"), "annotated": None, "plate_crop": None}

    if mode in ("PLATE", "FULL"):
        plate = read_plate(img, vbbox, plate_threshold)
        crop = plate.pop("crop")
        if crop is not None:
            ip.save_jpeg(out_dir / "plate.jpg", crop, 92)
            files["plate_crop"] = ip.rel(out_dir / "plate.jpg")
    if mode in ("CLEANLINESS", "FULL"):
        cleanliness = cleanliness_classifier.analyze(img, vbbox, clean_threshold)

    if vbbox or (plate and plate["bbox"]):
        ann = ip.draw_annotations(img, vbbox, plate["bbox"] if plate else None, plate["text"] if plate else None)
        ip.save_jpeg(out_dir / "annotated.jpg", ann, 85)
        files["annotated"] = ip.rel(out_dir / "annotated.jpg")

    return {
        "demo": False,
        "mode": mode,
        "file_type": "IMAGE",
        "storage_dir": ip.rel(out_dir),
        "files": files,
        "vehicle": ({"detected": True, "label": vehicle["label"], "confidence": round(vehicle["confidence"], 4),
                     "bbox": [round(v, 1) for v in vbbox], "count": len(vehicles)}
                    if vehicle else {"detected": False, "count": 0,
                                     "available": vehicle_detector.available}),
        "plate": plate,
        "cleanliness": cleanliness,
        "duration_ms": int((time.time() - t0) * 1000),
    }
