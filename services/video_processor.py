"""Traitement vidéo : extraction d'images, suivi, plaques, propreté, rapport par camion."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Callable

import cv2

from services import image_processor as ip
from services.engines import cleanliness_classifier, plate_detector, plate_recognizer
from services.pipeline import MIN_READABLE, read_plate
from services.plate_recognizer import normalize, plate_format, pretty
from services.truck_tracker import Track, TruckTracker

MAX_OCR_PER_TRACK = 10
Progress = Callable[..., None]


def probe(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ip.ValidationError("Vidéo illisible : le fichier ne peut pas être décodé.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 0 or fps > 240:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ip.ValidationError("Vidéo illisible : aucune image décodable.")
    return {"fps": round(fps, 2), "frames_total": total, "duration_s": round(total / fps, 2) if total else 0.0,
            "width": int(frame.shape[1]), "height": int(frame.shape[0]), "first_frame": frame}


def _plate_done(tr: Track, threshold: float) -> bool:
    cons = tr.plate_consensus()
    return bool(cons and cons["votes"] >= 3 and cons["score"] >= threshold)


def _plate_result(tr: Track, threshold: float) -> dict:
    cons = tr.plate_consensus()
    base = {"status": "NON_DETECTEE", "text": None, "confidence": 0.0, "format": None, "engine": plate_recognizer.name,
            "message": "Aucune plaque lisible sur les images de ce camion.", "votes": 0, "readings": 0, "timestamp_s": None}
    if not plate_detector.available:
        return {**base, "status": "INDISPONIBLE", "message": "Moteur OCR non installé."}
    if cons is None:
        return base
    res = {**base, "confidence": round(cons["score"], 4), "format": cons["format"], "votes": cons["votes"],
           "readings": cons["total"], "timestamp_s": round(cons["t"], 2)}
    if cons["score"] < MIN_READABLE:
        return {**res, "status": "ILLISIBLE", "format": None, "message": "Plaque présente mais illisible."}
    text = pretty(cons["norm"], cons["format"])
    if cons["score"] >= threshold and cons["format"] in ("SIV", "FNI"):
        return {**res, "status": "RECONNUE", "text": text,
                "message": f"Plaque reconnue ({cons['votes']} lecture(s) concordante(s) sur {cons['total']})."}
    return {**res, "status": "A_VERIFIER", "text": text,
            "message": "Lecture incertaine : vérification humaine nécessaire."}


def _clean_result(tr: Track, threshold: float) -> dict:
    """Agrège l'analyse de propreté sur les images où le camion est le plus visible."""
    results = []
    for _area, frame, bbox, _t in tr.clean_samples:
        r = cleanliness_classifier.analyze(frame, bbox, threshold)
        if r["category"]:
            results.append(r)
    if not results:
        return {"status": "IMPOSSIBLE", "category": None, "confidence": 0.0, "elements": [],
                "comment": "L'intérieur de la benne n'a pu être évalué sur aucune image de ce camion.",
                "engine": cleanliness_classifier.engine_name, "frames_used": 0}
    cat, n = Counter(r["category"] for r in results).most_common(1)[0]
    same = [r for r in results if r["category"] == cat]
    best = max(same, key=lambda r: r["confidence"])
    conf = sum(r["confidence"] for r in same) / len(same) * (n / len(results))
    status = best["status"] if n == len(results) else "A_VERIFIER"
    comment = best["comment"]
    if n != len(results):
        comment += f" Résultats divergents selon les images ({n}/{len(results)} concordantes)."
    return {"status": status, "category": cat, "confidence": round(conf, 4), "elements": best["elements"],
            "comment": comment, "engine": best["engine"], "frames_used": len(results)}


def process_video(path: Path, out_dir: Path, mode: str, sample_fps: float = 2.0, plate_threshold: float = 0.8,
                  clean_threshold: float = 0.7, progress: Progress = lambda **kw: None) -> dict:
    t0 = time.time()
    meta = probe(path)
    first = meta.pop("first_frame")
    ip.save_jpeg(out_dir / "thumb.jpg", ip.make_thumbnail(first), 80)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    step = max(1, int(round(meta["fps"] / max(0.1, sample_fps))))
    to_analyze = max(1, meta["frames_total"] // step) if meta["frames_total"] else 0
    tracker = TruckTracker()
    do_plate, do_clean = mode in ("PLATE", "FULL"), mode in ("CLEANLINESS", "FULL")

    cap = cv2.VideoCapture(str(path))
    idx = analyzed = 0
    keyframes = []  # utilisées si aucun véhicule n'est suivi
    progress(status="processing", frames_total=to_analyze, message="Analyse des images…")
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                t = idx / meta["fps"]
                analyzed += 1
                for tr in tracker.update(frame, t):
                    bbox = tr.last_bbox
                    if (do_plate and bbox and tr.ocr_attempts < MAX_OCR_PER_TRACK
                            and not _plate_done(tr, plate_threshold) and (bbox[2] - bbox[0]) >= 90):
                        tr.ocr_attempts += 1
                        p = read_plate(frame, bbox, plate_threshold)
                        if p["text"] or p["status"] == "ILLISIBLE":
                            norm = normalize(p["text"] or "")
                            tr.plate_reads.append({"t": t, "norm": norm, "format": plate_format(norm) if norm else None,
                                                   "score": p["confidence"], "crop": p["crop"]})
                if len(keyframes) < 6 and (not to_analyze or analyzed % max(1, to_analyze // 6) == 0):
                    keyframes.append((t, frame.copy()))
                pct = min(95, int(analyzed / to_analyze * 95)) if to_analyze else 0
                progress(progress=pct, frames_processed=analyzed, trucks_detected=len(tracker.tracks))
        idx += 1
    cap.release()

    progress(progress=96, message="Regroupement des détections et synthèse…")
    trucks = []
    for n, tr in enumerate(tracker.finalize(), start=1):
        files = {"snapshot": None, "plate_crop": None}
        plate = clean = None
        if tr.best_frame is not None:
            snap = ip.draw_annotations(tr.best_frame, tr.best_bbox)
            ip.save_jpeg(frames_dir / f"truck_{n}.jpg", ip.make_thumbnail(snap, 960), 85)
            files["snapshot"] = ip.rel(frames_dir / f"truck_{n}.jpg")
        if do_plate:
            cons = tr.plate_consensus()
            plate = _plate_result(tr, plate_threshold)
            if cons and cons["crop"] is not None and plate["status"] != "NON_DETECTEE":
                ip.save_jpeg(frames_dir / f"truck_{n}_plate.jpg", cons["crop"], 92)
                files["plate_crop"] = ip.rel(frames_dir / f"truck_{n}_plate.jpg")
        if do_clean:
            clean = _clean_result(tr, clean_threshold)
        detections = [{"t": round(r["t"], 2), "text": pretty(r["norm"], r["format"]) if r["norm"] and r["score"] >= MIN_READABLE else None,
                       "confidence": round(r["score"], 4)} for r in tr.plate_reads]
        trucks.append({"index": n, "track_id": tr.track_id, "vehicle_type": tr.label,
                       "first_seen_s": round(tr.first_t, 2), "last_seen_s": round(tr.last_t, 2),
                       "frames_seen": tr.hits, "plate": plate, "cleanliness": clean, "files": files,
                       "detections": detections})

    note = None
    if not trucks and keyframes:
        # Aucun véhicule suivi (vue de dessus, détecteur généraliste…) : analyse globale prudente.
        note = ("Aucun véhicule n'a pu être suivi dans cette vidéo. Une analyse globale a été réalisée "
                "sur quelques images ; le résultat doit être vérifié par un opérateur."
                if tracker.available else
                "Détecteur de véhicules indisponible : analyse globale sur quelques images, à vérifier.")
        t_mid, mid = keyframes[len(keyframes) // 2]
        files = {"snapshot": None, "plate_crop": None}
        ip.save_jpeg(frames_dir / "truck_1.jpg", ip.make_thumbnail(mid, 960), 85)
        files["snapshot"] = ip.rel(frames_dir / "truck_1.jpg")
        plate = clean = None
        if do_plate:
            reads = [read_plate(f, None, plate_threshold) for _t, f in keyframes]
            plate = max(reads, key=lambda p: (p["text"] is not None, p["confidence"]))
            crop = plate.pop("crop", None)
            if plate["status"] == "RECONNUE":
                plate["status"] = "A_VERIFIER"
                plate["message"] = "Plaque lue hors suivi de véhicule : vérification nécessaire."
            if crop is not None:
                ip.save_jpeg(frames_dir / "truck_1_plate.jpg", crop, 92)
                files["plate_crop"] = ip.rel(frames_dir / "truck_1_plate.jpg")
            plate.pop("bbox", None)
        if do_clean:
            clean = cleanliness_classifier.analyze(mid, None, clean_threshold)
            if clean["status"] == "OK":
                clean["status"] = "A_VERIFIER"
        trucks.append({"index": 1, "track_id": None, "vehicle_type": "Non identifié",
                       "first_seen_s": 0.0, "last_seen_s": meta["duration_s"], "frames_seen": 0,
                       "plate": plate, "cleanliness": clean, "files": files, "detections": []})

    return {
        "demo": False, "mode": mode, "file_type": "VIDEO", "storage_dir": ip.rel(out_dir),
        "files": {"original": ip.rel(path), "thumbnail": ip.rel(out_dir / "thumb.jpg")},
        "video": {**meta, "sample_fps": sample_fps},
        "frames_analyzed": analyzed, "trucks_detected": len(trucks) if not note else 0,
        "trucks": trucks, "note": note,
        "duration_ms": int((time.time() - t0) * 1000),
    }
