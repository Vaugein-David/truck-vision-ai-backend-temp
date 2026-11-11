"""Mode DEMO : résultats et images FICTIFS, toujours marqués `demo: true`.

Ce module n'est jamais appelé en mode réel. Il permet de présenter l'interface
sans modèle entraîné ; toutes les valeurs produites ici sont simulées.
"""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import cv2
import numpy as np

from services import image_processor as ip
from services.video_processor import probe

LETTERS = "ABCDEFGHJKLMNPQRSTVWXYZ"
ENGINE = "Simulation (mode DEMO)"
CLEAN_TEXT = {
    "PROPRE": (["Aucun résidu significatif"], "Benne vide, sans résidus significatifs visibles."),
    "MOYEN": (["Traces de terre sur le fond", "Résidus dans les coins"], "Quelques traces de terre et de résidus de matériaux."),
    "SALE": (["Boue sur le fond", "Matériaux restants", "Parois souillées"], "Présence importante de boue et de matériaux restants."),
}


def random_plate(rng: random.Random) -> str:
    l = lambda: "".join(rng.choice(LETTERS) for _ in range(2))
    return f"{l()}-{rng.randint(0, 999):03d}-{l()}"


def _plate(rng: random.Random, text: str | None = None) -> dict:
    conf = round(rng.uniform(0.62, 0.99), 4)
    ok = conf >= 0.8
    return {"status": "RECONNUE" if ok else "A_VERIFIER", "text": text or random_plate(rng), "confidence": conf,
            "bbox": None, "format": "SIV", "engine": ENGINE,
            "message": "Résultat fictif généré par le mode DEMO." if ok else "Lecture incertaine simulée (mode DEMO)."}


def _clean(rng: random.Random, category: str | None = None) -> dict:
    category = category or rng.choices(["PROPRE", "MOYEN", "SALE"], weights=[5, 3, 2])[0]
    conf = round(rng.uniform(0.6, 0.97), 4)
    elements, comment = CLEAN_TEXT[category]
    return {"status": "OK" if conf >= 0.7 else "A_VERIFIER", "category": category, "confidence": conf,
            "elements": elements, "comment": comment + " (Résultat fictif — mode DEMO.)", "engine": ENGINE}


def demo_image_result(path: Path, out_dir: Path, mode: str) -> dict:
    t0 = time.time()
    img = ip.load_image(path)
    ip.save_jpeg(out_dir / "thumb.jpg", ip.make_thumbnail(img), 80)
    rng = random.Random(hashlib.sha256(path.read_bytes()).hexdigest())
    time.sleep(0.6)
    return {
        "demo": True, "mode": mode, "file_type": "IMAGE", "storage_dir": ip.rel(out_dir),
        "files": {"original": ip.rel(path), "thumbnail": ip.rel(out_dir / "thumb.jpg"), "annotated": None, "plate_crop": None},
        "vehicle": {"detected": True, "label": "Camion (simulé)", "confidence": 0.9, "bbox": None, "count": 1},
        "plate": _plate(rng) if mode in ("PLATE", "FULL") else None,
        "cleanliness": _clean(rng) if mode in ("CLEANLINESS", "FULL") else None,
        "duration_ms": int((time.time() - t0) * 1000),
    }


def demo_video_result(path: Path, out_dir: Path, mode: str, progress) -> dict:
    t0 = time.time()
    meta = probe(path)
    first = meta.pop("first_frame")
    ip.save_jpeg(out_dir / "thumb.jpg", ip.make_thumbnail(first), 80)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    rng = random.Random(hashlib.sha256(path.read_bytes()[:1_000_000]).hexdigest())
    n_trucks = rng.randint(1, 3)
    steps = 30
    progress(status="processing", frames_total=steps, message="Simulation du traitement (mode DEMO)…")
    for i in range(steps):
        time.sleep(0.15)
        progress(progress=int((i + 1) / steps * 95), frames_processed=i + 1,
                 trucks_detected=min(n_trucks, 1 + i * n_trucks // steps))
    cap = cv2.VideoCapture(str(path))
    trucks = []
    dur = meta["duration_s"] or 1.0
    for n in range(1, n_trucks + 1):
        t_first = round(dur * (n - 1) / n_trucks, 2)
        t_last = round(min(dur, t_first + dur / n_trucks * 0.9), 2)
        cap.set(cv2.CAP_PROP_POS_MSEC, (t_first + t_last) / 2 * 1000)
        ok, frame = cap.read()
        files = {"snapshot": None, "plate_crop": None}
        if ok and frame is not None:
            ip.save_jpeg(frames_dir / f"truck_{n}.jpg", ip.make_thumbnail(frame, 960), 85)
            files["snapshot"] = ip.rel(frames_dir / f"truck_{n}.jpg")
        plate = _plate(rng) if mode in ("PLATE", "FULL") else None
        if plate:
            plate.update({"votes": 3, "readings": 4, "timestamp_s": round((t_first + t_last) / 2, 2)})
            plate.pop("bbox")
        trucks.append({"index": n, "track_id": n, "vehicle_type": "Camion (simulé)", "first_seen_s": t_first,
                       "last_seen_s": t_last, "frames_seen": steps // n_trucks, "plate": plate,
                       "cleanliness": _clean(rng) if mode in ("CLEANLINESS", "FULL") else None, "files": files,
                       "detections": ([{"t": plate["timestamp_s"], "text": plate["text"], "confidence": plate["confidence"]}] if plate else [])})
    cap.release()
    return {"demo": True, "mode": mode, "file_type": "VIDEO", "storage_dir": ip.rel(out_dir),
            "files": {"original": ip.rel(path), "thumbnail": ip.rel(out_dir / "thumb.jpg")},
            "video": {**meta, "sample_fps": 2.0}, "frames_analyzed": steps, "trucks_detected": n_trucks,
            "trucks": trucks, "note": "Résultats fictifs générés par le mode DEMO.",
            "duration_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------- images synthétiques
def draw_demo_truck(plate: str, level: str, rng: random.Random) -> tuple[np.ndarray, list[int]]:
    """Dessine une vue de dessus schématique d'une benne. Retourne (image, bbox de la plaque)."""
    W, H = 960, 640
    img = np.full((H, W, 3), (96, 98, 100), np.uint8)
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).integers(-10, 10, (H, W, 1), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cab = rng.choice([(40, 70, 170), (160, 90, 30), (40, 130, 60), (200, 200, 205), (30, 150, 220)])
    cv2.rectangle(img, (70, 210), (230, 430), cab, -1)                 # cabine
    cv2.rectangle(img, (90, 230), (150, 410), (60, 50, 45), -1)        # pare-brise
    cv2.rectangle(img, (250, 150), (870, 490), (120, 124, 128), -1)    # benne : parois
    cv2.rectangle(img, (272, 172), (848, 468), (72, 76, 80), -1)       # benne : fond
    for x in range(360, 848, 96):
        cv2.line(img, (x, 172), (x, 468), (64, 68, 72), 3)
    n_blobs = {"PROPRE": 0, "MOYEN": rng.randint(5, 8), "SALE": rng.randint(22, 30)}[level]
    for _ in range(n_blobs):
        c = (rng.randint(300, 820), rng.randint(200, 440))
        axes = (rng.randint(18, 60 if level == "SALE" else 34), rng.randint(12, 44 if level == "SALE" else 22))
        color = (rng.randint(25, 50), rng.randint(60, 95), rng.randint(105, 150))
        cv2.ellipse(img, c, axes, rng.randint(0, 180), 0, 360, color, -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    px, py, pw, ph = 395, 528, 330, 76                                  # plaque
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (245, 245, 245), -1)
    cv2.rectangle(img, (px, py), (px + pw, py + ph), (20, 20, 20), 2)
    cv2.rectangle(img, (px + 2, py + 2), (px + 30, py + ph - 2), (150, 60, 20), -1)
    cv2.putText(img, plate, (px + 42, py + 54), cv2.FONT_HERSHEY_DUPLEX, 1.25, (15, 15, 15), 3, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (W, 44), (40, 30, 20), -1)
    cv2.putText(img, "DEMO - IMAGE FICTIVE - TRUCK VISION AI", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img, [px, py, px + pw, py + ph]


def seed(count: int) -> list[dict]:
    """Génère `count` contrôles fictifs (image synthétique + résultat cohérent avec le dessin)."""
    rng = random.Random()
    fleet = [random_plate(rng) for _ in range(max(4, count // 4))]
    out = []
    for _ in range(count):
        _sid, d = ip.new_storage_dir("demo")
        plate = rng.choice(fleet)
        level = rng.choices(["PROPRE", "MOYEN", "SALE"], weights=[5, 3, 2])[0]
        mode = rng.choices(["FULL", "PLATE", "CLEANLINESS"], weights=[6, 2, 2])[0]
        img, pb = draw_demo_truck(plate, level, rng)
        ip.save_jpeg(d / "original.jpg", img)
        ip.save_jpeg(d / "thumb.jpg", ip.make_thumbnail(img), 80)
        ip.save_jpeg(d / "plate.jpg", ip.crop(img, pb, 0.08), 92)
        out.append({
            "demo": True, "mode": mode, "file_type": "IMAGE", "storage_dir": ip.rel(d),
            "files": {"original": ip.rel(d / "original.jpg"), "thumbnail": ip.rel(d / "thumb.jpg"), "annotated": None,
                      "plate_crop": ip.rel(d / "plate.jpg") if mode != "CLEANLINESS" else None},
            "vehicle": {"detected": True, "label": "Camion (simulé)", "confidence": 0.9, "bbox": None, "count": 1},
            "plate": {**_plate(rng, plate), "bbox": pb} if mode != "CLEANLINESS" else None,
            "cleanliness": _clean(rng, level) if mode != "PLATE" else None,
            "duration_ms": rng.randint(350, 1400),
        })
    return out
