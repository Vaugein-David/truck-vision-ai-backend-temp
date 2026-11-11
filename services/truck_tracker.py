"""Suivi des camions dans une vidéo avec ByteTrack (Ultralytics).

Chaque vidéo utilise sa propre instance YOLO afin que l'état du tracker ne soit
pas partagé entre deux traitements. Les pistes ayant lu la même plaque sont
fusionnées pour ne pas compter deux fois le même camion.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config
from services.engines import vehicle_detector


@dataclass
class Track:
    track_id: int
    label: str
    first_t: float
    last_t: float
    hits: int = 0
    best_area: float = 0.0
    best_frame: np.ndarray | None = None
    best_bbox: list | None = None
    best_t: float = 0.0
    last_bbox: list | None = None
    plate_reads: list = field(default_factory=list)      # {t, norm, format, score, crop}
    ocr_attempts: int = 0
    clean_samples: list = field(default_factory=list)    # (area, frame, bbox, t)

    def observe(self, t: float, frame: np.ndarray, bbox: list, keep_clean: int = 3) -> None:
        self.last_t = t
        self.last_bbox = bbox
        self.hits += 1
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area > self.best_area:
            self.best_area, self.best_frame, self.best_bbox, self.best_t = area, frame.copy(), bbox, t
        self.clean_samples.append((area, frame.copy(), bbox, t))
        self.clean_samples.sort(key=lambda s: s[0], reverse=True)
        del self.clean_samples[keep_clean:]

    def plate_consensus(self) -> dict | None:
        """Vote entre les lectures : texte au meilleur score cumulé."""
        if not self.plate_reads:
            return None
        groups: dict[str, list] = {}
        for r in self.plate_reads:
            groups.setdefault(r["norm"], []).append(r)
        norm, reads = max(groups.items(), key=lambda kv: sum(r["score"] for r in kv[1]))
        top = max(reads, key=lambda r: r["score"])
        return {"norm": norm, "format": top["format"], "score": top["score"], "votes": len(reads),
                "total": len(self.plate_reads), "crop": top["crop"], "t": top["t"]}


class TruckTracker:
    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.model = None
        if vehicle_detector.available:
            from ultralytics import YOLO

            self.model = YOLO(vehicle_detector.weights_path())
        self.classes = None if vehicle_detector.custom else list(config.VEHICLE_CLASSES.keys())

    @property
    def available(self) -> bool:
        return self.model is not None

    def update(self, frame: np.ndarray, t: float) -> list[Track]:
        """Analyse une image et retourne les pistes visibles."""
        if self.model is None:
            return []
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=self.classes,
                               conf=0.3, verbose=False)[0]
        visible = []
        if res.boxes is None or res.boxes.id is None:
            return visible
        for box, tid, cls in zip(res.boxes.xyxy.tolist(), res.boxes.id.tolist(), res.boxes.cls.tolist()):
            tid = int(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, vehicle_detector.label(int(cls)), t, t)
            tr.observe(t, frame, [float(v) for v in box])
            visible.append(tr)
        return visible

    def finalize(self, min_hits: int = 2) -> list[Track]:
        """Écarte les pistes fugaces et fusionne celles qui portent la même plaque."""
        tracks = [t for t in self.tracks.values() if t.hits >= min_hits] or list(self.tracks.values())
        merged: dict[str, Track] = {}
        out: list[Track] = []
        for tr in sorted(tracks, key=lambda t: t.first_t):
            cons = tr.plate_consensus()
            key = cons["norm"] if cons and cons["score"] >= 0.6 else None
            if key and key in merged:
                m = merged[key]
                m.first_t, m.last_t = min(m.first_t, tr.first_t), max(m.last_t, tr.last_t)
                m.hits += tr.hits
                m.plate_reads += tr.plate_reads
                m.clean_samples = sorted(m.clean_samples + tr.clean_samples, key=lambda s: s[0], reverse=True)[:3]
                if tr.best_area > m.best_area:
                    m.best_area, m.best_frame, m.best_bbox, m.best_t = tr.best_area, tr.best_frame, tr.best_bbox, tr.best_t
                continue
            if key:
                merged[key] = tr
            out.append(tr)
        return out
