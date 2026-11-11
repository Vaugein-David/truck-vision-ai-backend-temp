"""Préparation de l'intégration des caméras IP (non exposée dans l'interface en V1).

Fonctionnement cible :
- caméra d'ENTRÉE : plaque + propreté de la benne (mode FULL) ;
- caméra de SORTIE : plaque uniquement (mode PLATE) — la propreté n'est
  jamais analysée à la sortie.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ROLE_MODE = {"ENTREE": "FULL", "SORTIE": "PLATE"}


@dataclass
class CameraConfig:
    name: str
    url: str          # ex. rtsp://utilisateur:motdepasse@192.168.1.50:554/stream1
    role: str         # ENTREE | SORTIE

    @property
    def analysis_mode(self) -> str:
        return ROLE_MODE[self.role]


class CameraStream:
    """Lecture d'un flux RTSP/HTTP avec OpenCV."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.cfg.url)
        return bool(self.cap.isOpened())

    def grab(self) -> np.ndarray | None:
        if self.cap is None and not self.open():
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
