"""Évaluation de la propreté de la benne.

Deux moteurs :
1. Modèle spécialisé (models/cleanliness.pt, classification YOLO-cls entraînée sur
   les bennes du site ; classes attendues : propre / moyen / sale [/ non_visible]).
2. À défaut : estimation heuristique OpenCV (couleur terre/boue + texture).
   Ce moteur de démonstration n'est PAS un contrôle industriel fiable : sa
   confiance est plafonnée et le résultat est toujours marqué « À vérifier ».
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

import config
from services import image_processor as ip

CLASS_MAP = {
    "propre": "PROPRE", "clean": "PROPRE",
    "moyen": "MOYEN", "moyennement_propre": "MOYEN", "medium": "MOYEN",
    "sale": "SALE", "dirty": "SALE",
    "non_visible": None, "charge": None, "chargee": None, "loaded": None,
}
HEURISTIC_MAX_CONFIDENCE = 0.60


class CleanlinessClassifier:
    def __init__(self) -> None:
        self.model = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._loaded = False

    def load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True
            if config.CLEANLINESS_MODEL_CUSTOM.exists():
                try:
                    from ultralytics import YOLO

                    self.model = YOLO(str(config.CLEANLINESS_MODEL_CUSTOM))
                except Exception as exc:
                    self.error = str(exc)

    @property
    def engine_name(self) -> str:
        self.load()
        return "Modèle spécialisé (cleanliness.pt)" if self.model is not None else "Heuristique OpenCV de démonstration (non entraînée)"

    def analyze(self, img: np.ndarray, vehicle_bbox=None, threshold: float = 0.7) -> dict:
        """Retourne {status, category, confidence, elements, comment, engine}."""
        self.load()
        q = ip.quality_metrics(img)
        if min(q["width"], q["height"]) < 160:
            return self._impossible("Image trop petite pour évaluer l'intérieur de la benne.")
        if q["brightness"] < 35:
            return self._impossible("Image trop sombre : l'intérieur de la benne n'est pas visible.")
        if q["sharpness"] < 12:
            return self._impossible("Image trop floue pour évaluer la propreté.")

        roi_note = None
        if vehicle_bbox:
            roi = ip.crop(img, vehicle_bbox)
        else:
            h, w = img.shape[:2]
            roi = img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
            roi_note = "Aucun camion localisé automatiquement : estimation réalisée sur la zone centrale de l'image."
        if roi.size == 0:
            return self._impossible("Zone d'analyse vide.")

        if self.model is not None:
            return self._with_model(roi, threshold, roi_note)
        return self._heuristic(roi, roi_note)

    def _impossible(self, reason: str) -> dict:
        return {"status": "IMPOSSIBLE", "category": None, "confidence": 0.0, "elements": [],
                "comment": reason, "engine": self.engine_name}

    def _with_model(self, roi: np.ndarray, threshold: float, roi_note: str | None) -> dict:
        with self._lock:
            res = self.model.predict(roi, verbose=False)[0]
        idx, conf = int(res.probs.top1), float(res.probs.top1conf)
        name = str(res.names[idx]).lower().strip()
        category = CLASS_MAP.get(name, "?")
        if category is None:
            return self._impossible("L'intérieur de la benne n'est pas visible (benne chargée ou masquée).")
        if category == "?":
            return self._impossible(f"Classe « {name} » inconnue : vérifier le modèle de propreté.")
        status = "OK" if conf >= threshold and roi_note is None else "A_VERIFIER"
        comment = f"Classification par le modèle spécialisé : {name} ({conf:.0%})."
        if roi_note:
            comment += " " + roi_note
        return {"status": status, "category": category, "confidence": conf, "elements": [],
                "comment": comment, "engine": self.engine_name}

    def _heuristic(self, roi: np.ndarray, roi_note: str | None) -> dict:
        small = cv2.resize(roi, (320, int(320 * roi.shape[0] / roi.shape[1])), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        # Teintes terre / boue / rouille : brun-orangé, saturé, luminosité moyenne.
        earth = cv2.inRange(hsv, (4, 45, 35), (30, 255, 215))
        earth = cv2.morphologyEx(earth, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        earth_ratio = float(earth.mean() / 255)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
        edge_ratio = float(edges.mean() / 255)
        n_spots = max(0, cv2.connectedComponents(earth)[0] - 1)

        score = 0.85 * min(1.0, earth_ratio / 0.22) + 0.15 * min(1.0, edge_ratio / 0.18)
        lo, hi = 0.08, 0.50
        category = "PROPRE" if score < lo else "MOYEN" if score < hi else "SALE"
        # Confiance : distance aux frontières de décision, plafonnée (moteur non entraîné).
        margin = min(abs(score - lo), abs(score - hi)) if category == "MOYEN" else abs(score - (lo if category == "PROPRE" else hi))
        confidence = round(float(min(HEURISTIC_MAX_CONFIDENCE, 0.40 + margin)), 4)

        elements = []
        if earth_ratio >= 0.02:
            elements.append(f"Teintes terre / boue sur environ {earth_ratio:.0%} de la zone analysée")
        if n_spots >= 3 and earth_ratio >= 0.02:
            elements.append(f"{n_spots} zones de résidus distinctes")
        if edge_ratio >= 0.10:
            elements.append("Surface irrégulière (matériaux ou déchets possibles)")
        if not elements:
            elements.append("Aucun résidu significatif détecté")

        comment = {
            "PROPRE": "Peu ou pas de traces de terre ou de résidus détectées dans la zone analysée.",
            "MOYEN": "Quelques traces de terre, de boue ou de résidus sont visibles.",
            "SALE": "Présence importante de teintes terre/boue ou de matériaux dans la zone analysée.",
        }[category]
        comment += (" Estimation produite par l'heuristique de démonstration (aucun modèle spécialisé installé) : "
                    "elle ne distingue pas une benne chargée d'une benne sale et doit être validée par un opérateur.")
        if roi_note:
            comment += " " + roi_note
        return {"status": "A_VERIFIER", "category": category, "confidence": confidence, "elements": elements,
                "comment": comment, "engine": self.engine_name,
                "metrics": {"earth_ratio": round(earth_ratio, 4), "edge_ratio": round(edge_ratio, 4), "score": round(score, 4)}}

    def info(self) -> dict:
        self.load()
        return {
            "role": "Analyse de propreté des bennes",
            "name": self.engine_name,
            "loaded": True,
            "specialized": self.model is not None,
            "error": self.error,
        }
