"""Configuration du backend IA (variables d'environnement)."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", BASE_DIR / "storage")).resolve()
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models")).resolve()

# Clé partagée avec le frontend (côté serveur Next.js uniquement). Vide = pas de protection.
AI_API_KEY = os.getenv("AI_API_KEY", "")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

# Plafonds absolus (le frontend peut demander moins, jamais plus).
HARD_MAX_IMAGE_MB = int(os.getenv("HARD_MAX_IMAGE_MB", "25"))
HARD_MAX_VIDEO_MB = int(os.getenv("HARD_MAX_VIDEO_MB", "1024"))

# Moteur OCR : "auto" (PaddleOCR si installé, sinon RapidOCR/PP-OCR ONNX), "paddleocr" ou "rapidocr".
OCR_ENGINE = os.getenv("OCR_ENGINE", "auto")

# Poids des modèles (remplaçables par des modèles entraînés pour le site).
VEHICLE_MODEL_CUSTOM = MODELS_DIR / "vehicle_detector.pt"
VEHICLE_MODEL_DEFAULT = MODELS_DIR / "yolov8n.pt"      # COCO, téléchargé au 1er lancement
PLATE_MODEL_CUSTOM = MODELS_DIR / "plate_detector.pt"   # optionnel
CLEANLINESS_MODEL_CUSTOM = MODELS_DIR / "cleanliness.pt"  # optionnel (YOLO-cls)

# Classes COCO considérées comme véhicules : voiture, bus, camion.
VEHICLE_CLASSES = {2: "Véhicule léger", 5: "Bus / poids lourd", 7: "Camion"}

VERSION = "1.0.0-demo"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
