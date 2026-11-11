"""Instances uniques des moteurs IA (chargement paresseux)."""
from services.cleanliness_classifier import CleanlinessClassifier
from services.plate_detector import PlateDetector
from services.plate_recognizer import PlateRecognizer
from services.vehicle_detector import VehicleDetector

vehicle_detector = VehicleDetector()
plate_recognizer = PlateRecognizer()
plate_detector = PlateDetector(plate_recognizer)
cleanliness_classifier = CleanlinessClassifier()


def engines_info() -> dict:
    return {
        "vehicle_detector": vehicle_detector.info(),
        "plate_detector": plate_detector.info(),
        "plate_recognizer": plate_recognizer.info(),
        "cleanliness_classifier": cleanliness_classifier.info(),
        "tracker": {"role": "Suivi des camions (vidéo)", "name": "ByteTrack (Ultralytics)",
                    "loaded": vehicle_detector.available, "specialized": True, "error": None},
    }
