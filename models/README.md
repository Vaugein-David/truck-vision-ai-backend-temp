# Modèles IA

Déposez ici vos modèles entraînés ; ils sont pris en compte au redémarrage du backend.

| Fichier | Rôle | Par défaut (si absent) |
|---|---|---|
| `vehicle_detector.pt` | Détection des camions (YOLO) | `yolov8n.pt` COCO, téléchargé au 1er lancement |
| `plate_detector.pt` | Localisation des plaques (YOLO) | Détecteur de texte PP-OCR + filtrage par format |
| `cleanliness.pt` | Propreté des bennes (YOLO-cls) — classes `propre`, `moyen`, `sale`, `non_visible` | Heuristique OpenCV de démonstration, toujours « À vérifier » |
