"""Dépendances communes : protection par clé et réception sécurisée des fichiers."""
from __future__ import annotations

import hmac
import shutil
from pathlib import Path

from fastapi import Header, HTTPException, UploadFile

import config
from services import image_processor as ip


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if config.AI_API_KEY and not hmac.compare_digest(x_api_key or "", config.AI_API_KEY):
        raise HTTPException(status_code=401, detail="Clé d'API invalide ou absente.")


def store_upload(file: UploadFile, kind: str, max_mb: float | None) -> tuple[Path, Path]:
    """Valide le fichier puis l'enregistre sous un nom neutre. Retourne (fichier, dossier)."""
    hard = config.HARD_MAX_IMAGE_MB if kind == "image" else config.HARD_MAX_VIDEO_MB
    limit = min(max_mb or hard, hard)
    head = file.file.read(16)
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    try:
        ext = ip.validate_upload(file.filename or "", head, size, kind, limit)
    except ip.ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _sid, d = ip.new_storage_dir("analyses")
    dest = d / f"original{ext}"   # le nom fourni par l'utilisateur n'est jamais utilisé comme chemin
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return dest, d
