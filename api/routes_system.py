"""Santé du serveur, fichiers stockés, stockage, mode DEMO, rapports PDF."""
from __future__ import annotations

import os
import shutil

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

import config
from api.deps import require_key
from services import demo_engine, image_processor as ip, report_generator
from services.engines import engines_info

router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_key)])

MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
         ".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".webm": "video/webm"}


@router.get("/api/health")
def health():
    return {"status": "ok", "service": "TRUCK VISION AI - backend IA", "version": config.VERSION,
            "protected": bool(config.AI_API_KEY), "engines": engines_info(),
            "limits": {"max_image_mb": config.HARD_MAX_IMAGE_MB, "max_video_mb": config.HARD_MAX_VIDEO_MB}}


@protected.get("/files/{path:path}")
def get_file(path: str):
    try:
        p = ip.safe_path(path)
    except ip.ValidationError:
        raise HTTPException(status_code=403, detail="Chemin non autorisé.")
    media = MEDIA.get(p.suffix.lower())
    if not p.is_file() or media is None:
        raise HTTPException(status_code=404, detail="Fichier introuvable.")
    return FileResponse(p, media_type=media, headers={"X-Content-Type-Options": "nosniff"})


def _dir_size(path) -> tuple[int, int]:
    total = files = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                total += os.path.getsize(os.path.join(root, n))
                files += 1
            except OSError:
                pass
    return total, files


@protected.get("/api/storage/usage")
def storage_usage():
    disk = shutil.disk_usage(config.STORAGE_DIR)
    cats = {}
    for name in ("analyses", "demo", "jobs"):
        size, files = _dir_size(config.STORAGE_DIR / name)
        cats[name] = {"bytes": size, "files": files}
    return {"path": str(config.STORAGE_DIR), "used_bytes": sum(c["bytes"] for c in cats.values()),
            "categories": cats, "disk_total_bytes": disk.total, "disk_free_bytes": disk.free}


@protected.delete("/api/storage")
def storage_delete(dir: str = Query(..., description="Dossier d'analyse relatif au stockage")):
    parts = dir.strip("/").split("/")
    if len(parts) != 3 or parts[0] not in ("analyses", "demo"):
        raise HTTPException(status_code=400, detail="Dossier non supprimable.")
    try:
        return {"deleted": ip.delete_dir(dir)}
    except ip.ValidationError:
        raise HTTPException(status_code=403, detail="Chemin non autorisé.")


@protected.post("/api/demo/seed")
async def demo_seed(count: int = Query(24, ge=1, le=80)):
    return {"demo": True, "items": await run_in_threadpool(demo_engine.seed, count)}


@protected.post("/api/reports/analysis")
async def report_analysis(data: dict = Body(...)):
    pdf = await run_in_threadpool(report_generator.analysis_pdf, data)
    return Response(pdf, media_type="application/pdf")


@protected.post("/api/reports/summary")
async def report_summary(data: dict = Body(...)):
    pdf = await run_in_threadpool(report_generator.summary_pdf, data)
    return Response(pdf, media_type="application/pdf")
