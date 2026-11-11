"""Endpoints d'analyse : images, vidéos, tâches."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.deps import require_key, store_upload
from services import demo_engine, image_processor as ip, pipeline, video_processor
from workers import job_manager

router = APIRouter(prefix="/api/vision", tags=["vision"], dependencies=[Depends(require_key)])


def _mode(mode: str) -> str:
    mode = mode.upper()
    if mode not in pipeline.MODES:
        raise HTTPException(status_code=400, detail="Mode d'analyse inconnu (PLATE, CLEANLINESS ou FULL).")
    return mode


async def _image(file: UploadFile, mode: str, plate_threshold: float, clean_threshold: float,
                 max_mb: float | None, demo: bool) -> dict:
    path, d = store_upload(file, "image", max_mb)
    try:
        if demo:
            return await run_in_threadpool(demo_engine.demo_image_result, path, d, mode)
        return await run_in_threadpool(pipeline.analyze_image, path, d, mode, plate_threshold, clean_threshold)
    except ip.ValidationError as exc:
        ip.delete_dir(ip.rel(d))
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        ip.delete_dir(ip.rel(d))
        raise HTTPException(status_code=500, detail=f"Erreur pendant l'analyse : {exc}")


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), mode: str = Query("FULL"),
                        plate_threshold: float = Query(0.8, ge=0, le=1), clean_threshold: float = Query(0.7, ge=0, le=1),
                        max_mb: float | None = Query(None, gt=0), demo: bool = Query(False)):
    return await _image(file, _mode(mode), plate_threshold, clean_threshold, max_mb, demo)


@router.post("/recognize-plate")
async def recognize_plate(file: UploadFile = File(...), plate_threshold: float = Query(0.8, ge=0, le=1),
                          max_mb: float | None = Query(None, gt=0), demo: bool = Query(False)):
    return await _image(file, "PLATE", plate_threshold, 0.7, max_mb, demo)


@router.post("/analyze-cleanliness")
async def analyze_cleanliness(file: UploadFile = File(...), clean_threshold: float = Query(0.7, ge=0, le=1),
                              max_mb: float | None = Query(None, gt=0), demo: bool = Query(False)):
    return await _image(file, "CLEANLINESS", 0.8, clean_threshold, max_mb, demo)


@router.post("/analyze-video", status_code=202)
async def analyze_video(file: UploadFile = File(...), mode: str = Query("FULL"),
                        sample_fps: float = Query(2.0, gt=0, le=30),
                        plate_threshold: float = Query(0.8, ge=0, le=1), clean_threshold: float = Query(0.7, ge=0, le=1),
                        max_mb: float | None = Query(None, gt=0), demo: bool = Query(False)):
    mode = _mode(mode)
    path, d = store_upload(file, "video", max_mb)
    try:
        meta = await run_in_threadpool(video_processor.probe, path)
    except ip.ValidationError as exc:
        ip.delete_dir(ip.rel(d))
        raise HTTPException(status_code=400, detail=str(exc))
    ip.save_jpeg(d / "thumb.jpg", ip.make_thumbnail(meta.pop("first_frame")), 80)

    def task(progress):
        if demo:
            return demo_engine.demo_video_result(path, d, mode, progress)
        return video_processor.process_video(path, d, mode, sample_fps, plate_threshold, clean_threshold, progress)

    job = job_manager.create_job(task, mode=mode, demo=demo, storage_dir=ip.rel(d),
                                 files={"original": ip.rel(path), "thumbnail": ip.rel(d / "thumb.jpg")}, video=meta)
    return job


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    return job


@router.get("/jobs/{job_id}/results")
def job_results(job_id: str):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    if job["status"] == "failed":
        raise HTTPException(status_code=422, detail=job.get("error") or "Le traitement a échoué.")
    result = job_manager.get_result(job_id)
    if result is None:
        raise HTTPException(status_code=409, detail="Le traitement n'est pas terminé.")
    return result
