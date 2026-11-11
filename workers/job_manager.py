"""Gestion des tâches d'analyse vidéo en arrière-plan.

File d'attente en mémoire + un worker (une vidéo à la fois : adapté à une petite
machine). L'état et les résultats sont aussi écrits sur disque afin de rester
consultables après un redémarrage du serveur.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config

JOBS_DIR = config.STORAGE_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-worker")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _persist(job: dict) -> None:
    (JOBS_DIR / f"{job['job_id']}.json").write_text(json.dumps(job, ensure_ascii=False))


def create_job(task, **meta) -> dict:
    """`task(progress)` s'exécute en arrière-plan et retourne le résultat (dict)."""
    job = {"job_id": uuid.uuid4().hex, "status": "queued", "progress": 0, "frames_processed": 0,
           "frames_total": 0, "trucks_detected": 0, "message": "En attente de traitement…", "error": None,
           "created_at": time.time(), "finished_at": None, **meta}
    with _lock:
        _jobs[job["job_id"]] = job
        _persist(job)

    def progress(**kw) -> None:
        with _lock:
            job.update(kw)

    def run() -> None:
        progress(status="processing", message="Chargement de la vidéo…")
        try:
            result = task(progress)
            (JOBS_DIR / f"{job['job_id']}.result.json").write_text(json.dumps(result, ensure_ascii=False))
            progress(status="completed", progress=100, message="Traitement terminé.", finished_at=time.time())
        except Exception as exc:
            traceback.print_exc()
            msg = str(exc) if exc.__class__.__name__ == "ValidationError" else f"Erreur de traitement : {exc}"
            progress(status="failed", error=msg, message="Le traitement a échoué.", finished_at=time.time())
        with _lock:
            _persist(job)

    _executor.submit(run)
    return dict(job)


def get_job(job_id: str) -> dict | None:
    if not job_id.isalnum():
        return None
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    f = JOBS_DIR / f"{job_id}.json"
    if f.exists():
        job = json.loads(f.read_text())
        if job["status"] in ("queued", "processing"):  # serveur redémarré pendant le traitement
            job.update(status="failed", error="Le serveur IA a été redémarré pendant le traitement.")
        return job
    return None


def get_result(job_id: str) -> dict | None:
    if not job_id.isalnum():
        return None
    f = JOBS_DIR / f"{job_id}.result.json"
    return json.loads(f.read_text()) if f.exists() else None
