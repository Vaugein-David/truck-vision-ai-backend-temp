"""TRUCK VISION AI - backend IA (FastAPI).

Lancement :  uvicorn main:app --reload --port 8000
"""
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from api import routes_system, routes_vision
from services import engines

app = FastAPI(title="TRUCK VISION AI - Backend IA", version=config.VERSION)
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(routes_system.router)
app.include_router(routes_system.protected)
app.include_router(routes_vision.router)


@app.on_event("startup")
def warmup() -> None:
    """Charge les modèles en arrière-plan pour que la première analyse soit rapide."""
    threading.Thread(target=engines.engines_info, daemon=True).start()
