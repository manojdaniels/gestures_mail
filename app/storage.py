import os
import uuid
from typing import Dict
from .config import settings


def ensure_storage_dirs() -> None:
    for sub in ("uploads", "indexes", "outputs", "tokens", "logs", "meta", "slides"):
        os.makedirs(os.path.join(settings.storage_dir, sub), exist_ok=True)


def generate_document_id() -> str:
    return str(uuid.uuid4())


def get_paths(document_id: str) -> Dict[str, str]:
    base = settings.storage_dir
    return {
        "pdf_path": os.path.join(base, "uploads", f"{document_id}.pdf"),
        "index_dir": os.path.join(base, "indexes", document_id),
        "audio_path": os.path.join(base, "outputs", f"{document_id}.mp3"),
        "video_path": os.path.join(base, "outputs", f"{document_id}.mp4"),
        "slides_dir": os.path.join(base, "slides", document_id),
        "meta_path": os.path.join(base, "meta", f"{document_id}.json"),
    }