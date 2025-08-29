from __future__ import annotations
import os
from typing import Optional

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from PyPDF2 import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore

from ..logging_config import get_logger

logger = get_logger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Prefer pdfplumber for better layout parsing
    if pdfplumber is not None:
        try:
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text() or ""
                    text_parts.append(extracted)
            text = "\n".join(text_parts)
            if text.strip():
                logger.info("Extracted text using pdfplumber: %d chars", len(text))
                return text
        except Exception as e:  # pragma: no cover
            logger.warning("pdfplumber failed, falling back to PyPDF2: %s", e)

    # Fallback to PyPDF2
    if PdfReader is None:
        raise RuntimeError("No PDF extraction backend available")

    reader = PdfReader(pdf_path)
    text_parts = []
    for page in reader.pages:
        try:
            text_parts.append(page.extract_text() or "")
        except Exception:  # pragma: no cover
            continue
    text = "\n".join(text_parts)
    logger.info("Extracted text using PyPDF2: %d chars", len(text))
    return text