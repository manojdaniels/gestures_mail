from pydantic import BaseModel, Field
from typing import Optional, List


class UploadResponse(BaseModel):
    document_id: str
    message: str = Field(default="Upload accepted and processing started")


class ScheduleRequest(BaseModel):
    document_id: str
    title: str
    description: str
    publish_at_iso: str  # ISO 8601 datetime string in UTC
    tags: Optional[List[str]] = None
    category_id: Optional[str] = None


class RetrievalRequest(BaseModel):
    document_id: str
    query: Optional[str] = None
    target_minutes: int = 6


class GenerateVideoRequest(BaseModel):
    document_id: str
    title: str
    description: str
    target_minutes: int = 6