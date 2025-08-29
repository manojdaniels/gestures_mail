import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    storage_dir: str = Field(default=os.getenv("STORAGE_DIR", "/workspace/data"))
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))

    embedding_model: str = Field(default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

    openai_api_key: str | None = Field(default=os.getenv("OPENAI_API_KEY"))

    elevenlabs_api_key: str | None = Field(default=os.getenv("ELEVENLABS_API_KEY"))
    tts_voice: str = Field(default=os.getenv("TTS_VOICE", "en"))

    youtube_client_secrets_file: str = Field(default=os.getenv("YOUTUBE_CLIENT_SECRETS_FILE", "/workspace/secrets/client_secret.json"))
    youtube_token_file: str = Field(default=os.getenv("YOUTUBE_TOKEN_FILE", "/workspace/data/tokens/token.json"))
    youtube_default_category_id: str = Field(default=os.getenv("YOUTUBE_DEFAULT_CATEGORY_ID", "27"))

    timezone: str = Field(default=os.getenv("TIMEZONE", "UTC"))

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()