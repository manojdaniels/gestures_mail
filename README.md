## RAG Video Creator from PDFs (FastAPI)

This app ingests a PDF, builds a RAG index (FAISS + Sentence-Transformers), retrieves content for a 5–6 minute video, generates a script, converts it to voice-over audio, creates a video with slides, and optionally schedules automatic YouTube upload using APScheduler and the YouTube Data API v3.

### Features
- PDF upload via web UI
- Text extraction and chunking (LangChain text splitter)
- Embeddings (Sentence-Transformers) and FAISS vector store
- Retrieval sized for ~5–6 minutes (≈750–900 words)
- Script generation (OpenAI if available, fallback simple summarizer)
- TTS (gTTS by default; optional ElevenLabs)
- Video generation (MoviePy + Pillow)
- Scheduling for YouTube upload (APScheduler + YouTube Data API)
- Env-based configuration, logging, and modular services

### Quickstart

1. Create and populate a virtual environment, then install requirements:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy env template and fill values:
```bash
cp .env.example .env
```

3. Place your YouTube OAuth client secret at the path in `YOUTUBE_CLIENT_SECRETS_FILE` (e.g., `./secrets/client_secret.json`). The first upload triggers an OAuth flow; run locally once to generate `token.json` that will be reused on server.

4. Run the server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Open the web UI at `http://localhost:8000` and upload a PDF. Optionally enter a schedule date and time (UTC) for YouTube upload.

### Environment Variables
See `.env.example` for all variables. Key ones:
- STORAGE_DIR: Base directory for data (default `./data`)
- EMBEDDING_MODEL: Sentence-Transformers model (default `sentence-transformers/all-MiniLM-L6-v2`)
- OPENAI_API_KEY: Optional; enables LLM summarization
- ELEVENLABS_API_KEY: Optional for advanced TTS
- YOUTUBE_CLIENT_SECRETS_FILE: Path to OAuth client secrets JSON
- YOUTUBE_TOKEN_FILE: Where to store OAuth tokens (default `./data/tokens/token.json`)

### Deployment (AWS EC2)
- Provision Ubuntu host, install system packages (`ffmpeg`, `python3`, `git`). MoviePy requires `ffmpeg`.
- Clone repo, set up venv, `pip install -r requirements.txt`.
- Set environment variables (systemd unit or `.env`).
- Run with `uvicorn` or behind `gunicorn` + `uvicorn.workers.UvicornWorker` and a reverse proxy (Nginx).

### Notes
- The first YouTube upload requires OAuth consent; run locally where a browser is available, then copy `token.json` to the server.
- FAISS indices are stored on disk under `STORAGE_DIR/indexes/<doc_id>`.
- Logs go under `STORAGE_DIR/logs/`. 
