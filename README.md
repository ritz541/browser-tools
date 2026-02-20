# Gemini Live Browser Tools

FastAPI service for Gemini Live voice browsing with browser automation controls.

## Features

- Gemini Live audio session management
- Browser controls via API: navigate, click, type, fill forms, submit forms, scroll
- Minimal web UI for manual control
- Playwright-based visible Chromium browser

## Run

```bash
export GEMINI_API_KEY=your_key
export SCREENSHOT_INTERVAL_SECONDS=3
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000`.

## Main Endpoints

- `POST /api/v1/session/start`
- `GET /api/v1/session/status`
- `POST /api/v1/session/prompt`
- `POST /api/v1/session/stop`
- `POST /api/v1/audio/pause`
- `POST /api/v1/audio/resume`
- `POST /api/v1/browser/navigate`
- `POST /api/v1/browser/click`
- `POST /api/v1/browser/type`
- `POST /api/v1/browser/fill-form`
- `POST /api/v1/browser/submit`
- `POST /api/v1/browser/scroll`
- `GET /api/v1/browser/info`
- `GET /api/v1/browser/screenshot`
