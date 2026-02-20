import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import DEFAULT_MODEL
from app.schemas import (
    ClickRequest,
    FillFormRequest,
    NavigateRequest,
    PromptRequest,
    ScrollRequest,
    StartSessionRequest,
    SubmitRequest,
    TypeRequest,
)
from app.services.agent import VoiceBrowserAgent


app = FastAPI(title="Gemini Live Browser API", version="2.0.0")
app.state.agent: VoiceBrowserAgent | None = None

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


def get_agent() -> VoiceBrowserAgent:
    agent = app.state.agent
    if not agent:
        raise HTTPException(status_code=409, detail="session not started")
    return agent


def ensure_browser_ready(agent: VoiceBrowserAgent) -> None:
    if not agent.running:
        raise HTTPException(status_code=409, detail="session not running")
    if not agent.page:
        raise HTTPException(status_code=409, detail="browser page not ready")


@app.get("/")
async def index():
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health():
    return {"ok": True, "default_model": DEFAULT_MODEL}


@app.post("/api/v1/session/start")
async def start_session(payload: StartSessionRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")

    existing = app.state.agent
    if existing and existing.running:
        return {"message": "session already running", "status": existing.status()}

    agent = VoiceBrowserAgent(api_key=api_key, model=payload.model)
    await agent.start()
    app.state.agent = agent
    return {"message": "session started", "status": agent.status()}


@app.post("/api/v1/session/stop")
async def stop_session():
    agent = app.state.agent
    if not agent:
        return {"message": "no active session"}

    await agent.stop()
    app.state.agent = None
    return {"message": "session stopped"}


@app.get("/api/v1/session/status")
async def session_status():
    agent = app.state.agent
    if not agent:
        return {"running": False, "connected": False}
    return agent.status()


@app.post("/api/v1/session/prompt")
async def session_prompt(payload: PromptRequest):
    agent = get_agent()
    if not agent.session:
        raise HTTPException(status_code=409, detail="session not connected")

    await agent.send_prompt(payload.text)
    return {"message": "prompt sent"}


@app.post("/api/v1/audio/pause")
async def pause_listening():
    agent = get_agent()
    return agent.set_listening(False)


@app.post("/api/v1/audio/resume")
async def resume_listening():
    agent = get_agent()
    return agent.set_listening(True)


@app.post("/api/v1/browser/navigate")
async def browser_navigate(payload: NavigateRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    return await agent.navigate(payload.url)


@app.post("/api/v1/browser/click")
async def browser_click(payload: ClickRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    try:
        return await agent.click(selector=payload.selector, text=payload.text, nth=payload.nth)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/browser/type")
async def browser_type(payload: TypeRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    try:
        return await agent.type_input(
            selector=payload.selector,
            text=payload.text,
            clear=payload.clear,
            press_enter=payload.press_enter,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/browser/fill-form")
async def browser_fill_form(payload: FillFormRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    fields = [{"selector": f.selector, "value": f.value} for f in payload.fields]
    try:
        return await agent.fill_form(fields=fields, submit_selector=payload.submit_selector)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/browser/submit")
async def browser_submit(payload: SubmitRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    try:
        return await agent.submit_form(selector=payload.selector)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/v1/browser/scroll")
async def browser_scroll(payload: ScrollRequest):
    agent = get_agent()
    ensure_browser_ready(agent)
    try:
        return await agent.scroll(direction=payload.direction, amount=payload.amount)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/v1/browser/info")
async def browser_info():
    agent = get_agent()
    ensure_browser_ready(agent)
    return await agent.page_info()


@app.get("/api/v1/browser/screenshot")
async def browser_screenshot():
    agent = get_agent()
    ensure_browser_ready(agent)
    image_b64 = await agent.screenshot_base64()
    return {"mime_type": "image/jpeg", "image_base64": image_b64}


@app.on_event("shutdown")
async def shutdown_event():
    agent = app.state.agent
    if agent:
        await agent.stop()
        app.state.agent = None
