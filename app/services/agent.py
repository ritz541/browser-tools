import asyncio
import base64
import os
import sys
from contextlib import suppress
from typing import Any

import pyaudio
from google import genai
from google.genai import types
from playwright.async_api import Page, async_playwright

from app.config import SCREENSHOT_INTERVAL_SECONDS


class VoiceBrowserAgent:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.browser = None
        self.page: Page | None = None
        self.session = None
        self.playwright = None

        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.input_rate = 16000
        self.output_rate = 24000
        self.chunk_size = 1024
        self.pyaudio = pyaudio.PyAudio()

        self.audio_input_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)
        self.audio_output_queue: asyncio.Queue[bytes] = asyncio.Queue()

        self.running = False
        self.last_url = None
        self.playback_chunks = 0
        self.last_error: str | None = None
        self.last_model_text: str | None = None
        self.listening_enabled = True
        self.screenshot_interval_seconds = SCREENSHOT_INTERVAL_SECONDS
        self.last_screenshot_sent_at = 0.0
        self.last_tool_name: str | None = None

        self._task: asyncio.Task | None = None

    def _ensure_page(self) -> Page:
        if not self.page:
            raise RuntimeError("browser page is not available")
        return self.page

    def get_tools(self) -> list[types.FunctionDeclaration]:
        return [
            types.FunctionDeclaration(
                name="navigate_to",
                description="Navigate browser to URL",
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            ),
            types.FunctionDeclaration(
                name="search_google",
                description="Search Google by query",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="click_selector",
                description="Click an element by CSS selector",
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "nth": {"type": "integer"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="pause_listening",
                description="Pause microphone forwarding to model",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="resume_listening",
                description="Resume microphone forwarding to model",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="type_input",
                description="Type text into input by CSS selector",
                parameters={
                    "type": "object",
                    "properties": {
                        "selector": {"type": "string"},
                        "text": {"type": "string"},
                        "clear": {"type": "boolean"},
                        "press_enter": {"type": "boolean"},
                    },
                    "required": ["selector", "text"],
                },
            ),
            types.FunctionDeclaration(
                name="submit_form",
                description="Submit form, optionally by clicking submit button selector",
                parameters={
                    "type": "object",
                    "properties": {"selector": {"type": "string"}},
                },
            ),
            types.FunctionDeclaration(
                name="scroll_page",
                description="Scroll page up/down",
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string"},
                        "amount": {"type": "integer"},
                    },
                },
            ),
            types.FunctionDeclaration(
                name="go_back",
                description="Go back in browser history",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="get_page_info",
                description="Get current page URL/title",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    async def start_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=False)
        self.page = await self.browser.new_page(viewport={"width": 1280, "height": 800})
        print("Browser started")

    async def close_browser(self):
        if self.browser is not None:
            with suppress(Exception):
                await self.browser.close()
            self.browser = None
        if self.playwright is not None:
            with suppress(Exception):
                await self.playwright.stop()
            self.playwright = None
        self.page = None

    async def navigate(self, url: str) -> dict[str, Any]:
        page = self._ensure_page()
        await page.goto(url)
        return await self.page_info()

    async def click(self, selector: str | None = None, text: str | None = None, nth: int = 0) -> dict[str, Any]:
        page = self._ensure_page()
        if selector:
            locator = page.locator(selector)
            count = await locator.count()
            if count == 0:
                raise RuntimeError(f"selector not found: {selector}")
            await locator.nth(max(0, nth)).click()
            return {"status": "success", "action": "click", "selector": selector, "nth": nth}

        if text:
            link = page.get_by_role("link", name=text, exact=False).first
            with suppress(Exception):
                await link.click(timeout=3000)
                return {"status": "success", "action": "click", "text": text}
            await page.get_by_text(text, exact=False).first.click()
            return {"status": "success", "action": "click", "text": text}

        # Fallback for Google-like result pages.
        result_links = page.locator("a:has(h3)")
        count = await result_links.count()
        if count > 0:
            await result_links.nth(max(0, nth)).click()
            return {"status": "success", "action": "click", "selector": "a:has(h3)", "nth": nth}

        raise RuntimeError("either selector or text is required, or no result link found")

    async def type_input(self, selector: str, text: str, clear: bool = True, press_enter: bool = False) -> dict[str, Any]:
        page = self._ensure_page()
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=7000)
        if clear:
            await locator.fill("")
        await locator.type(text)
        if press_enter:
            await locator.press("Enter")
        return {"status": "success", "action": "type", "selector": selector, "typed": len(text)}

    async def fill_form(self, fields: list[dict[str, str]], submit_selector: str | None = None) -> dict[str, Any]:
        page = self._ensure_page()
        for field in fields:
            selector = field["selector"]
            value = field["value"]
            locator = page.locator(selector).first
            await locator.wait_for(state="visible", timeout=7000)
            await locator.fill(value)

        if submit_selector:
            await page.locator(submit_selector).first.click()

        return {"status": "success", "action": "fill_form", "fields": len(fields), "submitted": bool(submit_selector)}

    async def submit_form(self, selector: str | None = None) -> dict[str, Any]:
        page = self._ensure_page()
        if selector:
            await page.locator(selector).first.click()
            return {"status": "success", "action": "submit_form", "selector": selector}

        await page.evaluate(
            """
            () => {
              const active = document.activeElement;
              if (active && active.form) {
                active.form.requestSubmit();
                return;
              }
              const form = document.querySelector('form');
              if (!form) throw new Error('No form found');
              form.requestSubmit();
            }
            """
        )
        return {"status": "success", "action": "submit_form", "selector": None}

    async def scroll(self, direction: str = "down", amount: int = 500) -> dict[str, Any]:
        page = self._ensure_page()
        delta = amount if direction == "down" else -amount
        await page.evaluate(f"window.scrollBy(0, {delta})")
        return {"status": "success", "action": "scroll", "direction": direction, "amount": amount}

    async def go_back(self) -> dict[str, Any]:
        page = self._ensure_page()
        await page.go_back()
        return await self.page_info()

    async def page_info(self) -> dict[str, Any]:
        page = self._ensure_page()
        return {
            "status": "success",
            "url": page.url,
            "title": await page.title(),
        }

    async def screenshot_base64(self) -> str:
        page = self._ensure_page()
        raw = await page.screenshot(type="jpeg", quality=70)
        return base64.b64encode(raw).decode("utf-8")

    async def screenshot_loop(self):
        while self.running:
            try:
                if self.session and self.page:
                    url = self.page.url
                    now = asyncio.get_running_loop().time()
                    interval_elapsed = (now - self.last_screenshot_sent_at) >= self.screenshot_interval_seconds
                    if url != self.last_url or interval_elapsed:
                        self.last_url = url
                        img = await self.page.screenshot(type="jpeg", quality=70)
                        await self.session.send_realtime_input(
                            media=types.Blob(data=img, mime_type="image/jpeg")
                        )
                        self.last_screenshot_sent_at = now
                        print("📸 Screenshot sent")
                await asyncio.sleep(0.5)
            except Exception as e:
                print("Screenshot error:", e)
                await asyncio.sleep(1)

    async def handle_tool_call(self, function_call: Any):
        name = function_call.name
        args = function_call.args or {}
        call_id = function_call.id

        try:
            if self.last_tool_name == name and name in {
                "navigate_to",
                "search_google",
                "click_selector",
                "scroll_page",
            }:
                result = {
                    "status": "error",
                    "message": f"Tool repetition guard blocked duplicate call: {name}",
                }
                return types.FunctionResponse(id=call_id, name=name, response=result)
            if name == "navigate_to":
                result = await self.navigate(args["url"])
            elif name == "search_google":
                q = args["query"]
                result = await self.navigate(f"https://www.google.com/search?q={q.replace(' ', '+')}")
            elif name == "click_selector":
                result = await self.click(
                    selector=args.get("selector"),
                    text=args.get("text"),
                    nth=args.get("nth", 0),
                )
            elif name == "type_input":
                result = await self.type_input(
                    selector=args["selector"],
                    text=args["text"],
                    clear=args.get("clear", True),
                    press_enter=args.get("press_enter", False),
                )
            elif name == "submit_form":
                result = await self.submit_form(selector=args.get("selector"))
            elif name == "pause_listening":
                self.listening_enabled = False
                result = {"status": "success", "listening_enabled": self.listening_enabled}
            elif name == "resume_listening":
                self.listening_enabled = True
                result = {"status": "success", "listening_enabled": self.listening_enabled}
            elif name == "scroll_page":
                result = await self.scroll(direction=args.get("direction", "down"), amount=args.get("amount", 500))
            elif name == "go_back":
                result = await self.go_back()
            elif name == "get_page_info":
                result = await self.page_info()
            else:
                result = {"status": "error", "message": f"Unknown tool: {name}"}
            self.last_tool_name = name
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        return types.FunctionResponse(id=call_id, name=name, response=result)

    async def listen_microphone(self):
        stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.input_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )
        try:
            while self.running:
                try:
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    if self.listening_enabled:
                        await self.audio_input_queue.put(data)
                except Exception as e:
                    print("Mic error:", e)
                    await asyncio.sleep(0.1)
        finally:
            with suppress(Exception):
                stream.stop_stream()
            with suppress(Exception):
                stream.close()

    async def send_audio(self):
        while self.running:
            try:
                data = await self.audio_input_queue.get()
                if self.session is None:
                    continue
                await self.session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                )
            except Exception as e:
                print("Audio send error:", e)
                await asyncio.sleep(0.1)

    async def receive_audio(self):
        while self.running:
            if self.session is None:
                await asyncio.sleep(0.1)
                continue
            try:
                async for response in self.session.receive():
                    if not self.running:
                        break
                    if getattr(response, "text", None):
                        self.last_model_text = response.text
                        print(f"Gemini(text): {response.text}")
                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                self.last_model_text = part.text
                                print(f"Gemini(text): {part.text}")
                            if part.inline_data and part.inline_data.data:
                                data = part.inline_data.data
                                if isinstance(data, str):
                                    data = base64.b64decode(data)
                                await self.audio_output_queue.put(data)
                                self.playback_chunks += 1
                    if response.tool_call:
                        responses = [
                            await self.handle_tool_call(fc)
                            for fc in response.tool_call.function_calls
                        ]
                        await self.session.send_tool_response(function_responses=responses)
            except Exception as e:
                print("Receive error:", e)
                await asyncio.sleep(0.3)

    async def play_audio(self):
        stream = self.pyaudio.open(
            format=self.audio_format,
            channels=self.channels,
            rate=self.output_rate,
            output=True,
            **self._output_stream_kwargs(),
        )
        try:
            while self.running:
                try:
                    data = await self.audio_output_queue.get()
                    stream.write(data)
                except Exception as e:
                    print("Playback error:", e)
                    if self.last_model_text:
                        print(f"Fallback model text: {self.last_model_text}")
                    await asyncio.sleep(0.1)
        finally:
            with suppress(Exception):
                stream.stop_stream()
            with suppress(Exception):
                stream.close()

    async def _run(self):
        try:
            await self.start_browser()

            config = types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                tools=[types.Tool(function_declarations=self.get_tools())],
                system_instruction=types.Content(
                    parts=[
                        types.Part(
                            text=(
                                "You are a voice web assistant. You continuously receive screenshots of "
                                "the browser and should use visual context when describing pages or choosing tools. "
                                "Call tools only when necessary. Never repeat the same tool twice in a row "
                                "unless the user explicitly asks for repetition. Prefer precise selectors and use "
                                "Google result links via a:has(h3) when relevant."
                            )
                        )
                    ],
                ),
            )

            async with self.client.aio.live.connect(model=self.model, config=config) as session:
                self.session = session
                self.last_screenshot_sent_at = 0.0

                await asyncio.gather(
                    self.listen_microphone(),
                    self.send_audio(),
                    self.receive_audio(),
                    self.play_audio(),
                    self.screenshot_loop(),
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = str(e)
            raise
        finally:
            self.running = False
            self.session = None
            await self.close_browser()

    async def start(self):
        if self.running:
            return
        self.last_error = None
        self.running = True
        self._task = asyncio.create_task(self._run())
        # Wait until browser page is initialized or startup fails.
        for _ in range(100):
            if self.page is not None:
                return
            if self._task.done():
                exc = self._task.exception()
                if exc:
                    raise exc
                return
            await asyncio.sleep(0.1)
        raise RuntimeError("session startup timed out")

    async def stop(self):
        if not self.running:
            return

        self.running = False

        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        self.session = None
        await self.close_browser()

        with suppress(Exception):
            self.pyaudio.terminate()

    async def send_prompt(self, text: str):
        if not self.session:
            raise RuntimeError("session is not connected")
        await self.session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=text)]),
            turn_complete=True,
        )

    def set_listening(self, enabled: bool) -> dict[str, Any]:
        self.listening_enabled = enabled
        return {"status": "success", "listening_enabled": self.listening_enabled}

    def _output_stream_kwargs(self) -> dict[str, Any]:
        if not sys.platform.startswith("linux"):
            return {}

        env_index = os.getenv("PYAUDIO_OUTPUT_DEVICE_INDEX")
        if env_index:
            try:
                index = int(env_index)
                print(f"Using output device from env: {index}")
                return {"output_device_index": index}
            except ValueError:
                print(f"Invalid PYAUDIO_OUTPUT_DEVICE_INDEX: {env_index}")

        try:
            info = self.pyaudio.get_default_output_device_info()
            index = int(info["index"])
            print(f"Using default output device: {info.get('name', index)}")
            return {"output_device_index": index}
        except Exception:
            pass

        try:
            for i in range(self.pyaudio.get_device_count()):
                info = self.pyaudio.get_device_info_by_index(i)
                if int(info.get("maxOutputChannels", 0)) > 0:
                    print(f"Using fallback output device: {info.get('name', i)}")
                    return {"output_device_index": i}
        except Exception:
            pass

        return {}

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "connected": self.session is not None,
            "model": self.model,
            "playback_chunks": self.playback_chunks,
            "url": self.page.url if self.page else None,
            "last_error": self.last_error,
            "listening_enabled": self.listening_enabled,
            "screenshot_interval_seconds": self.screenshot_interval_seconds,
            "last_model_text": self.last_model_text,
        }
