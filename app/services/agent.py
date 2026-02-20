import asyncio
import base64
import os
import sys
import re
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
        self.last_tool_repeat_count = 0
        self.tool_failure_counts: dict[str, int] = {}
        self.approval_required = False
        self.pending_sensitive_tool: dict[str, Any] | None = None
        self.macro_steps: list[dict[str, Any]] = []

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
                name="go_back",
                description="Go back to previous page",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="scroll",
                description="Scroll page",
                parameters={
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["up", "down"]},
                        "amount": {"type": "number"},
                    },
                    "required": ["direction", "amount"],
                },
            ),
            types.FunctionDeclaration(
                name="take_screenshot",
                description="Capture current screen",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="click_button",
                description="Click a visible button by text",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            types.FunctionDeclaration(
                name="click_link",
                description="Click a link using part of its URL",
                parameters={
                    "type": "object",
                    "properties": {"href_contains": {"type": "string"}},
                    "required": ["href_contains"],
                },
            ),
            types.FunctionDeclaration(
                name="click_text",
                description="Click visible text on page",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            ),
            types.FunctionDeclaration(
                name="fill_by_label",
                description="Fill input by label text",
                parameters={
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["label", "value"],
                },
            ),
            types.FunctionDeclaration(
                name="fill_by_placeholder",
                description="Fill input by placeholder text",
                parameters={
                    "type": "object",
                    "properties": {
                        "placeholder": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["placeholder", "value"],
                },
            ),
            types.FunctionDeclaration(
                name="fill_by_role",
                description="Fill input by role (textbox, searchbox)",
                parameters={
                    "type": "object",
                    "properties": {
                        "role": {"type": "string"},
                        "value": {"type": "string"},
                    },
                    "required": ["role", "value"],
                },
            ),
            types.FunctionDeclaration(
                name="press_key",
                description="Press keyboard key",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            types.FunctionDeclaration(
                name="submit_form",
                description="Submit nearest form",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="extract_text",
                description="Extract visible page text",
                parameters={
                    "type": "object",
                    "properties": {"max_chars": {"type": "integer"}},
                },
            ),
            types.FunctionDeclaration(
                name="read_page_summary",
                description="Read full page text for summarization",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="detect_forms",
                description="Detect forms and fields",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="list_inputs",
                description="List all visible input-like fields",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="approve_sensitive_action",
                description="Approve previously blocked sensitive action",
                parameters={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="replay_macro",
                description="Replay previous recorded tool flow",
                parameters={"type": "object", "properties": {}},
            ),
        ]

    def _contains_sensitive_data(self, name: str, args: dict[str, Any]) -> bool:
        if name not in {"fill_by_label", "fill_by_placeholder", "fill_by_role"}:
            return False

        bag = " ".join(
            str(x).lower()
            for x in [args.get("label"), args.get("placeholder"), args.get("role"), args.get("value")]
            if x is not None
        )
        sensitive_terms = [
            "email",
            "phone",
            "password",
            "passcode",
            "otp",
            "bank",
            "card",
            "cvv",
            "ssn",
        ]
        if any(term in bag for term in sensitive_terms):
            return True
        if "@" in str(args.get("value", "")):
            return True
        return False

    async def _execute_tool(self, name: str, args: dict[str, Any], record_macro: bool = True) -> dict[str, Any]:
        page = self._ensure_page()

        if name == "navigate_to":
            await page.goto(args["url"], wait_until="networkidle")
            result = await self.page_info()
        elif name == "go_back":
            await page.go_back()
            result = await self.page_info()
        elif name == "scroll":
            amount = int(args["amount"])
            delta = amount if args["direction"] == "down" else -amount
            await page.evaluate(f"window.scrollBy(0, {delta})")
            result = {"status": "success", "direction": args["direction"], "amount": amount}
        elif name == "take_screenshot":
            img = await page.screenshot(type="jpeg", quality=80)
            result = {
                "status": "success",
                "image_base64": base64.b64encode(img).decode("utf-8"),
            }
        elif name == "click_button":
            await page.get_by_role("button", name=args["text"], exact=False).first.click()
            result = {"status": "success"}
        elif name == "click_link":
            href_fragment = self._safe_href_fragment(args["href_contains"])
            await page.locator(f'a[href*="{href_fragment}"]').first.click()
            result = {"status": "success"}
        elif name == "click_text":
            await page.get_by_text(args["text"], exact=False).first.click()
            result = {"status": "success"}
        elif name == "fill_by_label":
            await page.get_by_label(args["label"], exact=False).first.fill(args["value"])
            result = {"status": "success"}
        elif name == "fill_by_placeholder":
            await page.get_by_placeholder(args["placeholder"]).first.fill(args["value"])
            result = {"status": "success"}
        elif name == "fill_by_role":
            role = args["role"].strip().lower()
            if role not in {"textbox", "searchbox"}:
                raise RuntimeError("role must be one of: textbox, searchbox")
            await page.get_by_role(role).first.fill(args["value"])
            result = {"status": "success"}
        elif name == "press_key":
            await page.keyboard.press(args["key"])
            result = {"status": "success"}
        elif name == "submit_form":
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
            result = {"status": "success"}
        elif name == "extract_text":
            max_chars = int(args.get("max_chars", 4000))
            text = await page.evaluate(
                """(n) => document.body && document.body.innerText
                    ? document.body.innerText.slice(0, n)
                    : ''""",
                max_chars,
            )
            result = {"status": "success", "text": text}
        elif name == "read_page_summary":
            text = await page.evaluate(
                "() => document.body && document.body.innerText ? document.body.innerText : ''"
            )
            result = {"status": "success", "text": text[:6000]}
        elif name == "detect_forms":
            forms = await page.evaluate(
                """
                () => {
                  return Array.from(document.querySelectorAll("form")).map((form, i) => ({
                    index: i,
                    inputs: Array.from(form.querySelectorAll("input, textarea, select"))
                      .map(el => ({
                        name: el.name || null,
                        placeholder: el.placeholder || null,
                        type: el.type || el.tagName.toLowerCase()
                      }))
                  }));
                }
                """
            )
            result = {"status": "success", "forms": forms}
        elif name == "list_inputs":
            inputs = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll("input, textarea, select"))
                  .map(el => ({
                    label: el.labels && el.labels[0] ? el.labels[0].innerText : null,
                    placeholder: el.placeholder || null,
                    name: el.name || null,
                    type: el.type || el.tagName.toLowerCase()
                  }))
                """
            )
            result = {"status": "success", "inputs": inputs}
        elif name == "approve_sensitive_action":
            if not self.pending_sensitive_tool:
                result = {"status": "error", "message": "No pending sensitive action to approve."}
            else:
                pending = self.pending_sensitive_tool
                self.approval_required = False
                self.pending_sensitive_tool = None
                result = await self._execute_tool(
                    pending["name"], pending["args"], record_macro=False
                )
                result["approved"] = True
        elif name == "replay_macro":
            replay_steps = [
                step
                for step in self.macro_steps
                if step["name"] not in {"replay_macro", "approve_sensitive_action"}
            ]
            if not replay_steps:
                result = {"status": "error", "message": "No macro steps recorded yet."}
            else:
                executed = 0
                for step in replay_steps:
                    await self._execute_tool(step["name"], step["args"], record_macro=False)
                    executed += 1
                result = {"status": "success", "replayed_steps": executed}
        else:
            result = {"status": "error", "message": f"Unknown tool: {name}"}

        if record_macro and name not in {"replay_macro", "approve_sensitive_action"}:
            self.macro_steps.append({"name": name, "args": dict(args)})
        return result

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
            if self._contains_sensitive_data(name, args):
                self.approval_required = True
                self.pending_sensitive_tool = {"name": name, "args": dict(args)}
                result = {
                    "status": "error",
                    "message": (
                        "Sensitive action requires approval. Ask user to say or send: "
                        "'approve_sensitive_action'."
                    ),
                }
                return types.FunctionResponse(id=call_id, name=name, response=result)

            if name == self.last_tool_name:
                self.last_tool_repeat_count += 1
            else:
                self.last_tool_name = name
                self.last_tool_repeat_count = 1

            if self.last_tool_repeat_count > 2:
                result = {
                    "status": "error",
                    "message": f"Tool repetition guard blocked duplicate call: {name}",
                }
                return types.FunctionResponse(id=call_id, name=name, response=result)
            result = await self._execute_tool(name, args)
            self.tool_failure_counts[name] = 0
        except Exception as e:
            failures = self.tool_failure_counts.get(name, 0) + 1
            self.tool_failure_counts[name] = failures
            message = str(e)
            if failures >= 2:
                message = f"{message}. Tool failed twice; ask user what to do next."
            result = {"status": "error", "message": message}

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
                                "You are a voice-controlled browser assistant.\n\n"
                                "You can control a real web browser using tools.\n\n"
                                "CRITICAL RULES:\n"
                                "- You are controlling a real browser.\n"
                                "- You MUST ONLY use the provided tools.\n"
                                "- You MUST NOT generate CSS selectors, XPath, or Playwright code.\n"
                                "- You MUST NOT guess DOM structure.\n"
                                "- If an action fails twice, ask the user what to do next.\n"
                                "- If sensitive info is involved, request approval before proceeding.\n"
                                "- You can request extract_text(), list_inputs(), or detect_forms() to understand the page.\n"
                                "- Never repeat the same tool call more than twice.\n\n"
                                "Rules:\n"
                                "- NEVER invent CSS selectors or XPath.\n"
                                "- NEVER guess DOM structure.\n"
                                "- ONLY use the provided tools.\n"
                                "- Do NOT attempt to log in or enter passwords unless the user explicitly asks.\n"
                                "- Always ask before entering sensitive information (email, phone, OTP, password).\n"
                                "- If a tool fails twice, stop and ask the user what to do.\n"
                                "- Do NOT repeat the same tool call more than twice.\n"
                                "- Use screenshots to understand the page when unsure.\n\n"
                                "Advanced tools:\n"
                                "- Use extract_text / read_page_summary to read content.\n"
                                "- Use detect_forms / list_inputs before filling unknown forms.\n"
                                "- When sensitive action is blocked, call approve_sensitive_action only after user consent.\n"
                                "- Use replay_macro when the user asks to repeat the last flow.\n\n"
                                "How to act:\n"
                                "- Decide the next best action\n"
                                "- Call ONE tool at a time\n"
                                "- After navigation or form filling, briefly describe what you see\n"
                                "- Ask what the user wants to do next\n\n"
                                "Examples:\n"
                                "Example 1:\n"
                                "User: Open Google\n"
                                "Assistant: navigate_to(url=\"https://google.com\")\n\n"
                                "Example 2:\n"
                                "User: Search for Python tutorials\n"
                                "Assistant: fill_by_role(role=\"searchbox\", value=\"Python tutorials\")\n"
                                "Assistant: press_key(key=\"Enter\")\n\n"
                                "Example 3:\n"
                                "User: Fill my name as Ritesh\n"
                                "Assistant: fill_by_label(label=\"Name\", value=\"Ritesh\")\n\n"
                                "Example 4:\n"
                                "User: Click the first result\n"
                                "Assistant: click_link(href_contains=\"python\")\n\n"
                                "Example 5:\n"
                                "User: Scroll down\n"
                                "Assistant: scroll(direction=\"down\", amount=800)\n\n"
                                "Example 6:\n"
                                "User: Submit the form\n"
                                "Assistant: click_button(text=\"Submit\")"
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

    async def approve_sensitive_action(self) -> dict[str, Any]:
        return await self._execute_tool("approve_sensitive_action", {}, record_macro=False)

    async def replay_macro(self) -> dict[str, Any]:
        return await self._execute_tool("replay_macro", {}, record_macro=False)

    def set_listening(self, enabled: bool) -> dict[str, Any]:
        self.listening_enabled = enabled
        return {"status": "success", "listening_enabled": self.listening_enabled}

    def _safe_href_fragment(self, fragment: str) -> str:
        frag = fragment.replace("\\", "\\\\").replace('"', '\\"')
        # Keep fragments selector-safe and predictable.
        return re.sub(r"[\r\n\t]", "", frag)

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
            "approval_required": self.approval_required,
            "pending_sensitive_tool": self.pending_sensitive_tool,
            "macro_steps_count": len(self.macro_steps),
        }
