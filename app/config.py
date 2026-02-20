import os

DEFAULT_MODEL = os.getenv(
    "GEMINI_LIVE_MODEL",
    "gemini-2.5-flash-native-audio-preview-12-2025",
)
SCREENSHOT_INTERVAL_SECONDS = float(os.getenv("SCREENSHOT_INTERVAL_SECONDS", "3"))
DEFAULT_HOST = os.getenv("HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("PORT", "8000"))
