"""Settings for the Meta Model API, read from the environment."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


KEY_FILE = Path(os.environ.get("SAHARA_AI_KEY_FILE", Path.home() / ".config" / "sahara_ai" / "key"))


def _api_key():
    """MODEL_API_KEY from the environment, else from KEY_FILE (kept outside the repo)."""
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text().strip()
    return key


def _rate(name):
    v = os.environ.get(name)
    return float(v) if v else None


@dataclass
class Config:
    api_key: str = field(default_factory=_api_key)
    base_url: str = field(default_factory=lambda: os.environ.get("SAHARA_AI_BASE_URL", "https://api.meta.ai/v1").rstrip("/"))
    agent_model: str = field(default_factory=lambda: os.environ.get("SAHARA_AI_AGENT_MODEL", "muse-spark-1.3"))
    vision_model: str = field(default_factory=lambda: os.environ.get("SAHARA_AI_VISION_MODEL", "muse-spark-1.3"))
    sam_model: str = field(default_factory=lambda: os.environ.get("SAHARA_AI_SAM_MODEL", "sam-3.1"))
    voice_model: str = field(default_factory=lambda: os.environ.get("SAHARA_AI_VOICE_MODEL", "muse-voice-transcribe-1.0"))
    timeout_s: float = field(default_factory=lambda: float(os.environ.get("SAHARA_AI_TIMEOUT", "60")))
    # Force mock mode even with a key (offline development, CI).
    force_mock: bool = field(default_factory=lambda: os.environ.get("SAHARA_AI_MOCK", "") == "1")
    usage_log: Path = field(default_factory=lambda: Path(os.environ.get("SAHARA_AI_USAGE_LOG", ROOT / "logs" / "ai_usage.jsonl")))
    # USD per million tokens for the cost estimate in the usage log. Defaults
    # are Muse Spark's published launch prices ($1.25 in / $4.25 out); SAM 3.1
    # has no published token price, so its calls are counted, not costed.
    price_in: float = field(default_factory=lambda: _rate("SAHARA_AI_PRICE_IN") or 1.25)
    price_out: float = field(default_factory=lambda: _rate("SAHARA_AI_PRICE_OUT") or 4.25)
    # Hard stop on real API calls per day (all models), to protect credits.
    max_calls_per_day: int = field(default_factory=lambda: int(os.environ.get("SAHARA_AI_MAX_CALLS_PER_DAY", "300")))

    @property
    def mock(self):
        return self.force_mock or not self.api_key
