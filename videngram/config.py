"""
VidEngram Configuration
Central configuration for bridging Qwen2.5-Omni + EverMemOS.
All tunable parameters live here.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class QwenOmniConfig:
    """Qwen2.5-Omni served via vLLM-Omni or an external API (e.g. DashScope)."""
    base_url: str = os.getenv("QWEN_BASE_URL", "http://localhost:8091/v1")
    model: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Omni-7B")
    api_key: str = os.getenv("QWEN_API_KEY", "EMPTY")
    max_tokens: int = 512       # used in local (vLLM-Omni) mode
    max_tokens_api: int = 2048  # used in external API mode
    temperature: float = 0.3
    modalities: list = field(default_factory=lambda: ["text"])

    @property
    def is_local(self) -> bool:
        """True when QWEN_BASE_URL points to a local or port-forwarded vLLM-Omni server.

        Controls API request parameters only (vLLM-specific mm_processor_kwargs and
        max_tokens vs external API enable_thinking and max_tokens_api).

        File transport (file:// vs base64) is controlled separately by
        RemoteConfig.enabled: remote mode uses file:// paths on the server;
        local mode (including SSH port-forwarding) uses base64-encoded payloads.
        """
        return self.base_url.startswith(
            ("http://localhost", "http://127.0.0.1", "http://0.0.0.0")
        )


@dataclass
class EverMemOSConfig:
    """EverMemOS API server."""
    base_url: str = os.getenv("EVERMEMOS_BASE_URL", "http://localhost:8001")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    @property
    def memorize_url(self) -> str:
        return f"{self.base_url}/api/v1/memories"

    @property
    def search_url(self) -> str:
        return f"{self.base_url}/api/v1/memories/search"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass
class RemoteConfig:
    """SSH remote execution for segmenter and ffmpeg."""
    host: str = os.getenv("REMOTE_HOST", "")
    remote_work_dir: str = os.getenv("REMOTE_WORK_DIR", "")

    @property
    def enabled(self) -> bool:
        return bool(self.host and self.remote_work_dir)


@dataclass
class SegmenterConfig:
    """HippoMM-inspired temporal segmentation."""
    strategy: str = "adaptive"
    fixed_window_sec: float = 10.0   # fallback window for speech/lecture videos
    ssim_threshold: float = 0.65
    silence_threshold_db: float = -40.0
    min_segment_sec: float = 3.0    # 3 s minimum → captures a complete sentence
    max_segment_sec: float = 10.0    # 10 s cap → covers a full thought/argument
    analysis_fps: float = 2.0


@dataclass
class CaptionerConfig:
    """Qwen2.5-Omni captioning settings."""
    max_clip_duration: float = 30.0
    use_audio_in_video: bool = True
    caption_fields: list = field(default_factory=lambda: [
        "scene_description", "visible_text_or_signs",
        "people_and_actions", "dialogue_or_speech",
        "sounds_and_music", "emotional_tone",
        "notable_objects", "temporal_cues",
    ])


@dataclass
class ConsolidatorConfig:
    """HippoMM-inspired memory consolidation before writing to EverMemOS."""
    merge_similarity_threshold: float = 0.85
    episode_min_segments: int = 1   # minimum segments before model is asked to decide boundary
    episode_max_segments: int = 5   # hard upper bound; forces split regardless of model decision
    build_profiles: bool = True


@dataclass
class TranscriberConfig:
    """Whisper-compatible ASR for full-video speech transcription."""
    api_key: str = field(default_factory=lambda: os.getenv("TRANSCRIBER_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("TRANSCRIBER_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("TRANSCRIBER_MODEL", "whisper-1"))

    @property
    def enabled(self) -> bool:
        """Enabled iff an API key has been configured."""
        return bool(self.api_key)


@dataclass
class AgentConfig:
    """Agentic query orchestrator."""
    planning_llm_base_url: str = os.getenv("PLANNING_LLM_BASE_URL", "")
    planning_llm_model: str = os.getenv("PLANNING_LLM_MODEL", "")
    planning_llm_api_key: str = os.getenv("PLANNING_LLM_API_KEY", "")
    max_iterations: int = 5
    enable_video_grounding: bool = True

    def __post_init__(self):
        if not self.planning_llm_base_url:
            self.planning_llm_base_url = os.getenv(
                "QWEN_BASE_URL", "http://localhost:8091/v1"
            )
        if not self.planning_llm_model:
            self.planning_llm_model = os.getenv(
                "QWEN_MODEL", "Qwen/Qwen2.5-Omni-7B"
            )
        if not self.planning_llm_api_key:
            self.planning_llm_api_key = os.getenv("QWEN_API_KEY", "EMPTY")


@dataclass
class VidEngramConfig:
    """Top-level config aggregating all sub-configs."""
    qwen: QwenOmniConfig = field(default_factory=QwenOmniConfig)
    evermemos: EverMemOSConfig = field(default_factory=EverMemOSConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)
    consolidator: ConsolidatorConfig = field(default_factory=ConsolidatorConfig)
    transcriber: TranscriberConfig = field(default_factory=TranscriberConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Local work_dir is only used for storing metadata and small files
    work_dir: Path = Path(os.getenv("VIDENGRAM_WORK_DIR", "/tmp/videngram"))

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        # Create local directory normally (for metadata storage)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # The segments directory on the server is created by segmenter; don't touch it here

    def validate(self) -> list[str]:
        import shutil
        issues = []

        # ffmpeg is only needed when running segmenter locally
        if not self.remote.enabled:
            if not shutil.which("ffmpeg"):
                issues.append("[CRITICAL] ffmpeg not found in PATH — segmenter will fail")
            if not shutil.which("ffprobe"):
                issues.append("[CRITICAL] ffprobe not found in PATH — segmenter will fail")
        else:
            issues.append(
                f"[INFO] Remote segmenter enabled: {self.remote.host}, "
                f"work_dir={self.remote.remote_work_dir}"
            )

        for name, url in [
            ("Qwen base_url", self.qwen.base_url),
            ("EverMemOS base_url", self.evermemos.base_url),
        ]:
            if not url.startswith(("http://", "https://")):
                issues.append(f"[ERROR] {name} is not a valid URL: {url}")

        if ":8000/" in self.qwen.base_url or ":8000" == self.qwen.base_url[-5:]:
            issues.append(
                "[WARNING] Qwen base_url uses port 8000 — vLLM-Omni default is 8091. "
                "Did you mean http://localhost:8091/v1?"
            )

        if not self.agent.planning_llm_base_url:
            issues.append("[WARNING] No planning LLM configured — agent will use Qwen for reasoning")

        if self.remote.host and not self.remote.remote_work_dir:
            issues.append("[ERROR] REMOTE_HOST set but REMOTE_WORK_DIR is empty")

        return issues