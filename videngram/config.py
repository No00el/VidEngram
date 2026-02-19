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
    """Qwen2.5-Omni served via vLLM-Omni."""
    base_url: str = os.getenv("QWEN_BASE_URL", "http://localhost:8091/v1")
    model: str = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-Omni-7B")
    api_key: str = os.getenv("QWEN_API_KEY", "EMPTY")
    max_tokens: int = 2048
    temperature: float = 0.3
    # Text-only output for captioning (skip audio generation = faster)
    modalities: list = field(default_factory=lambda: ["text"])


@dataclass
class EverMemOSConfig:
    """EverMemOS API server.
    
    Supports both v1 (documented REST) and v3 (agentic) endpoints.
    We use v1 for storing and v1/search for retrieval by default,
    consistent with the official EverMemOS usage docs.
    """
    base_url: str = os.getenv("EVERMEMOS_BASE_URL", "http://localhost:8001")
    # LLM config for agentic retrieval (uses EverMemOS's internal LLM)
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    @property
    def memorize_url(self) -> str:
        """POST /api/v1/memories — store a single message."""
        return f"{self.base_url}/api/v1/memories"

    @property
    def search_url(self) -> str:
        """GET /api/v1/memories/search — retrieve memories."""
        return f"{self.base_url}/api/v1/memories/search"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}/health"


@dataclass
class SegmenterConfig:
    """HippoMM-inspired temporal segmentation."""
    # Strategy: "fixed" (fixed window) or "adaptive" (scene-change + silence)
    strategy: str = "adaptive"
    # Fixed window size in seconds
    fixed_window_sec: float = 30.0
    # Adaptive: SSIM threshold for scene change (lower = more sensitive)
    ssim_threshold: float = 0.65
    # Adaptive: audio silence threshold in dBFS
    silence_threshold_db: float = -40.0
    # Minimum / maximum segment duration (seconds)
    min_segment_sec: float = 5.0
    max_segment_sec: float = 60.0
    # Frame sampling rate for SSIM computation
    analysis_fps: float = 2.0


@dataclass
class CaptionerConfig:
    """Qwen2.5-Omni captioning settings."""
    # Max video segment duration the model sees at once (seconds)
    max_clip_duration: float = 30.0
    # Whether to include audio analysis in captions
    use_audio_in_video: bool = True
    # Structured caption fields to request
    caption_fields: list = field(default_factory=lambda: [
        "scene_description", "visible_text_or_signs",
        "people_and_actions", "dialogue_or_speech",
        "sounds_and_music", "emotional_tone",
        "notable_objects", "temporal_cues",
    ])


@dataclass
class ConsolidatorConfig:
    """HippoMM-inspired memory consolidation before writing to EverMemOS."""
    # Cosine similarity threshold: merge segments above this
    merge_similarity_threshold: float = 0.85
    # Episode grouping: max segments per episode summary
    episode_max_segments: int = 10
    # Whether to generate entity profiles from consolidated memories
    build_profiles: bool = True


@dataclass
class AgentConfig:
    """Agentic query orchestrator."""
    # LLM for the agent's planning/reasoning (can be same as Qwen or external)
    planning_llm_base_url: str = os.getenv("PLANNING_LLM_BASE_URL", "")
    planning_llm_model: str = os.getenv("PLANNING_LLM_MODEL", "")
    planning_llm_api_key: str = os.getenv("PLANNING_LLM_API_KEY", "")
    # Max ReAct iterations before forcing an answer
    max_iterations: int = 5
    # Whether to use video grounding (extract + re-analyze clips)
    enable_video_grounding: bool = True

    def __post_init__(self):
        # Fall back to Qwen for planning if not configured
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
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)
    consolidator: ConsolidatorConfig = field(default_factory=ConsolidatorConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    # Working directory for temp files (extracted clips, etc.)
    work_dir: Path = Path(os.getenv("VIDENGRAM_WORK_DIR", "/tmp/videngram"))
    # Time scale: 1 video-second = N virtual-seconds for EverMemOS timestamps
    # This spreads video time across a wider datetime range so EverMemOS's
    # temporal reasoning can distinguish fine-grained video moments.
    time_scale_factor: int = 60
    # Base datetime for virtual timestamps (arbitrary anchor)
    base_datetime: str = "2025-01-01T00:00:00+00:00"

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> list[str]:
        """Validate configuration, returning a list of warnings/errors.
        
        Call this before running the pipeline to catch misconfigurations early.
        """
        import shutil
        issues = []

        # Check ffmpeg availability
        if not shutil.which("ffmpeg"):
            issues.append("[CRITICAL] ffmpeg not found in PATH — segmenter will fail")
        if not shutil.which("ffprobe"):
            issues.append("[CRITICAL] ffprobe not found in PATH — segmenter will fail")

        # Check URL format
        for name, url in [
            ("Qwen base_url", self.qwen.base_url),
            ("EverMemOS base_url", self.evermemos.base_url),
        ]:
            if not url.startswith(("http://", "https://")):
                issues.append(f"[ERROR] {name} is not a valid URL: {url}")

        # Check port consistency
        if ":8000/" in self.qwen.base_url or ":8000" == self.qwen.base_url[-5:]:
            issues.append(
                "[WARNING] Qwen base_url uses port 8000 — vLLM-Omni default is 8091. "
                "Did you mean http://localhost:8091/v1?"
            )

        # Check time scale sanity
        if self.time_scale_factor < 1:
            issues.append("[ERROR] time_scale_factor must be >= 1")
        if self.time_scale_factor > 3600:
            issues.append("[WARNING] time_scale_factor > 3600 may cause datetime overflow for long videos")

        # Check agent config
        if not self.agent.planning_llm_base_url:
            issues.append("[WARNING] No planning LLM configured — agent will use Qwen for reasoning")

        return issues
