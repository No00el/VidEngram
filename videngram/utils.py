"""
VidEngram Utilities
Shared data classes, helpers, and timestamp mapping logic.
"""
import hashlib
import logging
import subprocess
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("videngram")


# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class VideoSegment:
    """A temporal slice of the source video."""
    segment_id: str
    video_path: str          # Path to source video
    start_sec: float         # Start time in video (seconds)
    end_sec: float           # End time in video (seconds)
    clip_path: Optional[str] = None  # Path to extracted .mp4 clip
    asr_text: Optional[str] = None   # Whisper transcript for this segment's time window

    def __post_init__(self):
        self.start_sec = float(self.start_sec)
        self.end_sec = float(self.end_sec)

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def timestamp_label(self) -> str:
        """Human-readable timestamp like '2:30 - 3:00'."""
        return f"{_fmt_time(self.start_sec)} - {_fmt_time(self.end_sec)}"


@dataclass
class Caption:
    """Structured caption for a video segment, produced by Qwen2.5-Omni."""
    segment_id: str
    raw_text: str                       # Full caption text
    structured: dict = field(default_factory=dict)  # Parsed fields
    start_sec: float = 0.0
    end_sec: float = 0.0

    def __post_init__(self):
        self.start_sec = float(self.start_sec)
        self.end_sec = float(self.end_sec)


@dataclass
class ConsolidatedMemory:
    """Post-consolidation memory unit ready for EverMemOS ingestion.
    Maps to HippoMM's ThetaEvent concept."""
    memory_id: str
    content: str              # Enriched text for EverMemOS content field
    start_sec: float          # Earliest timestamp in this memory
    end_sec: float            # Latest timestamp
    memory_type: str = "segment"  # "segment" | "episode_summary" | "entity_register"
    source_segments: list = field(default_factory=list)  # List of segment_ids
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryResult:
    """A single result from EverMemOS retrieval."""
    content: str
    score: float = 0.0
    memory_type: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def timestamp_range(self) -> Optional[tuple]:
        """Extract (start_sec, end_sec) from content if present.

        Handles all content formats produced by the consolidator and agent:
          - [Video 1:30 - 3:00] ...          (segment memories)
          - [Episode 0:00 - 5:00] ...         (episode summaries)
          - [Video analysis 1:30 - 3:00]      (agent grounding)
          - [Video 1:30:45 - 1:32:00]         (H:MM:SS for long videos)
        """
        import re

        def _parse_ts(ts: str) -> float:
            parts = ts.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return int(parts[0]) * 60 + int(parts[1])

        match = re.search(
            r"\[(?:Video|Episode)(?:\s+analysis)?\s+(\d+:\d{2}(?::\d{2})?)\s*-\s*(\d+:\d{2}(?::\d{2})?)\]",
            self.content,
        )
        if match:
            return _parse_ts(match.group(1)), _parse_ts(match.group(2))
        return None


@dataclass
class AgentAction:
    """An action taken by the agentic orchestrator."""
    tool: str
    input_params: dict
    output: str
    reasoning: str = ""


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    sources: list = field(default_factory=list)   # List of MemoryResult
    actions: list = field(default_factory=list)    # List of AgentAction
    grounded_clips: list = field(default_factory=list)  # Paths to relevant clips


# ── FFmpeg Helpers ────────────────────────────────────────────────────────

def extract_clip(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> str:
    """Extract a video clip using ffmpeg. Returns output path."""
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac",
        "-loglevel", "error",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def generate_id(prefix: str, *parts) -> str:
    """Generate a deterministic short ID from parts."""
    raw = "|".join(str(p) for p in parts)
    h = hashlib.md5(raw.encode()).hexdigest()[:10]
    return f"{prefix}_{h}"


# ── Formatting ────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_minutes(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS timecode (e.g. '1:30', '10:05')."""
    return _fmt_time(seconds)


# ── HTTP Session ─────────────────────────────────────────────────────────

def create_http_session(retries: int = 3, backoff: float = 0.5):
    """Create an HTTP session with retry logic and connection pooling."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
