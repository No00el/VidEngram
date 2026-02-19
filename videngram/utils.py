"""
VidEngram Utilities
Shared data classes, helpers, and timestamp mapping logic.
"""
import hashlib
import logging
import subprocess
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
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
    memory_type: str = "segment"  # "segment" | "episode_summary" | "entity_profile"
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
          - [Video 1.5min - 3.0min] ...       (segment memories)
          - [Episode 0.0min - 5.0min] ...      (episode summaries)
          - [Video analysis 1.5min - 3.0min]   (agent grounding)
        """
        import re
        # Match: [Video|Episode (optional "analysis")] Xmin - Ymin]
        match = re.search(
            r"\[(?:Video|Episode)(?:\s+analysis)?\s+(\d+\.?\d*)min\s*-\s*(\d+\.?\d*)min\]",
            self.content,
        )
        if match:
            return float(match.group(1)) * 60, float(match.group(2)) * 60
        # Fallback: try space-separated format [Video X - Y min]
        match = re.search(
            r"\[Video\s+(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*min\]",
            self.content,
        )
        if match:
            return float(match.group(1)) * 60, float(match.group(2)) * 60
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


# ── Timestamp Mapping ─────────────────────────────────────────────────────

def video_sec_to_datetime(
    video_sec: float,
    base_datetime: str = "2025-01-01T00:00:00+00:00",
    time_scale_factor: int = 60,
) -> str:
    """Map video seconds → virtual datetime for EverMemOS create_time.

    We scale video time so that 1 video-second = `time_scale_factor`
    virtual-seconds. This gives EverMemOS's temporal reasoning enough
    spread to distinguish events that are only seconds apart in the video.

    Example: video_sec=90 with scale=60 → base + 5400 seconds = 1.5 hours later.
    """
    base = datetime.fromisoformat(base_datetime)
    delta = timedelta(seconds=video_sec * time_scale_factor)
    return (base + delta).isoformat()


def datetime_to_video_sec(
    dt_str: str,
    base_datetime: str = "2025-01-01T00:00:00+00:00",
    time_scale_factor: int = 60,
) -> float:
    """Inverse of video_sec_to_datetime."""
    base = datetime.fromisoformat(base_datetime)
    dt = datetime.fromisoformat(dt_str)
    total_sec = (dt - base).total_seconds()
    return total_sec / time_scale_factor


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
    """Format as decimal minutes like '1.5min'."""
    return f"{seconds / 60:.1f}min"
