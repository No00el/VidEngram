"""
VidEngram — Hippocampal Video Memory via Qwen2.5-Omni + EverMemOS
═══════════════════════════════════════════════════════════════════

Bridges multimodal video understanding (vLLM-Omni / Qwen2.5-Omni)
with long-term structured memory (EverMemOS) using a hippocampal-inspired
encoding-consolidation-retrieval pipeline and an agentic ReAct query loop.

Quick Start:
    from videngram import VidEngramPipeline

    pipe = VidEngramPipeline()
    pipe.ingest("lecture.mp4")
    ans = pipe.query("What was discussed in the first 5 minutes?", "lecture.mp4")
    print(ans.answer)
"""

__version__ = "0.1.0"

from .config import (
    VidEngramConfig,
    QwenOmniConfig,
    EverMemOSConfig,
    SegmenterConfig,
    CaptionerConfig,
    ConsolidatorConfig,
    AgentConfig,
)
from .utils import (
    VideoSegment,
    Caption,
    ConsolidatedMemory,
    MemoryResult,
    AgentAction,
    AgentResponse,
)
from .pipeline import VidEngramPipeline
from .segmenter import VideoSegmenter
from .captioner import Captioner
from .consolidator import Consolidator
from .memory_writer import MemoryWriter
from .memory_reader import MemoryReader
from .agent import VidEngramAgent

__all__ = [
    "VidEngramPipeline",
    "VidEngramConfig", "QwenOmniConfig", "EverMemOSConfig",
    "SegmenterConfig", "CaptionerConfig", "ConsolidatorConfig", "AgentConfig",
    "VideoSegmenter", "Captioner", "Consolidator",
    "MemoryWriter", "MemoryReader", "VidEngramAgent",
    "VideoSegment", "Caption", "ConsolidatedMemory",
    "MemoryResult", "AgentAction", "AgentResponse",
]
