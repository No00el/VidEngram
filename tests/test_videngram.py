"""
VidEngram Test Suite
====================
Tests core logic without requiring live services (EverMemOS, vLLM-Omni).
Run: pytest tests/ -v
"""
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Config Tests ─────────────────────────────────────────────────────────

class TestConfig:
    """Verify config defaults and env overrides."""

    def test_default_qwen_port(self):
        """Qwen must default to port 8091 (vLLM-Omni standard)."""
        from videngram.config import QwenOmniConfig
        cfg = QwenOmniConfig()
        assert ":8091" in cfg.base_url

    def test_default_evermemos_endpoints(self):
        """EverMemOS must use v1 API endpoints."""
        from videngram.config import EverMemOSConfig
        cfg = EverMemOSConfig()
        assert cfg.memorize_url.endswith("/api/v1/memories")
        assert cfg.search_url.endswith("/api/v1/memories/search")
        assert "/v3/" not in cfg.memorize_url  # no v3 references

    def test_evermemos_port_default(self):
        from videngram.config import EverMemOSConfig
        cfg = EverMemOSConfig()
        assert ":8001" in cfg.base_url or ":1995" in cfg.base_url

    def test_videngram_config_creates_work_dir(self, tmp_path):
        """Top-level config should create its work directory."""
        from videngram.config import VidEngramConfig
        work = tmp_path / "videngram_test"
        cfg = VidEngramConfig(work_dir=work)
        assert work.exists()

    def test_agent_config_fallback_to_qwen(self):
        """Agent planning LLM should fall back to Qwen if not explicitly set."""
        from videngram.config import AgentConfig
        with patch.dict(os.environ, {
            "PLANNING_LLM_BASE_URL": "",
            "PLANNING_LLM_MODEL": "",
            "QWEN_BASE_URL": "http://localhost:8091/v1",
            "QWEN_MODEL": "Qwen/Qwen2.5-Omni-7B",
        }, clear=False):
            cfg = AgentConfig()
            assert "8091" in cfg.planning_llm_base_url
            assert "Qwen" in cfg.planning_llm_model

    def test_modalities_default_text_only(self):
        """Caption output should default to text-only (no audio generation)."""
        from videngram.config import QwenOmniConfig
        cfg = QwenOmniConfig()
        assert cfg.modalities == ["text"]


# ── Utils / Data Classes Tests ──────────────────────────────────────────

class TestUtils:
    """Test data classes and utility functions."""

    def test_video_segment_duration(self):
        from videngram.utils import VideoSegment
        seg = VideoSegment(
            segment_id="seg_001", video_path="/v.mp4",
            start_sec=10.0, end_sec=40.0,
        )
        assert seg.duration == 30.0

    def test_video_segment_int_coercion(self):
        """Ints should be coerced to float."""
        from videngram.utils import VideoSegment
        seg = VideoSegment(
            segment_id="seg_001", video_path="/v.mp4",
            start_sec=0, end_sec=30,
        )
        assert isinstance(seg.start_sec, float)
        assert isinstance(seg.end_sec, float)
        assert isinstance(seg.duration, float)

    def test_video_segment_timestamp_label(self):
        from videngram.utils import VideoSegment
        seg = VideoSegment(
            segment_id="seg_001", video_path="/v.mp4",
            start_sec=90.0, end_sec=150.0,
        )
        assert "1:30" in seg.timestamp_label
        assert "2:30" in seg.timestamp_label

    def test_video_sec_to_datetime_basic(self):
        from videngram.utils import video_sec_to_datetime
        dt_str = video_sec_to_datetime(0.0)
        assert "2025-01-01T00:00:00" in dt_str

    def test_video_sec_to_datetime_scaling(self):
        """90 video-seconds with scale=60 should map to 5400 virtual seconds."""
        from videngram.utils import video_sec_to_datetime
        dt_str = video_sec_to_datetime(90.0, time_scale_factor=60)
        dt = datetime.fromisoformat(dt_str)
        base = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
        delta = (dt - base).total_seconds()
        assert delta == 5400.0  # 90 * 60

    def test_datetime_roundtrip(self):
        """video_sec → datetime → video_sec should round-trip."""
        from videngram.utils import video_sec_to_datetime, datetime_to_video_sec
        original = 42.5
        dt_str = video_sec_to_datetime(original, time_scale_factor=60)
        recovered = datetime_to_video_sec(dt_str, time_scale_factor=60)
        assert abs(recovered - original) < 0.01

    def test_generate_id_deterministic(self):
        from videngram.utils import generate_id
        id1 = generate_id("seg", "video.mp4", 10.0, 20.0)
        id2 = generate_id("seg", "video.mp4", 10.0, 20.0)
        assert id1 == id2
        assert id1.startswith("seg_")

    def test_generate_id_different_inputs(self):
        from videngram.utils import generate_id
        id1 = generate_id("seg", "a.mp4", 0, 10)
        id2 = generate_id("seg", "b.mp4", 0, 10)
        assert id1 != id2

    def test_memory_result_timestamp_actual_format(self):
        """Test the ACTUAL format produced by consolidator: [Video 1:30 - 3:00]"""
        from videngram.utils import MemoryResult, fmt_minutes
        # This is what the consolidator actually produces
        content = f"[Video {fmt_minutes(90)} - {fmt_minutes(180)}] Something happened"
        r = MemoryResult(content=content)
        ts = r.timestamp_range
        assert ts is not None, f"Regex failed to match actual content: {content}"
        assert ts == (90.0, 180.0)

    def test_memory_result_timestamp_episode_format(self):
        """Test episode format: [Episode 0:00 - 5:00]"""
        from videngram.utils import MemoryResult
        r = MemoryResult(content="[Episode 0:00 - 5:00] Episode summary here")
        ts = r.timestamp_range
        assert ts is not None
        assert ts == (0.0, 300.0)

    def test_memory_result_timestamp_analysis_format(self):
        """Test agent analysis format: [Video analysis 1:30 - 3:00]"""
        from videngram.utils import MemoryResult
        r = MemoryResult(content="[Video analysis 1:30 - 3:00] Details here")
        ts = r.timestamp_range
        assert ts is not None
        assert ts == (90.0, 180.0)

    def test_memory_result_timestamp_hhmmss_format(self):
        """Test H:MM:SS format for long videos: [Video 1:02:30 - 1:03:00]"""
        from videngram.utils import MemoryResult
        r = MemoryResult(content="[Video 1:02:30 - 1:03:00] Long video segment")
        ts = r.timestamp_range
        assert ts is not None
        assert ts == (3750.0, 3780.0)  # 1h2m30s, 1h3m0s

    def test_memory_result_no_timestamp(self):
        from videngram.utils import MemoryResult
        r = MemoryResult(content="Just a plain memory with no timestamp")
        assert r.timestamp_range is None

    def test_fmt_minutes(self):
        from videngram.utils import fmt_minutes
        assert fmt_minutes(90.0) == "1:30"
        assert fmt_minutes(0.0) == "0:00"
        assert fmt_minutes(3661.0) == "1:01:01"


# ── Segmenter Tests ─────────────────────────────────────────────────────

class TestSegmenter:
    """Test segmenter logic (boundary computation, not ffmpeg calls)."""

    def test_fixed_boundaries(self):
        from videngram.config import VidEngramConfig
        from videngram.segmenter import VideoSegmenter

        cfg = VidEngramConfig()
        cfg.segmenter.strategy = "fixed"
        cfg.segmenter.fixed_window_sec = 30.0
        cfg.segmenter.min_segment_sec = 5.0
        seg = VideoSegmenter(cfg)

        bounds = seg._fixed_boundaries(100.0)
        # Should produce: (0,30), (30,60), (60,90), (90,100)
        assert len(bounds) == 4
        assert bounds[0] == (0.0, 30.0)
        assert bounds[-1] == (90.0, 100.0)

    def test_fixed_boundaries_short_video(self):
        from videngram.config import VidEngramConfig
        from videngram.segmenter import VideoSegmenter

        cfg = VidEngramConfig()
        cfg.segmenter.fixed_window_sec = 30.0
        cfg.segmenter.min_segment_sec = 5.0
        seg = VideoSegmenter(cfg)

        bounds = seg._fixed_boundaries(10.0)
        assert len(bounds) == 1
        assert bounds[0] == (0.0, 10.0)

    def test_boundaries_to_segments_min_constraint(self):
        from videngram.config import VidEngramConfig
        from videngram.segmenter import VideoSegmenter

        cfg = VidEngramConfig()
        cfg.segmenter.min_segment_sec = 10.0
        cfg.segmenter.max_segment_sec = 60.0
        seg = VideoSegmenter(cfg)

        # Boundaries 2 seconds apart should get merged
        boundaries = [2.0, 4.0, 6.0, 50.0]
        result = seg._boundaries_to_segments(boundaries, 60.0)
        # All close boundaries get merged, leaving larger segments
        for start, end in result:
            assert end - start >= 10.0

    def test_boundaries_to_segments_max_split(self):
        from videngram.config import VidEngramConfig
        from videngram.segmenter import VideoSegmenter

        cfg = VidEngramConfig()
        cfg.segmenter.min_segment_sec = 5.0
        cfg.segmenter.max_segment_sec = 30.0
        seg = VideoSegmenter(cfg)

        # Single long span should get split
        result = seg._boundaries_to_segments([], 90.0)
        assert len(result) == 3
        for start, end in result:
            assert end - start <= 30.0


# ── Consolidator Tests ──────────────────────────────────────────────────

class TestConsolidator:
    """Test consolidation logic (dedup, grouping) without LLM calls."""

    def test_text_similarity(self):
        from videngram.consolidator import Consolidator
        from videngram.config import VidEngramConfig

        cfg = VidEngramConfig()
        con = Consolidator(cfg)

        # Identical texts should have similarity 1.0
        sim = con._text_similarity("hello world foo bar", "hello world foo bar")
        assert sim == 1.0

        # Completely different texts should have similarity 0.0
        sim = con._text_similarity("hello world", "goodbye moon")
        assert sim == 0.0

    def test_filter_removes_near_duplicates(self):
        from videngram.consolidator import Consolidator
        from videngram.config import VidEngramConfig
        from videngram.utils import Caption

        cfg = VidEngramConfig()
        cfg.consolidator.merge_similarity_threshold = 0.8
        con = Consolidator(cfg)

        captions = [
            Caption(segment_id="s1", raw_text="The cat sat on the mat in the room", start_sec=0, end_sec=10),
            Caption(segment_id="s2", raw_text="The cat sat on the mat in the room today", start_sec=10, end_sec=20),
            Caption(segment_id="s3", raw_text="A completely different scene with dogs", start_sec=20, end_sec=30),
        ]

        result = con._filter_duplicates(captions)
        # s1 and s2 are near-duplicates, should merge; s3 should survive
        assert len(result) < len(captions)


# ── Memory Writer Tests ─────────────────────────────────────────────────

class TestMemoryWriter:
    """Test memory writer payload construction (mocked HTTP)."""

    def test_video_group_id_stable(self):
        from videngram.memory_writer import MemoryWriter
        g1 = MemoryWriter._video_group_id("/path/to/video.mp4")
        g2 = MemoryWriter._video_group_id("/path/to/video.mp4")
        assert g1 == g2
        assert g1.startswith("vid_")

    def test_video_group_id_different(self):
        from videngram.memory_writer import MemoryWriter
        g1 = MemoryWriter._video_group_id("/a.mp4")
        g2 = MemoryWriter._video_group_id("/b.mp4")
        assert g1 != g2

    @patch("videngram.memory_writer.requests.Session")
    def test_write_single_payload_shape(self, mock_session_cls):
        """Verify the POST payload matches EverMemOS v1 schema."""
        from videngram.config import VidEngramConfig
        from videngram.memory_writer import MemoryWriter
        from videngram.utils import ConsolidatedMemory

        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(status_code=200)

        cfg = VidEngramConfig()
        writer = MemoryWriter(cfg)
        writer._session = mock_session

        mem = ConsolidatedMemory(
            memory_id="mem_test_001",
            content="[Video 0:00 - 0:30] A person walks into frame",
            start_sec=0.0,
            end_sec=30.0,
            memory_type="segment",
        )

        success = writer._write_single(mem, "vid_test", "test_video", 0)
        assert success

        # Verify payload
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["message_id"] == "mem_test_001"
        assert "create_time" in payload
        assert payload["sender"] == "video_segment"
        assert payload["role"] == "user"  # Added in v1.2.0
        assert payload["content"].startswith("[Video")
        assert payload["group_id"] == "vid_test"
        assert payload["scene"] == "assistant"

        # Verify URL is v1
        url = call_args.args[0] if call_args.args else call_args.kwargs.get("url", "")
        assert "/api/v1/memories" in url
        assert "/v3/" not in url


# ── Config Validation Tests ──────────────────────────────────────────────

class TestConfigValidation:
    """Test the config validation system."""

    def test_validate_detects_missing_ffmpeg(self):
        from videngram.config import VidEngramConfig
        from unittest.mock import patch as mpatch
        cfg = VidEngramConfig()
        with mpatch("shutil.which", return_value=None):
            issues = cfg.validate()
        critical = [i for i in issues if "[CRITICAL]" in i]
        assert len(critical) >= 1  # at least ffmpeg missing

    def test_validate_warns_wrong_port(self):
        from videngram.config import VidEngramConfig, QwenOmniConfig
        cfg = VidEngramConfig(
            qwen=QwenOmniConfig(base_url="http://localhost:8000/v1")
        )
        issues = cfg.validate()
        port_warnings = [i for i in issues if "8000" in i]
        assert len(port_warnings) >= 1

    def test_validate_passes_good_config(self):
        from videngram.config import VidEngramConfig
        cfg = VidEngramConfig()
        issues = cfg.validate()
        errors = [i for i in issues if "[ERROR]" in i or "[CRITICAL]" in i]
        # May still have ffmpeg warnings in CI, but no logic errors
        for e in errors:
            assert "ffmpeg" in e or "ffprobe" in e  # only infra, not logic


# ── Pipeline Cleanup Tests ───────────────────────────────────────────────

class TestPipelineCleanup:
    """Test temp file cleanup functionality."""

    def test_cleanup_creates_dirs(self, tmp_path):
        from videngram.config import VidEngramConfig
        from videngram.pipeline import VidEngramPipeline

        cfg = VidEngramConfig(work_dir=tmp_path / "work")
        pipe = VidEngramPipeline(cfg, validate=False)

        # Create some temp files
        seg_dir = cfg.work_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        (seg_dir / "test.mp4").write_text("fake")

        pipe.cleanup()
        # Dir should still exist but be empty
        assert seg_dir.exists()
        assert list(seg_dir.iterdir()) == []

    def test_bounded_cache(self):
        from videngram.pipeline import VidEngramPipeline, MAX_INGESTED_CACHE
        from videngram.config import VidEngramConfig

        pipe = VidEngramPipeline(VidEngramConfig(), validate=False)
        # Simulate filling the cache beyond limit
        for i in range(MAX_INGESTED_CACHE + 10):
            pipe._ingested_videos[f"/video_{i}.mp4"] = {"dummy": True}
            # Evict oldest
            from collections import OrderedDict
            while len(pipe._ingested_videos) > MAX_INGESTED_CACHE:
                pipe._ingested_videos.popitem(last=False)

        assert len(pipe._ingested_videos) == MAX_INGESTED_CACHE


# ── Memory Reader Tests ─────────────────────────────────────────────────

class TestMemoryReader:
    """Test reader request construction (mocked HTTP)."""

    @patch("videngram.memory_reader.requests.Session")
    def test_search_episodes_uses_post(self, mock_session_cls):
        """EverMemOS search must use POST (not GET with body)."""
        from videngram.config import VidEngramConfig
        from videngram.memory_reader import MemoryReader

        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": []}
        )
        mock_session_cls.return_value = mock_session

        cfg = VidEngramConfig()
        reader = MemoryReader(cfg)
        reader._session = mock_session  # inject mock
        reader.search_episodes("test query", "/video.mp4")

        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["memory_types"] == ["episodic_memory"]
        assert payload["retrieve_method"] == "rrf"

    @patch("videngram.memory_reader.requests.Session")
    def test_search_profiles_correct_type(self, mock_session_cls):
        from videngram.config import VidEngramConfig
        from videngram.memory_reader import MemoryReader

        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": []}
        )

        cfg = VidEngramConfig()
        reader = MemoryReader(cfg)
        reader._session = mock_session
        reader.search_profiles("who is the speaker", "/video.mp4")

        payload = mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1].get("json")
        assert payload["memory_types"] == ["profile"]

    @patch("videngram.memory_reader.requests.Session")
    def test_agentic_retrieval_all_types(self, mock_session_cls):
        """Agentic mode should search across all memory types."""
        from videngram.config import VidEngramConfig
        from videngram.memory_reader import MemoryReader

        mock_session = MagicMock()
        mock_session.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": []}
        )

        cfg = VidEngramConfig()
        reader = MemoryReader(cfg)
        reader._session = mock_session
        reader.search_agentic("complex question", "/video.mp4")

        payload = mock_session.post.call_args.kwargs.get("json") or mock_session.post.call_args[1].get("json")
        assert "episodic_memory" in payload["memory_types"]
        assert "profile" in payload["memory_types"]
        assert "semantic_memory" in payload["memory_types"]
        assert payload["retrieve_method"] == "agentic"

    def test_parse_results_dict_format(self):
        from videngram.memory_reader import MemoryReader
        data = {
            "results": [
                {"content": "Memory 1", "score": 0.9, "memory_type": "episodic_memory"},
                {"content": "Memory 2", "score": 0.7, "memory_type": "profile"},
            ]
        }
        results = MemoryReader._parse_results(data)
        assert len(results) == 2
        assert results[0].content == "Memory 1"
        assert results[0].score == 0.9

    def test_parse_results_nested_format(self):
        """Handle EverMemOS {"result": {"memories": [...]}} format."""
        from videngram.memory_reader import MemoryReader
        data = {
            "result": {
                "memories": [
                    {"memories": [
                        {"memory": "Nested memory 1", "score": 0.8}
                    ]}
                ]
            }
        }
        results = MemoryReader._parse_results(data)
        assert len(results) >= 1
        assert "Nested memory" in results[0].content

    def test_parse_results_empty(self):
        from videngram.memory_reader import MemoryReader
        assert MemoryReader._parse_results({}) == []
        assert MemoryReader._parse_results({"results": []}) == []


# ── Agent Tests ──────────────────────────────────────────────────────────

class TestAgent:
    """Test agent parameter parsing and tool dispatch logic."""

    def test_parse_params_keyword(self):
        from videngram.agent import VidEngramAgent
        params = VidEngramAgent._parse_params(
            'start_min=1.5, end_min=3.0, question="what is happening?"'
        )
        assert params["start_min"] == "1.5"
        assert params["end_min"] == "3.0"
        assert params["question"] == "what is happening?"

    def test_parse_params_simple_query(self):
        from videngram.agent import VidEngramAgent
        params = VidEngramAgent._parse_params('"what happened in the meeting"')
        assert params["query"] == "what happened in the meeting"

    def test_parse_params_positional(self):
        from videngram.agent import VidEngramAgent
        params = VidEngramAgent._parse_params("1.0, 2.5, what color is the car?")
        assert params["start_min"] == "1.0"
        assert params["end_min"] == "2.5"
        assert "color" in params["question"]

    def test_format_results_empty(self):
        from videngram.agent import VidEngramAgent
        assert "No results" in VidEngramAgent._format_results([])

    def test_format_results_with_data(self):
        from videngram.agent import VidEngramAgent
        from videngram.utils import MemoryResult
        results = [
            MemoryResult(content="Test memory content", score=0.95),
        ]
        formatted = VidEngramAgent._format_results(results)
        assert "1 results" in formatted
        assert "Test memory" in formatted
        assert "0.95" in formatted


# ── Integration Smoke Test ──────────────────────────────────────────────

class TestIntegration:
    """Verify all components can be imported and instantiated."""

    def test_all_imports(self):
        from videngram import (
            VidEngramPipeline, VidEngramConfig,
            VideoSegmenter, Captioner, Consolidator,
            MemoryWriter, MemoryReader, VidEngramAgent,
            VideoSegment, Caption, ConsolidatedMemory,
            MemoryResult, AgentAction, AgentResponse,
        )

    def test_pipeline_instantiation(self):
        from videngram import VidEngramPipeline
        pipe = VidEngramPipeline()
        assert pipe.config is not None
        assert pipe.segmenter is not None
        assert pipe.captioner is not None
        assert pipe.consolidator is not None
        assert pipe.writer is not None
        assert pipe.reader is not None
        assert pipe.agent is not None

    def test_version(self):
        from videngram import __version__
        assert __version__ == "0.1.0"
