"""
VidEngram Runtime Validation
==============================
Traces every execution path with mocked externals to verify the code
actually runs end-to-end without real ffmpeg, vLLM-Omni, or EverMemOS.

This catches issues that unit tests miss:
- Import chains that break at runtime
- Method signature mismatches between caller and callee
- Missing attributes accessed deep in call chains
- Data flow issues (wrong types passed between components)
"""

import json
import sys
import traceback
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  ✅ {name}")
        PASS += 1
    except Exception as e:
        print(f"  ❌ {name}")
        traceback.print_exc()
        print()
        FAIL += 1


# ════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  VidEngram Runtime Validation")
print("=" * 60)

# ── 1. CONFIG CHAIN ──────────────────────────────────────────────

print("\n1. Configuration chain:")

def test_config_full_creation():
    from videngram.config import VidEngramConfig
    cfg = VidEngramConfig()
    # Verify all sub-configs exist and have correct types
    assert isinstance(cfg.qwen.base_url, str)
    assert isinstance(cfg.evermemos.base_url, str)
    assert isinstance(cfg.segmenter.strategy, str)
    assert isinstance(cfg.captioner.caption_fields, list)
    assert isinstance(cfg.consolidator.merge_similarity_threshold, float)
    assert isinstance(cfg.agent.max_iterations, int)
    assert isinstance(cfg.work_dir, Path)
    assert cfg.work_dir.exists()
    # Verify URLs are well-formed
    assert cfg.evermemos.memorize_url == "http://localhost:8001/api/v1/memories"
    assert cfg.evermemos.search_url == "http://localhost:8001/api/v1/memories/search"
    assert cfg.evermemos.health_url == "http://localhost:8001/health"
    # Agent should have fallen back to Qwen
    assert "8091" in cfg.agent.planning_llm_base_url

check("Full config creation + property access", test_config_full_creation)

def test_config_validation():
    from videngram.config import VidEngramConfig
    cfg = VidEngramConfig()
    issues = cfg.validate()
    assert isinstance(issues, list)
    for issue in issues:
        assert isinstance(issue, str)
        assert issue.startswith("[")

check("Config validation returns list of strings", test_config_validation)


# ── 2. UTILS DATA FLOW ──────────────────────────────────────────

print("\n2. Utils / data class flow:")

def test_data_classes_serializable():
    from videngram.utils import (
        VideoSegment, Caption, ConsolidatedMemory,
        MemoryResult, AgentAction, AgentResponse,
    )
    seg = VideoSegment("s1", "/v.mp4", 0.0, 30.0, "/clip.mp4")
    assert seg.duration == 30.0
    assert "0:00" in seg.timestamp_label
    assert "0:30" in seg.timestamp_label

    cap = Caption("s1", "A dog runs", start_sec=0.0, end_sec=30.0)
    assert cap.raw_text == "A dog runs"

    mem = ConsolidatedMemory("m1", "[Video 0:00 - 0:30] Dog", 0.0, 30.0, "segment", ["s1"])
    assert mem.memory_type == "segment"

    mr = MemoryResult("[Video 1:30 - 3:00] Something", 0.9, "episodic_memory")
    assert mr.timestamp_range == (90.0, 180.0)

    mr2 = MemoryResult("No timestamps here")
    assert mr2.timestamp_range is None

    act = AgentAction("search_episodes", {"query": "dog"}, "Found 3 results")
    resp = AgentResponse("The dog runs at 0:15", [mr], [act], ["/clip.mp4"])
    assert resp.answer.startswith("The dog")

check("All data classes construct + properties work", test_data_classes_serializable)

def test_timestamp_mapping_invertible():
    from videngram.utils import video_sec_to_datetime, datetime_to_video_sec
    for sec in [0.0, 1.0, 42.5, 90.0, 3600.0]:
        dt = video_sec_to_datetime(sec, time_scale_factor=60)
        recovered = datetime_to_video_sec(dt, time_scale_factor=60)
        assert abs(recovered - sec) < 0.001, f"Roundtrip failed: {sec} → {dt} → {recovered}"

check("Timestamp roundtrip for multiple values", test_timestamp_mapping_invertible)


# ── 3. SEGMENTER ─────────────────────────────────────────────────

print("\n3. Segmenter execution path:")

def test_segmenter_fixed_full_path():
    """Trace: segment() → _fixed_boundaries() → extract_clip() for each segment."""
    from videngram.config import VidEngramConfig
    from videngram.segmenter import VideoSegmenter

    cfg = VidEngramConfig()
    cfg.segmenter.strategy = "fixed"
    cfg.segmenter.fixed_window_sec = 30.0
    seg = VideoSegmenter(cfg)

    # Mock ffmpeg calls
    with patch("videngram.segmenter.get_video_duration", return_value=65.0), \
         patch("videngram.segmenter.extract_clip", return_value="/tmp/clip.mp4"):

        segments = seg.segment("/fake/video.mp4")
        # 65s / 30s = 2 full + 1 partial (5s, above min_segment_sec)
        assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
        assert segments[0].start_sec == 0.0
        assert segments[0].end_sec == 30.0
        assert segments[1].start_sec == 30.0
        assert segments[2].end_sec == 65.0
        # All should have clip_path set
        for s in segments:
            assert s.clip_path is not None

check("Fixed segmenter full path", test_segmenter_fixed_full_path)

def test_segmenter_adaptive_full_path():
    """Trace: segment() → _adaptive_boundaries() → _detect_scene_changes + _detect_silence."""
    from videngram.config import VidEngramConfig
    from videngram.segmenter import VideoSegmenter

    cfg = VidEngramConfig()
    cfg.segmenter.strategy = "adaptive"
    seg = VideoSegmenter(cfg)

    # Mock both detection methods and clip extraction
    with patch("videngram.segmenter.get_video_duration", return_value=120.0), \
         patch("videngram.segmenter.extract_clip", return_value="/tmp/clip.mp4"), \
         patch.object(seg, "_detect_scene_changes", return_value=[30.0, 60.0, 90.0]), \
         patch.object(seg, "_detect_silence", return_value=[45.0, 75.0]):

        segments = seg.segment("/fake/video.mp4")
        assert len(segments) > 0
        # Verify boundaries were merged and segments are non-overlapping
        for i in range(len(segments) - 1):
            assert segments[i].end_sec <= segments[i + 1].start_sec + 0.001

check("Adaptive segmenter full path", test_segmenter_adaptive_full_path)

def test_segmenter_ffmpeg_failure():
    """Verify graceful fallback when ffmpeg extract_clip fails."""
    from videngram.config import VidEngramConfig
    from videngram.segmenter import VideoSegmenter
    import subprocess

    cfg = VidEngramConfig()
    cfg.segmenter.strategy = "fixed"
    cfg.segmenter.fixed_window_sec = 30.0
    seg = VideoSegmenter(cfg)

    with patch("videngram.segmenter.get_video_duration", return_value=35.0), \
         patch("videngram.segmenter.extract_clip", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):

        segments = seg.segment("/fake/video.mp4")
        # 35s / 30s = (0-30) + (30-35) = 2 segments (5s >= min_segment_sec)
        assert len(segments) == 2, f"Expected 2 segments, got {len(segments)}"
        # ALL clip paths should be None on ffmpeg failure
        for s in segments:
            assert s.clip_path is None, f"Expected None clip_path, got {s.clip_path}"

check("Segmenter handles ffmpeg failure gracefully", test_segmenter_ffmpeg_failure)


# ── 4. CAPTIONER ─────────────────────────────────────────────────

print("\n4. Captioner execution path:")

def test_captioner_single_segment():
    """Trace: caption_segment() → OpenAI client.chat.completions.create()."""
    from videngram.config import VidEngramConfig
    from videngram.captioner import Captioner
    from videngram.utils import VideoSegment

    cfg = VidEngramConfig()

    with patch("videngram.captioner.OpenAI") as mock_openai_cls, \
         patch("videngram.captioner.AsyncOpenAI"):

        # Mock the OpenAI response chain
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A person walks into a bright room with white walls."
        mock_client.chat.completions.create.return_value = mock_response

        cap = Captioner(cfg)

        # Create a segment with a "real" clip path
        seg = VideoSegment("seg_001", "/v.mp4", 0.0, 30.0, "/tmp/clip.mp4")

        # Mock Path.exists() for clip check
        with patch("videngram.captioner.Path.exists", return_value=True):
            result = cap.caption_segment(seg, seg_index=0, total_segments=3)

        assert result.segment_id == "seg_001"
        assert result.raw_text == "A person walks into a bright room with white walls."
        assert result.start_sec == 0.0
        assert result.end_sec == 30.0

        # Verify the actual API call shape
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "Qwen/Qwen2.5-Omni-7B"
        assert call_args.kwargs["extra_body"] == {"modalities": ["text"]}
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        # User content should be a list (multimodal)
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "video_url"
        assert user_content[1]["type"] == "text"

check("Caption single segment + verify API call shape", test_captioner_single_segment)

def test_captioner_missing_clip():
    """caption_segment should return placeholder when clip is missing."""
    from videngram.config import VidEngramConfig
    from videngram.captioner import Captioner
    from videngram.utils import VideoSegment

    cfg = VidEngramConfig()
    with patch("videngram.captioner.OpenAI"), patch("videngram.captioner.AsyncOpenAI"):
        cap = Captioner(cfg)
        seg = VideoSegment("seg_001", "/v.mp4", 0.0, 30.0, clip_path=None)
        result = cap.caption_segment(seg)
        assert "[Clip unavailable]" in result.raw_text

check("Captioner handles missing clip", test_captioner_missing_clip)

def test_captioner_api_error():
    """caption_segment should return error placeholder on API failure."""
    from videngram.config import VidEngramConfig
    from videngram.captioner import Captioner
    from videngram.utils import VideoSegment

    cfg = VidEngramConfig()
    with patch("videngram.captioner.OpenAI") as mock_cls, \
         patch("videngram.captioner.AsyncOpenAI"):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("Connection refused")

        cap = Captioner(cfg)
        seg = VideoSegment("seg_001", "/v.mp4", 0.0, 30.0, "/tmp/clip.mp4")
        with patch("videngram.captioner.Path.exists", return_value=True):
            result = cap.caption_segment(seg)
        assert "[Caption error:" in result.raw_text

check("Captioner handles API error gracefully", test_captioner_api_error)

def test_captioner_caption_all():
    """caption_all() loops correctly and passes indices."""
    from videngram.config import VidEngramConfig
    from videngram.captioner import Captioner
    from videngram.utils import VideoSegment

    cfg = VidEngramConfig()
    with patch("videngram.captioner.OpenAI") as mock_cls, \
         patch("videngram.captioner.AsyncOpenAI"):
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Caption text"
        mock_client.chat.completions.create.return_value = mock_response

        cap = Captioner(cfg)
        segments = [
            VideoSegment(f"s{i}", "/v.mp4", i*30.0, (i+1)*30.0, f"/tmp/s{i}.mp4")
            for i in range(3)
        ]
        with patch("videngram.captioner.Path.exists", return_value=True):
            captions = cap.caption_all(segments, parallel=False)

        assert len(captions) == 3
        assert captions[0].segment_id == "s0"
        assert captions[2].segment_id == "s2"
        # Verify create was called 3 times
        assert mock_client.chat.completions.create.call_count == 3

check("caption_all loops correctly", test_captioner_caption_all)


# ── 5. CONSOLIDATOR ──────────────────────────────────────────────

print("\n5. Consolidator execution path:")

def test_consolidator_full_pipeline():
    """Trace: consolidate() → _filter_duplicates → _create_segment_memories → _create_episode_summaries → _extract_profiles."""
    from videngram.config import VidEngramConfig
    from videngram.consolidator import Consolidator
    from videngram.utils import Caption

    cfg = VidEngramConfig()
    cfg.consolidator.episode_max_segments = 2  # Small for testing
    cfg.consolidator.build_profiles = True

    with patch("videngram.consolidator.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        # Mock LLM responses for episode summary AND profile extraction
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        # First call = episode summary, subsequent = more episodes, last = profile
        mock_response.choices[0].message.content = "Episode summary about a meeting."
        mock_client.chat.completions.create.return_value = mock_response

        con = Consolidator(cfg)

        captions = [
            Caption("s1", "A person enters a conference room and sits down.", 0.0, 30.0),
            Caption("s2", "The person opens a laptop and starts presenting slides.", 30.0, 60.0),
            Caption("s3", "A completely different outdoor scene with trees and birds.", 60.0, 90.0),
        ]

        memories = con.consolidate(captions)

        # Should have: segment memories + episode summaries + profiles
        segment_mems = [m for m in memories if m.memory_type == "segment"]
        episode_mems = [m for m in memories if m.memory_type == "episode_summary"]

        assert len(segment_mems) == 3, f"Expected 3 segments, got {len(segment_mems)}"
        assert len(episode_mems) >= 1, f"Expected ≥1 episodes, got {len(episode_mems)}"

        # Verify segment memories have timestamp prefix
        for sm in segment_mems:
            assert sm.content.startswith("[Video ")
            assert "min" in sm.content

        # Verify episode memories have episode prefix
        for em in episode_mems:
            assert em.content.startswith("[Episode ")

check("Full consolidation pipeline", test_consolidator_full_pipeline)

def test_consolidator_dedup_doesnt_mutate():
    """Verify _filter_duplicates doesn't modify original captions."""
    from videngram.config import VidEngramConfig
    from videngram.consolidator import Consolidator
    from videngram.utils import Caption

    cfg = VidEngramConfig()
    cfg.consolidator.merge_similarity_threshold = 0.8

    with patch("videngram.consolidator.OpenAI"):
        con = Consolidator(cfg)

    # Create two very similar captions
    cap1 = Caption("s1", "the cat sat on the mat in the room", 0.0, 10.0)
    cap2 = Caption("s2", "the cat sat on the mat in the room today", 10.0, 20.0)
    cap3 = Caption("s3", "a dog ran in the park chasing a ball", 20.0, 30.0)

    original_end = cap1.end_sec  # Should stay 10.0

    filtered = con._filter_duplicates([cap1, cap2, cap3])

    # Original cap1 should NOT have been mutated
    assert cap1.end_sec == original_end, f"Original mutated: cap1.end_sec = {cap1.end_sec}"

check("Dedup doesn't mutate originals", test_consolidator_dedup_doesnt_mutate)

def test_consolidator_empty_input():
    from videngram.config import VidEngramConfig
    from videngram.consolidator import Consolidator
    with patch("videngram.consolidator.OpenAI"):
        con = Consolidator(VidEngramConfig())
    assert con.consolidate([]) == []

check("Consolidator handles empty input", test_consolidator_empty_input)


# ── 6. MEMORY WRITER ─────────────────────────────────────────────

print("\n6. Memory writer execution path:")

def test_writer_write_memories_full():
    """Trace: write_memories() → _write_single() × N → _video_group_id()."""
    from videngram.config import VidEngramConfig
    from videngram.memory_writer import MemoryWriter
    from videngram.utils import ConsolidatedMemory

    cfg = VidEngramConfig()
    writer = MemoryWriter(cfg)

    mock_session = MagicMock()
    mock_session.post.return_value = MagicMock(status_code=200)
    # _write_single now uses _get_session() (thread-local), not _session directly
    writer._get_session = lambda: mock_session

    memories = [
        ConsolidatedMemory(f"m{i}", f"[Video {i//2}:{(i%2)*30:02d}] Content {i}", i*30.0, (i+1)*30.0, "segment")
        for i in range(3)
    ]

    # Override sleep to speed up test
    with patch("videngram.memory_writer.time.sleep"):
        stats = writer.write_memories(memories, "/path/to/my_video.mp4",
                                       wait_for_indexing=0)

    assert stats["total"] == 3
    assert stats["success"] == 3
    assert stats["failed"] == 0
    assert mock_session.post.call_count == 3

    # Verify first call payload shape
    first_call = mock_session.post.call_args_list[0]
    payload = first_call.kwargs.get("json") or first_call[1].get("json")
    assert "message_id" in payload
    assert "create_time" in payload
    assert "sender" in payload
    assert "role" in payload
    assert payload["role"] == "user"
    assert "content" in payload
    assert "group_id" in payload
    assert payload["group_id"].startswith("vid_my_video_")
    assert payload["scene"] == "assistant"
    # Verify URL
    url = first_call.args[0] if first_call.args else first_call.kwargs.get("url", "")
    assert "/api/v1/memories" in url

check("write_memories full path + payload shape", test_writer_write_memories_full)

def test_writer_partial_failure():
    """Some writes succeed, some fail."""
    from videngram.config import VidEngramConfig
    from videngram.memory_writer import MemoryWriter
    from videngram.utils import ConsolidatedMemory

    cfg = VidEngramConfig()
    writer = MemoryWriter(cfg)

    mock_session = MagicMock()
    # First succeeds, second fails, third succeeds
    responses = [
        MagicMock(status_code=200),
        MagicMock(status_code=500, text="Internal Server Error"),
        MagicMock(status_code=201),
    ]
    mock_session.post.side_effect = responses
    # _write_single now uses _get_session() (thread-local), not _session directly
    writer._get_session = lambda: mock_session

    memories = [
        ConsolidatedMemory(f"m{i}", f"Content {i}", i*30.0, (i+1)*30.0, "segment")
        for i in range(3)
    ]

    with patch("videngram.memory_writer.time.sleep"):
        stats = writer.write_memories(memories, "/v.mp4", wait_for_indexing=0)

    assert stats["success"] == 2
    assert stats["failed"] == 1

check("Writer handles partial failures", test_writer_partial_failure)


# ── 7. MEMORY READER ─────────────────────────────────────────────

print("\n7. Memory reader execution path:")

def test_reader_search_episodes_full():
    """Trace: search_episodes() → _retrieve_lightweight() → session.post() → _parse_results()."""
    from videngram.config import VidEngramConfig
    from videngram.memory_reader import MemoryReader

    cfg = VidEngramConfig()
    reader = MemoryReader(cfg)

    mock_session = MagicMock()
    mock_session.post.return_value = MagicMock(
        status_code=200,
        json=lambda: {"results": [
            {"content": "[Video 1:00 - 2:00] Meeting discussion", "score": 0.92, "memory_type": "episodic_memory"},
            {"content": "[Video 2:00 - 3:00] Product demo", "score": 0.85, "memory_type": "episodic_memory"},
        ]}
    )
    reader._session = mock_session

    results = reader.search_episodes("meeting discussion", "/video.mp4", mode="rrf", top_k=5)

    assert len(results) == 2
    assert results[0].content.startswith("[Video")
    assert results[0].score == 0.92
    assert results[1].score == 0.85

    # Verify POST was used (not GET)
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["memory_types"] == ["episodic_memory"]
    assert payload["retrieve_method"] == "rrf"
    assert payload["query"] == "meeting discussion"

check("search_episodes full path + verify POST", test_reader_search_episodes_full)

def test_reader_agentic_fallback():
    """Agentic search falls back to lightweight on failure."""
    from videngram.config import VidEngramConfig
    from videngram.memory_reader import MemoryReader

    cfg = VidEngramConfig()
    reader = MemoryReader(cfg)

    call_count = [0]
    def mock_post(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call (agentic) fails
            return MagicMock(status_code=500)
        else:
            # Second call (fallback rrf) succeeds
            return MagicMock(status_code=200, json=lambda: {"results": [
                {"content": "Fallback result", "score": 0.7}
            ]})

    mock_session = MagicMock()
    mock_session.post.side_effect = mock_post
    reader._session = mock_session

    results = reader.search_agentic("complex question", "/v.mp4")
    assert len(results) == 1
    assert results[0].content == "Fallback result"
    assert call_count[0] == 2  # agentic + fallback

check("Agentic search fallback to lightweight", test_reader_agentic_fallback)


# ── 8. AGENT ─────────────────────────────────────────────────────

print("\n8. Agent execution path:")

def test_agent_react_loop():
    """Trace full ReAct loop: query() → LLM → ACTION parsing → tool exec → OBSERVATION → ANSWER."""
    from videngram.config import VidEngramConfig
    from videngram.agent import VidEngramAgent

    cfg = VidEngramConfig()

    with patch("videngram.agent.OpenAI") as mock_openai_cls, \
         patch("videngram.captioner.OpenAI"), \
         patch("videngram.captioner.AsyncOpenAI"):

        mock_llm = MagicMock()
        mock_openai_cls.return_value = mock_llm

        # Simulate ReAct loop: THINK → ACTION → OBSERVATION → ANSWER
        call_number = [0]
        def llm_response(*args, **kwargs):
            call_number[0] += 1
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock()]
            if call_number[0] == 1:
                # First: think + search
                mock_resp.choices[0].message.content = (
                    'THINK: I need to find what happened in the meeting.\n'
                    'ACTION: search_episodes("meeting discussion")'
                )
            elif call_number[0] == 2:
                # Second: got results, now answer
                mock_resp.choices[0].message.content = (
                    'THINK: I found relevant memories about the meeting.\n'
                    'ANSWER: The meeting discussed quarterly results at 2:30 into the video.'
                )
            return mock_resp

        mock_llm.chat.completions.create.side_effect = llm_response

        agent = VidEngramAgent(cfg)

        # Mock the reader that agent calls internally
        with patch.object(agent.reader, "search_episodes") as mock_search:
            mock_search.return_value = [
                MagicMock(content="[Video 2:30 - 3:00] Quarterly meeting", score=0.9)
            ]

            response = agent.query("What happened in the meeting?", "/video.mp4")

        assert "quarterly" in response.answer.lower() or "meeting" in response.answer.lower()
        assert len(response.actions) == 1
        assert response.actions[0].tool == "search_episodes"
        # LLM should have been called exactly 2 times (think+action, then answer)
        assert mock_llm.chat.completions.create.call_count == 2

check("Agent ReAct loop (THINK → ACTION → ANSWER)", test_agent_react_loop)

def test_agent_look_at_video_tool():
    """Trace look_at_video tool: extract clip → analyze with Qwen."""
    from videngram.config import VidEngramConfig
    from videngram.agent import VidEngramAgent

    cfg = VidEngramConfig()

    with patch("videngram.agent.OpenAI"), \
         patch("videngram.captioner.OpenAI"), \
         patch("videngram.captioner.AsyncOpenAI"):

        agent = VidEngramAgent(cfg)

        with patch("videngram.agent.extract_clip", return_value="/tmp/clip.mp4"), \
             patch.object(agent.captioner, "analyze_clip", return_value="A red car is parked on the street."):

            result_text, clip_path = agent._tool_look_at_video(
                {"start_min": "1.5", "end_min": "2.0", "question": "What color is the car?"},
                "/video.mp4"
            )

        assert "red car" in result_text
        assert clip_path is not None

check("look_at_video tool execution", test_agent_look_at_video_tool)

def test_agent_fallback_on_llm_error():
    """Agent falls back to simple retrieve-answer when LLM errors."""
    from videngram.config import VidEngramConfig
    from videngram.agent import VidEngramAgent

    cfg = VidEngramConfig()

    with patch("videngram.agent.OpenAI") as mock_cls, \
         patch("videngram.captioner.OpenAI"), \
         patch("videngram.captioner.AsyncOpenAI"):

        mock_llm = MagicMock()
        mock_cls.return_value = mock_llm

        # First call to planning LLM fails
        mock_llm.chat.completions.create.side_effect = [
            Exception("Connection timeout"),  # Planning fails
            MagicMock(choices=[MagicMock(message=MagicMock(content="Fallback answer"))]),  # Fallback succeeds
        ]

        agent = VidEngramAgent(cfg)
        with patch.object(agent.reader, "search_episodes", return_value=[
            MagicMock(content="Some memory", score=0.8)
        ]):
            response = agent.query("test question", "/v.mp4")

        # Should get a fallback answer (not crash)
        assert response.answer is not None
        assert len(response.answer) > 0

check("Agent fallback on LLM error", test_agent_fallback_on_llm_error)


# ── 9. FULL PIPELINE ─────────────────────────────────────────────

print("\n9. Full pipeline integration:")

def test_pipeline_ingest_full():
    """Trace complete: ingest() → segment → caption → consolidate → write."""
    from videngram.config import VidEngramConfig
    from videngram.pipeline import VidEngramPipeline

    cfg = VidEngramConfig()

    with patch("videngram.captioner.OpenAI") as mock_cap_cls, \
         patch("videngram.captioner.AsyncOpenAI"), \
         patch("videngram.consolidator.OpenAI") as mock_con_cls, \
         patch("videngram.agent.OpenAI"):

        # Mock caption LLM
        mock_cap = MagicMock()
        mock_cap_cls.return_value = mock_cap
        cap_resp = MagicMock()
        cap_resp.choices = [MagicMock()]
        cap_resp.choices[0].message.content = "A person presents slides about Q3 results."
        mock_cap.chat.completions.create.return_value = cap_resp

        # Mock consolidator LLM
        mock_con = MagicMock()
        mock_con_cls.return_value = mock_con
        con_resp = MagicMock()
        con_resp.choices = [MagicMock()]
        con_resp.choices[0].message.content = "ENTITY: Speaker — presents quarterly results"
        mock_con.chat.completions.create.return_value = con_resp

        pipe = VidEngramPipeline(cfg, validate=False)

        # Mock segmenter
        with patch.object(pipe.segmenter, "segment") as mock_seg, \
             patch.object(pipe.writer, "check_health", return_value=True), \
             patch.object(pipe.writer, "write_memories") as mock_write, \
             patch.object(pipe.writer, "_write_single", return_value=True) as mock_write_single, \
             patch("videngram.captioner.Path.exists", return_value=True):

            from videngram.utils import VideoSegment
            mock_seg.return_value = [
                VideoSegment("s0", "/v.mp4", 0.0, 30.0, "/tmp/s0.mp4"),
                VideoSegment("s1", "/v.mp4", 30.0, 60.0, "/tmp/s1.mp4"),
            ]
            mock_write.return_value = {"total": 3, "success": 3, "failed": 0}

            stats = pipe.ingest("/v.mp4")

        assert stats["segments"] == 2
        assert stats["captions"] == 2
        assert stats["memories_total"] > 0
        assert "total_time" in stats
        # Segment memories are streamed via _write_single during captioning
        assert mock_write_single.call_count == 2, (
            f"Expected 2 streaming segment writes, got {mock_write_single.call_count}"
        )
        assert stats["memories_segments"] == 2
        # write_memories called once for higher-order memories (episodes + entity)
        mock_write.assert_called_once()
        memories_arg = mock_write.call_args[0][0]
        # Higher-order memories should NOT include segment type (already streamed)
        types = {m.memory_type for m in memories_arg}
        assert "segment" not in types, f"Segment memories should be streamed, not in write_memories: {types}"
        assert "episode_summary" in types or len(memories_arg) == 0

check("Full ingest pipeline (segment → caption → consolidate → write)", test_pipeline_ingest_full)

def test_pipeline_query_full():
    """Trace: query() → agent.query() → ReAct loop."""
    from videngram.config import VidEngramConfig
    from videngram.pipeline import VidEngramPipeline
    from videngram.utils import AgentResponse

    cfg = VidEngramConfig()

    with patch("videngram.captioner.OpenAI"), \
         patch("videngram.captioner.AsyncOpenAI"), \
         patch("videngram.consolidator.OpenAI"), \
         patch("videngram.agent.OpenAI"):

        pipe = VidEngramPipeline(cfg, validate=False)

        mock_resp = AgentResponse(
            answer="The meeting discussed Q3 results at 2:30.",
            sources=[],
            actions=[],
        )
        with patch.object(pipe.agent, "query", return_value=mock_resp):
            result = pipe.query("What was discussed?", "/v.mp4")

        assert result.answer == "The meeting discussed Q3 results at 2:30."
        # Chat history should be updated
        assert len(pipe._chat_history) == 2  # user + assistant

check("Full query pipeline", test_pipeline_query_full)

def test_pipeline_health_check_failure():
    """Pipeline raises clear error when EverMemOS is down."""
    from videngram.config import VidEngramConfig
    from videngram.pipeline import VidEngramPipeline

    cfg = VidEngramConfig()
    with patch("videngram.captioner.OpenAI"), \
         patch("videngram.captioner.AsyncOpenAI"), \
         patch("videngram.consolidator.OpenAI"), \
         patch("videngram.agent.OpenAI"):

        pipe = VidEngramPipeline(cfg, validate=False)

        with patch.object(pipe.writer, "check_health", return_value=False):
            try:
                pipe.ingest("/v.mp4")
                assert False, "Should have raised ConnectionError"
            except ConnectionError as e:
                assert "EverMemOS" in str(e)

check("Pipeline raises clear error when EverMemOS is down", test_pipeline_health_check_failure)


# ── 10. CLI ──────────────────────────────────────────────────────

print("\n10. CLI entry points:")

def test_cli_main_parseable():
    """Verify CLI argparse structure doesn't crash on --help."""
    from demo.cli import main
    import io
    from contextlib import redirect_stdout, redirect_stderr

    with patch("sys.argv", ["cli", "--help"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # --help exits with 0

check("CLI --help doesn't crash", test_cli_main_parseable)

def test_cli_no_command():
    """CLI with no command prints help and exits 1."""
    from demo.cli import main
    with patch("sys.argv", ["cli"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 1

check("CLI no-command exits 1", test_cli_no_command)


# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  Results: {PASS} passed, {FAIL} failed")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
