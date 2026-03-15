"""
VidEngram Pipeline
End-to-end orchestration: Video → Memory → Query

Usage:
    from videngram.pipeline import VidEngramPipeline

    pipeline = VidEngramPipeline()
    
    # Ingest a video into memory
    stats = pipeline.ingest("path/to/video.mp4")
    
    # Ask questions
    response = pipeline.query("What was discussed in the first 5 minutes?", "path/to/video.mp4")
"""
import json
import logging
import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import VidEngramConfig
from .segmenter import VideoSegmenter
from .captioner import Captioner
from .consolidator import Consolidator
from .transcriber import SpeechTranscriber
from .memory_writer import MemoryWriter
from .memory_reader import MemoryReader
from .agent import VidEngramAgent
from .utils import AgentResponse, fmt_minutes
from .visualizer import generate_tsne_plot

logger = logging.getLogger("videngram.pipeline")

# Maximum number of ingested videos to cache stats for (prevents unbounded growth)
MAX_INGESTED_CACHE = 50


class VidEngramPipeline:
    """End-to-end pipeline for video memory and querying."""

    def __init__(self, config: VidEngramConfig = None, validate: bool = True):
        self.config = config or VidEngramConfig()
        
        # Validate config upfront
        if validate:
            issues = self.config.validate()
            for issue in issues:
                if issue.startswith("[CRITICAL]"):
                    logger.error(issue)
                elif issue.startswith("[ERROR]"):
                    logger.error(issue)
                else:
                    logger.warning(issue)
        
        self.segmenter = VideoSegmenter(self.config)
        self.captioner = Captioner(self.config)
        self.consolidator = Consolidator(self.config)
        self.transcriber = SpeechTranscriber(self.config)
        self.writer = MemoryWriter(self.config)
        self.reader = MemoryReader(self.config)
        self.agent = VidEngramAgent(self.config)
        # Bounded LRU cache for ingestion stats
        self._ingested_videos: OrderedDict[str, dict] = OrderedDict()
        self._chat_history: list[dict] = []

    def ingest(
        self,
        video_path: str,
        parallel_caption: bool = False,
        skip_if_exists: bool = True,
        on_caption_ready=None,
    ) -> dict:
        """Ingest a video: ASR → segment → caption → consolidate → write to EverMemOS.

        This is the full hippocampal encoding pipeline:
          1. Automatic Speech Recognition (transcriber) — timestamps drive segmentation
          2. Temporal Pattern Separation (segmenter, ASR-guided)
          3. Perceptual Encoding (captioner via Qwen2.5-Omni)
          4. Memory Consolidation (dedup + episodes + profiles)
          5. Memory Storage (EverMemOS)

        Args:
            video_path: Path to video file
            parallel_caption: Use async captioning (faster but more GPU load)
            skip_if_exists: Skip if video already ingested
            on_caption_ready: Optional callable(event_type, seg_or_segs, data).
                Called with ("segments", segments_list, None) after Stage 2, and
                ("caption", segment, (index, caption)) after each caption in Stage 3.
                Used by the demo CLI to stream structured events to the UI.

        Returns:
            Stats dict with segment/memory counts and timing
        """
        video_path = str(Path(video_path).resolve())

        if skip_if_exists and video_path in self._ingested_videos:
            logger.info(f"Video already ingested: {video_path}")
            return self._ingested_videos[video_path]

        # Check EverMemOS health
        if not self.writer.check_health():
            raise ConnectionError(
                "EverMemOS is not reachable. Start it with: "
                "cd EverMemOS && docker compose up -d && uv run python src/run.py"
            )

        t0 = time.time()
        stats = {"video_path": video_path}

        # Precompute video identity — shared by all streaming write paths below
        group_id = MemoryWriter._video_group_id(video_path)
        group_name = Path(video_path).stem

        # Speech streaming state (populated right after ASR when transcriber is on)
        speech_write_executor: ThreadPoolExecutor | None = None
        speech_write_futures: list = []
        speech_memories_list: list = []

        # Step 1/5: Automatic Speech Recognition (ASR)
        # Runs first so Whisper timestamps can guide segmentation in Step 2.
        # Speech cues are saved to disk immediately so the frontend can display
        # subtitles as soon as SEGMENTS_JSON arrives (before video sync starts).
        logger.info("=" * 60)
        t_asr = time.time()
        speech_segments: list[dict] = []
        if self.config.transcriber.enabled:
            logger.info("Step 1/5: Automatic Speech Recognition (ASR)")
            try:
                speech_segments = self.transcriber.transcribe(video_path)
                stats["speech_segments"] = len(speech_segments)
                logger.info(f"  Got {len(speech_segments)} speech segment(s)")
            except Exception as e:
                logger.warning(f"ASR failed (non-fatal): {e}")
                stats["speech_segments"] = 0
            if speech_segments:
                self._save_speech_cues_file(speech_segments, video_path)
                speech_memories_list = self.transcriber.build_speech_memories(
                    speech_segments, video_path
                )
                if speech_memories_list:
                    speech_write_executor = ThreadPoolExecutor(max_workers=2)
                    for sp_idx, sp_mem in enumerate(speech_memories_list):
                        speech_write_futures.append(
                            speech_write_executor.submit(
                                self.writer._write_single,
                                sp_mem, group_id, group_name, sp_idx,
                            )
                        )
                    logger.info(
                        f"  Streaming {len(speech_memories_list)} speech memories "
                        "to EverMemOS in background"
                    )
        else:
            logger.info("Step 1/5: Skipping ASR (TRANSCRIBER_API_KEY not set)")
            stats["speech_segments"] = 0
        stats["speech_time"] = time.time() - t_asr

        # Step 2/5: Segment video (ASR-guided when available, silence-detection fallback)
        logger.info("=" * 60)
        logger.info("Step 2/5: Segmenting video")
        t_seg = time.time()
        segments = self.segmenter.segment(
            video_path,
            asr_segments=speech_segments or None,
        )

        # Populate asr_text for each segment: concatenate all Whisper sentences
        # whose time window overlaps with the segment. Since segments are
        # ASR-guided, alignment is tight; overlap matching handles edge cases
        # where sentence boundaries don't perfectly match segment boundaries.
        if speech_segments:
            for seg in segments:
                overlapping = [
                    s["text"].strip()
                    for s in speech_segments
                    if s["end_sec"] > seg.start_sec and s["start_sec"] < seg.end_sec
                    and s["text"].strip()
                ]
                if overlapping:
                    seg.asr_text = " ".join(overlapping)

        stats["segments"] = len(segments)
        stats["segment_time"] = time.time() - t_seg

        # Notify caller of segment list — triggers SEGMENTS_JSON → frontend video sync.
        # ASR is complete at this point so speech cues are already on disk.
        if on_caption_ready is not None:
            on_caption_ready("segments", segments, None)

        # Step 3/5: Caption segments with Qwen2.5-Omni
        # Segment memories are streamed to EverMemOS in background as each
        # caption arrives, overlapping with subsequent Qwen caption calls.
        t1 = time.time()
        logger.info(f"Step 3/5: Captioning {len(segments)} segments (streaming ingest)")

        seg_write_executor = ThreadPoolExecutor(max_workers=3)
        seg_write_futures: list = []
        seg_memories_list: list = []  # kept for t-SNE and stats

        def _cap_cb(index, seg, cap):
            # Immediately stream this segment's memory to EverMemOS (non-blocking)
            mem = self.consolidator.make_segment_memory(cap)
            if mem is not None:
                seg_memories_list.append(mem)
                seg_write_futures.append(
                    seg_write_executor.submit(
                        self.writer._write_single, mem, group_id, group_name, index
                    )
                )
            if on_caption_ready is not None:
                on_caption_ready("caption", seg, (index, cap))

        captions = self.captioner.caption_all(
            segments,
            parallel=parallel_caption,
            on_caption=_cap_cb,
        )
        stats["captions"] = len(captions)
        stats["caption_time"] = time.time() - t1

        # Wait for all streaming segment writes to finish
        seg_write_executor.shutdown(wait=True)
        streaming_happened = len(seg_write_futures) > 0
        seg_success = sum(1 for f in seg_write_futures if f.result())
        seg_failed = len(seg_write_futures) - seg_success
        if streaming_happened:
            logger.info(
                f"Streaming segment writes complete: "
                f"{seg_success}/{len(seg_write_futures)} succeeded"
            )

        # Step 4/5: Consolidate (dedup + episodes + entity register)
        # Segment memories already in EverMemOS — skip regenerating them.
        # Falls back to include_segments=True when parallel_caption=True
        # (async captioner doesn't fire on_caption, so streaming didn't happen).
        t3 = time.time()
        logger.info("Step 4/5: Consolidating memories (episodes + entity register)")
        higher_order = self.consolidator.consolidate(
            captions, include_segments=not streaming_happened
        )

        stats["memories_segments"] = (
            len(seg_write_futures) if streaming_happened
            else sum(1 for m in higher_order if m.memory_type == "segment")
        )
        stats["memories_episodes"] = sum(
            1 for m in higher_order if m.memory_type == "episode_summary"
        )
        stats["memories_profiles"] = sum(
            1 for m in higher_order if m.memory_type == "entity_register"
        )
        stats["memories_speech"] = len(speech_write_futures)
        stats["memories_total"] = (
            stats["memories_segments"] + len(higher_order) + len(speech_write_futures)
        )
        stats["consolidate_time"] = time.time() - t3

        # Ensure speech streaming writes are done before the final write step.
        # By the time consolidation finishes (~30-60s of LLM calls), speech writes
        # (submitted right after ASR) have had ample time to complete.
        if speech_write_executor is not None:
            speech_write_executor.shutdown(wait=True)
        speech_success = sum(1 for f in speech_write_futures if f.result())
        speech_failed = len(speech_write_futures) - speech_success
        if speech_write_futures:
            logger.info(
                f"Streaming speech writes complete: "
                f"{speech_success}/{len(speech_write_futures)} succeeded"
            )

        # Step 5/5: Write episode summaries + entity register to EverMemOS.
        # Segment and speech memories are already indexed (streamed earlier),
        # so we only wait for indexing of the higher-order memories written here.
        t4 = time.time()
        logger.info(f"Step 5/5: Writing {len(higher_order)} higher-order memories to EverMemOS")
        write_stats = self.writer.write_memories(higher_order, video_path)
        # Combine all three streaming streams with higher-order write stats
        stats["total"] = (
            len(seg_write_futures) + len(speech_write_futures) + write_stats.get("total", 0)
        )
        stats["success"] = seg_success + speech_success + write_stats.get("success", 0)
        stats["failed"] = seg_failed + speech_failed + write_stats.get("failed", 0)
        stats["write_time"] = time.time() - t4
        stats["total_time"] = time.time() - t0

        # Visualization: embed all memories and plot t-SNE
        generate_tsne_plot(seg_memories_list + speech_memories_list + higher_order)

        # Save local cues files for fast server-side cache loading.
        # Pass pre-consolidation captions so every spoken segment is preserved
        # (the consolidator may merge adjacent segments and drop speech from the
        # second one; using raw captions avoids that loss).
        # Note: speech cues were already saved after ASR above; only save the
        # main cues file here.
        self._save_cues_file(captions, higher_order, video_path)

        self._ingested_videos[video_path] = stats
        # Evict oldest if cache exceeds limit
        while len(self._ingested_videos) > MAX_INGESTED_CACHE:
            self._ingested_videos.popitem(last=False)

        logger.info(f"Ingestion complete in {stats['total_time']:.1f}s")
        self._print_stats(stats)
        return stats

    def _save_cues_file(self, captions: list, memories: list, video_path: str) -> None:
        """Save cues to a local JSON file for instant server-side cache loading.

        Uses PRE-CONSOLIDATION captions for subtitle entries so that every spoken
        segment is represented individually — the consolidator may merge adjacent
        segments with similar visual content and discard the second segment's audio
        transcript, creating gaps in subtitle coverage.  By using raw captions here
        each original segment keeps its own DIALOGUE field intact.

        Episode summaries from consolidation are appended as additional entries;
        the server sorts by span-length ascending so the short per-segment entries
        always win over the wider episode entries when the frontend looks up a cue
        at a specific timestamp.

        File path: {work_dir}/{group_id}_cues.json
        """
        group_id = MemoryWriter._video_group_id(video_path)
        cues_file = self.config.work_dir / f"{group_id}_cues.json"
        cues = []

        # One entry per original caption (pre-dedup) — ensures complete subtitle coverage
        error_prefixes = ("[Caption error", "[Clip unavailable]", "[Analysis error")
        for cap in captions:
            if cap.raw_text.startswith(error_prefixes):
                continue
            cues.append({
                "content": (
                    f"[Video {fmt_minutes(cap.start_sec)} - {fmt_minutes(cap.end_sec)}] "
                    f"{cap.raw_text}"
                ),
                "start_sec": cap.start_sec,
                "end_sec": cap.end_sec,
            })

        # Append episode summaries for richer scene descriptions in the scene panel
        for m in memories:
            if m.memory_type == "episode_summary":
                cues.append({
                    "content": m.content,
                    "start_sec": m.start_sec,
                    "end_sec": m.end_sec,
                })

        try:
            cues_file.write_text(json.dumps(cues, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Saved {len(cues)} cues to {cues_file} "
                        f"({len(captions)} captions + "
                        f"{len(cues) - len(captions)} episode summaries)")
        except Exception as e:
            logger.warning(f"Failed to save cues file: {e}")

    def _save_speech_cues_file(
        self, speech_segments: list[dict], video_path: str
    ) -> None:
        """Save Whisper transcription segments to {group_id}_speech.json.

        Format: [{"start_sec": float, "end_sec": float, "text": str}, ...]
        Loaded by server.py /speech_cues and used by the frontend for subtitle
        display (higher priority than memory DIALOGUE fields).
        """
        group_id = MemoryWriter._video_group_id(video_path)
        speech_file = self.config.work_dir / f"{group_id}_speech.json"
        try:
            speech_file.write_text(
                json.dumps(speech_segments, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                f"Saved {len(speech_segments)} speech segments to {speech_file}"
            )
        except Exception as e:
            logger.warning(f"Failed to save speech cues file: {e}")

    def query(
        self,
        question: str,
        video_path: str,
        multi_turn: bool = True,
    ) -> AgentResponse:
        """Ask a question about an ingested video.

        Uses the agentic ReAct loop with tool access:
        - search_episodes: Fast memory lookup
        - search_profiles: Entity/speaker profiles
        - search_deep: Multi-hop agentic retrieval
        - look_at_video: Extract + re-analyze video clip (grounding)
        - get_timeline: Chronological event listing

        Args:
            question: Natural language question
            video_path: Path to the source video
            multi_turn: Maintain conversation history for follow-ups

        Returns:
            AgentResponse with answer, sources, actions taken
        """
        video_path = str(Path(video_path).resolve())
        history = self._chat_history if multi_turn else None

        response = self.agent.query(question, video_path, chat_history=history)

        # Update conversation history
        if multi_turn:
            self._chat_history.append({"role": "user", "content": question})
            self._chat_history.append({
                "role": "assistant", "content": response.answer
            })
            # Keep history bounded
            if len(self._chat_history) > 20:
                self._chat_history = self._chat_history[-10:]

        return response

    def clear_history(self):
        """Clear multi-turn conversation history."""
        self._chat_history.clear()

    def cleanup(self, video_path: str = None):
        """Clean up temporary files (extracted clips, agent clips).
        
        Args:
            video_path: If given, clean only this video's temp files.
                        If None, clean all temp files in work_dir.
        """
        work_dir = self.config.work_dir
        if video_path:
            # Clean specific video segments
            segments_dir = work_dir / "segments"
            agent_clips_dir = work_dir / "agent_clips"
            for d in (segments_dir, agent_clips_dir):
                if d.exists():
                    for f in d.glob("*.mp4"):
                        try:
                            f.unlink()
                        except OSError:
                            pass
            logger.info(f"Cleaned temp files for video")
        else:
            # Clean everything
            for subdir in ("segments", "agent_clips"):
                d = work_dir / subdir
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
                    d.mkdir(exist_ok=True)
            logger.info(f"Cleaned all temp files in {work_dir}")

    @staticmethod
    def _print_stats(stats: dict):
        """Pretty-print ingestion stats."""
        print("\n" + "=" * 50)
        print("  VidEngram Ingestion Summary")
        print("=" * 50)
        print(f"  Video:     {Path(stats['video_path']).name}")
        print(f"  Segments:  {stats.get('segments', 0)}")
        print(f"  Captions:  {stats.get('captions', 0)}")
        print(f"  Memories:")
        print(f"    Segments:  {stats.get('memories_segments', 0)}")
        print(f"    Episodes:  {stats.get('memories_episodes', 0)}")
        print(f"    Profiles:  {stats.get('memories_profiles', 0)}")
        print(f"    Speech:    {stats.get('memories_speech', 0)}")
        print(f"    Total:     {stats.get('memories_total', 0)}")
        print(f"  Written:   {stats.get('success', 0)}/{stats.get('total', 0)}")
        print(f"  Time:")
        print(f"    Transcribe:  {stats.get('speech_time', 0):.1f}s")
        print(f"    Segment:     {stats.get('segment_time', 0):.1f}s")
        print(f"    Caption:     {stats.get('caption_time', 0):.1f}s")
        print(f"    Consolidate: {stats.get('consolidate_time', 0):.1f}s")
        print(f"    Write:       {stats.get('write_time', 0):.1f}s")
        print(f"    Total:       {stats.get('total_time', 0):.1f}s")
        print("=" * 50)
