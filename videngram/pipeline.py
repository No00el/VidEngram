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
import logging
import shutil
import time
from collections import OrderedDict
from pathlib import Path

from .config import VidEngramConfig
from .segmenter import VideoSegmenter
from .captioner import Captioner
from .consolidator import Consolidator
from .memory_writer import MemoryWriter
from .memory_reader import MemoryReader
from .agent import VidEngramAgent
from .utils import AgentResponse

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
    ) -> dict:
        """Ingest a video: segment → caption → consolidate → write to EverMemOS.

        This is the full hippocampal encoding pipeline:
          1. Temporal Pattern Separation (segmenter)
          2. Perceptual Encoding (captioner via Qwen2.5-Omni)
          3. Memory Consolidation (dedup + episodes + profiles)
          4. Memory Storage (EverMemOS)

        Args:
            video_path: Path to video file
            parallel_caption: Use async captioning (faster but more GPU load)
            skip_if_exists: Skip if video already ingested

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

        # Step 1: Segment video
        logger.info("=" * 60)
        logger.info(f"Step 1/4: Segmenting video")
        segments = self.segmenter.segment(video_path)
        stats["segments"] = len(segments)
        stats["segment_time"] = time.time() - t0

        # Step 2: Caption segments with Qwen2.5-Omni
        t1 = time.time()
        logger.info(f"Step 2/4: Captioning {len(segments)} segments")
        captions = self.captioner.caption_all(segments, parallel=parallel_caption)
        stats["captions"] = len(captions)
        stats["caption_time"] = time.time() - t1

        # Step 3: Consolidate (dedup + episodes + profiles)
        t2 = time.time()
        logger.info(f"Step 3/4: Consolidating memories")
        memories = self.consolidator.consolidate(captions)
        stats["memories_total"] = len(memories)
        stats["memories_segments"] = sum(
            1 for m in memories if m.memory_type == "segment"
        )
        stats["memories_episodes"] = sum(
            1 for m in memories if m.memory_type == "episode_summary"
        )
        stats["memories_profiles"] = sum(
            1 for m in memories if m.memory_type == "entity_profile"
        )
        stats["consolidate_time"] = time.time() - t2

        # Step 4: Write to EverMemOS
        t3 = time.time()
        logger.info(f"Step 4/4: Writing {len(memories)} memories to EverMemOS")
        write_stats = self.writer.write_memories(memories, video_path)
        stats.update(write_stats)
        stats["write_time"] = time.time() - t3
        stats["total_time"] = time.time() - t0

        self._ingested_videos[video_path] = stats
        # Evict oldest if cache exceeds limit
        while len(self._ingested_videos) > MAX_INGESTED_CACHE:
            self._ingested_videos.popitem(last=False)
        
        logger.info(f"Ingestion complete in {stats['total_time']:.1f}s")
        self._print_stats(stats)
        return stats

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
        print(f"    Total:     {stats.get('memories_total', 0)}")
        print(f"  Written:   {stats.get('success', 0)}/{stats.get('total', 0)}")
        print(f"  Time:")
        print(f"    Segment:     {stats.get('segment_time', 0):.1f}s")
        print(f"    Caption:     {stats.get('caption_time', 0):.1f}s")
        print(f"    Consolidate: {stats.get('consolidate_time', 0):.1f}s")
        print(f"    Write:       {stats.get('write_time', 0):.1f}s")
        print(f"    Total:       {stats.get('total_time', 0):.1f}s")
        print("=" * 50)
