"""
VidEngram Segmenter
HippoMM-inspired temporal pattern separation for video streams.

Implements two strategies:
  - fixed: uniform time windows (simple, predictable)
  - adaptive: scene-change (SSIM) + audio silence detection (HippoMM §3.1)

The adaptive strategy mimics hippocampal pattern separation by detecting
perceptual boundaries where the visual or auditory scene shifts.
"""
import logging
import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Optional

from .config import VidEngramConfig
from .utils import VideoSegment, generate_id, get_video_duration, extract_clip

logger = logging.getLogger("videngram.segmenter")


class VideoSegmenter:
    """Segments video into temporally coherent units."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.segmenter
        self.work_dir = config.work_dir / "segments"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def segment(self, video_path: str) -> list[VideoSegment]:
        """Segment video into a list of VideoSegments.

        Returns segments sorted by start time. Each segment has its
        clip extracted to work_dir for later captioning.
        """
        duration = get_video_duration(video_path)
        logger.info(f"Video duration: {duration:.1f}s, strategy: {self.cfg.strategy}")

        if self.cfg.strategy == "fixed":
            boundaries = self._fixed_boundaries(duration)
        elif self.cfg.strategy == "adaptive":
            boundaries = self._adaptive_boundaries(video_path, duration)
        else:
            raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

        # Create segments and extract clips
        segments = []
        for i, (start, end) in enumerate(boundaries):
            seg_id = generate_id("seg", video_path, start, end)
            clip_path = str(self.work_dir / f"{seg_id}.mp4")

            try:
                extract_clip(video_path, start, end, clip_path)
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to extract clip {seg_id}: {e}")
                clip_path = None

            segments.append(VideoSegment(
                segment_id=seg_id,
                video_path=video_path,
                start_sec=start,
                end_sec=end,
                clip_path=clip_path,
            ))

        logger.info(f"Created {len(segments)} segments")
        return segments

    # ── Fixed Window Strategy ─────────────────────────────────────────

    def _fixed_boundaries(self, duration: float) -> list[tuple[float, float]]:
        """Split into uniform windows."""
        window = self.cfg.fixed_window_sec
        boundaries = []
        start = 0.0
        while start < duration:
            end = min(start + window, duration)
            if end - start >= self.cfg.min_segment_sec:
                boundaries.append((start, end))
            start = end
        return boundaries

    # ── Adaptive Strategy (HippoMM-inspired) ──────────────────────────

    def _adaptive_boundaries(
        self, video_path: str, duration: float
    ) -> list[tuple[float, float]]:
        """Detect scene boundaries using visual change + audio silence.

        Logic (mirrors HippoMM's temporal pattern separation):
        1. Sample frames at analysis_fps, compute SSIM between consecutive frames
        2. Detect audio silence periods
        3. Place boundaries at timestamps where EITHER:
           - SSIM drops below threshold (visual scene change), OR
           - Audio silence is detected (speaker pause / scene break)
        4. Enforce min/max segment duration constraints
        """
        # Step 1: Detect visual scene changes via ffmpeg scene filter
        visual_boundaries = self._detect_scene_changes(video_path, duration)

        # Step 2: Detect audio silence boundaries
        silence_boundaries = self._detect_silence(video_path)

        # Step 3: Merge both boundary sets and sort
        all_boundaries = sorted(set(visual_boundaries + silence_boundaries))
        logger.info(
            f"Raw boundaries: {len(visual_boundaries)} visual, "
            f"{len(silence_boundaries)} audio → {len(all_boundaries)} merged"
        )

        # Step 4: Convert boundary timestamps to (start, end) pairs
        #         with min/max duration enforcement
        return self._boundaries_to_segments(all_boundaries, duration)

    def _detect_scene_changes(
        self, video_path: str, duration: float
    ) -> list[float]:
        """Use ffmpeg's scene detection filter.

        The 'select' filter with scene detection outputs frame timestamps
        where the visual scene changes significantly. This replaces HippoMM's
        frame-by-frame SSIM computation with a more efficient native approach.
        """
        # scene filter uses change score (0=identical, 1=completely different)
        threshold = 1.0 - self.cfg.ssim_threshold
        # Escape special characters in path for lavfi filter syntax
        escaped_path = video_path.replace("'", r"\'").replace(" ", r"\ ")
        cmd = [
            "ffprobe",
            "-v", "error",
            "-f", "lavfi",
            "-i", f"movie={escaped_path},select='gt(scene,{threshold})'",
            "-show_entries", "frame=pts_time",
            "-of", "json",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                # Fallback: use fixed intervals if scene detection fails
                logger.warning("Scene detection failed, using fixed fallback")
                return [
                    i * self.cfg.fixed_window_sec
                    for i in range(1, int(duration / self.cfg.fixed_window_sec))
                ]
            frames = json.loads(result.stdout).get("frames", [])
            return [float(f["pts_time"]) for f in frames if "pts_time" in f]
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning(f"Scene detection error: {e}, using fixed fallback")
            return [
                i * self.cfg.fixed_window_sec
                for i in range(1, int(duration / self.cfg.fixed_window_sec))
            ]

    def _detect_silence(self, video_path: str) -> list[float]:
        """Detect audio silence periods using ffmpeg silencedetect filter.

        Returns midpoint timestamps of detected silence periods.
        These correspond to natural breaks in speech/music — analogous to
        HippoMM's audio energy threshold detection.
        """
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-af", f"silencedetect=noise={self.cfg.silence_threshold_db}dB:d=0.5",
            "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            # Parse silence_start and silence_end from stderr
            boundaries = []
            lines = result.stderr.split("\n")
            silence_start = None
            for line in lines:
                if "silence_start:" in line:
                    try:
                        silence_start = float(
                            line.split("silence_start:")[1].strip().split()[0]
                        )
                    except (IndexError, ValueError):
                        pass
                elif "silence_end:" in line and silence_start is not None:
                    try:
                        silence_end = float(
                            line.split("silence_end:")[1].strip().split()[0]
                        )
                        # Use midpoint of silence as boundary
                        midpoint = (silence_start + silence_end) / 2
                        boundaries.append(midpoint)
                        silence_start = None
                    except (IndexError, ValueError):
                        pass
            return boundaries
        except subprocess.TimeoutExpired:
            logger.warning("Silence detection timed out")
            return []

    def _boundaries_to_segments(
        self, boundaries: list[float], duration: float
    ) -> list[tuple[float, float]]:
        """Convert sorted boundary timestamps to (start, end) pairs.

        Enforces min_segment_sec and max_segment_sec constraints:
        - If a segment would be shorter than min, skip the boundary
        - If a segment would be longer than max, split it at max intervals
        """
        min_dur = self.cfg.min_segment_sec
        max_dur = self.cfg.max_segment_sec

        # Add video start and end as implicit boundaries
        points = [0.0] + [b for b in boundaries if 0 < b < duration] + [duration]

        # First pass: merge boundaries that are too close together
        filtered = [points[0]]
        for p in points[1:]:
            if p - filtered[-1] >= min_dur:
                filtered.append(p)
        if filtered[-1] < duration:
            filtered.append(duration)

        # Second pass: split segments that are too long
        segments = []
        for i in range(len(filtered) - 1):
            start, end = filtered[i], filtered[i + 1]
            seg_duration = end - start

            if seg_duration <= max_dur:
                if seg_duration >= min_dur:
                    segments.append((start, end))
            else:
                # Split into sub-segments of max_dur
                sub_start = start
                while sub_start < end:
                    sub_end = min(sub_start + max_dur, end)
                    if sub_end - sub_start >= min_dur:
                        segments.append((sub_start, sub_end))
                    sub_start = sub_end

        return segments
