"""
VidEngram Segmenter
HippoMM-inspired temporal pattern separation for video streams.

When RemoteConfig.enabled=True, all ffmpeg/ffprobe commands are executed on the server via SSH,
and segments are stored on the server. The captioner subsequently calls Qwen vLLM directly with the server path.
"""
import logging
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import VidEngramConfig
from .utils import VideoSegment, generate_id, get_video_duration, extract_clip

logger = logging.getLogger("videngram.segmenter")


class VideoSegmenter:

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.segmenter
        self.remote = config.remote

        if self.remote.enabled:
            # work_dir is a path on the server (string, not Path, to avoid local mkdir)
            self.work_dir = self.remote.remote_work_dir.rstrip("/") + "/segments"
            self._ssh(f"mkdir -p {self.work_dir}")
            logger.info(f"Remote segmenter: {self.remote.host}, work_dir={self.work_dir}")
        else:
            self.work_dir_path = config.work_dir / "segments"
            self.work_dir_path.mkdir(parents=True, exist_ok=True)
            self.work_dir = str(self.work_dir_path)

    # ── SSH helpers ────────────────────────────────────────────────────

    def _ssh(self, remote_cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
        """Execute command on the server, return CompletedProcess."""
        result = subprocess.run(
            ["ssh", self.remote.host, remote_cmd],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.warning(f"SSH command failed: {remote_cmd}\n{result.stderr}")
        return result

    def _ssh_json(self, remote_cmd: str, timeout: int = 300) -> dict:
        """Execute command on the server, parse stdout as JSON."""
        result = self._ssh(remote_cmd, timeout=timeout)
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from remote command: {remote_cmd}")
            return {}

    # ── Public API ─────────────────────────────────────────────────────

    def segment(
        self,
        video_path: str,
        asr_segments: list[dict] | None = None,
    ) -> list[VideoSegment]:
        """Segment video and return a list of VideoSegment objects.

        asr_segments: Whisper transcription results [{start_sec, end_sec, text}, ...].
                      If provided (and non-empty), uses the ASR-first strategy: speech
                      sentence boundaries are the primary cut points, with SSIM scene
                      changes supplemented in non-speech gaps (>= min_segment_sec);
                      if None or empty list, falls back to the silence-detection strategy.

        In remote mode: duration queries, ffmpeg clipping, and scene detection all run on the server.
        The returned VideoSegment.clip_path is a server-side path passed directly to Qwen by the captioner.
        """
        if self.remote.enabled:
            video_path = video_path.replace("/System/Volumes/Data", "")

        duration = self._get_duration(video_path)
        logger.info(f"Video duration: {duration:.1f}s, strategy: {self.cfg.strategy}")

        if self.cfg.strategy == "fixed":
            boundaries = self._fixed_boundaries(duration)
        elif self.cfg.strategy == "adaptive":
            if asr_segments:
                logger.info(
                    f"Using ASR-first segmentation "
                    f"({len(asr_segments)} Whisper segment(s))"
                )
                boundaries = self._asr_boundaries(video_path, duration, asr_segments)
            else:
                logger.info(
                    "ASR segments unavailable; falling back to silence-detection"
                )
                boundaries = self._silence_boundaries(video_path, duration)
        else:
            raise ValueError(f"Unknown strategy: {self.cfg.strategy}")

        # Pre-generate all (seg_id, clip_path, start, end) tuples to preserve order
        pending = []
        for start, end in boundaries:
            seg_id = generate_id("seg", video_path, start, end)
            clip_path = f"{self.work_dir}/{seg_id}.mp4"
            pending.append((seg_id, clip_path, start, end))

        # Extract clips in parallel — each writes to a unique seg_id path, no shared state
        def _do_extract(item):
            seg_id, clip_path, start, end = item
            try:
                self._extract_clip(video_path, start, end, clip_path)
                logger.debug(f"Clip extracted: {clip_path}")
                return seg_id, clip_path
            except Exception as e:
                logger.warning(f"Failed to extract clip {seg_id}: {e}")
                return seg_id, None

        clip_results: dict[str, str | None] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for seg_id, result_path in pool.map(_do_extract, pending):
                clip_results[seg_id] = result_path

        # Assemble segments in original boundary order (pool.map preserves order too,
        # but we use clip_results dict for clarity)
        segments = [
            VideoSegment(
                segment_id=seg_id,
                video_path=video_path,
                start_sec=start,
                end_sec=end,
                clip_path=clip_results[seg_id],
            )
            for seg_id, _, start, end in pending
        ]

        logger.info(f"Created {len(segments)} segments")
        return segments

    # ── Duration ───────────────────────────────────────────────────────

    def _get_duration(self, video_path: str) -> float:
        if self.remote.enabled:
            cmd = (
                f"ffprobe -v error -show_entries format=duration "
                f"-of json '{video_path}'"
            )
            data = self._ssh_json(cmd)
            return float(data.get("format", {}).get("duration", 0))
        else:
            return get_video_duration(video_path)

    # ── Clip extraction ────────────────────────────────────────────────

    def _extract_clip(self, video_path: str, start: float, end: float, clip_path: str):
        if self.remote.enabled:
            # Use input-side -ss (input-side seeking) + re-encoding instead of -c copy.
            # -c copy requires keyframe alignment: old silence-detection cut points happen to
            # fall at silence (where encoders often insert I-frames), but ASR sentence
            # boundaries are unrelated to keyframes — -c copy would cause seconds of missing
            # content between adjacent clips. Re-encoding is slower but frame-accurate.
            duration = end - start
            cmd = (
                f"ffmpeg -y "
                f"-ss {start} "
                f"-i '{video_path}' "
                f"-t {duration} "
                f"-c:v libx264 -preset fast "
                f"-c:a aac "
                f"'{clip_path}' 2>/dev/null"
            )
            result = self._ssh(cmd, timeout=300)
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd)
        else:
            extract_clip(video_path, start, end, clip_path)

    # ── Fixed Window ───────────────────────────────────────────────────

    def _fixed_boundaries(self, duration: float) -> list[tuple[float, float]]:
        window = self.cfg.fixed_window_sec
        boundaries = []
        start = 0.0
        while start < duration:
            end = min(start + window, duration)
            if end - start >= self.cfg.min_segment_sec:
                boundaries.append((start, end))
            start = end
        return boundaries

    # ── ASR-first segmentation ─────────────────────────────────────────

    # Sentence-ending punctuation (English + CJK)
    _SENTENCE_END = frozenset('.!?。！？')

    # Hard maximum duration for a single sentence-grouped segment.
    # Sentences shorter than this are kept whole even when they exceed
    # max_segment_sec.  Only segments longer than this are force-split at
    # the nearest SSIM boundary (or uniformly as a last resort).
    _SENTENCE_HARD_MAX_SEC = 15.0

    # Minimum number of words that must be accumulated in the current group
    # before a sentence-ending punctuation triggers a cut.  Prevents prosodic
    # pause fragments ("Today." / "This chip." / "Exactly.") from being treated
    # as sentence ends when the speaker is simply emphasising mid-sentence.
    _SENTENCE_MIN_WORDS = 5

    def _group_into_sentences(self, asr_segments: list[dict]) -> list[dict]:
        """Merge Whisper sub-sentence segments into complete sentences.

        A sentence ends when a segment ends with sentence-final punctuation (.!?。！？)
        and the accumulated word count >= _SENTENCE_MIN_WORDS; if the word count is
        insufficient, accumulation continues to avoid treating emphasis pauses
        ("Today." / "This chip.") as sentence boundaries. Segments without punctuation
        keep accumulating until the next valid sentence-end boundary or end of sequence.

        Returns: [{start_sec, end_sec, text}, ...]
        """
        sentences: list[dict] = []
        group: list[dict] = []

        for seg in asr_segments:
            group.append(seg)
            text = seg.get("text", "").strip()
            group_text = " ".join(s.get("text", "") for s in group).strip()
            word_count = len(group_text.split())
            if text and text[-1] in self._SENTENCE_END and word_count >= self._SENTENCE_MIN_WORDS:
                sentences.append({
                    "start_sec": group[0]["start_sec"],
                    "end_sec": group[-1]["end_sec"],
                    "text": group_text,
                })
                group = []

        # Trailing remainder segments (no punctuation or insufficient word count) kept as one sentence
        if group:
            sentences.append({
                "start_sec": group[0]["start_sec"],
                "end_sec": group[-1]["end_sec"],
                "text": " ".join(s.get("text", "") for s in group).strip(),
            })

        return sentences

    def _asr_boundaries(
        self, video_path: str, duration: float, asr_segments: list[dict]
    ) -> list[tuple[float, float]]:
        """ASR-first adaptive segmentation (sentence-boundary-priority version).

        First merges Whisper sub-sentence segments into complete sentences, using each
        sentence's end_sec as the primary cut point, with SSIM scene change boundaries
        supplemented in non-speech gaps (>= min_segment_sec).
        min/max constraints:
          - Too short (< min) -> merge with the shorter adjacent segment
          - Exceeds _SENTENCE_HARD_MAX_SEC (15s) -> split with SSIM cut points, uniform cut if none available
        """
        min_dur = self.cfg.min_segment_sec
        sentence_max_dur = self._SENTENCE_HARD_MAX_SEC

        asr = sorted(asr_segments, key=lambda s: s["start_sec"])

        # Merge Whisper segments into sentences
        sentences = self._group_into_sentences(asr)

        logger.info(
            f"ASR-first: {len(sentences)} sentence group(s) as cut points "
            f"({len(asr)} Whisper segment(s))"
        )

        # Always compute SSIM boundaries (used for gap analysis + long-segment splitting)
        visual_boundaries = self._detect_scene_changes(video_path, duration)

        # Identify non-speech gaps (based on sentence boundaries, including head and tail)
        gaps: list[tuple[float, float]] = []
        if sentences[0]["start_sec"] >= min_dur:
            gaps.append((0.0, sentences[0]["start_sec"]))
        for i in range(len(sentences) - 1):
            gap_start = sentences[i]["end_sec"]
            gap_end = sentences[i + 1]["start_sec"]
            if gap_end - gap_start >= min_dur:
                gaps.append((gap_start, gap_end))
        if duration - sentences[-1]["end_sec"] >= min_dur:
            gaps.append((sentences[-1]["end_sec"], duration))

        logger.debug(
            f"  {len(gaps)} non-speech gap(s) ≥{min_dur:.0f}s, "
            f"{len(visual_boundaries)} SSIM boundary candidate(s)"
        )

        # Primary cut points: end_sec of each sentence
        cut_points: set[float] = set()
        for sent in sentences:
            cut_points.add(sent["end_sec"])

        # Supplementary cut points: SSIM scene changes within non-speech gaps
        for gap_start, gap_end in gaps:
            in_gap = [v for v in visual_boundaries if gap_start < v < gap_end]
            if in_gap:
                cut_points.update(in_gap)
                logger.debug(
                    f"  Gap [{gap_start:.1f}s-{gap_end:.1f}s]: "
                    f"+{len(in_gap)} SSIM cut(s)"
                )
            else:
                logger.debug(
                    f"  Gap [{gap_start:.1f}s-{gap_end:.1f}s]: no SSIM cuts found"
                )

        # Convert cut points to a list of (start, end) segments
        points = sorted(
            [0.0] + [p for p in cut_points if 0 < p < duration] + [duration]
        )
        segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
        logger.debug(f"Initial segments from sentence cut points: {len(segments)}")

        # Merge short segments (< min_dur)
        segments = self._merge_short_segments(segments, min_dur)
        logger.debug(f"After short-segment merge: {len(segments)}")

        # Split overly long segments (> sentence_max_dur = 15s): prefer SSIM cut points, uniform cut if none
        segments = self._split_long_segments(segments, sentence_max_dur, visual_boundaries)
        # Splitting may produce new short tail segments; run one more merge pass to ensure no gaps
        segments = self._merge_short_segments(segments, min_dur)
        logger.info(f"ASR-first final segment count: {len(segments)}")

        return segments

    def _merge_short_segments(
        self,
        segments: list[tuple[float, float]],
        min_dur: float,
    ) -> list[tuple[float, float]]:
        """Merge segments shorter than min_dur into their shorter adjacent neighbor.

        Each iteration finds the shortest non-conforming segment and merges it with
        the shorter neighbor, until all segments satisfy the constraint or only one remains.
        """
        segs = list(segments)

        while len(segs) > 1:
            short = [i for i in range(len(segs)) if segs[i][1] - segs[i][0] < min_dur]
            if not short:
                break

            # Process the shortest segment first
            idx = min(short, key=lambda i: segs[i][1] - segs[i][0])
            has_left = idx > 0
            has_right = idx < len(segs) - 1

            if has_left and has_right:
                left_dur = segs[idx - 1][1] - segs[idx - 1][0]
                right_dur = segs[idx + 1][1] - segs[idx + 1][0]
                if left_dur <= right_dur:
                    merged = (segs[idx - 1][0], segs[idx][1])
                    segs[idx - 1: idx + 1] = [merged]
                else:
                    merged = (segs[idx][0], segs[idx + 1][1])
                    segs[idx: idx + 2] = [merged]
            elif has_left:
                merged = (segs[idx - 1][0], segs[idx][1])
                segs[idx - 1: idx + 1] = [merged]
            else:
                merged = (segs[idx][0], segs[idx + 1][1])
                segs[idx: idx + 2] = [merged]

        return segs

    def _split_long_segments(
        self,
        segments: list[tuple[float, float]],
        max_dur: float,
        visual_boundaries: list[float],
    ) -> list[tuple[float, float]]:
        """Split segments exceeding max_dur using SSIM cut points (uniform cut if none available)."""
        result = []
        for start, end in segments:
            if end - start <= max_dur:
                result.append((start, end))
            else:
                result.extend(self._split_long_segment(start, end, visual_boundaries))
        return result

    # ── Silence-detection fallback (original adaptive strategy) ────────

    def _silence_boundaries(
        self, video_path: str, duration: float
    ) -> list[tuple[float, float]]:
        """Silence-detection segmentation using audio pauses as primary and scene changes as auxiliary cues (fallback when ASR is unavailable).

        1. All silence_end timestamps serve as primary boundaries; after min/max filtering, base segments are formed.
        2. If a segment exceeds max_segment_sec (indicating prolonged speech between two pauses),
           recursively split it near the midpoint using visual boundaries; fall back to uniform cutting if none exist.
        """
        visual_boundaries = self._detect_scene_changes(video_path, duration)
        silence_intervals = self._detect_silence(video_path)
        silence_ends = sorted(set(se for _, se in silence_intervals))

        logger.info(
            f"Raw boundaries: {len(silence_ends)} audio (primary silence_end), "
            f"{len(visual_boundaries)} visual (auxiliary)"
        )
        return self._boundaries_to_segments(
            silence_ends, duration, preferred_splits=visual_boundaries
        )

    def _split_long_segment(
        self,
        start: float,
        end: float,
        preferred_splits: list[float],
    ) -> list[tuple[float, float]]:
        """Recursively split an overly long segment into sub-segments not exceeding max_segment_sec.

        Preferentially selects the visual boundary closest to the segment midpoint as the split point,
        then recursively processes sub-segments. Falls back to uniform cutting if no visual boundaries exist.
        """
        max_dur = self.cfg.max_segment_sec

        if end - start <= max_dur:
            return [(start, end)]

        candidates = [v for v in preferred_splits if start < v < end]
        if candidates:
            mid = (start + end) / 2.0
            split_at = min(candidates, key=lambda v: abs(v - mid))
            return (
                self._split_long_segment(start, split_at, preferred_splits)
                + self._split_long_segment(split_at, end, preferred_splits)
            )

        # No visual boundaries available; fall back to uniform cutting
        result = []
        sub_start = start
        while sub_start < end:
            sub_end = min(sub_start + max_dur, end)
            result.append((sub_start, sub_end))
            sub_start = sub_end
        return result

    def _detect_scene_changes(self, video_path: str, duration: float) -> list[float]:
        threshold = 1.0 - self.cfg.ssim_threshold
        escaped = video_path.replace("'", r"\'").replace(" ", r"\ ")

        if self.remote.enabled:
            cmd = (
                f"ffprobe -v error -f lavfi "
                f"-i \"movie={escaped},select='gt(scene,{threshold})'\" "
                f"-show_entries frame=pts_time -of json"
            )
            data = self._ssh_json(cmd, timeout=300)
        else:
            import subprocess as sp
            result = sp.run(
                [
                    "ffprobe", "-v", "error", "-f", "lavfi",
                    "-i", f"movie={escaped},select='gt(scene,{threshold})'",
                    "-show_entries", "frame=pts_time", "-of", "json",
                ],
                capture_output=True, text=True, timeout=300
            )
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                data = {}

        frames = data.get("frames", [])
        if not frames:
            logger.warning("Scene detection returned no frames, using fixed fallback")
            return [
                i * self.cfg.fixed_window_sec
                for i in range(1, int(duration / self.cfg.fixed_window_sec))
            ]
        return [float(f["pts_time"]) for f in frames if "pts_time" in f]

    def _detect_silence(self, video_path: str) -> list[tuple[float, float]]:
        """Return a list of silence intervals [(silence_start, silence_end), ...]."""
        noise = self.cfg.silence_threshold_db

        if self.remote.enabled:
            # Redirect stderr to stdout to capture silencedetect output
            cmd = (
                f"ffmpeg -i '{video_path}' "
                f"-af silencedetect=noise={noise}dB:d=0.5 "
                f"-f null - 2>&1"
            )
            result = self._ssh(cmd, timeout=300)
            stderr_text = result.stdout  # because of 2>&1
        else:
            import subprocess as sp
            result = sp.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-af", f"silencedetect=noise={noise}dB:d=0.5",
                    "-f", "null", "-",
                ],
                capture_output=True, text=True, timeout=300
            )
            stderr_text = result.stderr

        intervals = []
        silence_start = None
        for line in stderr_text.split("\n"):
            if "silence_start:" in line:
                try:
                    silence_start = float(line.split("silence_start:")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            elif "silence_end:" in line and silence_start is not None:
                try:
                    silence_end = float(line.split("silence_end:")[1].strip().split()[0])
                    intervals.append((silence_start, silence_end))
                    silence_start = None
                except (IndexError, ValueError):
                    pass
        return intervals

    def _boundaries_to_segments(
        self,
        boundaries: list[float],
        duration: float,
        preferred_splits: list[float] | None = None,
    ) -> list[tuple[float, float]]:
        """Convert a list of boundaries into a list of (start, end) segments.

        preferred_splits: preferred split points (visual boundaries) used when a segment
                          exceeds max_segment_sec. When None, falls back to uniform cutting
                          (the fixed strategy path does not pass this parameter).
        """
        min_dur = self.cfg.min_segment_sec
        max_dur = self.cfg.max_segment_sec

        points = [0.0] + [b for b in boundaries if 0 < b < duration] + [duration]

        # Filter out boundaries with spacing < min_dur
        filtered = [points[0]]
        for p in points[1:]:
            if p - filtered[-1] >= min_dur:
                filtered.append(p)
        if filtered[-1] < duration:
            filtered.append(duration)

        segments = []
        for i in range(len(filtered) - 1):
            start, end = filtered[i], filtered[i + 1]
            if end - start <= max_dur:
                if end - start >= min_dur:
                    segments.append((start, end))
                else:
                    # Too short: merge into the previous segment; keep as-is if no previous segment (handled by merge later)
                    if segments:
                        segments[-1] = (segments[-1][0], end)
                    else:
                        segments.append((start, end))
            elif preferred_splits is not None:
                # Prefer visual boundaries for recursive splitting near the midpoint
                segments.extend(
                    self._split_long_segment(start, end, preferred_splits)
                )
            else:
                # fixed strategy: fall back to uniform cutting
                sub_start = start
                while sub_start < end:
                    sub_end = min(sub_start + max_dur, end)
                    segments.append((sub_start, sub_end))
                    sub_start = sub_end

        # Tail segments shorter than min_dur (from the preferred_splits path) are merged into adjacent segments
        segments = self._merge_short_segments(segments, min_dur)
        return segments