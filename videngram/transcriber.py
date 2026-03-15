"""
VidEngram Speech Transcriber
Extracts audio from a video (full-length, time-chunked) and transcribes via
a Whisper-compatible API.

Chunking strategy
-----------------
At 16 kHz mono PCM WAV the rule of thumb is ~1.9 MB/min.
_CHUNK_DURATION_SEC = 600 (10 min) → ~18 MB per chunk, safely under the
25 MB upload limit of most Whisper-compatible APIs.

Remote mode
-----------
When RemoteConfig.enabled is True the video lives on an SSH-accessible server.
For each chunk we:
  1. SSH: ffmpeg extracts the slice → /tmp/ve_chunk_<hash>.wav on remote
  2. SCP: pull the wav to a local tempdir
  3. Call the Whisper API with the local file
  4. SSH: rm the remote temp file
"""
import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import VidEngramConfig
from .utils import ConsolidatedMemory, generate_id, fmt_minutes

logger = logging.getLogger("videngram.transcriber")

# 30 seconds per chunk → ~0.9 MB WAV, well under API limits.
# Shorter chunks give chunk-level timestamp accuracy (±30 s) when the API model
# does not support verbose_json (e.g. gpt-4o-mini-transcribe).  Models that DO
# support verbose_json (e.g. whisper-1) return sentence-level timestamps
# regardless of chunk size, so 30 s is safe for both code paths.
_CHUNK_DURATION_SEC = 30


class SpeechTranscriber:
    """Transcribes full-video audio via a Whisper-compatible API.

    Splits the audio into _CHUNK_DURATION_SEC chunks to respect API upload
    size limits.  Supports both local and SSH-remote video files.
    """

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.transcriber
        self.remote = config.remote
        self.client = OpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def transcribe(self, video_path: str) -> list[dict]:
        """Transcribe all speech from a video file.

        Returns:
            [{"start_sec": float, "end_sec": float, "text": str}, ...]
            Empty list if transcription fails or no speech found.
        """
        duration = self._get_duration(video_path)
        if not duration or duration <= 0:
            logger.warning(f"Could not determine video duration for {video_path}")
            return []

        n_chunks = max(
            1,
            int(duration // _CHUNK_DURATION_SEC)
            + (1 if duration % _CHUNK_DURATION_SEC else 0),
        )
        logger.info(
            f"Transcribing {duration:.0f}s of audio in {n_chunks} chunk(s) "
            f"({_CHUNK_DURATION_SEC}s each)"
        )

        all_segments: list[dict] = []
        for i in range(n_chunks):
            start = i * _CHUNK_DURATION_SEC
            chunk_dur = min(_CHUNK_DURATION_SEC, duration - start)
            if chunk_dur <= 0:
                break
            logger.info(
                f"  Chunk [{i+1}/{n_chunks}] "
                f"{fmt_minutes(start)} – {fmt_minutes(start + chunk_dur)}"
            )
            segments = self._transcribe_chunk(video_path, start, chunk_dur, offset_sec=start)
            all_segments.extend(segments)

        logger.info(f"Transcription complete: {len(all_segments)} segment(s)")
        return all_segments

    def build_speech_memories(
        self, segments: list[dict], video_path: str
    ) -> list[ConsolidatedMemory]:
        """Convert raw transcription segments into ConsolidatedMemory objects
        ready for EverMemOS ingestion (memory_type='speech')."""
        from .memory_writer import MemoryWriter

        group_id = MemoryWriter._video_group_id(video_path)
        memories = []
        for seg in segments:
            start = seg["start_sec"]
            end = seg["end_sec"]
            text = seg["text"].strip()
            if not text:
                continue
            content = f"[Video {fmt_minutes(start)} - {fmt_minutes(end)}] TRANSCRIPT: {text}"
            memories.append(
                ConsolidatedMemory(
                    memory_id=generate_id("sp", group_id, start),
                    content=content,
                    start_sec=start,
                    end_sec=end,
                    memory_type="speech",
                    source_segments=[],
                    metadata={"transcription": True},
                )
            )
        return memories

    # ── Path helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_remote_path(path: str) -> str:
        """Reverse the macOS Path.resolve() expansion of remote server paths.

        On macOS, /home is a symlink to /System/Volumes/Data/home.
        Path.resolve() (called in pipeline.py) silently expands remote paths
        like /home/user/... to /System/Volumes/Data/home/user/..., which do
        not exist on the Linux remote server.  Strip that prefix so SSH
        commands receive the correct path.
        """
        _MACOS_DATA_PREFIX = "/System/Volumes/Data"
        if path.startswith(_MACOS_DATA_PREFIX):
            return path[len(_MACOS_DATA_PREFIX):]
        return path

    # ── Duration probe ────────────────────────────────────────────────────

    def _get_duration(self, video_path: str) -> Optional[float]:
        """Return video duration in seconds via ffprobe (local or remote)."""
        remote_path = self._to_remote_path(video_path)
        cmd_parts = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", remote_path,
        ]
        try:
            if self.remote.enabled:
                # Quote the path so spaces/special chars in filenames are handled correctly
                ssh_cmd = (
                    f"ffprobe -v error -show_entries format=duration "
                    f"-of json '{remote_path}'"
                )
                result = subprocess.run(
                    ["ssh", self.remote.host, ssh_cmd],
                    capture_output=True, text=True, timeout=30,
                )
            else:
                result = subprocess.run(
                    cmd_parts, capture_output=True, text=True, timeout=30,
                )
            if result.returncode != 0:
                logger.warning(f"ffprobe failed: {result.stderr[:200]}")
                return None
            info = json.loads(result.stdout)
            return float(info["format"]["duration"])
        except Exception as e:
            logger.warning(f"Duration probe failed: {e}")
            return None

    # ── Chunk extraction + transcription ──────────────────────────────────

    def _transcribe_chunk(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: float,
        offset_sec: float,
    ) -> list[dict]:
        """Extract one audio chunk, call the API, return offset-adjusted segments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_wav = str(Path(tmpdir) / "chunk.wav")
            ok = self._extract_audio_chunk(video_path, start_sec, duration_sec, local_wav)
            if not ok:
                return []
            return self._call_api(local_wav, offset_sec, duration_sec)

    def _extract_audio_chunk(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: float,
        local_wav_path: str,
    ) -> bool:
        """Dispatch to local or remote audio extraction."""
        if self.remote.enabled:
            return self._extract_remote(video_path, start_sec, duration_sec, local_wav_path)
        return self._extract_local(video_path, start_sec, duration_sec, local_wav_path)

    def _extract_local(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: float,
        out_path: str,
    ) -> bool:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", out_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            ok = result.returncode == 0 and Path(out_path).exists()
            if not ok:
                logger.warning(f"Local audio extraction failed: {result.stderr[:200]}")
            return ok
        except Exception as e:
            logger.warning(f"Local audio extraction error: {e}")
            return False

    def _extract_remote(
        self,
        video_path: str,
        start_sec: float,
        duration_sec: float,
        local_wav_path: str,
    ) -> bool:
        """Extract audio on remote server via SSH, SCP back, clean up remote file."""
        chunk_hash = hashlib.md5(f"{video_path}:{start_sec}".encode()).hexdigest()[:8]
        remote_wav = f"/tmp/ve_chunk_{chunk_hash}.wav"

        # Step 1: extract on remote (use un-resolved path for SSH command)
        remote_video = self._to_remote_path(video_path)
        ffmpeg_cmd = (
            f"ffmpeg -y -ss {start_sec} -t {duration_sec} "
            f"-i '{remote_video}' -vn -ac 1 -ar 16000 -f wav '{remote_wav}' 2>/dev/null"
        )
        try:
            result = subprocess.run(
                ["ssh", self.remote.host, ffmpeg_cmd],
                capture_output=True, timeout=180,
            )
            if result.returncode != 0:
                logger.warning(f"Remote ffmpeg failed for chunk at {start_sec}s")
                return False
        except Exception as e:
            logger.warning(f"Remote ffmpeg exception: {e}")
            return False

        # Step 2: SCP back to local
        success = False
        try:
            result = subprocess.run(
                ["scp", f"{self.remote.host}:{remote_wav}", local_wav_path],
                capture_output=True, timeout=180,
            )
            success = result.returncode == 0 and Path(local_wav_path).exists()
            if not success:
                logger.warning(f"SCP failed: {result.stderr[:200]}")
        except Exception as e:
            logger.warning(f"SCP exception: {e}")
        finally:
            # Step 3: always clean up remote temp file
            try:
                subprocess.run(
                    ["ssh", self.remote.host, f"rm -f {remote_wav}"],
                    capture_output=True, timeout=15,
                )
            except Exception:
                pass

        return success

    # ── Whisper API ───────────────────────────────────────────────────────

    def _call_api(self, wav_path: str, offset_sec: float, chunk_duration: float) -> list[dict]:
        """Call the Whisper-compatible transcription API.

        Tries verbose_json first (per-segment timestamps).  If the model does
        not support verbose_json (e.g. gpt-4o-mini-transcribe), falls back to
        text format and returns one segment covering the whole chunk so that
        subtitles are at least accurate to ±chunk_duration seconds.
        max_retries=0 is used for the verbose_json attempt so that a permanent
        incompatibility error does not trigger the SDK's exponential-backoff
        retry loop.
        """
        # ── Attempt 1: verbose_json — exact segment-level timestamps ─────────
        try:
            with open(wav_path, "rb") as f:
                response = self.client.with_options(max_retries=0).audio.transcriptions.create(
                    file=f,
                    model=self.cfg.model,
                    response_format="verbose_json",
                )
            raw_segments = getattr(response, "segments", None) or []
            if not raw_segments:
                logger.debug("Whisper returned no segments (silent or empty chunk)")
                return []
            result = []
            for seg in raw_segments:
                text = (getattr(seg, "text", None) or "").strip()
                if not text:
                    continue
                start = float(getattr(seg, "start", 0.0)) + offset_sec
                end = float(getattr(seg, "end", start)) + offset_sec
                result.append({"start_sec": start, "end_sec": end, "text": text})
            return result
        except Exception as e:
            if "verbose_json" not in str(e):
                logger.warning(f"Whisper API call failed: {e}")
                return []
            logger.info(
                "Model does not support verbose_json; using text format "
                f"(timestamps accurate to ±{int(chunk_duration)}s)"
            )

        # ── Fallback: text format — chunk-level timestamps ────────────────────
        try:
            with open(wav_path, "rb") as f:
                text = self.client.audio.transcriptions.create(
                    file=f,
                    model=self.cfg.model,
                    response_format="text",
                )
            text = str(text).strip()
            if not text:
                logger.debug("Whisper (text format) returned empty response")
                return []
            return [{"start_sec": offset_sec, "end_sec": offset_sec + chunk_duration, "text": text}]
        except Exception as e:
            logger.warning(f"Whisper API call (text format) failed: {e}")
            return []
