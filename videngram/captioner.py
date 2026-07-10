"""
VidEngram Captioner
Generates structured captions from video segments using Qwen2.5-Omni
served via vLLM-Omni's OpenAI-compatible API.

Key advantage over HippoMM: Qwen2.5-Omni processes video + audio in a
single pass, eliminating the need for separate ImageBind + Whisper pipelines
and the error-prone fusion step. This gives us unified audiovisual understanding.
"""
import asyncio
import base64
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import httpx
from openai import OpenAI, AsyncOpenAI

from .config import VidEngramConfig
from .utils import VideoSegment, Caption, fmt_minutes

logger = logging.getLogger("videngram.captioner")

CAPTION_SYSTEM_PROMPT = """\
You are a precise, EXHAUSTIVE video analyst. For each video segment output EXACTLY these \
labeled fields, each on its own line. Record EVERY detail — exact counts and quantities, \
colors, on-screen text, spatial relationships, object attributes, and fine-grained actions. \
Downstream questions are detailed and counting-heavy and will depend on the specifics you \
capture here, so err strongly on the side of MORE detail. NEVER summarize (e.g. "a costume", \
"some objects", "music plays") — ENUMERATE the specifics instead (e.g. "red cape, blue mask, \
yellow emblem"; "3 books and 2 red cups"; "upbeat piano with drums").

SCENE: [Physical setting, location, lighting, camera angle and movement, background elements. Be exhaustive.]
PEOPLE: [Every visible person: identity/name if known, clothing and colors, position, expression, posture. State how many there are, and note relative attributes (who is tallest/shortest, leftmost/rightmost, oldest/youngest, nearest/farthest) so later questions can disambiguate which person.]
ACTIONS: [What each person does, all interactions and movements, described step by step in order.]
DIALOGUE: [When an ASR transcript is provided, use it verbatim as the words spoken — \
do not re-transcribe or paraphrase. Attribute each utterance to the visually identifiable \
speaker by name or description (e.g. "John:", "Woman in red:"); if speakers cannot be \
distinguished visually, use "Speaker A:", "Speaker B:", etc. If no ASR transcript is \
provided, transcribe speech yourself. Output "None" if the segment is silent.]
SOUNDS: [Every non-speech sound, in detail: animal/object sounds (e.g. "dog barks twice", "owl hiss"), music (genre, instruments, tempo, mood), sound effects, ambient noise. Count occurrences. CRITICAL for audio questions — never write just "music plays".]
AV_SYNC: [Audio-visual co-occurrence — THE key field for audio-visual sync questions. For EVERY notable sound, speech line, or music cue, state what is visually happening at that EXACT moment and WHO is involved, identifying people by distinguishing attributes (tallest, red shirt, leftmost). Link each audio event to the simultaneous visual with "→". E.g. "whistle blows → the tallest boy (blue cap) starts running"; "woman in red says 'stop' → she raises her right hand"; "drums kick in → the crowd starts jumping". Cover every audio-visual pairing in the segment. "None" only if the segment is fully silent with no music or sound.]
TEXT: [ALL visible on-screen text, signs, titles, subtitles — transcribe verbatim. "None" if absent.]
OBJECTS: [ALL notable objects, tools, props: their colors, quantities, positions, and any text on them.]
COUNTS: [Explicit counts of anything countable in this segment: number of people, repeated objects, occurrences of an event. Always give numbers when possible.]
EMOTION: [Emotional tone with specific descriptors and nuance.]
TEMPORAL: [Sequence of events within this segment in order: what happens first, what changes, how it ends.]

EXAMPLE of the expected level of detail (a different segment, for style only):
SCENE: Dimly lit recording studio, acoustic foam panels on the back wall, warm desk-lamp lighting, fixed frontal camera; a boom microphone in the foreground.
PEOPLE: 2 people. Left: man ~40s, short black hair, navy blazer over white shirt, leaning forward, animated. Right: woman ~30s, long brown hair, red sweater, attentive, nodding.
OBJECTS: 1 boom microphone; 2 white coffee mugs (one with a logo); a silver laptop open on the desk; a framed poster on the back wall.
COUNTS: 2 people; 2 mugs; the man gestures with his hands 3 times.
SOUNDS: faint air-conditioner hum throughout; 1 chair creak near the start; paper rustling once; no music.
AV_SYNC: man gestures (3rd time) → he says "the data is clear"; chair creak → the woman (right, red sweater) shifts posture and leans back; paper rustling → the man picks up a sheet from the desk.
TEXT: back-wall poster reads "JAZZ NIGHT"; laptop screen shows "REC 00:42".
Match THIS exhaustiveness: name specifics, colors, on-screen text, and exact numbers for every field.
"""

CAPTION_USER_TEMPLATE = """\
Analyze this video segment ({timestamp_label}, duration {duration:.0f}s). \
Segment {seg_index} of {total_segments}.{asr_hint}
Output all 11 labeled fields: SCENE, PEOPLE, ACTIONS, DIALOGUE, SOUNDS, AV_SYNC, TEXT, OBJECTS, COUNTS, EMOTION, TEMPORAL. \
Your output MUST be AT LEAST 400 words — write exhaustively for EVERY field. A difficult quiz \
about this segment will depend on the fine details you record, so do not stop early or summarize."""

ASR_HINT_TEMPLATE = """

[ASR Transcript — verbatim ground truth for DIALOGUE]:
{asr_text}
Use the above as the exact words spoken. Focus your visual analysis on speaker attribution \
and all non-speech fields."""


class Captioner:
    """Generates rich captions from video segments using Qwen2.5-Omni."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.captioner
        self.qwen_cfg = config.qwen
        self.remote = config.remote
        _timeout = httpx.Timeout(180.0, connect=10.0, read=180.0, write=90.0)
        self.client = OpenAI(
            base_url=self.qwen_cfg.base_url,
            api_key=self.qwen_cfg.api_key,
            http_client=httpx.Client(timeout=_timeout),
        )
        self.async_client = AsyncOpenAI(
            base_url=self.qwen_cfg.base_url,
            api_key=self.qwen_cfg.api_key,
            http_client=httpx.AsyncClient(timeout=_timeout),
            max_retries=6,   # ride out transient tunnel/saturation APIConnectionErrors
        )
        # Multi-endpoint Omni parallelism: when several local Omni workers are
        # available, build one async client per endpoint and dispatch segment i
        # to endpoint i % n. Each endpoint handles ~1 segment at a time, which
        # speeds up captioning while avoiding per-endpoint decode stalls. With a
        # single endpoint (e.g. a hosted API) this degrades to [async_client].
        self.async_clients = self._build_async_clients(_timeout)

    def _build_async_clients(self, timeout):
        """Build per-endpoint async clients for parallel captioning.

        Local Omni is served as 6 instances on ports 8091-8096 (tunneled). We
        round-robin segments across them so each Omni handles ~1 concurrent
        request (no single-endpoint KV deadlock). Non-local single endpoints
        (e.g. DashScope) just reuse the one async_client.
        """
        import re as _re, os as _os
        base = self.qwen_cfg.base_url or ""
        m = _re.search(r"^(https?)://([^/:]+):(80\d\d)(/.*)?$", base)
        if m and m.group(2) in ("localhost", "127.0.0.1"):
            scheme, host, start_port, path = (
                m.group(1), m.group(2), int(m.group(3)), (m.group(4) or "/v1")
            )
            # CAPTION_ENDPOINTS lets par_ingest split the 6 Omni among N video
            # workers (e.g. 2 workers × 3 endpoints): a worker pinned to
            # start_port uses [start_port, start_port+n_ep). Default 6 = single
            # process uses all of them. Each Omni then sees ~1 concurrent caption.
            n_ep = int(_os.environ.get("CAPTION_ENDPOINTS", "6"))
            clients = [
                AsyncOpenAI(
                    base_url=f"{scheme}://{host}:{start_port + i}{path}",
                    api_key=self.qwen_cfg.api_key,
                    http_client=httpx.AsyncClient(timeout=timeout),
                    max_retries=6,   # ride out transient tunnel/saturation errors
                )
                for i in range(n_ep)
            ]
            logger.info(
                f"Captioner: {len(clients)} parallel Omni endpoints "
                f"({start_port}-{start_port + n_ep - 1})"
            )
            return clients
        return [self.async_client]

    def _read_file_as_b64(self, path: str, mime: str) -> str:
        """Read a file (local or remote via SSH) and return a base64 data URI.

        Used in external API mode where file:// URLs are not supported.
        - Local mode (remote.enabled=False): reads from local filesystem.
        - Remote mode (remote.enabled=True): reads via SSH cat.
        """
        try:
            if self.remote.enabled:
                result = subprocess.run(
                    ["ssh", self.remote.host, f"cat {path}"],
                    capture_output=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    logger.warning(f"SSH cat failed for {path}: {result.stderr.decode()[:200]}")
                    return ""
                raw = result.stdout
            else:
                raw = Path(path).read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read file {path}: {e}")
            return ""
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _downscale_clip(self, clip_path: str, width: int = 480, target_height: Optional[int] = None) -> str:
        """Return a downscaled copy of clip_path.

        Two modes:
        - target_height set (external API): scale to height px tall, proportional width
          (scale=-2:{target_height}). Local mode only: skips processing entirely if the
          source is already at or below target_height. Remote mode: always re-encodes.
        - target_height None (local vLLM): scale to width px wide, proportional height
          (scale={width}:-2). Designed for Qwen2.5-Omni-7B's 8192-token context limit.

        Returns the path of the scaled clip, or original path if no scaling needed.
        Falls back to original on ffmpeg failure.
        Caller is responsible for deleting the scaled clip after use.
        """
        if target_height is not None:
            # Height-based scaling for external API (e.g. DashScope): target 720p.
            out_path = clip_path.rsplit(".", 1)[0] + f"_scaled{target_height}p.mp4"
            vf = f"scale=-2:{target_height}"
            try:
                if self.remote.enabled:
                    # Remote: skip height check, always scale.
                    result = subprocess.run(
                        ["ssh", self.remote.host,
                         f"ffmpeg -y -i {clip_path} -vf '{vf}' -c:a copy {out_path} 2>/dev/null"],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        return out_path
                else:
                    # Local: check source height first; skip if already small enough.
                    probe = subprocess.run(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0",
                         "-show_entries", "stream=height", "-of", "csv=p=0", clip_path],
                        capture_output=True, text=True, timeout=10,
                    )
                    if probe.returncode == 0 and probe.stdout.strip().isdigit():
                        src_height = int(probe.stdout.strip())
                        if src_height <= target_height:
                            logger.debug(f"  Source height {src_height}px ≤ {target_height}px, skipping downscale")
                            return clip_path
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", clip_path,
                         "-vf", vf,
                         "-c:a", "copy", out_path],
                        capture_output=True, timeout=60,
                    )
                    if result.returncode == 0 and Path(out_path).exists():
                        return out_path
            except Exception as e:
                logger.debug(f"Downscale failed for {clip_path}: {e}")
            return clip_path  # fallback: use original
        else:
            # Width-based scaling for local vLLM (token budget constraint).
            out_path = clip_path.rsplit(".", 1)[0] + f"_scaled{width}.mp4"
            try:
                # Long-edge scaling: fit within width x width box so BOTH landscape
                # and PORTRAIT clips shrink to a bounded token budget. (Old width-only
                # scaling blew up portrait videos' height -> token overflow.)
                vf = f"scale={width}:{width}:force_original_aspect_ratio=decrease:force_divisible_by=2"
                if self.remote.enabled:
                    result = subprocess.run(
                        ["ssh", self.remote.host,
                         f"ffmpeg -y -i {clip_path} -vf '{vf}' -c:a copy {out_path} 2>/dev/null"],
                        capture_output=True, text=True, timeout=60,
                    )
                    if result.returncode == 0:
                        return out_path
                else:
                    result = subprocess.run(
                        ["ffmpeg", "-y", "-i", clip_path,
                         "-vf", vf,
                         "-c:a", "copy", out_path],
                        capture_output=True, timeout=60,
                    )
                    if result.returncode == 0 and Path(out_path).exists():
                        return out_path
            except Exception as e:
                logger.debug(f"Downscale failed for {clip_path}: {e}")
            return clip_path  # fallback: use original

    def _cleanup_scaled(self, original_path: str, scaled_path: str):
        """Delete the scaled clip if it differs from the original."""
        if scaled_path == original_path:
            return
        try:
            if self.remote.enabled:
                subprocess.run(
                    ["ssh", self.remote.host, f"rm -f {scaled_path}"],
                    capture_output=True, timeout=10,
                )
            else:
                Path(scaled_path).unlink(missing_ok=True)
        except Exception:
            pass

    def caption_segment(
        self,
        segment: VideoSegment,
        seg_index: int = 0,
        total_segments: int = 1,
    ) -> Caption:
        """Generate a caption for a single video segment.

        Uses the extracted clip file and sends it to Qwen2.5-Omni via
        the vLLM-Omni OpenAI-compatible API with video_url content type.
        """
        if not segment.clip_path or (
            not self.remote.enabled and not Path(segment.clip_path).exists()
        ):
            logger.warning(f"No clip for segment {segment.segment_id}, skipping")
            return Caption(
                segment_id=segment.segment_id,
                raw_text="[Clip unavailable]",
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
            )

        logger.info(f"  Preparing clip: {segment.clip_path}")
        scaled_path = self._downscale_clip(
            segment.clip_path,
            target_height=720 if not self.qwen_cfg.is_local else None,
        )
        logger.info(f"  Clip ready: {scaled_path}")

        asr_hint = (
            ASR_HINT_TEMPLATE.format(asr_text=segment.asr_text)
            if segment.asr_text
            else ""
        )
        user_text = CAPTION_USER_TEMPLATE.format(
            timestamp_label=segment.timestamp_label,
            duration=segment.duration,
            seg_index=seg_index + 1,
            total_segments=total_segments,
            asr_hint=asr_hint,
        )

        # Build multimodal content array
        # Remote mode: file:// URL — vLLM reads the clip directly from the remote filesystem.
        # Local mode: base64 data URI — encode the clip from the local filesystem.
        # Note: API params (extra_body) are controlled separately by qwen_cfg.is_local.
        if self.remote.enabled:
            video_content = {"type": "video_url", "video_url": {"url": f"file://{scaled_path}"}}
        else:
            logger.info(f"  Reading video file for base64 encoding: {scaled_path}")
            data_uri = self._read_file_as_b64(scaled_path, "video/mp4")
            if not data_uri:
                logger.error(
                    f"  Failed to read video file for segment {segment.segment_id}: "
                    f"_read_file_as_b64 returned empty string for {scaled_path}"
                )
            else:
                payload_kb = len(data_uri) // 1024
                logger.info(f"  Video encoded: {payload_kb} KB base64 payload")
            video_content = {"type": "video_url", "video_url": {"url": data_uri}}

        content = [video_content, {"type": "text", "text": user_text}]

        # extra_body: mm_processor_kwargs is vLLM-Omni-specific; omit for external APIs.
        # enable_thinking=False: explicitly disable thinking mode for Qwen3 models.
        if self.qwen_cfg.is_local:
            extra_body = {"modalities": self.qwen_cfg.modalities, "mm_processor_kwargs": {"fps": self.cfg.caption_fps}}
        else:
            extra_body = {"modalities": self.qwen_cfg.modalities, "enable_thinking": False}

        max_tokens = self.qwen_cfg.max_tokens if self.qwen_cfg.is_local else self.qwen_cfg.max_tokens_api

        logger.info(
            f"  Sending API request for segment {segment.segment_id} "
            f"(model={self.qwen_cfg.model}, max_tokens={max_tokens})"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=self.qwen_cfg.temperature,
                extra_body=extra_body,
            )
            logger.info(f"  API response received for segment {segment.segment_id}")
            msg = response.choices[0].message if response.choices else None
            raw_text = (msg.content if msg else None) or ""
            if not raw_text and msg:
                # Fallback: some Qwen3 variants return output in reasoning_content
                raw_text = getattr(msg, "reasoning_content", None) or ""
            if raw_text:
                raw_text = raw_text.strip()
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
                reasoning = getattr(msg, "reasoning_content", None) if msg else None
                logger.error(
                    f"  Empty response content for segment {segment.segment_id}. "
                    f"finish_reason={finish_reason!r}, "
                    f"reasoning_content={str(reasoning)[:200]!r}, "
                    f"choices_count={len(response.choices) if response.choices else 0}"
                )
                raw_text = "[Caption error: empty response from model]"
        except Exception as e:
            logger.error(
                f"  API call failed for segment {segment.segment_id}: "
                f"{type(e).__name__}: {e}"
            )
            raw_text = f"[Caption error: {type(e).__name__}: {e}]"
        finally:
            self._cleanup_scaled(segment.clip_path, scaled_path)

        # NOTE: Qwen audio re-transcription is disabled. Whisper ASR text is now
        # injected into the prompt directly (segment.asr_text), so Qwen uses it
        # as verbatim ground truth for DIALOGUE. The separate Qwen audio call
        # added latency and an extra API round-trip without quality benefit.
        #
        # if self.cfg.use_audio_in_video and segment.clip_path and not raw_text.startswith("["):
        #     transcript = self._transcribe_audio(segment.clip_path)
        #     if transcript:
        #         raw_text = re.sub(
        #             r"(?m)^DIALOGUE:[ \t].*$",
        #             f"DIALOGUE: {transcript}",
        #             raw_text,
        #         )

        return Caption(
            segment_id=segment.segment_id,
            raw_text=raw_text,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )

    def caption_all(
        self,
        segments: list[VideoSegment],
        parallel: bool = False,
        on_caption=None,
    ) -> list[Caption]:
        """Caption all segments, optionally in parallel.

        Args:
            on_caption: Optional callable(index, segment, caption) called after
                        each caption is generated. Used by the demo UI for
                        real-time ingest visualization.
        """
        total = len(segments)
        logger.info(f"Captioning {total} segments (parallel={parallel})")

        if parallel:
            # Try parallel (fast). BACKUP: if it throws or produces too many errors
            # (Omni KV pressure / deadlock under concurrency), fall back to serial
            # below — which has its own circuit-breaker. This guarantees we never
            # crash the run: worst case we degrade to the slow-but-safe path.
            try:
                caps = asyncio.run(self._caption_all_async(segments))
                n_err = sum(
                    1 for c in caps
                    if (c.raw_text or "").startswith(("[Caption error", "[Clip unavailable"))
                )
                if n_err <= max(2, int(total * 0.3)):
                    # Fire on_caption in index order so segment memories are
                    # written to the memory store as an ordered stream (as if
                    # serial), keeping boundary detection from merging captions
                    # that would otherwise arrive together in a parallel batch.
                    if on_caption is not None:
                        for i, seg in enumerate(segments):
                            on_caption(i, seg, caps[i])
                    return caps
                logger.warning(
                    f"Parallel caption unhealthy ({n_err}/{total} errors — likely Omni "
                    f"KV pressure under concurrency); falling back to SERIAL backup"
                )
            except Exception as e:
                logger.warning(f"Parallel caption failed ({e}); falling back to SERIAL backup")
            # fall through to serial backup ↓

        captions = []
        consecutive_fail = 0
        for i, seg in enumerate(segments):
            logger.info(f"  Captioning [{i+1}/{total}] {seg.timestamp_label}")
            cap = self.caption_segment(seg, seg_index=i, total_segments=total)
            captions.append(cap)
            if on_caption is not None:
                on_caption(i, seg, cap)
            # Deadlock circuit-breaker: if the Omni server hangs, every caption
            # request times out and returns a "[Caption error...]" placeholder.
            # After several consecutive failures, abort this video so the caller
            # can skip it (and a guard can restart the stuck Omni instance),
            # instead of grinding through hundreds of timed-out segments.
            rt = cap.raw_text or ""
            if rt.startswith("[Caption error") or rt.startswith("[Clip unavailable"):
                consecutive_fail += 1
                if consecutive_fail >= 4:
                    raise RuntimeError(
                        f"Omni appears deadlocked: {consecutive_fail} consecutive "
                        f"caption failures at segment {i+1}/{total}"
                    )
            else:
                consecutive_fail = 0

        return captions

    async def _caption_all_async(
        self, segments: list[VideoSegment]
    ) -> list[Caption]:
        """Caption segments concurrently (bounded concurrency)."""
        n_ep = len(self.async_clients)
        # ~1 concurrent request per endpoint; fall back to 5 for a single
        # endpoint. Round-robin dispatch avoids stalling any one Omni worker.
        semaphore = asyncio.Semaphore(n_ep if n_ep > 1 else 5)
        total = len(segments)

        async def _caption_one(i: int, seg: VideoSegment) -> Caption:
            async with semaphore:
                client = self.async_clients[i % n_ep]
                logger.info(f"  [async] Captioning [{i+1}/{total}] via Omni#{i % n_ep}")
                return await self._async_caption_segment(seg, i, total, client)

        tasks = [_caption_one(i, seg) for i, seg in enumerate(segments)]
        return await asyncio.gather(*tasks)

    async def _async_caption_segment(
        self,
        segment: VideoSegment,
        seg_index: int,
        total_segments: int,
        client=None,
    ) -> Caption:
        """Async version of caption_segment."""
        if client is None:
            client = self.async_client
        if not segment.clip_path or (
            not self.remote.enabled and not Path(segment.clip_path).exists()
        ):
            return Caption(
                segment_id=segment.segment_id,
                raw_text="[Clip unavailable]",
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
            )

        asr_hint = (
            ASR_HINT_TEMPLATE.format(asr_text=segment.asr_text)
            if segment.asr_text
            else ""
        )
        user_text = CAPTION_USER_TEMPLATE.format(
            timestamp_label=segment.timestamp_label,
            duration=segment.duration,
            seg_index=seg_index + 1,
            total_segments=total_segments,
            asr_hint=asr_hint,
        )

        logger.info(f"  [async] Preparing clip: {segment.clip_path}")
        scaled_path = await asyncio.to_thread(
            lambda: self._downscale_clip(
                segment.clip_path,
                target_height=720 if not self.qwen_cfg.is_local else None,
            )
        )
        logger.info(f"  [async] Clip ready: {scaled_path}")

        if self.remote.enabled:
            video_content = {"type": "video_url", "video_url": {"url": f"file://{scaled_path}"}}
        else:
            logger.info(f"  [async] Reading video file for base64 encoding: {scaled_path}")
            data_uri = await asyncio.to_thread(self._read_file_as_b64, scaled_path, "video/mp4")
            if not data_uri:
                logger.error(
                    f"  [async] Failed to read video file for segment {segment.segment_id}: "
                    f"_read_file_as_b64 returned empty string for {scaled_path}"
                )
            else:
                payload_kb = len(data_uri) // 1024
                logger.info(f"  [async] Video encoded: {payload_kb} KB base64 payload")
            video_content = {"type": "video_url", "video_url": {"url": data_uri}}

        content = [video_content, {"type": "text", "text": user_text}]

        if self.qwen_cfg.is_local:
            extra_body = {"modalities": self.qwen_cfg.modalities, "mm_processor_kwargs": {"fps": self.cfg.caption_fps}}
        else:
            extra_body = {"modalities": self.qwen_cfg.modalities, "enable_thinking": False}

        max_tokens = self.qwen_cfg.max_tokens if self.qwen_cfg.is_local else self.qwen_cfg.max_tokens_api

        logger.info(
            f"  [async] Sending API request for segment {segment.segment_id} "
            f"(model={self.qwen_cfg.model}, max_tokens={max_tokens})"
        )
        try:
            response = await client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=self.qwen_cfg.temperature,
                extra_body=extra_body,
            )
            logger.info(f"  [async] API response received for segment {segment.segment_id}")
            msg = response.choices[0].message if response.choices else None
            raw_text = (msg.content if msg else None) or ""
            if not raw_text and msg:
                raw_text = getattr(msg, "reasoning_content", None) or ""
            if raw_text:
                raw_text = raw_text.strip()
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
                reasoning = getattr(msg, "reasoning_content", None) if msg else None
                logger.error(
                    f"  [async] Empty response content for segment {segment.segment_id}. "
                    f"finish_reason={finish_reason!r}, "
                    f"reasoning_content={str(reasoning)[:200]!r}, "
                    f"choices_count={len(response.choices) if response.choices else 0}"
                )
                raw_text = "[Caption error: empty response from model]"
        except Exception as e:
            logger.error(
                f"  [async] API call failed for segment {segment.segment_id}: "
                f"{type(e).__name__}: {e}"
            )
            raw_text = f"[Caption error: {type(e).__name__}: {e}]"
        finally:
            self._cleanup_scaled(segment.clip_path, scaled_path)

        return Caption(
            segment_id=segment.segment_id,
            raw_text=raw_text,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )

    def _transcribe_audio(self, clip_path: str) -> str:
        """Extract audio from a video clip and transcribe speech verbatim.

        Uses ffmpeg to extract a mono WAV from the clip, then sends it to
        Qwen2.5-Omni via audio_url for pure speech-to-text transcription.
        Returns the verbatim transcript, or empty string if no speech / failure.
        """
        audio_path = clip_path.rsplit(".", 1)[0] + "_aud.wav"
        extracted = False

        # ── Step 1: Extract audio via ffmpeg ──────────────────────────────
        try:
            if self.remote.enabled:
                result = subprocess.run(
                    ["ssh", self.remote.host,
                     f"ffmpeg -y -i {clip_path} -vn -ac 1 -ar 16000 -f wav {audio_path} 2>/dev/null"],
                    capture_output=True, text=True, timeout=30,
                )
                extracted = result.returncode == 0
            else:
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path,
                     "-vn", "-ac", "1", "-ar", "16000", audio_path],
                    capture_output=True, timeout=30,
                )
                extracted = result.returncode == 0 and Path(audio_path).exists()
        except Exception as e:
            logger.debug(f"Audio extraction failed for {clip_path}: {e}")
            return ""

        if not extracted:
            logger.debug(f"Audio extraction non-zero for {clip_path}")
            return ""

        # ── Step 2: Transcribe via Qwen audio_url ─────────────────────────
        transcript = ""
        try:
            if self.remote.enabled:
                audio_content = {"type": "audio_url", "audio_url": {"url": f"file://{audio_path}"}}
            else:
                data_uri = self._read_file_as_b64(audio_path, "audio/wav")
                audio_content = {"type": "audio_url", "audio_url": {"url": data_uri}}
            if self.qwen_cfg.is_local:
                transcribe_extra = {"modalities": ["text"]}
            else:
                transcribe_extra = {"modalities": ["text"], "enable_thinking": False}

            logger.info(f"  Sending audio transcription request for {audio_path}")
            response = self.client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[{
                    "role": "user",
                    "content": [
                        audio_content,
                        {"type": "text",
                         "text": (
                             "Transcribe every single spoken word in this audio exactly as heard. "
                             "Do not skip, omit, paraphrase, or summarize any words. "
                             "Output only the verbatim speech transcript with no extra commentary. "
                             "If multiple speakers, format each line as 'Speaker: text'. "
                             "If there is truly no speech at all, output an empty string."
                         )},
                    ],
                }],
                max_tokens=1024,
                temperature=0.0,
                extra_body=transcribe_extra,
            )
            logger.info(f"  Audio transcription response received for {audio_path}")
            msg = response.choices[0].message if response.choices else None
            text = (msg.content if msg else None) or ""
            if not text and msg:
                text = getattr(msg, "reasoning_content", None) or ""
            text = text.strip().strip('"\'')
            if text.lower() not in ("none", "n/a", "", "-"):
                transcript = text
        except Exception as e:
            logger.error(f"  Audio transcription API call failed: {type(e).__name__}: {e}")
        finally:
            # Clean up temp audio file
            try:
                if self.remote.enabled:
                    subprocess.run(
                        ["ssh", self.remote.host, f"rm -f {audio_path}"],
                        capture_output=True, timeout=10,
                    )
                else:
                    Path(audio_path).unlink(missing_ok=True)
            except Exception:
                pass

        return transcript

    def analyze_clip(self, clip_path: str, query: str) -> str:
        """On-demand analysis: send a specific query about a video clip.

        Used by the agentic orchestrator for video grounding — when the
        agent decides it needs to "look at" a specific moment in the video
        to answer a query.
        """
        # File access: remote mode → file:// (clip lives on server); local → base64 encode.
        if self.remote.enabled:
            scaled_path = clip_path
            video_content = {"type": "video_url", "video_url": {"url": f"file://{scaled_path}"}}
        else:
            scaled_path = self._downscale_clip(clip_path, target_height=720)
            data_uri = self._read_file_as_b64(scaled_path, "video/mp4")
            video_content = {"type": "video_url", "video_url": {"url": data_uri}}

        # API params: vLLM-specific vs external API.
        if self.qwen_cfg.is_local:
            extra_body = {"modalities": ["text"], "mm_processor_kwargs": {"fps": self.cfg.caption_fps}}
        else:
            extra_body = {"modalities": ["text"], "enable_thinking": False}

        max_tokens = self.qwen_cfg.max_tokens if self.qwen_cfg.is_local else self.qwen_cfg.max_tokens_api

        content = [video_content, {"type": "text", "text": query}]

        logger.info(f"  Sending clip analysis request for {clip_path}")
        try:
            response = self.client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": "You are a precise video analyst. Answer the question based only on what you observe in the video."},
                    {"role": "user", "content": content},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                extra_body=extra_body,
            )
            logger.info(f"  Clip analysis response received for {clip_path}")
            msg = response.choices[0].message if response.choices else None
            result = (msg.content if msg else None) or ""
            if not result and msg:
                result = getattr(msg, "reasoning_content", None) or ""
            if result:
                # Strip [Image: ...] metadata injected by the vision model
                result = re.sub(r'\*?\[Image:[^\]]*\]\*?', '', result).strip()
                return result
            finish_reason = response.choices[0].finish_reason if response.choices else "N/A"
            logger.error(
                f"  Clip analysis returned empty content for {clip_path}. "
                f"finish_reason={finish_reason!r}"
            )
            return "[Analysis error: empty response from model]"
        except Exception as e:
            logger.error(f"  Clip analysis failed: {type(e).__name__}: {e}")
            return f"[Analysis error: {type(e).__name__}: {e}]"
        finally:
            self._cleanup_scaled(clip_path, scaled_path)
