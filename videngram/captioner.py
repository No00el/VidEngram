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
from pathlib import Path
from typing import Optional

from openai import OpenAI, AsyncOpenAI

from .config import VidEngramConfig
from .utils import VideoSegment, Caption, fmt_minutes

logger = logging.getLogger("videngram.captioner")

CAPTION_SYSTEM_PROMPT = """\
You are a precise video analyst. For each video segment, produce a structured \
caption covering ALL of the following aspects. Be specific and factual. \
Include timestamps relative to the segment when notable events occur.

Required fields:
- SCENE: Physical setting, lighting, camera angle
- PEOPLE: Who is visible, their appearance, expressions, actions
- DIALOGUE/SPEECH: Transcribe any spoken words verbatim (mark speaker if identifiable)
- SOUNDS: Background music, sound effects, ambient noise
- TEXT: Any visible text, signs, titles, captions on screen
- OBJECTS: Notable objects, tools, props
- EMOTION: Overall emotional tone or mood
- TEMPORAL: What changes during this segment, any cause-effect relationships

Format your response as a cohesive paragraph (not bullet points). \
Start with the scene, then describe what happens chronologically. \
Weave in all the above aspects naturally. Be concise but thorough."""

CAPTION_USER_TEMPLATE = """\
Analyze this video segment ({timestamp_label}, duration {duration:.0f}s) \
in detail. This is segment {seg_index} of {total_segments} from a longer video. \
Provide a comprehensive structured caption covering scene, people, dialogue, \
sounds, text, objects, emotion, and temporal progression."""


class Captioner:
    """Generates rich captions from video segments using Qwen2.5-Omni."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.captioner
        self.qwen_cfg = config.qwen
        self.client = OpenAI(
            base_url=self.qwen_cfg.base_url,
            api_key=self.qwen_cfg.api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.qwen_cfg.base_url,
            api_key=self.qwen_cfg.api_key,
        )

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
        if not segment.clip_path or not Path(segment.clip_path).exists():
            logger.warning(f"No clip for segment {segment.segment_id}, skipping")
            return Caption(
                segment_id=segment.segment_id,
                raw_text="[Clip unavailable]",
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
            )

        user_text = CAPTION_USER_TEMPLATE.format(
            timestamp_label=segment.timestamp_label,
            duration=segment.duration,
            seg_index=seg_index + 1,
            total_segments=total_segments,
        )

        # Build multimodal content array
        content = [
            # Video input via local file URL
            {
                "type": "video_url",
                "video_url": {"url": f"file://{segment.clip_path}"},
            },
            {"type": "text", "text": user_text},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=self.qwen_cfg.max_tokens,
                temperature=self.qwen_cfg.temperature,
                # Text-only output — skip audio generation for speed
                extra_body={"modalities": self.qwen_cfg.modalities},
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Caption failed for {segment.segment_id}: {e}")
            raw_text = f"[Caption error: {e}]"

        return Caption(
            segment_id=segment.segment_id,
            raw_text=raw_text,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )

    def caption_all(
        self, segments: list[VideoSegment], parallel: bool = False
    ) -> list[Caption]:
        """Caption all segments, optionally in parallel."""
        total = len(segments)
        logger.info(f"Captioning {total} segments (parallel={parallel})")

        if parallel:
            return asyncio.run(self._caption_all_async(segments))

        captions = []
        for i, seg in enumerate(segments):
            logger.info(f"  Captioning [{i+1}/{total}] {seg.timestamp_label}")
            cap = self.caption_segment(seg, seg_index=i, total_segments=total)
            captions.append(cap)

        return captions

    async def _caption_all_async(
        self, segments: list[VideoSegment]
    ) -> list[Caption]:
        """Caption segments concurrently (bounded concurrency)."""
        semaphore = asyncio.Semaphore(4)  # Max 4 concurrent requests
        total = len(segments)

        async def _caption_one(i: int, seg: VideoSegment) -> Caption:
            async with semaphore:
                logger.info(f"  [async] Captioning [{i+1}/{total}]")
                return await self._async_caption_segment(seg, i, total)

        tasks = [_caption_one(i, seg) for i, seg in enumerate(segments)]
        return await asyncio.gather(*tasks)

    async def _async_caption_segment(
        self,
        segment: VideoSegment,
        seg_index: int,
        total_segments: int,
    ) -> Caption:
        """Async version of caption_segment."""
        if not segment.clip_path or not Path(segment.clip_path).exists():
            return Caption(
                segment_id=segment.segment_id,
                raw_text="[Clip unavailable]",
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
            )

        user_text = CAPTION_USER_TEMPLATE.format(
            timestamp_label=segment.timestamp_label,
            duration=segment.duration,
            seg_index=seg_index + 1,
            total_segments=total_segments,
        )

        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"file://{segment.clip_path}"},
            },
            {"type": "text", "text": user_text},
        ]

        try:
            response = await self.async_client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=self.qwen_cfg.max_tokens,
                temperature=self.qwen_cfg.temperature,
                extra_body={"modalities": self.qwen_cfg.modalities},
            )
            raw_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Async caption failed for {segment.segment_id}: {e}")
            raw_text = f"[Caption error: {e}]"

        return Caption(
            segment_id=segment.segment_id,
            raw_text=raw_text,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
        )

    def analyze_clip(self, clip_path: str, query: str) -> str:
        """On-demand analysis: send a specific query about a video clip.

        Used by the agentic orchestrator for video grounding — when the
        agent decides it needs to "look at" a specific moment in the video
        to answer a query.
        """
        content = [
            {
                "type": "video_url",
                "video_url": {"url": f"file://{clip_path}"},
            },
            {"type": "text", "text": query},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.qwen_cfg.model,
                messages=[
                    {"role": "system", "content": "You are a precise video analyst. Answer the question based only on what you observe in the video."},
                    {"role": "user", "content": content},
                ],
                max_tokens=self.qwen_cfg.max_tokens,
                temperature=0.2,
                extra_body={"modalities": ["text"]},
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Clip analysis failed: {e}")
            return f"[Analysis error: {e}]"
