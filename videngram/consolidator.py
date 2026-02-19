"""
VidEngram Consolidator
HippoMM-inspired memory consolidation pipeline.

Raw captions → Consolidated segment memories + Episode summaries + Entity profiles

This is a KEY novelty component. Instead of dumping raw captions directly
into EverMemOS (which would just be naive RAG), we perform three stages of
consolidation inspired by hippocampal memory formation:

1. FILTER: Remove near-duplicate segments (pattern separation)
2. EPISODE: Group related segments into episode summaries (consolidation)
3. PROFILE: Extract recurring entities/themes across episodes (semantic replay)

This creates a HIERARCHICAL memory structure that EverMemOS can leverage
for multi-hop reasoning — it's not just searching captions, but understanding
narrative arcs and entity relationships across the entire video.
"""
import logging
import re
from typing import Optional

from openai import OpenAI

from .config import VidEngramConfig
from .utils import Caption, ConsolidatedMemory, generate_id, fmt_minutes

logger = logging.getLogger("videngram.consolidator")

EPISODE_SUMMARY_PROMPT = """\
You are summarizing a group of consecutive video segments into a single episode.
Create a coherent episode summary that:
1. Captures the main narrative arc or theme
2. Notes key participants and their roles
3. Highlights important information, decisions, or events
4. Preserves temporal flow (what happened first, then, finally)
5. Notes any recurring elements or callbacks to earlier moments

Segment captions:
{segment_texts}

Write a concise episode summary (3-5 sentences). Start with the time range \
and main theme, then describe the progression."""

PROFILE_EXTRACTION_PROMPT = """\
Analyze these episode summaries from a video and extract entity profiles.
For each recurring person, topic, or concept, create a brief profile.

Episode summaries:
{episode_texts}

For each entity, provide:
- Name/identifier
- Role or significance in the video
- Key attributes, opinions, or behaviors observed
- How they evolve or change across the video

Format each profile as: "ENTITY: [name] — [profile description]"
List the most significant entities first. Maximum 5 entities."""


class Consolidator:
    """Transforms raw captions into hierarchical consolidated memories."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.consolidator
        self.qwen_cfg = config.qwen

        # Use GPT (via LiteLLM) for text-only tasks if configured,
        # otherwise fall back to Qwen2.5-Omni
        agent_cfg = config.agent
        has_text_llm = (
            agent_cfg.planning_llm_base_url
            and "localhost:8091" not in agent_cfg.planning_llm_base_url
        )
        if has_text_llm:
            self.text_client = OpenAI(
                base_url=agent_cfg.planning_llm_base_url,
                api_key=agent_cfg.planning_llm_api_key,
            )
            self.text_model = agent_cfg.planning_llm_model
            self._text_extra = {}
            logger.info(f"Consolidator using external LLM for text tasks: {self.text_model}")
        else:
            self.text_client = OpenAI(
                base_url=self.qwen_cfg.base_url,
                api_key=self.qwen_cfg.api_key,
            )
            self.text_model = self.qwen_cfg.model
            self._text_extra = {"extra_body": {"modalities": ["text"]}}
            logger.info("Consolidator using Qwen for text tasks (no external LLM configured)")

    def consolidate(
        self, captions: list[Caption]
    ) -> list[ConsolidatedMemory]:
        """Full consolidation pipeline: filter → episode → profile.

        Returns a flat list of ConsolidatedMemory objects ready for
        EverMemOS ingestion, including:
        - Individual segment memories (after dedup filtering)
        - Episode summaries spanning multiple segments
        - Entity profiles (if build_profiles=True)
        """
        if not captions:
            return []

        logger.info(f"Consolidating {len(captions)} captions")

        # Stage 1: Filter near-duplicate segments
        filtered = self._filter_duplicates(captions)
        logger.info(f"  After dedup: {len(filtered)} segments (removed {len(captions) - len(filtered)})")

        # Stage 2: Create per-segment memories (enriched with timestamp prefix)
        segment_memories = self._create_segment_memories(filtered)

        # Stage 3: Group into episodes and generate episode summaries
        episode_memories = self._create_episode_summaries(filtered)
        logger.info(f"  Created {len(episode_memories)} episode summaries")

        # Stage 4: Extract entity profiles across episodes
        profile_memories = []
        if self.cfg.build_profiles and episode_memories:
            profile_memories = self._extract_profiles(episode_memories)
            logger.info(f"  Extracted {len(profile_memories)} entity profiles")

        all_memories = segment_memories + episode_memories + profile_memories
        logger.info(f"  Total consolidated memories: {len(all_memories)}")
        return all_memories

    # ── Stage 1: Deduplication ────────────────────────────────────────

    def _filter_duplicates(self, captions: list[Caption]) -> list[Caption]:
        """Remove near-duplicate captions using simple text similarity.

        Mirrors HippoMM's consolidation step where similar ShortTermMemory
        objects are filtered using cosine similarity on ImageBind embeddings.
        Here we use a simpler approach: Jaccard similarity on word sets.
        (In production, you'd use embeddings — but this avoids an extra
        model dependency and works surprisingly well for caption text.)
        """
        if len(captions) <= 1:
            return list(captions)

        threshold = self.cfg.merge_similarity_threshold
        # Work on copies to avoid mutating originals
        from dataclasses import replace
        filtered = [replace(captions[0])]

        for cap in captions[1:]:
            # Compare with previous caption (adjacent dedup)
            sim = self._text_similarity(filtered[-1].raw_text, cap.raw_text)
            if sim < threshold:
                filtered.append(replace(cap))
            else:
                # Merge: extend previous caption's time range (on our copy)
                logger.debug(
                    f"  Merging segment {cap.segment_id} into "
                    f"{filtered[-1].segment_id} (sim={sim:.2f})"
                )
                filtered[-1].end_sec = cap.end_sec

        return filtered

    @staticmethod
    def _text_similarity(text_a: str, text_b: str) -> float:
        """Jaccard similarity on word sets."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)

    # ── Stage 2: Segment Memories ─────────────────────────────────────

    def _create_segment_memories(
        self, captions: list[Caption]
    ) -> list[ConsolidatedMemory]:
        """Create enriched segment-level memories.

        Each caption gets a timestamp prefix so EverMemOS can leverage it
        during retrieval. The prefix format matches what the agent's
        context-grounding module expects.
        """
        memories = []
        for cap in captions:
            # Prefix with video timestamp for grounding
            content = (
                f"[Video {fmt_minutes(cap.start_sec)} - {fmt_minutes(cap.end_sec)}] "
                f"{cap.raw_text}"
            )
            memories.append(ConsolidatedMemory(
                memory_id=generate_id("mem", cap.segment_id),
                content=content,
                start_sec=cap.start_sec,
                end_sec=cap.end_sec,
                memory_type="segment",
                source_segments=[cap.segment_id],
            ))
        return memories

    # ── Stage 3: Episode Summaries ────────────────────────────────────

    def _create_episode_summaries(
        self, captions: list[Caption]
    ) -> list[ConsolidatedMemory]:
        """Group segments into episodes and generate LLM summaries.

        This mirrors HippoMM's ThetaEvent creation: multiple ShortTermMemory
        objects are grouped and a VLM generates an abstract semantic summary.
        Here we use Qwen2.5-Omni (text mode) to summarize caption groups.
        """
        max_per_episode = self.cfg.episode_max_segments
        episodes = []

        # Group captions into chunks
        for chunk_start in range(0, len(captions), max_per_episode):
            chunk = captions[chunk_start:chunk_start + max_per_episode]
            if not chunk:
                continue

            start_sec = chunk[0].start_sec
            end_sec = chunk[-1].end_sec
            segment_ids = [c.segment_id for c in chunk]

            # Build segment text block for the LLM
            segment_texts = "\n\n".join(
                f"[{fmt_minutes(c.start_sec)} - {fmt_minutes(c.end_sec)}]: {c.raw_text}"
                for c in chunk
            )

            # Generate episode summary via LLM (GPT or Qwen text mode)
            try:
                response = self.text_client.chat.completions.create(
                    model=self.text_model,
                    messages=[
                        {"role": "system", "content": "You are a precise video summarizer."},
                        {"role": "user", "content": EPISODE_SUMMARY_PROMPT.format(
                            segment_texts=segment_texts
                        )},
                    ],
                    max_tokens=512,
                    temperature=0.3,
                    **self._text_extra,
                )
                summary = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Episode summary generation failed: {e}")
                # Fallback: concatenate first sentences of each caption
                summary = " ".join(
                    c.raw_text.split(".")[0] + "." for c in chunk
                )

            content = (
                f"[Episode {fmt_minutes(start_sec)} - {fmt_minutes(end_sec)}] "
                f"{summary}"
            )

            episodes.append(ConsolidatedMemory(
                memory_id=generate_id("ep", start_sec, end_sec),
                content=content,
                start_sec=start_sec,
                end_sec=end_sec,
                memory_type="episode_summary",
                source_segments=segment_ids,
            ))

        return episodes

    # ── Stage 4: Entity Profiles ──────────────────────────────────────

    def _extract_profiles(
        self, episodes: list[ConsolidatedMemory]
    ) -> list[ConsolidatedMemory]:
        """Extract recurring entity profiles from episode summaries.

        Mirrors HippoMM's cross-episode entity tracking and EverMemOS's
        profile building. Identifies people, topics, and concepts that
        appear across multiple episodes.
        """
        episode_texts = "\n\n".join(
            f"Episode ({fmt_minutes(ep.start_sec)}-{fmt_minutes(ep.end_sec)}): "
            f"{ep.content}"
            for ep in episodes
        )

        try:
            response = self.text_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "You extract entity profiles from video summaries."},
                    {"role": "user", "content": PROFILE_EXTRACTION_PROMPT.format(
                        episode_texts=episode_texts
                    )},
                ],
                max_tokens=1024,
                temperature=0.3,
                **self._text_extra,
            )
            profile_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Profile extraction failed: {e}")
            return []

        # Parse "ENTITY: name — description" format
        profiles = []
        for line in profile_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Try to parse structured format
            match = re.match(r"(?:ENTITY:\s*)?(.+?)\s*[—–-]\s*(.+)", line)
            if match:
                entity_name = match.group(1).strip()
                description = match.group(2).strip()
                content = f"[Entity Profile: {entity_name}] {description}"
                profiles.append(ConsolidatedMemory(
                    memory_id=generate_id("prof", entity_name),
                    content=content,
                    start_sec=episodes[0].start_sec,
                    end_sec=episodes[-1].end_sec,
                    memory_type="entity_profile",
                    source_segments=[],
                    metadata={"entity_name": entity_name},
                ))

        return profiles
