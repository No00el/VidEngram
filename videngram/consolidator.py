"""
VidEngram Consolidator
HippoMM-inspired memory consolidation pipeline.

Raw captions → Consolidated segment memories + Episode summaries + Entity Register

This is a KEY novelty component. Instead of dumping raw captions directly
into EverMemOS (which would just be naive RAG), we perform three stages of
consolidation inspired by hippocampal memory formation:

1. FILTER: Remove near-duplicate segments (pattern separation)
2. EPISODE: Group related segments into episode summaries (consolidation)
3. ENTITY REGISTER: Two-step entity resolution across episodes (semantic replay)
   - Step A: Extract entity mentions from each episode individually
   - Step B: Resolve/merge cross-episode mentions into unified entity entries

This creates a HIERARCHICAL memory structure that EverMemOS can leverage
for multi-hop reasoning — it's not just searching captions, but understanding
narrative arcs and entity relationships across the entire video.
"""
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from openai import OpenAI

from .config import VidEngramConfig
from .utils import Caption, ConsolidatedMemory, generate_id, fmt_minutes

logger = logging.getLogger("videngram.consolidator")

EPISODE_BOUNDARY_PROMPT = """\
You are analyzing consecutive video segments to determine episode boundaries.

Current episode so far:
{current_segments}

New incoming segment:
{new_segment}

Does the new segment continue the current episode, or does it start a new episode?
Start a new episode if the new segment shows a clear topic/theme shift, a new scene \
or location, a natural narrative break, or a significant change in participants or activity.

Respond with exactly one word: CONTINUE or NEW_EPISODE"""

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

ENTITY_EXTRACTION_PROMPT = """\
Extract all persons and notable entities mentioned in this episode summary.

Episode:
{episode_text}

For each person or entity present, output one line in this exact format:
ENTITY: [name or description] | TIMES: [time range from the episode, e.g. "1:12-3:30"] | ROLE: [brief description of what they do or who they are]

Use the time range that is already present in the episode text.
Only include actual persons or named entities (not generic objects or abstract topics).
If no persons or entities are present, output: NO_ENTITIES"""

ENTITY_RESOLUTION_PROMPT = """\
The following persons and entities were extracted from different episodes of the same video.
Some entries may refer to the same person described with different names or descriptions.

{entity_list}

Merge entries that refer to the same person. For each unique person or entity, output a block using this exact format:

ENTITY: [canonical name or clearest identifier]
ALIASES: [all other names/descriptions used across episodes, comma-separated, or "none"]
APPEARS: [all time ranges from all episodes where they appear, comma-separated, e.g. "0:30-2:18, 5:06-8:12"]
PROFILE: [unified description covering their role, appearance, behaviors, and key moments across the video]
---

Rules:
- Only merge entries you are confident refer to the same person
- When uncertain, keep them as separate entries
- Choose the most specific and recognizable identifier as the canonical name
- Include every time range where this entity appears"""


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

    _ERROR_PREFIXES = ("[Caption error", "[Clip unavailable]", "[Analysis error")

    @staticmethod
    def make_segment_memory(cap: Caption) -> Optional[ConsolidatedMemory]:
        """Create a single segment memory from one caption (no dedup).

        Returns None for error/placeholder captions.
        Used by the pipeline for streaming ingest: as each caption arrives,
        its segment memory is written to EverMemOS immediately without waiting
        for the rest of the captions to complete.
        """
        if cap.raw_text.startswith(Consolidator._ERROR_PREFIXES):
            return None
        content = (
            f"[Video {fmt_minutes(cap.start_sec)} - {fmt_minutes(cap.end_sec)}] "
            f"{cap.raw_text}"
        )
        return ConsolidatedMemory(
            memory_id=generate_id("mem", cap.segment_id),
            content=content,
            start_sec=cap.start_sec,
            end_sec=cap.end_sec,
            memory_type="segment",
            source_segments=[cap.segment_id],
        )

    def consolidate(
        self,
        captions: list[Caption],
        include_segments: bool = True,
    ) -> list[ConsolidatedMemory]:
        """Full consolidation pipeline: filter → episode → entity register.

        Returns a flat list of ConsolidatedMemory objects ready for
        EverMemOS ingestion, including:
        - Individual segment memories (after dedup filtering) — omitted when
          include_segments=False, e.g. when already streamed via
          make_segment_memory() during captioning.
        - Episode summaries spanning multiple segments
        - Entity register entries (if build_profiles=True)
        """
        if not captions:
            return []

        # Drop error/placeholder captions before consolidation so they are
        # never stored in EverMemOS (they contain no real video information).
        valid = [c for c in captions
                 if not c.raw_text.startswith(self._ERROR_PREFIXES)]
        if len(valid) < len(captions):
            logger.warning(
                f"Skipped {len(captions) - len(valid)} error captions "
                f"({len(valid)} valid remain)"
            )
        captions = valid
        if not captions:
            logger.warning("All captions were errors — nothing to consolidate")
            return []

        logger.info(f"Consolidating {len(captions)} captions")

        # Stage 1: Filter near-duplicate segments
        filtered = self._filter_duplicates(captions)
        logger.info(f"  After dedup: {len(filtered)} segments (removed {len(captions) - len(filtered)})")

        # Stage 2: Create per-segment memories (enriched with timestamp prefix)
        # Skipped when include_segments=False (already streamed during captioning)
        segment_memories = self._create_segment_memories(filtered) if include_segments else []

        # Stage 3: Group into episodes and generate episode summaries
        episode_memories = self._create_episode_summaries(filtered)
        logger.info(f"  Created {len(episode_memories)} episode summaries")

        # Stage 4: Build entity register via two-step entity resolution
        register_memories = []
        if self.cfg.build_profiles and episode_memories:
            register_memories = self._build_entity_register(episode_memories)
            logger.info(f"  Built {len(register_memories)} entity register entries")

        all_memories = segment_memories + episode_memories + register_memories
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
        """Group segments into model-determined episodes and generate LLM summaries.

        Instead of a fixed segment count, the model incrementally decides after
        each new segment whether to continue the current episode or start a new one.

        Hard constraints:
        - episode_min_segments: buffer must reach this size before the model is
          asked. Default=1 means the model is consulted from the 2nd segment on.
        - episode_max_segments: forces a split regardless of model decision.
        - If the final buffer is below episode_min_segments AND merging into
          the preceding episode wouldn't exceed max_seg, it is merged in.
          With the default min=1 this condition never triggers (non-empty
          buffer always has len >= 1 = min_seg), so the trailing chunk always
          becomes its own episode.
        """
        min_seg = self.cfg.episode_min_segments
        max_seg = self.cfg.episode_max_segments

        chunks: list[list[Caption]] = []
        current_buffer: list[Caption] = []

        for cap in captions:
            if len(current_buffer) < min_seg:
                # Haven't hit the minimum yet — just accumulate
                current_buffer.append(cap)
            elif len(current_buffer) >= max_seg:
                # Hit the hard upper bound — force a split
                logger.debug(
                    f"  Episode hard split at {len(current_buffer)} segments "
                    f"(max={max_seg})"
                )
                chunks.append(current_buffer)
                current_buffer = [cap]
            else:
                # Ask the model whether this segment belongs to the current episode
                if self._should_start_new_episode(current_buffer, cap):
                    logger.debug(
                        f"  Model decided new episode after "
                        f"{fmt_minutes(current_buffer[-1].end_sec)} "
                        f"({len(current_buffer)} segments)"
                    )
                    chunks.append(current_buffer)
                    current_buffer = [cap]
                else:
                    current_buffer.append(cap)

        # Flush the remaining buffer
        if current_buffer:
            if len(current_buffer) < min_seg and chunks:
                # Too few segments — merge into the previous episode
                logger.debug(
                    f"  Final buffer has only {len(current_buffer)} segment(s) "
                    f"(< min={min_seg}); merging into previous episode"
                )
                chunks[-1].extend(current_buffer)
            else:
                chunks.append(current_buffer)

        if not chunks:
            return []
        with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
            # map() preserves order, so episode chronology is maintained
            return list(pool.map(self._summarize_chunk, chunks))

    def _should_start_new_episode(
        self, buffer: list[Caption], new_cap: Caption
    ) -> bool:
        """Ask the model whether new_cap starts a new episode or continues the current one."""
        current_segments = "\n\n".join(
            f"[{fmt_minutes(c.start_sec)} - {fmt_minutes(c.end_sec)}]: {c.raw_text}"
            for c in buffer
        )
        new_segment = (
            f"[{fmt_minutes(new_cap.start_sec)} - {fmt_minutes(new_cap.end_sec)}]: "
            f"{new_cap.raw_text}"
        )
        try:
            response = self.text_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "You are a video structure analyst."},
                    {"role": "user", "content": EPISODE_BOUNDARY_PROMPT.format(
                        current_segments=current_segments,
                        new_segment=new_segment,
                    )},
                ],
                max_tokens=8,
                temperature=0.0,
                **self._text_extra,
            )
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith("NEW")
        except Exception as e:
            logger.warning(f"Episode boundary detection failed: {e}; defaulting to CONTINUE")
            return False

    def _summarize_chunk(self, chunk: list[Caption]) -> ConsolidatedMemory:
        """Generate an episode summary for a group of captions."""
        start_sec = chunk[0].start_sec
        end_sec = chunk[-1].end_sec
        segment_ids = [c.segment_id for c in chunk]

        segment_texts = "\n\n".join(
            f"[{fmt_minutes(c.start_sec)} - {fmt_minutes(c.end_sec)}]: {c.raw_text}"
            for c in chunk
        )

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
            summary = " ".join(c.raw_text.split(".")[0] + "." for c in chunk)

        content = (
            f"[Episode {fmt_minutes(start_sec)} - {fmt_minutes(end_sec)}] "
            f"{summary}"
        )
        return ConsolidatedMemory(
            memory_id=generate_id("ep", start_sec, end_sec),
            content=content,
            start_sec=start_sec,
            end_sec=end_sec,
            memory_type="episode_summary",
            source_segments=segment_ids,
        )

    # ── Stage 4: Entity Register ──────────────────────────────────────

    def _build_entity_register(
        self, episodes: list[ConsolidatedMemory]
    ) -> list[ConsolidatedMemory]:
        """Two-step entity resolution: extract per episode, then resolve across episodes.

        Step A: For each episode summary, extract all entity mentions (name, time, role).
        Step B: Feed all mentions to LLM to merge cross-episode references into
                unified entity_register entries with canonical names and full time ranges.
        """
        # Step A: Extract entities from each episode in parallel (episodes are independent)
        with ThreadPoolExecutor(max_workers=min(4, len(episodes))) as pool:
            all_episode_entities: list[list[dict]] = list(
                pool.map(self._extract_entities_from_episode, episodes)
            )
        for ep, entities in zip(episodes, all_episode_entities):
            logger.debug(
                f"  Episode [{fmt_minutes(ep.start_sec)}-{fmt_minutes(ep.end_sec)}]: "
                f"{len(entities)} entity mention(s)"
            )

        total_raw = sum(len(e) for e in all_episode_entities)
        if total_raw == 0:
            logger.info("  No entity mentions found in any episode")
            return []

        logger.info(
            f"  Extracted {total_raw} raw entity mentions across "
            f"{len(episodes)} episodes — resolving..."
        )

        # Step B: Resolve and merge across all episodes
        return self._resolve_entities(all_episode_entities, episodes)

    def _extract_entities_from_episode(
        self, episode: ConsolidatedMemory
    ) -> list[dict]:
        """Extract entity mentions from a single episode summary.

        Returns a list of dicts with keys: name, times, role.
        """
        try:
            response = self.text_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract person and entity mentions from video episode summaries.",
                    },
                    {
                        "role": "user",
                        "content": ENTITY_EXTRACTION_PROMPT.format(
                            episode_text=episode.content
                        ),
                    },
                ],
                max_tokens=512,
                temperature=0.1,
                **self._text_extra,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(
                f"Entity extraction failed for episode "
                f"[{fmt_minutes(episode.start_sec)}-{fmt_minutes(episode.end_sec)}]: {e}"
            )
            return []

        if "NO_ENTITIES" in raw.upper():
            return []

        entities = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or "ENTITY:" not in line:
                continue
            match = re.match(
                r"ENTITY:\s*(.+?)\s*\|\s*TIMES:\s*(.+?)\s*\|\s*ROLE:\s*(.+)",
                line,
            )
            if match:
                entities.append({
                    "name": match.group(1).strip(),
                    "times": match.group(2).strip(),
                    "role": match.group(3).strip(),
                })
        return entities

    def _resolve_entities(
        self,
        all_episode_entities: list[list[dict]],
        episodes: list[ConsolidatedMemory],
    ) -> list[ConsolidatedMemory]:
        """Merge and resolve entity mentions across all episodes.

        Feeds all per-episode entity mentions to the LLM to group references
        to the same person and produce unified entity_register entries.
        """
        lines = []
        for ep, ep_entities in zip(episodes, all_episode_entities):
            for ent in ep_entities:
                lines.append(
                    f"Episode [{fmt_minutes(ep.start_sec)}-{fmt_minutes(ep.end_sec)}] "
                    f"| ENTITY: {ent['name']} | TIMES: {ent['times']} | ROLE: {ent['role']}"
                )

        if not lines:
            return []

        try:
            response = self.text_client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You resolve and unify entity identities across video episode descriptions.",
                    },
                    {
                        "role": "user",
                        "content": ENTITY_RESOLUTION_PROMPT.format(
                            entity_list="\n".join(lines)
                        ),
                    },
                ],
                max_tokens=2048,
                temperature=0.1,
                **self._text_extra,
            )
            resolved_text = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Entity resolution failed: {e}")
            return []

        return self._parse_entity_register(resolved_text, episodes)

    def _parse_entity_register(
        self,
        resolved_text: str,
        episodes: list[ConsolidatedMemory],
    ) -> list[ConsolidatedMemory]:
        """Parse the LLM's resolved entity output into ConsolidatedMemory objects."""
        registers = []

        # Split on the --- separator between entity blocks
        blocks = re.split(r"\n?---\n?", resolved_text)

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            entity_match = re.search(r"^ENTITY:\s*(.+)$", block, re.MULTILINE)
            aliases_match = re.search(r"^ALIASES:\s*(.+)$", block, re.MULTILINE)
            appears_match = re.search(r"^APPEARS:\s*(.+)$", block, re.MULTILINE)

            # PROFILE may span multiple lines — capture everything after "PROFILE:"
            profile_idx = block.find("PROFILE:")
            profile_desc = (
                block[profile_idx + len("PROFILE:"):].strip()
                if profile_idx >= 0
                else ""
            )

            if not entity_match or not profile_desc:
                logger.debug(f"  Skipping incomplete entity block: {block[:80]!r}")
                continue

            canonical_name = entity_match.group(1).strip()
            aliases_str = aliases_match.group(1).strip() if aliases_match else "none"
            appears_str = appears_match.group(1).strip() if appears_match else ""

            # Parse time ranges from APPEARS to get overall start/end
            time_pairs = re.findall(r"(\d+\.?\d*)min\s*-\s*(\d+\.?\d*)min", appears_str)
            if time_pairs:
                start_sec = min(float(p[0]) * 60 for p in time_pairs)
                end_sec = max(float(p[1]) * 60 for p in time_pairs)
            else:
                start_sec = episodes[0].start_sec
                end_sec = episodes[-1].end_sec

            # Build aliases list (filtering placeholder values)
            aliases = [
                a.strip()
                for a in aliases_str.split(",")
                if a.strip().lower() not in ("none", "")
            ]

            # Build EverMemOS content: header with canonical name + time ranges, then profile
            appears_label = appears_str if appears_str else (
                f"{fmt_minutes(start_sec)}-{fmt_minutes(end_sec)}"
            )
            alias_clause = f" Also known as: {', '.join(aliases)}." if aliases else ""
            content = (
                f"[Entity Register: {canonical_name} | {appears_label}] "
                f"{profile_desc}{alias_clause}"
            )

            registers.append(ConsolidatedMemory(
                memory_id=generate_id("reg", canonical_name),
                content=content,
                start_sec=start_sec,
                end_sec=end_sec,
                memory_type="entity_register",
                source_segments=[],
                metadata={
                    "entity_name": canonical_name,
                    "aliases": aliases,
                    "appearances": appears_str,
                },
            ))

        logger.info(f"  Parsed {len(registers)} entity register entries")
        return registers
