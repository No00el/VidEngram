"""
VidEngram Agent — Agentic Query Orchestrator
═══════════════════════════════════════════════════════════════

This is the CORE NOVELTY of VidEngram. Instead of a simple retrieve→answer
pipeline, the agent uses a ReAct-style reasoning loop with multiple tools:

  THINK → ACT (choose tool) → OBSERVE → THINK → ... → ANSWER

Available tools:
  1. search_episodes  — Search video episode memories (fast, lightweight)
  2. search_profiles  — Look up entity/speaker profiles
  3. search_deep      — Agentic multi-hop retrieval (thorough, slower)
  4. look_at_clip    — Extract + analyze a specific video moment (grounding)
  5. get_timeline     — List all events in a time range

This makes VidEngram truly AGENTIC:
  - The agent PLANS which tools to use based on query complexity
  - It can COMBINE memory retrieval with direct video analysis
  - It ITERATES if initial retrieval is insufficient
  - It GROUNDS answers in specific timestamps (context-grounding)

Design highlights:
  - Not just RAG over captions — it's a reasoning agent with tool access
  - Combines EverMemOS structured memory with on-demand video understanding
  - Hippocampal dual-pathway: fast retrieval first, then deep recall if needed
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import VidEngramConfig
from .utils import (
    AgentAction, AgentResponse, MemoryResult,
    extract_clip, fmt_minutes,
)
from .memory_reader import MemoryReader
from .captioner import Captioner

logger = logging.getLogger("videngram.agent")

PLANNING_SYSTEM_PROMPT = """\
You are VidEngram, an intelligent video memory assistant. You answer questions \
about a video by using your memory system and video analysis tools.

You have access to these tools:

1. search_episodes(query) → Search video memories for relevant episodes/segments.
   Returns text snippets with [Video M:SS - N:SS] timestamps.
   USE FOR: factual questions, "what happened", finding specific moments.

2. search_speech(query) → BM25 keyword search over Whisper speech transcripts.
   Returns exact spoken words with [Video M:SS - N:SS] timestamps.
   USE FOR: "when did speaker say X", finding exact words/phrases, keyword lookup.
   PREFER over search_episodes for word-exact or temporal queries.

3. search_profiles(query) → Look up the entity register: unified profiles of people/entities
   identified across the video, including all time ranges they appear.
   USE FOR: "who is", "tell me about", character/speaker questions, "when does X appear".

4. search_deep(query) → Deep multi-hop search across all memory types.
   SLOWER but more thorough. USE FOR: complex questions requiring reasoning
   across multiple video segments, cross-references, temporal logic.

5. look_at_clip(start_min, end_min, question) → Extract and analyze a specific
   video moment. Sends the actual clip to the vision model for fresh analysis.
   USE FOR: when memory doesn't have enough detail, visual verification,
   "what exactly does X look like", counting objects, reading text on screen.

6. get_timeline(start_min, end_min) → List all memorized events in a time range.
   USE FOR: "what happened between X and Y", chronological questions.

## Instructions:

### Core principle — answer as soon as the evidence allows; do NOT over-search.
After EVERY tool result, FIRST ask yourself: "Does what I already have let me answer the question?"
If yes, output ANSWER immediately — do NOT call more tools just to be thorough. One or two
well-targeted searches answer most questions; calling every tool is a mistake, not diligence.
For MULTIPLE-CHOICE: the moment the evidence favors one option, commit with `ANSWER: X`. A confident
answer from partial evidence beats extra tool calls. Escalate to another tool ONLY when your current
results are empty or clearly do not contain the fact asked.

- Think step-by-step about what information you need
- For KEYWORD/TEMPORAL questions ("when did speaker say X", "at what time did Y happen"):
  ALWAYS start with search_speech(query) — it does BM25 exact-word search on transcripts.
- For FACTUAL/VISUAL questions: start with the single most relevant tool (search_episodes for
  facts/visuals, search_speech for exact spoken words). If its result already answers the question,
  ANSWER. Add a second tool ONLY if the first is empty or clearly misses the specific fact asked.
- For questions about specific spoken facts (exact book titles, tool names, costs, dates,
  abbreviation definitions): call search_speech with the key content words (not the
  speaker's name) — episode memories may paraphrase while speech has the exact wording.
- For overview questions ("what is the video about", "summarize the video", "what happens in the video"):
  use get_timeline(0, 999) first to get a full chronological event list, then synthesize an answer
- Only if your first search returns nothing useful, try ONE alternative (different query phrasing,
  search_profiles, search_deep, or get_timeline). Never chain extra tools once you already have an answer.
- Answer in the SAME LANGUAGE as the user's question
- ALWAYS cite timestamps for every factual claim about the video using EXACTLY this format: [Video M:SS - N:SS]
  where M:SS is minutes:seconds (e.g. [Video 0:35 - 1:20]).
  Use the EXACT timestamp range from the retrieved memory snippet — do NOT artificially shrink or extend it.
  NEVER invent or estimate timestamp values — only cite timestamps you actually retrieved from memory.
- Timestamp granularity rules:
  * For SPECIFIC-FACT questions (what did X say, why did Y happen, what does Z look like, etc.):
    - You MUST cite a [Video M:SS - N:SS] segment-level timestamp that pinpoints the exact moment.
    - [Episode M:SS - N:SS] timestamps span large portions of the video and are NOT acceptable for specific facts.
    - Prefer a precise [Video ...] segment timestamp. If you only have [Episode ...] results but they already
      support the answer, ANSWER with it. Escalate to another tool only when you have no usable evidence yet.
  * For OVERVIEW/SUMMARY questions (what is the video about, summarize, what happens):
    - Use the video's total duration as the single timestamp (provided in Video metadata above).
    - Or cite each major event with its own [Video ...] segment timestamp separately.
- Only when you still have NO usable evidence after searching may you call look_at_clip on the most
  relevant segment. If you already have partial evidence, prefer to ANSWER with it rather than add a tool call.
- For MULTIPLE-CHOICE questions: NEVER answer that you cannot determine or that information is
  unavailable. You MUST commit to the single most likely option using whatever partial evidence
  and reasoning you have — a best guess always beats refusing. End with `ANSWER: X`.
- For video-specific facts prefer the video memories over general knowledge
- Answer in plain prose. Do NOT use markdown formatting (no bullet points, no bold, no headers).
- When expanding abbreviations (RLVR, DCLM, GRPO, etc.): call search_speech with the
  abbreviation itself to find its definition. Use the definition verbatim from the retrieved
  speech. Do NOT guess from your general knowledge.
- For specific book titles mentioned in the video: call search_speech for the title words
  (not the author's name). Use the EXACT title from retrieved speech — do not substitute
  another book by the same author even if you know of one.
- Dates and times spoken aloud by speakers in the video ARE real facts and MUST be reported
  accurately when retrieved from speech transcripts (e.g. "January 2025", "five million dollars").
  Do NOT suppress or replace these with vague phrases.
- NEVER cite EverMemOS metadata timestamps (ISO format like "2025-12-16T00:00:00" or upload
  timestamps injected by the storage system) — these are not video event dates.

## Response Format:
For each step, output EXACTLY one of:

THINK: [your reasoning about what to do next]
ACTION: tool_name(param1, param2, ...)
ANSWER: [your final answer to the user]

Start with THINK, then alternate ACTION and THINK. End with ANSWER."""


class VidEngramAgent:
    """Agentic orchestrator for video understanding queries."""

    def __init__(self, config: VidEngramConfig):
        self.config = config
        self.agent_cfg = config.agent
        self.reader = MemoryReader(config)
        self.captioner = Captioner(config)
        self.llm = OpenAI(
            base_url=self.agent_cfg.planning_llm_base_url,
            api_key=self.agent_cfg.planning_llm_api_key,
        )

    def query(
        self,
        question: str,
        video_path: str,
        chat_history: Optional[list[dict]] = None,
        step_callback=None,
        video_duration: Optional[float] = None,
        search_scope: str = "current",
    ) -> AgentResponse:
        """Answer a question about a video using the agentic reasoning loop.

        Args:
            question: User's natural language question
            video_path: Path to the source video
            chat_history: Optional previous Q&A turns for multi-turn context
            video_duration: Total video duration in seconds (injected into system prompt)

        Returns:
            AgentResponse with answer, sources, actions taken, and grounded clips
        """
        logger.info(f"Agent query: '{question}'")
        actions: list[AgentAction] = []
        sources: list[MemoryResult] = []
        grounded_clips: list[str] = []

        # Build system prompt, prepending video metadata if duration is known
        system_prompt = PLANNING_SYSTEM_PROMPT
        if video_duration is not None:
            from .utils import _fmt_time
            duration_mmss = _fmt_time(video_duration)
            end_min_label = max(1, int(video_duration / 60) + 1)
            system_prompt = system_prompt.replace("get_timeline(0, 999)", f"get_timeline(0, {end_min_label})")
            system_prompt = (
                f"Video metadata: total duration is [Video 0:00 - {duration_mmss}]. "
                f"Do NOT mention the video duration in your answer unless the user explicitly asks about it.\n\n"
                + system_prompt
            )

        # Store scope for tool execution
        self._search_scope = search_scope
        from .memory_writer import MemoryWriter as _MW
        self._current_group_id = _MW._video_group_id(video_path)

        # --- FAST MODE: single-shot RAG (1 retrieval pass + 1 LLM call, no ReAct loop) ---
        # VE_FAST=1 skips the multi-step ReAct loop: retrieve episodes (+ exact-word
        # speech) deterministically, then answer in a single LLM call. Trades a bit of
        # accuracy for ~1 LLM call instead of 3-4 (much lower latency + cost).
        if os.getenv("VE_FAST", "0") == "1":
            if step_callback is not None:
                step_callback("Fast mode: single-shot retrieve + answer…")
            omit_gid = search_scope == "all"
            results = self.reader.search_episodes(
                question, video_path, mode=os.getenv("VE_EPISODES_MODE", "hybrid"),
                top_k=int(os.getenv("VE_TOPK_EPISODES", "15")), omit_group_id=omit_gid,
            ) or []
            try:
                sp = self.reader.search_speech_bm25(
                    question, video_path, top_k=5, omit_group_id=omit_gid,
                ) or []
                results = results + sp
            except Exception as e:
                logger.warning(f"fast-mode speech search failed: {e}")
            fast_sources = (
                self.reader.tag_cross_video_content(results, self._current_group_id)
                if results else []
            )
            return self._fallback_answer(question, video_path, [], fast_sources)

        # When scope is "all", tell the model it can access memories from ALL ingested videos
        if search_scope == "all":
            system_prompt = (
                "IMPORTANT: You are operating in ALL VIDEOS mode. Your memory tools "
                "search across ALL previously ingested videos, not just the current one. "
                "You MUST use your search tools — do NOT assume you lack access to other videos.\n\n"
            ) + system_prompt + (
                "\n\nALL VIDEOS MODE RULE (highest priority — overrides all earlier instructions):\n"
                "Memories retrieved from OTHER videos do NOT carry timestamps. "
                "Do NOT invent or cite any timestamp for content from other videos. "
                "Only use [Video M:SS - N:SS] format for the CURRENTLY LOADED video. "
                "For content from other videos, simply state the information without any timestamp citation."
            )

        # Build conversation for the planning LLM
        messages = [{"role": "system", "content": system_prompt}]

        # Add chat history for multi-turn context
        if chat_history:
            for turn in chat_history[-4:]:  # Last 4 turns to stay in context
                messages.append(turn)

        # User question
        messages.append({"role": "user", "content": question})

        # For summary/overview questions, pre-fetch all memories and inject as initial observation
        # so the agent has full context instead of relying on keyword search
        if self._is_summary_question(question):
            if step_callback is not None:
                step_callback("Fetching all video memories for overview…")
            broad_results = self.reader.search_episodes(
                "video scenes events dialogue actions", video_path, top_k=20,
                omit_group_id=(search_scope == "all"),
            )
            if broad_results:
                broad_results = self.reader.tag_cross_video_content(
                    broad_results, self._current_group_id
                )
                sources.extend(broad_results)
                obs = "OBSERVATION (pre-fetched overview):\n" + self._format_results(broad_results)
                messages.append({"role": "assistant", "content": "THINK: This is an overview question. I will use the pre-fetched timeline to synthesize a summary."})
                messages.append({"role": "user", "content": obs})

        # ReAct loop
        for iteration in range(self.agent_cfg.max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.agent_cfg.max_iterations}")

            # Get LLM's next step
            try:
                extra = {}
                if "qwen" in self.agent_cfg.planning_llm_model.lower():
                    extra["extra_body"] = {"modalities": ["text"]}
                if "gpt-5" in self.agent_cfg.planning_llm_model.lower():
                    extra["reasoning_effort"] = "minimal"
                response = self.llm.chat.completions.create(
                    model=self.agent_cfg.planning_llm_model,
                    messages=messages,
                    max_tokens=2048,
                    temperature=(1 if "gpt-5" in self.agent_cfg.planning_llm_model.lower() else 0.2),
                    **extra,
                )
                llm_output = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Planning LLM error: {type(e).__name__}: {e}", exc_info=True)
                return self._fallback_answer(question, video_path, actions, sources)

            messages.append({"role": "assistant", "content": llm_output})
            logger.debug(f"  LLM output: {llm_output[:200]}")

            # Parse the output for THINK / ACTION / ANSWER
            # Priority: ACTION first (execute tool), then ANSWER (return result)
            # This prevents premature termination when GPT outputs both in one turn
            action_match = re.search(
                r"ACTION:\s*(\w+)\(([^)]*)\)", llm_output
            )
            if action_match:
                tool_name = action_match.group(1)
                params_str = action_match.group(2)

                # Notify caller of this step
                if step_callback is not None:
                    step_desc = self._describe_step(tool_name, params_str)
                    step_callback(step_desc)

                # Execute the tool
                tool_result, new_sources, new_clips = self._execute_tool(
                    tool_name, params_str, video_path, video_duration
                )

                actions.append(AgentAction(
                    tool=tool_name,
                    input_params={"raw": params_str},
                    output=tool_result[:500],
                    reasoning=llm_output.split("ACTION:")[0].strip(),
                ))
                sources.extend(new_sources)
                grounded_clips.extend(new_clips)

                # Feed observation back to LLM
                observation = f"OBSERVATION:\n{tool_result}"
                messages.append({"role": "user", "content": observation})

            elif "ANSWER:" in llm_output:
                answer_text = self._normalize_timestamps(
                    llm_output.split("ANSWER:")[-1].strip()
                )
                # In All Videos mode, remove all timestamps from the answer
                if search_scope == "all":
                    answer_text = self._strip_all_timestamps(answer_text)
                answer_text = self._strip_calendar_dates_from_answer(answer_text)

                # --- Forced video grounding fallback ---
                # If look_at_clip hasn't been used yet and sources are insufficient,
                # intercept the answer, run look_at_clip, and re-ask the LLM.
                # Disabled in "all" mode — grounding the current video is wrong for cross-video queries.
                look_at_clip_used = any(a.tool == "look_at_clip" for a in actions)
                _force_look = os.getenv("VE_FORCE_LOOK", "0") == "1"
                if not look_at_clip_used and (_force_look or self._sources_insufficient(sources)) and search_scope != "all":
                    if not self.agent_cfg.enable_video_grounding:
                        logger.warning(
                            "Insufficient sources but video grounding is disabled "
                            "(enable_video_grounding=False) — answer may be incomplete."
                        )
                    else:
                        if step_callback is not None:
                            step_callback("Looking at video for more context…")
                        grounding = self._forced_video_grounding(
                            question, sources, video_path, video_duration
                        )
                        if grounding:
                            result_text, clip_path = grounding
                            if clip_path:
                                grounded_clips.append(clip_path)
                            actions.append(AgentAction(
                                tool="look_at_clip",
                                input_params={"raw": "auto_fallback"},
                                output=result_text[:500],
                                reasoning="[Auto-fallback: insufficient memory results]",
                            ))
                            messages.append({"role": "user", "content": (
                                f"OBSERVATION (forced video grounding):\n{result_text}\n\n"
                                "Revise your ANSWER using this visual information. "
                                "Output ANSWER: followed by your updated answer."
                            )})
                            try:
                                extra = {}
                                if "qwen" in self.agent_cfg.planning_llm_model.lower():
                                    extra["extra_body"] = {"modalities": ["text"]}
                                if "gpt-5" in self.agent_cfg.planning_llm_model.lower():
                                    extra["reasoning_effort"] = "minimal"
                                gr = self.llm.chat.completions.create(
                                    model=self.agent_cfg.planning_llm_model,
                                    messages=messages,
                                    max_tokens=2048,
                                    temperature=(1 if "gpt-5" in self.agent_cfg.planning_llm_model.lower() else 0.2),
                                    **extra,
                                )
                                gr_output = gr.choices[0].message.content.strip()
                                if "ANSWER:" in gr_output:
                                    answer_text = self._normalize_timestamps(
                                        gr_output.split("ANSWER:")[-1].strip()
                                    )
                                    if search_scope == "all":
                                        answer_text = self._strip_all_timestamps(answer_text)
                                    answer_text = self._strip_calendar_dates_from_answer(answer_text)
                            except Exception as e:
                                logger.warning(f"Forced grounding LLM re-run failed: {e}")
                # --- End forced grounding ---

                return AgentResponse(
                    answer=answer_text,
                    sources=sources,
                    actions=actions,
                    grounded_clips=grounded_clips,
                )

            elif "THINK:" in llm_output:
                # LLM is still thinking, continue
                messages.append({
                    "role": "user",
                    "content": "Continue. Use an ACTION or provide your ANSWER.",
                })
            else:
                # Treat as answer
                ans = self._normalize_timestamps(llm_output)
                if search_scope == "all":
                    ans = self._strip_all_timestamps(ans)
                ans = self._strip_calendar_dates_from_answer(ans)
                return AgentResponse(
                    answer=ans,
                    sources=sources,
                    actions=actions,
                    grounded_clips=grounded_clips,
                )

        # Max iterations reached — force an answer
        return self._fallback_answer(question, video_path, actions, sources)

    # ── Tool Execution ────────────────────────────────────────────────

    def _execute_tool(
        self, tool_name: str, params_str: str, video_path: str, video_duration: Optional[float] = None
    ) -> tuple[str, list[MemoryResult], list[str]]:
        """Execute a tool and return (result_text, new_sources, new_clips)."""
        params = self._parse_params(params_str)
        new_sources: list[MemoryResult] = []
        new_clips: list[str] = []

        try:
            omit_gid = getattr(self, "_search_scope", "current") == "all"
            cur_gid = getattr(self, "_current_group_id", "")

            if tool_name == "search_episodes":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_episodes(query, video_path, mode=os.getenv("VE_EPISODES_MODE", "hybrid"), top_k=int(os.getenv("VE_TOPK_EPISODES", "15")), omit_group_id=omit_gid)
                results = self.reader.tag_cross_video_content(results, cur_gid)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "search_profiles":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_profiles(query, video_path, top_k=3, omit_group_id=omit_gid)
                results = self.reader.tag_cross_video_content(results, cur_gid)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "search_deep":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_agentic(query, video_path, top_k=int(os.getenv("VE_TOPK_DEEP", "25")), omit_group_id=omit_gid)
                results = self.reader.tag_cross_video_content(results, cur_gid)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "look_at_clip":
                result_text, clip_path = self._tool_look_at_clip(
                    params, video_path, video_duration
                )
                if clip_path:
                    new_clips.append(clip_path)

            elif tool_name == "search_speech":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_speech_bm25(query, video_path, top_k=12, omit_group_id=omit_gid)
                results = self.reader.tag_cross_video_content(results, cur_gid)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "get_timeline":
                result_text = self._tool_get_timeline(params, video_path, video_duration)

            else:
                result_text = f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {type(e).__name__}: {e}", exc_info=True)
            result_text = f"Tool error ({tool_name}): {type(e).__name__}: {e}"

        return result_text, new_sources, new_clips

    def _tool_look_at_clip(
        self, params: dict, video_path: str, video_duration: Optional[float] = None
    ) -> tuple[str, Optional[str]]:
        """Extract a video clip and analyze it with Qwen2.5-Omni.

        This is the "context-grounding" feature — the agent can look at
        specific video moments to verify or enrich its answer.
        """

        if not self.agent_cfg.enable_video_grounding:
            return "Video grounding disabled.", None

        question = params.get("question", "Describe what you see and hear in detail.")
        max_sec = video_duration if video_duration is not None else None

        # MEMORY-GROUNDED clip selection: locate the segment via retrieval timestamps
        # rather than the agent's own start/end guesses (which are unreliable — often
        # out of range, e.g. "3-4min" for a 2min video). Retrieve the question, take
        # the highest-scored hit's timestamp, look at that 30s window.
        start_sec = end_sec = None
        try:
            hits = self.reader.search_episodes(question, video_path, top_k=4)
            for r in sorted(hits, key=lambda r: r.score, reverse=True):
                if getattr(r, "timestamp_range", None):
                    s, e = r.timestamp_range
                    start_sec = s
                    end_sec = e if (e - s) <= 30.0 else s + 30.0
                    break
        except Exception:
            pass

        # Fallback: agent-provided window; if out of range, video midpoint ±15s.
        if start_sec is None:
            start_min = self.parse_min(params.get("start_min", 0))
            end_min = self.parse_min(params.get("end_min", start_min + 0.5))
            start_sec, end_sec = start_min * 60, end_min * 60
            if max_sec is not None and start_sec >= max_sec:
                mid = max_sec / 2
                start_sec, end_sec = max(0.0, mid - 15), min(max_sec, mid + 15)

        if max_sec is not None:
            start_sec = min(start_sec, max(0.0, max_sec - 1.0))
            end_sec = min(end_sec, max_sec)
        if end_sec <= start_sec:
            end_sec = start_sec + 30.0 if max_sec is None else min(start_sec + 30.0, max_sec)
        start_min, end_min = start_sec / 60.0, end_sec / 60.0

        # Extract clip
        clip_dir = self.config.work_dir / "agent_clips"
        clip_dir.mkdir(exist_ok=True)
        clip_path = str(clip_dir / f"look_{start_min:.1f}_{end_min:.1f}.mp4")

        try:
            extract_clip(video_path, start_sec, end_sec, clip_path)
        except Exception as e:
            return f"Failed to extract clip: {e}", None

        # Analyze with Qwen2.5-Omni
        analysis = self.captioner.analyze_clip(clip_path, question)
        result = (
            f"[Video analysis {fmt_minutes(start_sec)} - {fmt_minutes(end_sec)}]\n"
            f"{analysis}"
        )
        return result, clip_path

    def _tool_get_timeline(self, params: dict, video_path: str, video_duration: Optional[float] = None) -> str:
        """Get all memorized events in a time range."""
        max_min = (video_duration / 60) if video_duration is not None else 999
        start_min = self.parse_min(params.get("start_min", 0))
        end_min = min(self.parse_min(params.get("end_min", 999)), max_min)

        is_overview = end_min >= 900

        # Use a content-oriented query rather than a time-based one which rarely matches
        query = "video scenes events dialogue actions" if is_overview else f"events {start_min:.0f} to {end_min:.0f} minutes"
        results = self.reader.search_episodes(query, video_path, top_k=20)

        # For overview: include everything; for specific range: filter by timestamp
        in_range = []
        no_ts = []
        for r in results:
            ts = r.timestamp_range
            if ts:
                start_sec, end_sec = ts
                if is_overview or (start_sec / 60 >= start_min - 0.5 and end_sec / 60 <= end_min + 0.5):
                    in_range.append((start_sec, r))
            elif is_overview:
                no_ts.append(r)

        in_range.sort(key=lambda x: x[0])

        if not in_range and not no_ts:
            return f"No events found between {start_min:.0f}min and {end_min:.0f}min."

        label = "Full timeline" if is_overview else f"Timeline ({start_min:.0f}min - {end_min:.0f}min)"
        lines = [f"{label}:"]
        for sec, r in in_range:
            lines.append(f"  {fmt_minutes(sec)}: {self._strip_injected_dates(r.content[:200])}")
        for r in no_ts:
            lines.append(f"  [timestamp unknown]: {self._strip_injected_dates(r.content[:200])}")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────

    _SUMMARY_KEYWORDS = (
        "总结", "概括", "概述", "summarize", "summary", "overview",
        "what is the video about", "what's the video about",
        "tell me about the video", "what does the video", "what happens in the video",
        "what happened in the video", "整体", "介绍一下这个视频", "这个视频讲的是",
    )

    @classmethod
    def _is_summary_question(cls, question: str) -> bool:
        q = question.lower()
        return any(kw in q for kw in cls._SUMMARY_KEYWORDS)

    @staticmethod
    def _describe_step(tool_name: str, params_str: str) -> str:
        """Return a human-readable description of a tool invocation."""
        p = params_str.strip().strip("'\"")
        if tool_name == "search_episodes":
            return f"Searching memories: {p}"
        elif tool_name == "search_speech":
            return f"Searching speech transcripts: {p}"
        elif tool_name == "search_profiles":
            return f"Looking up entity register: {p}"
        elif tool_name == "search_deep":
            return f"Deep searching: {p}"
        elif tool_name == "look_at_clip":
            return "Examining video footage"
        elif tool_name == "get_timeline":
            return "Scanning video timeline"
        else:
            return f"Running {tool_name}({p})"

    def _fallback_answer(
        self,
        question: str,
        video_path: str,
        actions: list[AgentAction],
        sources: list[MemoryResult],
    ) -> AgentResponse:
        """Fallback: do a simple retrieve-and-answer when ReAct loop exhausts."""
        logger.info("  Using fallback retrieve-and-answer")

        if not sources:
            omit_gid = getattr(self, "_search_scope", "current") == "all"
            cur_gid = getattr(self, "_current_group_id", "")
            results = self.reader.search_episodes(question, video_path, top_k=8, omit_group_id=omit_gid)
            if not results:
                results = self.reader.search_agentic(question, video_path, top_k=10, omit_group_id=omit_gid)
            sources = self.reader.tag_cross_video_content(results, cur_gid)

        context = "\n".join(
            f"- {self._strip_injected_dates(r.content[:800])}" for r in sources[:12]
        )

        try:
            extra = {}
            if "qwen" in self.agent_cfg.planning_llm_model.lower():
                extra["extra_body"] = {"modalities": ["text"]}
            if "gpt-5" in self.agent_cfg.planning_llm_model.lower():
                extra["reasoning_effort"] = "minimal"
            response = self.llm.chat.completions.create(
                model=self.agent_cfg.planning_llm_model,
                messages=[
                    {"role": "system", "content": (
                        "You are a video analyst. Answer the user's question based on the retrieved video memories. "
                        "For every factual claim, cite timestamps using EXACTLY [Video M:SS - N:SS] format "
                        "(e.g. [Video 0:35 - 1:20]) — keep ranges short and precise. "
                        "NEVER mention real-world calendar dates (e.g. December 16, 2025, in January 2025) in your answer; only use [Video M:SS - N:SS] or phrases like 'at one point in the video'. "
                        "If the retrieved memories are genuinely insufficient, briefly say so and summarize "
                        "whatever partial information is available."
                    )},
                    {"role": "user", "content": (
                        f"Question: {question}\n\n"
                        f"Retrieved video memories:\n{context}"
                    )},
                ],
                max_tokens=1024,
                temperature=(1 if "gpt-5" in self.agent_cfg.planning_llm_model.lower() else 0.3),
                **extra,
            )
            answer = self._normalize_timestamps(response.choices[0].message.content.strip())
            if getattr(self, "_search_scope", "current") == "all":
                answer = self._strip_all_timestamps(answer)
            answer = self._strip_calendar_dates_from_answer(answer)
        except Exception as e:
            answer = f"I found {len(sources)} relevant memories but couldn't generate an answer: {e}"

        return AgentResponse(
            answer=answer,
            sources=sources,
            actions=actions,
        )

    @staticmethod
    def _parse_params(params_str: str) -> dict:
        """Parse tool parameters from the LLM's ACTION output.

        Handles formats like:
          - "some query text"
          - "query", 1.5, 3.0
          - start_min=1.5, end_min=3.0, question="what color is the car?"
        """
        params = {}
        params_str = params_str.strip()

        # Try key=value format first
        kv_pattern = re.findall(r'(\w+)\s*=\s*(?:"([^"]*)"|([\d.]+))', params_str)
        if kv_pattern:
            for key, str_val, num_val in kv_pattern:
                params[key] = str_val if str_val else num_val
            return params

        # Try positional format: first string is query, numbers are start/end
        parts = [p.strip().strip("'\"") for p in params_str.split(",")]
        if len(parts) == 1:
            params["query"] = parts[0]
        elif len(parts) == 2:
            try:
                params["start_min"] = parts[0]
                params["end_min"] = parts[1]
            except ValueError:
                params["query"] = params_str
        elif len(parts) >= 3:
            params["start_min"] = parts[0]
            params["end_min"] = parts[1]
            params["question"] = parts[2]
        else:
            params["query"] = params_str

        return params

    @staticmethod
    def _sources_insufficient(sources: list[MemoryResult]) -> bool:
        """Return True if sources are empty or all have low relevance scores.

        Used to decide whether to force a look_at_clip fallback.
        Score threshold of 0.3 is calibrated for hybrid/embedding retrieval (0-1 range).
        BM25 scores are typically >> 0.3 for any real match, so they won't false-trigger.
        """
        if not sources:
            return True
        scored = [r.score for r in sources if r.score > 0]
        if not scored:
            return False  # No scores available — can't judge, assume sufficient
        return max(scored) < 0.3

    @staticmethod
    def _strip_injected_dates(text: str) -> str:
        """Remove EverMemOS-injected metadata timestamps from memory content.

        Only strips machine-generated ISO timestamps. Human-readable dates that
        speakers mention in the video are legitimate facts and must be preserved
        so the LLM can cite them accurately.
        """
        # ISO datetime with T separator: 2026-01-01T00:00:00+00:00
        text = re.sub(
            r'\b(?:19|20)\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s,\]]*',
            '',
            text,
        )
        # "at 12:00 AM UTC" EverMemOS tail artifact
        text = re.sub(
            r',?\s*at\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s+UTC',
            '',
            text,
            flags=re.IGNORECASE,
        )
        # ISO date-only when preceded by upload/index/created context
        text = re.sub(
            r'(?:uploaded?|indexed?|created?|stored?)\s+(?:on\s+)?(?:19|20)\d{2}-\d{2}-\d{2}',
            '',
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r'  +', ' ', text).strip()
        return text

    @staticmethod
    def _format_results(results: list[MemoryResult]) -> str:
        """Format retrieval results for the LLM observation."""
        if not results:
            return "No results found."

        lines = [f"Found {len(results)} results:"]
        for i, r in enumerate(results):
            score_str = f" (score={r.score:.2f})" if r.score else ""
            content = VidEngramAgent._strip_injected_dates(r.content[:300])
            lines.append(f"  [{i+1}]{score_str} {content}")
        return "\n".join(lines)
    
    @staticmethod
    def _normalize_timestamps(text: str) -> str:
        """Wrap bare MM:SS timecodes in canonical [Video M:SS - N:SS] brackets.

        Handles:
          MM:SS              →  [Video M:SS - N:SS]   (±3 s window around single point)
          HH:MM:SS           →  [Video H:MM:SS - H:MM:SS]
          MM:SS - MM:SS      →  [Video M:SS - N:SS]
          from MM:SS to MM:SS
        Skips text already in [Video ...] or [Episode ...] format.
        """
        if not text:
            return text

        def _sec_to_ts(sec: float) -> str:
            h = int(sec // 3600)
            m = int((sec % 3600) // 60)
            s = int(sec % 60)
            if h > 0:
                return f"{h}:{m:02d}:{s:02d}"
            return f"{m}:{s:02d}"

        def _parse_mmss(hh_or_mm: str, mm_or_ss: str, ss: str = None) -> float:
            if ss is not None:
                return int(hh_or_mm) * 3600 + int(mm_or_ss) * 60 + float(ss)
            return int(hh_or_mm) * 60 + float(mm_or_ss)

        result = text

        # 0. [Video M:SS] or [Episode M:SS] with no end time → expand to ±1s range
        bracket_single_re = re.compile(
            r'\[(Video|Episode)(?:\s+analysis)?\s+(\d{1,2}:\d{2}(?::\d{2})?)\]',
            re.IGNORECASE,
        )
        def _replace_bracket_single(m):
            kind = m.group(1)
            ts_str = m.group(2)
            # Parse the timestamp to seconds
            parts = ts_str.split(":")
            if len(parts) == 3:
                sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                sec = int(parts[0]) * 60 + float(parts[1])
            s_start = max(0.0, sec - 1)
            s_end = sec + 1
            return f"[{kind} {_sec_to_ts(s_start)} - {_sec_to_ts(s_end)}]"
        result = bracket_single_re.sub(_replace_bracket_single, result)

        # 1. Range: MM:SS - MM:SS  or  MM:SS to MM:SS
        range_re = re.compile(
            r'\b(\d{1,2}):(\d{2})\s*(?:-|to)\s*(\d{1,2}):(\d{2})\b'
        )
        def _replace_range(m):
            start = m.start()
            preceding = result[max(0, start - 80):start]
            open_count = preceding.count('[') - preceding.count(']')
            if open_count > 0:
                return m.group(0)
            s1 = _parse_mmss(m.group(1), m.group(2))
            s2 = _parse_mmss(m.group(3), m.group(4))
            return f"[Video {_sec_to_ts(s1)} - {_sec_to_ts(s2)}]"
        result = range_re.sub(_replace_range, result)

        # 2. Single MM:SS — only when NOT already inside [Video ...] / [Episode ...]
        single_re = re.compile(
            r'(?:^|(?<=\s)|(?<=[(\[,]))(\d{1,2}):(\d{2})(?=\s|$|[)\].,;])'
        )
        def _replace_single(m):
            start = m.start()
            preceding = result[max(0, start - 80):start]
            open_count = preceding.count('[') - preceding.count(']')
            if open_count > 0:
                return m.group(0)
            sec = _parse_mmss(m.group(1), m.group(2))
            s_start = max(0.0, sec - 3)
            s_end = sec + 3
            return f"[Video {_sec_to_ts(s_start)} - {_sec_to_ts(s_end)}]"
        result = single_re.sub(_replace_single, result)

        return result

    @staticmethod
    def _strip_cross_video_timestamps(text: str) -> str:
        """Remove any [SomeName M:SS - N:SS] patterns that are not [Video ...] or [Episode ...].

        Belt-and-suspenders for ALL VIDEOS mode: the LLM should not emit cross-video
        timestamps (memories arrive without them), but if it hallucinates one, strip it.
        """
        return re.sub(
            r'\[(?!(?:Video|Episode)(?:\s+analysis)?\s+\d)(?:[^\]]+\d+:\d{2}[^\]]*)\]',
            '',
            text,
        ).strip()

    @staticmethod
    def _strip_all_timestamps(text: str) -> str:
        """Remove all [Video/Episode M:SS - N:SS] and [Video/Episode M:SS] timestamp citations.

        Used in ALL VIDEOS mode so answers contain no timestamps (direct deletion).
        """
        if not text:
            return text
        # Range: [Video 0:06 - 0:11], [Episode analysis 1:30 - 2:00], H:MM:SS supported
        text = re.sub(
            r"\[(?:Video|Episode)(?:\s+analysis)?\s+\d{1,2}:\d{2}(?::\d{2})?\s*-\s*\d{1,2}:\d{2}(?::\d{2})?\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Single: [Video 0:06], [Episode 1:30]
        text = re.sub(
            r"\[(?:Video|Episode)(?:\s+analysis)?\s+\d{1,2}:\d{2}(?::\d{2})?\]",
            "",
            text,
            flags=re.IGNORECASE,
        )
        # Collapse multiple spaces and clean up
        text = re.sub(r"  +", " ", text).strip()
        return text

    _NEUTRAL_PHRASE = "at one point in the video"

    @classmethod
    def _strip_calendar_dates_from_answer(cls, text: str) -> str:
        """Remove EverMemOS metadata timestamps from the answer.

        Only strips machine-generated ISO timestamps injected by the storage system.
        Human-readable dates spoken in the video (e.g. "January 2025", "in 2025") are
        legitimate facts and must NOT be removed.
        """
        if not text:
            return text
        neutral = cls._NEUTRAL_PHRASE
        # ISO datetime: 2025-12-16T00:00:00+00:00 or 2026-01-01T00:00:00
        text = re.sub(
            r"\b(?:19|20)\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[^\s,\]]*",
            neutral,
            text,
        )
        # ISO date-only when followed by EverMemOS context words (upload, index, created)
        text = re.sub(
            r"(?:uploaded?|indexed?|created?|stored?)\s+(?:on\s+)?"
            r"(?:19|20)\d{2}-\d{2}-\d{2}",
            neutral,
            text,
            flags=re.IGNORECASE,
        )
        # "at 12:00 AM UTC" tail (EverMemOS metadata artifact)
        text = re.sub(
            r",\s*at\s+\d{1,2}:\d{2}\s*(?:AM|PM)\s+UTC",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"  +", " ", text).strip()
        return text

    def _forced_video_grounding(
        self,
        question: str,
        sources: list[MemoryResult],
        video_path: str,
        video_duration: Optional[float],
    ) -> Optional[tuple[str, Optional[str]]]:
        """Pick the best clip to examine and call look_at_clip.

        Clip selection:
        - Has sources with timestamps → use the highest-scored one.
          Window = [start_sec, start_sec + 30s], or segment's own end_sec if < 30s.
        - No timestamps anywhere → fall back to the video midpoint ±15s.
        """
        MAX_WINDOW_SEC = 30.0

        best_start_sec: Optional[float] = None
        best_end_sec: Optional[float] = None

        # Sort by score descending; pick the first source that carries a timestamp
        for r in sorted(sources, key=lambda r: r.score, reverse=True):
            ts = r.timestamp_range
            if ts:
                start_sec, end_sec = ts
                segment_len = end_sec - start_sec
                best_start_sec = start_sec
                best_end_sec = end_sec if segment_len <= MAX_WINDOW_SEC else start_sec + MAX_WINDOW_SEC
                break

        # Fallback: video midpoint
        if best_start_sec is None:
            if not video_duration:
                logger.warning("Forced grounding: no sources with timestamps and no video_duration — skipping.")
                return None
            mid = video_duration / 2
            best_start_sec = max(0.0, mid - 15.0)
            best_end_sec = min(video_duration, mid + 15.0)

        # Clamp to video duration
        if video_duration:
            best_end_sec = min(best_end_sec, video_duration)

        logger.info(
            f"  Forced video grounding: [{best_start_sec:.1f}s - {best_end_sec:.1f}s] "
            f"({'from source timestamp' if sources else 'midpoint fallback'})"
        )

        params = {
            "start_min": best_start_sec / 60.0,
            "end_min": best_end_sec / 60.0,
            "question": question,
        }
        return self._tool_look_at_clip(params, video_path)

    @staticmethod
    def parse_min(val, default=0):
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace("min", "").replace("s", "").strip())
        except (ValueError, AttributeError):
            logger.warning(f"parse_min: could not convert {val!r} to float, using default={default}")
            return float(default)
