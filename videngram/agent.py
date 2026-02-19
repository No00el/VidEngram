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
  4. look_at_video    — Extract + analyze a specific video moment (grounding)
  5. get_timeline     — List all events in a time range

This makes VidEngram truly AGENTIC:
  - The agent PLANS which tools to use based on query complexity
  - It can COMBINE memory retrieval with direct video analysis
  - It ITERATES if initial retrieval is insufficient
  - It GROUNDS answers in specific timestamps (context-grounding)

Competition differentiators:
  - Not just RAG over captions — it's a reasoning agent with tool access
  - Combines EverMemOS structured memory with on-demand video understanding
  - Hippocampal dual-pathway: fast retrieval first, then deep recall if needed
"""
import json
import logging
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
   Returns text snippets with [Video X.Xmin - Y.Ymin] timestamps.
   USE FOR: factual questions, "what happened", finding specific moments.

2. search_profiles(query) → Look up profiles of people/entities in the video.
   USE FOR: "who is", "tell me about", character/speaker questions.

3. search_deep(query) → Deep multi-hop search across all memory types.
   SLOWER but more thorough. USE FOR: complex questions requiring reasoning
   across multiple video segments, cross-references, temporal logic.

4. look_at_video(start_min, end_min, question) → Extract and analyze a specific
   video moment. Sends the actual clip to the vision model for fresh analysis.
   USE FOR: when memory doesn't have enough detail, visual verification,
   "what exactly does X look like", counting objects, reading text on screen.

5. get_timeline(start_min, end_min) → List all memorized events in a time range.
   USE FOR: "what happened between X and Y", chronological questions.

## Instructions:
- Think step-by-step about what information you need
- Start with fast search (search_episodes), escalate to search_deep or look_at_video only if needed
- You can call multiple tools before answering
- Always cite video timestamps in your final answer (e.g., "at 3:20...")
- If you can't find the answer, say so honestly

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
    ) -> AgentResponse:
        """Answer a question about a video using the agentic reasoning loop.

        Args:
            question: User's natural language question
            video_path: Path to the source video
            chat_history: Optional previous Q&A turns for multi-turn context

        Returns:
            AgentResponse with answer, sources, actions taken, and grounded clips
        """
        logger.info(f"Agent query: '{question}'")
        actions: list[AgentAction] = []
        sources: list[MemoryResult] = []
        grounded_clips: list[str] = []

        # Build conversation for the planning LLM
        messages = [{"role": "system", "content": PLANNING_SYSTEM_PROMPT}]

        # Add chat history for multi-turn context
        if chat_history:
            for turn in chat_history[-4:]:  # Last 4 turns to stay in context
                messages.append(turn)

        # User question
        messages.append({"role": "user", "content": question})

        # ReAct loop
        for iteration in range(self.agent_cfg.max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.agent_cfg.max_iterations}")

            # Get LLM's next step
            try:
                extra = {}
                if "qwen" in self.agent_cfg.planning_llm_model.lower():
                    extra["extra_body"] = {"modalities": ["text"]}
                response = self.llm.chat.completions.create(
                    model=self.agent_cfg.planning_llm_model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.2,
                    **extra,
                )
                llm_output = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Planning LLM error: {e}")
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

                # Execute the tool
                tool_result, new_sources, new_clips = self._execute_tool(
                    tool_name, params_str, video_path
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
                answer_text = llm_output.split("ANSWER:")[-1].strip()
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
                return AgentResponse(
                    answer=llm_output,
                    sources=sources,
                    actions=actions,
                    grounded_clips=grounded_clips,
                )

        # Max iterations reached — force an answer
        return self._fallback_answer(question, video_path, actions, sources)

    # ── Tool Execution ────────────────────────────────────────────────

    def _execute_tool(
        self, tool_name: str, params_str: str, video_path: str
    ) -> tuple[str, list[MemoryResult], list[str]]:
        """Execute a tool and return (result_text, new_sources, new_clips)."""
        params = self._parse_params(params_str)
        new_sources: list[MemoryResult] = []
        new_clips: list[str] = []

        try:
            if tool_name == "search_episodes":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_episodes(query, video_path, top_k=5)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "search_profiles":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_profiles(query, video_path, top_k=3)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "search_deep":
                query = params.get("query", params_str.strip("'\""))
                results = self.reader.search_agentic(query, video_path, top_k=10)
                new_sources = results
                result_text = self._format_results(results)

            elif tool_name == "look_at_video":
                result_text, clip_path = self._tool_look_at_video(
                    params, video_path
                )
                if clip_path:
                    new_clips.append(clip_path)

            elif tool_name == "get_timeline":
                result_text = self._tool_get_timeline(params, video_path)

            else:
                result_text = f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            result_text = f"Tool error: {e}"

        return result_text, new_sources, new_clips

    def _tool_look_at_video(
        self, params: dict, video_path: str
    ) -> tuple[str, Optional[str]]:
        """Extract a video clip and analyze it with Qwen2.5-Omni.

        This is the "context-grounding" feature — the agent can look at
        specific video moments to verify or enrich its answer.
        """
        if not self.agent_cfg.enable_video_grounding:
            return "Video grounding disabled.", None

        start_min = float(params.get("start_min", 0))
        end_min = float(params.get("end_min", start_min + 0.5))
        question = params.get("question", "Describe what you see and hear in detail.")

        start_sec = start_min * 60
        end_sec = end_min * 60

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

    def _tool_get_timeline(self, params: dict, video_path: str) -> str:
        """Get all memorized events in a time range."""
        start_min = float(params.get("start_min", 0))
        end_min = float(params.get("end_min", 999))

        # Search for all events in this time range
        query = f"events between {start_min:.0f} and {end_min:.0f} minutes"
        results = self.reader.search_episodes(query, video_path, top_k=20)

        # Filter to time range and sort
        in_range = []
        for r in results:
            ts = r.timestamp_range
            if ts:
                start_sec, end_sec = ts
                if start_sec / 60 >= start_min - 0.5 and end_sec / 60 <= end_min + 0.5:
                    in_range.append((start_sec, r))

        in_range.sort(key=lambda x: x[0])

        if not in_range:
            return f"No events found between {start_min:.0f}min and {end_min:.0f}min."

        lines = [f"Timeline ({start_min:.0f}min - {end_min:.0f}min):"]
        for sec, r in in_range:
            lines.append(f"  {fmt_minutes(sec)}: {r.content[:150]}")
        return "\n".join(lines)

    # ── Helpers ───────────────────────────────────────────────────────

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
            results = self.reader.search_episodes(question, video_path, top_k=5)
            sources = results

        context = "\n".join(
            f"- {r.content[:300]}" for r in sources[:5]
        )

        try:
            extra = {}
            if "qwen" in self.agent_cfg.planning_llm_model.lower():
                extra["extra_body"] = {"modalities": ["text"]}
            response = self.llm.chat.completions.create(
                model=self.agent_cfg.planning_llm_model,
                messages=[
                    {"role": "system", "content": (
                        "You are a video analyst. Answer the question based on "
                        "the retrieved video memories below. Cite timestamps. "
                        "If the memories don't contain the answer, say so."
                    )},
                    {"role": "user", "content": (
                        f"Question: {question}\n\n"
                        f"Retrieved video memories:\n{context}"
                    )},
                ],
                max_tokens=1024,
                temperature=0.3,
                **extra,
            )
            answer = response.choices[0].message.content.strip()
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
    def _format_results(results: list[MemoryResult]) -> str:
        """Format retrieval results for the LLM observation."""
        if not results:
            return "No results found."

        lines = [f"Found {len(results)} results:"]
        for i, r in enumerate(results):
            score_str = f" (score={r.score:.2f})" if r.score else ""
            lines.append(f"  [{i+1}]{score_str} {r.content[:300]}")
        return "\n".join(lines)
