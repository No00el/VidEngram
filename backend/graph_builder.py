"""
VidEngram Graph Builder
Extracts a relationship graph from video memories using an LLM.
Runs as a background asyncio task after ingest completes.
"""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("videngram.graph_builder")

# ── Module-level state ────────────────────────────────────────────────────────
_graph_status: str = "idle"   # "idle" | "pending" | "ready" | "error"
_graph_data: Optional[dict] = None
_active_task: Optional[asyncio.Task] = None

# All-videos graph state (separate from per-video state)
_all_graph_status: str = "idle"
_all_graph_data: Optional[dict] = None
_all_active_task: Optional[asyncio.Task] = None

_ALL_CACHE_FILENAME = "graph_cache_all.json"


def get_status() -> str:
    return _graph_status


def get_data() -> Optional[dict]:
    return _graph_data


def get_all_status() -> str:
    return _all_graph_status


def get_all_data() -> Optional[dict]:
    return _all_graph_data


def clear():
    """Reset graph state. Call before a new ingest starts."""
    global _graph_status, _graph_data, _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
    _graph_status = "idle"
    _graph_data = None
    _active_task = None


def clear_all():
    """Reset all-videos graph state."""
    global _all_graph_status, _all_graph_data, _all_active_task
    if _all_active_task and not _all_active_task.done():
        _all_active_task.cancel()
    _all_graph_status = "idle"
    _all_graph_data = None
    _all_active_task = None


def load_cached(work_dir: Path, group_id: str) -> bool:
    """Load a pre-built graph from per-video disk cache. Returns True if found."""
    global _graph_status, _graph_data, _active_task
    if _active_task and not _active_task.done():
        _active_task.cancel()
    _active_task = None

    cache_file = work_dir / f"graph_cache_{group_id}.json"
    if not cache_file.exists():
        _graph_status = "idle"
        _graph_data = None
        return False
    try:
        _graph_data = json.loads(cache_file.read_text(encoding="utf-8"))
        _graph_status = "ready"
        logger.info(f"[graph] Loaded cached graph for {group_id}")
        return True
    except Exception as e:
        logger.warning(f"[graph] Cache load error for {group_id}: {e}")
        _graph_status = "idle"
        _graph_data = None
        return False


def load_all_cached(work_dir: Path) -> bool:
    """Load the pre-built all-videos graph from disk cache. Returns True if found."""
    global _all_graph_status, _all_graph_data, _all_active_task
    cache_file = work_dir / _ALL_CACHE_FILENAME
    if not cache_file.exists():
        _all_graph_status = "idle"
        _all_graph_data = None
        return False
    try:
        _all_graph_data = json.loads(cache_file.read_text(encoding="utf-8"))
        _all_graph_status = "ready"
        logger.info("[graph_all] Loaded cached all-videos graph")
        return True
    except Exception as e:
        logger.warning(f"[graph_all] Cache load error: {e}")
        _all_graph_status = "idle"
        _all_graph_data = None
        return False


def trigger_build(
    work_dir: Path,
    group_id: str,
    evermemos_url: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> asyncio.Task:
    """Schedule a graph build as a background asyncio task. Returns the task."""
    global _active_task
    _active_task = asyncio.create_task(
        _build_graph(work_dir, group_id, evermemos_url, llm_base_url, llm_api_key, llm_model),
        name="graph_builder",
    )
    return _active_task


def trigger_build_all(
    work_dir: Path,
    group_ids_and_names: list[tuple[str, str]],
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> asyncio.Task:
    """Schedule an all-videos graph build. group_ids_and_names is [(group_id, filename), ...]."""
    global _all_active_task
    if _all_active_task and not _all_active_task.done():
        _all_active_task.cancel()
    _all_active_task = asyncio.create_task(
        _build_graph_all(work_dir, group_ids_and_names, llm_base_url, llm_api_key, llm_model),
        name="graph_builder_all",
    )
    return _all_active_task


# ── Core build logic ──────────────────────────────────────────────────────────

async def _build_graph(
    work_dir: Path,
    group_id: str,
    evermemos_url: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> None:
    global _graph_status, _graph_data
    _graph_status = "pending"
    cache_file = work_dir / f"graph_cache_{group_id}.json"

    try:
        # 1. Load all memories
        memories = await _load_memories(work_dir, group_id, evermemos_url)
        if not memories:
            logger.warning("[graph] No memories found — cannot build graph")
            _graph_status = "error"
            return

        # 2. Format memories into LLM prompt text
        content_str = _format_memories(memories)
        logger.info(f"[graph] Formatted {len(memories)} memories ({len(content_str)} chars) for LLM")

        # 3. Call LLM to extract graph
        from openai import AsyncOpenAI
        llm = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key or "EMPTY")
        graph_data = await _extract_graph(content_str, llm, llm_model)

        # 4. Cache to disk and update state
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _graph_data = graph_data
        _graph_status = "ready"
        n_nodes = len(graph_data.get("nodes", []))
        n_edges = len(graph_data.get("edges", []))
        logger.info(f"[graph] Build complete: {n_nodes} nodes, {n_edges} edges")

    except asyncio.CancelledError:
        logger.info("[graph] Build task cancelled")
        _graph_status = "idle"
    except Exception as e:
        logger.error(f"[graph] Build error: {e}", exc_info=True)
        _graph_status = "error"


async def _build_graph_all(
    work_dir: Path,
    group_ids_and_names: list[tuple[str, str]],
    llm_base_url: str,
    llm_api_key: str,
    llm_model: str,
) -> None:
    global _all_graph_status, _all_graph_data
    _all_graph_status = "pending"
    cache_file = work_dir / _ALL_CACHE_FILENAME

    try:
        # 1. Load memories from each video's cues file (cap at 50 per video)
        all_sections: list[str] = []
        total_memories = 0
        for group_id, filename in group_ids_and_names:
            cues_file = work_dir / f"{group_id}_cues.json"
            if not cues_file.exists():
                logger.info(f"[graph_all] No cues file for {group_id} ({filename}), skipping")
                continue
            try:
                cues = json.loads(cues_file.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"[graph_all] Cues read error for {group_id}: {e}")
                continue
            memories = []
            seen: set[str] = set()
            for cue in cues:
                content = cue.get("content", "").strip()
                if not content:
                    continue
                key = content[:100]
                if key in seen:
                    continue
                seen.add(key)
                memories.append({
                    "content": content,
                    "start_sec": float(cue.get("start_sec", 0)),
                    "end_sec": float(cue.get("end_sec", 0)),
                })
                if len(memories) >= 50:
                    break
            if not memories:
                continue
            section = f"=== Video: {filename} ===\n\n" + _format_memories(memories)
            all_sections.append(section)
            total_memories += len(memories)
            logger.info(f"[graph_all] Loaded {len(memories)} memories from {filename}")

        if not all_sections:
            logger.warning("[graph_all] No memories found across any video — cannot build graph")
            _all_graph_status = "error"
            return

        content_str = "\n\n" + ("=" * 60) + "\n\n".join(all_sections)
        logger.info(f"[graph_all] Formatted {total_memories} total memories from {len(all_sections)} videos")

        # 2. Call LLM
        from openai import AsyncOpenAI
        llm = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key or "EMPTY")
        graph_data = await _extract_graph_all(content_str, llm, llm_model)

        # 3. Cache to disk and update state
        work_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _all_graph_data = graph_data
        _all_graph_status = "ready"
        n_nodes = len(graph_data.get("nodes", []))
        n_edges = len(graph_data.get("edges", []))
        logger.info(f"[graph_all] Build complete: {n_nodes} nodes, {n_edges} edges")

    except asyncio.CancelledError:
        logger.info("[graph_all] Build task cancelled")
        _all_graph_status = "idle"
    except Exception as e:
        logger.error(f"[graph_all] Build error: {e}", exc_info=True)
        _all_graph_status = "error"


# ── Memory loading ────────────────────────────────────────────────────────────

async def _load_memories(
    work_dir: Path, group_id: str, evermemos_url: str
) -> list[dict]:
    memories: list[dict] = []
    seen: set[str] = set()

    # ── Priority 1: local cues file (segment captions with exact timestamps) ──
    cues_file = work_dir / f"{group_id}_cues.json"
    if cues_file.exists():
        try:
            cues = json.loads(cues_file.read_text(encoding="utf-8"))
            for cue in cues:
                content = cue.get("content", "").strip()
                if not content:
                    continue
                key = content[:100]
                if key in seen:
                    continue
                seen.add(key)
                memories.append({
                    "content": content,
                    "start_sec": float(cue.get("start_sec", 0)),
                    "end_sec": float(cue.get("end_sec", 0)),
                })
            logger.info(f"[graph] Loaded {len(memories)} segment memories from cues file")
        except Exception as e:
            logger.warning(f"[graph] Cues file read error: {e}")

    # ── Priority 2: EverMemOS profiles + episode summaries ───────────────────
    try:
        search_url = f"{evermemos_url}/api/v1/memories/search"
        queries = [
            "entity person character object location event concept organization",
            "video scene description dialogue events",
        ]
        async with httpx.AsyncClient(timeout=30.0) as client:
            for query in queries:
                resp = await client.get(search_url, params={
                    "query": query,
                    "group_id": group_id,
                    "top_k": 50,
                })
                if resp.status_code != 200:
                    continue
                items = _parse_evermemos_response(resp.json())
                added = 0
                for content, _ in items:
                    content = content.strip()
                    if not content or len(content) < 20:
                        continue
                    key = content[:100]
                    if key in seen:
                        continue
                    seen.add(key)
                    memories.append({
                        "content": content,
                        "start_sec": None,
                        "end_sec": None,
                    })
                    added += 1
                logger.info(f"[graph] Added {added} EverMemOS memories (query={query!r})")
    except Exception as e:
        logger.warning(f"[graph] EverMemOS fetch error: {e}")

    return memories


def _parse_evermemos_response(response_data: dict) -> list[tuple[str, Optional[str]]]:
    """Minimal EverMemOS response parser — extracts (content, timestamp) pairs."""
    results: list[tuple[str, Optional[str]]] = []
    inner = response_data
    if isinstance(response_data, dict) and "result" in response_data:
        inner = response_data["result"]
    if not isinstance(inner, dict):
        return results

    def _extract(mem: dict) -> Optional[tuple[str, Optional[str]]]:
        c = (
            mem.get("episode") or mem.get("content") or mem.get("text") or
            mem.get("summary") or mem.get("atomic_fact") or ""
        )
        if isinstance(c, list):
            c = "; ".join(str(x) for x in c if x)
        c = str(c).strip()
        if not c or "[Caption error" in c or "[Clip unavailable]" in c:
            return None
        ts = mem.get("timestamp") or mem.get("create_time") or mem.get("message_create_time")
        return (c, ts)

    for key in ("memories", "original_data", "pending_messages"):
        items = inner.get(key, [])
        if not isinstance(items, list):
            continue
        for group_dict in items:
            if not isinstance(group_dict, dict):
                if isinstance(group_dict, str) and group_dict:
                    results.append((group_dict, None))
                continue
            # Handle {type_name: [MemCell, ...]} grouping
            for val in group_dict.values():
                if isinstance(val, list):
                    for mem in val:
                        if isinstance(mem, dict):
                            pair = _extract(mem)
                            if pair:
                                results.append(pair)
                        elif isinstance(mem, str) and mem:
                            results.append((mem, None))
            # Also handle flat memory dicts
            if any(k in group_dict for k in ("episode", "content", "text", "summary", "atomic_fact")):
                pair = _extract(group_dict)
                if pair:
                    results.append(pair)

    return results


# ── Formatting ────────────────────────────────────────────────────────────────

def _sec_to_label(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def _format_memories(memories: list[dict]) -> str:
    """Format memories as structured text for the LLM prompt."""
    parts: list[str] = []
    for mem in memories:
        content = mem["content"]
        start = mem.get("start_sec")
        end = mem.get("end_sec")
        if start is not None and end is not None:
            label = f"{_sec_to_label(start)} - {_sec_to_label(end)}"
            parts.append(f"[{label}]\n{content}")
        else:
            parts.append(f"[Profile/Summary]\n{content}")
    return "\n\n---\n\n".join(parts)


# ── LLM extraction ────────────────────────────────────────────────────────────

def _repair_truncated_json(raw: str) -> Optional[dict]:
    """
    Try to salvage a truncated JSON response from the LLM.

    Strategy:
    1. Extract the nodes array up to the last complete object (ends with '}').
    2. Extract the edges array the same way.
    3. Extract color_map if present.
    4. Reconstruct a valid dict from whatever was recovered.

    Returns a dict (possibly with fewer nodes/edges than intended) or None if
    nothing could be parsed at all.
    """
    result: dict = {"nodes": [], "edges": [], "color_map": {}}

    # ── nodes ────────────────────────────────────────────────────────────────
    nodes_match = re.search(r'"nodes"\s*:\s*\[', raw)
    if nodes_match:
        start = nodes_match.end()
        # Find the last complete node object before the truncation
        segment = raw[start:]
        # Collect complete {...} blocks
        depth = 0
        obj_start = None
        objects = []
        for i, ch in enumerate(segment):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(segment[obj_start:i + 1])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == ']' and depth == 0:
                break  # end of array
        result["nodes"] = objects
        logger.info(f"[graph] Truncation repair: recovered {len(objects)} nodes")

    # ── edges ────────────────────────────────────────────────────────────────
    edges_match = re.search(r'"edges"\s*:\s*\[', raw)
    if edges_match:
        start = edges_match.end()
        segment = raw[start:]
        depth = 0
        obj_start = None
        objects = []
        for i, ch in enumerate(segment):
            if ch == '{':
                if depth == 0:
                    obj_start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and obj_start is not None:
                    try:
                        obj = json.loads(segment[obj_start:i + 1])
                        objects.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = None
            elif ch == ']' and depth == 0:
                break
        result["edges"] = objects

    # ── color_map ────────────────────────────────────────────────────────────
    cm_match = re.search(r'"color_map"\s*:\s*(\{[^}]*\})', raw)
    if cm_match:
        try:
            result["color_map"] = json.loads(cm_match.group(1))
        except json.JSONDecodeError:
            pass

    return result if result["nodes"] else None


async def _extract_graph(content_str: str, llm, model: str) -> dict:
    """Call the LLM to extract nodes, edges, and color mappings."""
    system_prompt = (
        "You are an expert knowledge graph builder for video content analysis. "
        "Extract a comprehensive relationship graph from video memory descriptions. "
        "ALL output must be in English (translate names and descriptions from other languages). "
        "Respond ONLY with valid JSON — no markdown fences, no explanation."
    )

    user_prompt = f"""Analyze the following video memory content and build a relationship graph.

The content contains time-stamped scene descriptions including scenes, dialogue, objects, people, events, and locations.

EXTRACTION RULES:
1. Extract up to 150 nodes. Include all significant entities: people, objects, locations, events, concepts, organizations, technologies, etc.
2. Node names must be in English (translate from other languages)
3. For each node, list which video timestamps it appears in (reference the [M:SS - M:SS] brackets in the content)
4. Identify meaningful relationships between nodes as directed edges
5. Dynamically determine node type categories based on what you find (e.g. Person, Object, Location, Event, Concept, Organization, Technology, etc.)
6. Assign a visually distinct hex color to each type category
7. Keep descriptions to ONE short sentence. Keep edge labels to 1-4 words. This is critical to avoid truncation.

OUTPUT FORMAT (valid JSON only, no other text):
{{
  "nodes": [
    {{
      "id": "unique_snake_case_id",
      "name": "Entity Name in English",
      "type": "TypeName",
      "description": "One short sentence description.",
      "timestamps": [
        {{"label": "0:00 - 1:30", "start_sec": 0.0}}
      ]
    }}
  ],
  "edges": [
    {{
      "source": "node_id_1",
      "target": "node_id_2",
      "label": "relationship"
    }}
  ],
  "color_map": {{
    "TypeName": "#hexcolor"
  }}
}}

VIDEO MEMORIES:
{content_str}"""

    response = await llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=16000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the LLM included them
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Response may have been truncated — try to salvage what was parsed
        data = _repair_truncated_json(raw)
        if data is None:
            raise ValueError(f"LLM returned unparseable output: {raw[:300]}")

    # Ensure required keys exist
    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    data.setdefault("color_map", {})

    # Sanitize edges — remove any that reference non-existent node IDs
    valid_ids = {n["id"] for n in data["nodes"] if isinstance(n, dict) and "id" in n}
    data["edges"] = [
        e for e in data["edges"]
        if isinstance(e, dict)
        and e.get("source") in valid_ids
        and e.get("target") in valid_ids
    ]

    return data


async def _extract_graph_all(content_str: str, llm, model: str) -> dict:
    """Like _extract_graph but instructs LLM to synthesize across multiple videos."""
    system_prompt = (
        "You are an expert knowledge graph builder for multi-video content analysis. "
        "Extract a unified relationship graph from memory descriptions of multiple videos. "
        "ALL output must be in English (translate names and descriptions from other languages). "
        "Respond ONLY with valid JSON — no markdown fences, no explanation."
    )

    user_prompt = f"""Analyze the following memory content from MULTIPLE videos and build a unified relationship graph.

Each section begins with '=== Video: <filename> ===' followed by time-stamped scene descriptions.

EXTRACTION RULES:
1. Extract up to 150 nodes total. Include all significant entities across ALL videos: people, objects, locations, events, concepts, organizations, technologies, etc.
2. Node names must be in English (translate from other languages)
3. For each node, list which video timestamps it appears in (reference the [M:SS - M:SS] brackets)
4. Identify meaningful relationships between nodes as directed edges, including cross-video connections
5. Dynamically determine node type categories based on what you find (e.g. Person, Object, Location, Event, Concept, Organization, Technology, etc.)
6. Assign a visually distinct hex color to each type category
7. Keep descriptions to ONE short sentence. Keep edge labels to 1-4 words. This is critical to avoid truncation.

OUTPUT FORMAT (valid JSON only, no other text):
{{
  "nodes": [
    {{
      "id": "unique_snake_case_id",
      "name": "Entity Name in English",
      "type": "TypeName",
      "description": "One short sentence description.",
      "timestamps": [
        {{"label": "0:00 - 1:30", "start_sec": 0.0}}
      ]
    }}
  ],
  "edges": [
    {{
      "source": "node_id_1",
      "target": "node_id_2",
      "label": "relationship"
    }}
  ],
  "color_map": {{
    "TypeName": "#hexcolor"
  }}
}}

VIDEO MEMORIES:
{content_str}"""

    response = await llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=16000,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = _repair_truncated_json(raw)
        if data is None:
            raise ValueError(f"LLM returned unparseable output: {raw[:300]}")

    data.setdefault("nodes", [])
    data.setdefault("edges", [])
    data.setdefault("color_map", {})

    valid_ids = {n["id"] for n in data["nodes"] if isinstance(n, dict) and "id" in n}
    data["edges"] = [
        e for e in data["edges"]
        if isinstance(e, dict)
        and e.get("source") in valid_ids
        and e.get("target") in valid_ids
    ]

    return data
