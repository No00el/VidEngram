from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pathlib import Path
from datetime import datetime, timezone
import base64
import json, re, httpx, asyncio, os, sys, hashlib, shutil
from typing import Optional

# ── Graph builder (co-located in backend/) ────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
import graph_builder as _graph_builder

# Load videngram/.env so EVERMEMOS_BASE_URL / QWEN_BASE_URL are available
_env_file = Path(__file__).parent.parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(_env_file)
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────
VLLM_URL = os.environ.get("VLLM_SERVER") or os.environ.get("QWEN_BASE_URL", "http://localhost:8091/v1")
EVERMEMOS_URL = (
    os.environ.get("EVERMEMOS_SERVER")
    or os.environ.get("EVERMEMOS_BASE_URL", "http://localhost:8001")
)


def _is_local_qwen_url(url: str) -> bool:
    return url.startswith(("http://localhost", "http://127.0.0.1", "http://0.0.0.0"))


def _resolve_qwen_api_key() -> str:
    return os.environ.get("QWEN_API_KEY") or os.environ.get("VLLM_API_KEY") or ""


_QWEN_API_KEY = _resolve_qwen_api_key()


def _ensure_llm_ready():
    if not _QWEN_API_KEY and not _is_local_qwen_url(VLLM_URL):
        raise HTTPException(
            status_code=500,
            detail="QWEN_API_KEY is required when QWEN_BASE_URL is not a local vLLM endpoint.",
        )


def _decode_cli_event(line: str, prefix: str) -> Optional[dict]:
    """Decode structured stdout events from demo.cli (supports plain + *_B64 variants)."""
    variants = (
        (f"{prefix}_B64: ", True),
        (f"{prefix}: ", False),
    )
    for marker, is_b64 in variants:
        if line.startswith(marker):
            payload = line[len(marker):]
            try:
                if is_b64:
                    raw = base64.b64decode(payload.encode("utf-8")).decode("utf-8")
                else:
                    raw = payload
                return json.loads(raw)
            except Exception as exc:
                print(f"[ingest] Failed to decode {prefix}: {exc}")
                return None
    return None


FRONTEND_DIR = Path(
    os.environ.get(
        "FRONTEND_DIR",
        str(Path(__file__).parent.parent / "frontend")
    )
)

# Directory containing videngram package (where `python -m demo.cli` is run)
VIDENGRAM_DIR = Path(
    os.environ.get(
        "VIDENGRAM_DIR",
        str(Path(__file__).parent.parent)
    )
)

# Default video path for ingest — overridden by /upload on each session.
# Leave empty so the server refuses to ingest before a file is uploaded.
VIDEO_DEFAULT_PATH = os.environ.get("VIDEO_DEFAULT_PATH", "")

# Local path served to the browser
VIDEO_LOCAL_PATH = Path(
    os.environ.get(
        "VIDEO_LOCAL_PATH",
        str(Path(__file__).parent.parent.parent / "example" / "nvidia_2min.mp4")
    )
)

# Memory-augmented analysis controls (tune as needed)
MEMORY_WINDOW_SEC = float(os.environ.get("MEMORY_WINDOW_SEC", "15"))  # +/- seconds
MEMORY_TOP_K = int(os.environ.get("MEMORY_TOP_K", "6"))
MAX_MEMORY_CHARS = int(os.environ.get("MAX_MEMORY_CHARS", "1200"))

# Must match memory_writer.py: base_dt = datetime(2026, 1, 1) + timedelta(seconds=start_sec)
# Scale factor is 1 (1 video second = 1 datetime second, no scaling applied).
_BASE_DATETIME = os.environ.get("BASE_DATETIME", "2026-01-01T00:00:00+00:00")
_TIME_SCALE_FACTOR = int(os.environ.get("TIME_SCALE_FACTOR", "1"))

# Remote execution settings — must match videngram/config.py RemoteConfig
_REMOTE_HOST = os.environ.get("REMOTE_HOST", "")
_REMOTE_WORK_DIR = os.environ.get("REMOTE_WORK_DIR", "")

# Work directory for local cues files (shared with VidEngram pipeline)
_WORK_DIR = Path(os.environ.get("VIDENGRAM_WORK_DIR", "/tmp/videngram"))

# Upload directory for user-uploaded videos
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(_WORK_DIR / "uploads")))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(base_url=VLLM_URL, api_key=_QWEN_API_KEY or "EMPTY")

# Active video state — updated by /upload; empty until first upload
_active_video_remote_path: str = VIDEO_DEFAULT_PATH
_active_video_local_path: str = VIDEO_DEFAULT_PATH  # always the local upload path

# ── Video history (persisted to disk) ─────────────────────────────────────
_VIDEO_HISTORY_FILE = _WORK_DIR / "video_history.json"


def _load_history() -> list[dict]:
    """Load video history from disk. Returns [] if not found or corrupt."""
    try:
        if _VIDEO_HISTORY_FILE.exists():
            return json.loads(_VIDEO_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history_entry(entry: dict) -> None:
    """Append or update a video history entry by group_id."""
    _WORK_DIR.mkdir(parents=True, exist_ok=True)
    history = _load_history()
    # Replace existing entry with same group_id, or append new
    history = [e for e in history if e.get("group_id") != entry["group_id"]]
    history.append(entry)
    _VIDEO_HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def _generate_thumbnail(local_path: str, group_id: str) -> bool:
    """Extract first frame from video and save as {group_id}_thumb.jpg."""
    thumb_path = _WORK_DIR / f"{group_id}_thumb.jpg"
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-ss", "1", "-i", local_path,
            "-vframes", "1", "-q:v", "2", "-vf", "scale=240:-1",
            str(thumb_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        return thumb_path.exists()
    except Exception as e:
        print(f"[thumb] Error generating thumbnail: {e}")
        return False

# LLM config for graph builder (re-uses the same LLM_* env vars as EverMemOS)
_LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("PLANNING_LLM_BASE_URL", "")
_LLM_API_KEY  = os.environ.get("LLM_API_KEY")  or os.environ.get("PLANNING_LLM_API_KEY", "")
_LLM_MODEL    = os.environ.get("LLM_MODEL")    or os.environ.get("PLANNING_LLM_MODEL", "gpt-4o")

# ── Memory helper functions ───────────────────────────────────────────────

def _video_group_id(path: str) -> str:
    """
    Reproduce pipeline.py's group_id computation — must stay in sync.
    pipeline.py resolves the video path before hashing, so we do the same.
    """
    name = Path(path).stem
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = path
    h = hashlib.md5(resolved.encode()).hexdigest()[:8]
    return f"vid_{name}_{h}"


def _parse_evermemos_items_with_meta(
    response_data,
) -> list[tuple[str, Optional[str]]]:
    """Extract (content, create_time) pairs from EverMemOS v1 search response.

    EverMemOS actual search response format:
      {
        "status": "ok",
        "result": {
          "memories": [{"episodic_memory": [{memory_type, timestamp, summary, episode, ...}, ...]}, ...],
          "original_data": [...],
          "pending_messages": [{content, message_create_time, ...}, ...]
        }
      }

    memories is List[Dict[type_name, List[MemCell]]] — keyed by memory TYPE, not group_id.
    pending_messages contain original raw content (our [Video M:SS - N:SS] format).
    """
    results: list[tuple[str, Optional[str]]] = []

    # Unwrap envelope
    inner = response_data
    if isinstance(response_data, dict) and "result" in response_data:
        inner = response_data["result"]

    if not isinstance(inner, dict):
        # Bare list fallback
        for item in (inner if isinstance(inner, list) else []):
            if isinstance(item, str) and item:
                results.append((item, None))
            elif isinstance(item, dict):
                c = item.get("content") or item.get("text") or item.get("memory", "")
                if c:
                    results.append((c, item.get("create_time") or item.get("timestamp")))
        return results

    def _extract_from_mem(mem: dict) -> Optional[tuple[str, Optional[str]]]:
        """Extract (content, time) from a single MemCell or raw message dict."""
        # MemCell types: episodic_memory has episode/summary; event_log has atomic_fact
        c = (
            mem.get("episode") or
            mem.get("summary") or
            mem.get("atomic_fact") or
            mem.get("content") or
            mem.get("memory") or
            mem.get("text") or
            mem.get("subject") or
            ""
        )
        if isinstance(c, list):
            c = "; ".join(str(x) for x in c if x)
        c = str(c).strip()
        if not c or "[Caption error" in c or "[Clip unavailable]" in c:
            return None
        ct = (
            mem.get("timestamp") or
            mem.get("create_time") or
            mem.get("created_at") or
            mem.get("message_create_time")
        )
        return (c, ct)

    def _parse_grouped(grouped: list):
        """Parse List[Dict[type_name, List[MemCell]]] structure."""
        for group_dict in grouped:
            if not isinstance(group_dict, dict):
                continue
            for mem_list in group_dict.values():
                if not isinstance(mem_list, list):
                    continue
                for mem in mem_list:
                    if isinstance(mem, str) and mem:
                        results.append((mem, None))
                    elif isinstance(mem, dict):
                        pair = _extract_from_mem(mem)
                        if pair:
                            results.append(pair)

    # 1. Process MemCell memories (episodic_memory, event_log, foresight)
    raw_mems = inner.get("memories", [])
    if isinstance(raw_mems, list):
        _parse_grouped(raw_mems)

    # 2. Process original_data (raw conversation data — preserves [Video M:SS - N:SS] prefix)
    orig = inner.get("original_data", [])
    if isinstance(orig, list):
        _parse_grouped(orig)

    # 3. Process pending_messages (unconsumed — always have original content + message_create_time)
    pending = inner.get("pending_messages", [])
    if isinstance(pending, list):
        for pm in pending:
            if isinstance(pm, str) and pm:
                results.append((pm, None))
            elif isinstance(pm, dict):
                c = pm.get("content", "")
                if c and "[Caption error" not in c and "[Clip unavailable]" not in c:
                    ct = (
                        pm.get("message_create_time") or
                        pm.get("create_time") or
                        pm.get("created_at")
                    )
                    results.append((c, ct))

    # 4. Legacy format fallbacks (older EverMemOS versions)
    if not results:
        for key in ("results", "data"):
            items = inner.get(key, [])
            if isinstance(items, list) and items:
                for item in items:
                    if isinstance(item, str) and item:
                        results.append((item, None))
                    elif isinstance(item, dict):
                        pair = _extract_from_mem(item)
                        if pair:
                            results.append(pair)
                break

    return results


def _create_time_to_video_sec(create_time_str: str) -> Optional[float]:
    """Reverse video_sec_to_datetime: EverMemOS create_time → video seconds."""
    try:
        from datetime import timezone
        base = datetime.fromisoformat(_BASE_DATETIME)
        dt = datetime.fromisoformat(create_time_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        return (dt - base).total_seconds() / _TIME_SCALE_FACTOR
    except Exception:
        return None


def _extract_ts_range(content: str):
    """Extract (start_sec, end_sec) from [Video/Episode M:SS - N:SS] prefix."""
    m = re.search(
        r'\[(?:Video|Episode)(?:\s+analysis)?\s+(\d+:\d{2}(?::\d{2})?)\s*-\s*(\d+:\d{2}(?::\d{2})?)\]',
        content,
    )
    if m:
        def _parse(ts: str) -> float:
            parts = ts.split(":")
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            return int(parts[0]) * 60 + int(parts[1])
        return _parse(m.group(1)), _parse(m.group(2))
    return None


def _extract_scene_dialogue(content: str) -> tuple[str, str]:
    """
    Extract (scene, dialogue) from a memory content string.
    Supports:
      - 'Scene: ...' / 'Dialogue: ...'
      - older 'dialogue/speech: ...'
      - fallback to quoted speech
    """
    body = re.sub(r'^\[[^\]]+\]\s*', '', content).strip()

    # Field boundary lookahead: only stop at all-caps labels like SOUNDS:, PEOPLE:, etc.
    # Must NOT stop at mixed-case speaker labels like "Speaker 1:" or "Alice:" inside transcripts.
    _FIELD_BOUNDARY = r'(?=\n\s*[A-Z]{2,}[^a-z\n]*\s*[:\-]|\Z)'

    dialogue = ""
    for pat in [
        r'(?i)dialogue\s*[:\-]\s*(.+?)' + _FIELD_BOUNDARY,
        r'(?i)(?:dialogue(?:[/_]or[/_]speech)?|speech)\s*[:\-]\s*(.+?)' + _FIELD_BOUNDARY,
        r'"([^"]{10,})"',
    ]:
        m = re.search(pat, body, re.DOTALL)
        if m:
            dialogue = m.group(1).strip().strip('"\'')
            if dialogue.lower() in ("none", "n/a", "-", "–", ""):
                dialogue = ""
            break

    scene = ""
    m = re.search(
        r'(?i)scene(?:[/_]description)?\s*[:\-]\s*(.+?)' + _FIELD_BOUNDARY,
        body, re.DOTALL,
    )
    if m:
        scene = m.group(1).strip()

    # Include PEOPLE field so character descriptions appear in the scene panel
    people = ""
    m_people = re.search(
        r'(?i)people\s*[:\-]\s*(.+?)' + _FIELD_BOUNDARY,
        body, re.DOTALL,
    )
    if m_people:
        people_text = m_people.group(1).strip()
        if people_text.lower() not in ("none", "n/a", "-", "–", ""):
            people = people_text

    if people:
        scene = (scene + "\n\n" + people) if scene else people

    if not scene:
        scene = body

    return scene, dialogue


# ── Memory cache (timestamped memories) ───────────────────────────────────
_mem_cache: list[tuple[float, float, str]] | None = None
_cache_dirty = True   # True => reload on next access
_cache_empty_retries = 0
_MAX_EMPTY_RETRIES = 5    # keep retrying while EverMemOS index is still building


async def _build_cache(group_id: str) -> list[tuple[float, float, str]]:
    """
    Build the timestamp → content cache.

    Priority:
      1. Local cues file ({work_dir}/{group_id}_cues.json) — saved by pipeline during ingest.
         This is the most reliable source: exact timestamps, original content format.
      2. EverMemOS search — fallback for already-ingested videos without a local file.
         Fixes vs. previous implementation:
           - top_k ≤ 100 (EverMemOS max)
           - No memory_type filter (wrong param; omit to get all types)
           - Correct response parsing (memories are List[Dict[type, List[MemCell]]])
           - create_time fallback via MemCell timestamp field

    Sort by (span ascending, start ascending) so shorter/more-specific memories
    win when multiple ranges overlap the same timestamp.
    """
    entries: list[tuple[float, float, str]] = []

    # ── Priority 1: Local cues file ──────────────────────────────────────
    cues_file = _WORK_DIR / f"{group_id}_cues.json"
    if cues_file.exists():
        try:
            cues_data = json.loads(cues_file.read_text(encoding="utf-8"))
            for cue in cues_data:
                s = float(cue["start_sec"])
                e = float(cue["end_sec"])
                c = cue.get("content", "")
                if c:
                    entries.append((s, e, c))
            if entries:
                print(f"[cache] Loaded {len(entries)} entries from local cues file {cues_file}")
                entries.sort(key=lambda x: (x[1] - x[0], x[0]))
                return entries
            print(f"[cache] Local cues file empty, falling back to EverMemOS")
        except Exception as ex:
            print(f"[cache] Local cues file read error: {ex}")
            entries = []

    # ── Priority 2: EverMemOS search ─────────────────────────────────────
    search_url = f"{EVERMEMOS_URL}/api/v1/memories/search"

    # EverMemOS v1: top_k max=100; omit memory_types to get all types.
    # Try a broad descriptive query first, then a minimal one as fallback.
    query_variants = [
        {"query": "video scene description dialogue events", "group_id": group_id, "top_k": 100},
        {"query": "video",                                   "group_id": group_id, "top_k": 100},
    ]

    for params in query_variants:
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                resp = await c.get(search_url, params=params)
            if resp.status_code != 200:
                print(f"[cache] EverMemOS {resp.status_code}: {resp.text[:200]}")
                continue

            raw = resp.json()
            items_with_meta = _parse_evermemos_items_with_meta(raw)
            print(f"[cache] EverMemOS returned {len(items_with_meta)} parsed items "
                  f"(query={params['query']!r})")

            for content, create_time in items_with_meta:
                # Primary: exact [Video/Episode M:SS - N:SS] prefix
                rng = _extract_ts_range(content)
                # Fallback: MemCell timestamp → reverse map to video seconds
                if not rng and create_time:
                    vid_sec = _create_time_to_video_sec(create_time)
                    if vid_sec is not None and vid_sec >= 0:
                        half = 5.0
                        rng = (max(0.0, vid_sec - half), vid_sec + half)
                if rng:
                    entries.append((rng[0], rng[1], content))

            if entries:
                print(f"[cache] Built {len(entries)} timestamped entries from EverMemOS")
                break
            else:
                print(f"[cache] 0 timestamped entries from {len(items_with_meta)} items "
                      f"(query={params['query']!r})")

        except Exception as e:
            print(f"[cache] Build error: {e}")

    if not entries:
        print(f"[cache] WARNING: empty cache for group_id={group_id}")

    entries.sort(key=lambda x: (x[1] - x[0], x[0]))
    return entries


async def _ensure_cache(group_id: str) -> list[tuple[float, float, str]]:
    """Ensure the in-process cache is loaded and up to date.

    Unlike a simple dirty-flag cache, we keep retrying while the result is
    empty (up to _MAX_EMPTY_RETRIES times).  This handles the window between
    ingest completion and EverMemOS finishing its Elasticsearch/Milvus index
    build, during which search may legitimately return 0 results.
    """
    global _mem_cache, _cache_dirty, _cache_empty_retries
    needs_rebuild = (
        _cache_dirty
        or _mem_cache is None
        or (len(_mem_cache) == 0 and _cache_empty_retries < _MAX_EMPTY_RETRIES)
    )
    if needs_rebuild:
        _mem_cache = await _build_cache(group_id)
        _cache_dirty = False
        if _mem_cache:
            _cache_empty_retries = 0   # success — reset retry counter
        else:
            _cache_empty_retries += 1
    return _mem_cache or []


def _compact_memory_text(content: str) -> str:
    """Compact a memory into a short snippet for prompting."""
    prefix = ""
    m = re.match(r'^\[[^\]]+\]', content.strip())
    if m:
        prefix = m.group(0)
    scene, dialogue = _extract_scene_dialogue(content)

    parts = []
    if prefix:
        parts.append(prefix)
    if scene:
        parts.append(f"Scene: {scene}")
    if dialogue:
        parts.append(f"Dialogue: {dialogue}")

    s = " | ".join(parts).strip()
    if len(s) > 300:
        s = s[:297] + "..."
    return s


async def _retrieve_memories_for_ts(group_id: str, ts_sec: float) -> str:
    """
    Retrieve relevant memories near ts_sec and format them as a short context block.
    Prefer overlap with [ts-window, ts+window]. Fallback to nearest segments.
    """
    entries = await _ensure_cache(group_id)
    if not entries:
        return ""

    w0 = max(ts_sec - MEMORY_WINDOW_SEC, 0.0)
    w1 = ts_sec + MEMORY_WINDOW_SEC

    overlapping = []
    for start_sec, end_sec, content in entries:
        if end_sec >= w0 and start_sec <= w1:
            overlapping.append((start_sec, end_sec, content))

    selected = overlapping[:MEMORY_TOP_K]

    if not selected:
        by_dist = []
        for start_sec, end_sec, content in entries:
            dist = min(abs(ts_sec - start_sec), abs(ts_sec - end_sec))
            by_dist.append((dist, start_sec, end_sec, content))
        by_dist.sort(key=lambda x: x[0])
        selected = [(a, b, c) for _, a, b, c in by_dist[:MEMORY_TOP_K]]

    lines = []
    total_chars = 0
    for _, _, content in selected:
        snippet = _compact_memory_text(content)
        if not snippet:
            continue
        if total_chars + len(snippet) + 2 > MAX_MEMORY_CHARS:
            break
        lines.append(f"- {snippet}")
        total_chars += len(snippet) + 2

    return "\n".join(lines).strip()


def _sec_to_mmss(sec: float) -> str:
    """Convert seconds to [MM:SS] string."""
    total = int(sec)
    return f"[{total // 60:02d}:{total % 60:02d}]"


async def _retrieve_global_memories_for_connection(group_id: str) -> list[tuple[str, str]]:
    """
    Retrieve a temporally-diverse sample of all video memories for cross-segment
    connection analysis.

    Returns K = max(1, total // 5) entries sampled evenly across the timeline,
    each as a (mmss_label, snippet) tuple so the model can cite exact timestamps.
    """
    entries = await _ensure_cache(group_id)
    if not entries:
        return []

    # Sort by start time for temporal spread
    sorted_entries = sorted(entries, key=lambda x: x[0])
    total = len(sorted_entries)
    k = max(1, total // 5)

    # Evenly sample k indices across the sorted list
    if k >= total:
        selected = sorted_entries
    else:
        step = total / k
        selected = [sorted_entries[int(i * step)] for i in range(k)]

    result = []
    for start_sec, _end_sec, content in selected:
        label = _sec_to_mmss(start_sec)
        snippet = _compact_memory_text(content)
        if snippet:
            result.append((label, snippet))

    return result


# ── /ingest — SSE stream of ingest subprocess output ──────────────────────
@app.post("/ingest")
async def ingest_video():
    """Run `python -m demo.cli -v ingest <video>` and stream output as SSE."""
    _ensure_llm_ready()
    if not _active_video_remote_path:
        raise HTTPException(status_code=400, detail="No video uploaded yet. Please upload a video first.")
    # Clear stale cues files so the server doesn't serve old memories during re-ingest
    group_id = _video_group_id(_active_video_remote_path)
    for suffix in ("_cues.json", "_speech.json"):
        stale = _WORK_DIR / f"{group_id}{suffix}"
        try:
            stale.unlink(missing_ok=True)
        except Exception:
            pass
    global _mem_cache, _cache_dirty
    _mem_cache = None
    _cache_dirty = True

    # Clear stale graph cache from previous ingest (per-video file)
    _graph_builder.clear()
    stale_graph = _WORK_DIR / f"graph_cache_{group_id}.json"
    try:
        stale_graph.unlink(missing_ok=True)
    except Exception:
        pass

    async def run():
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1"}
            if _QWEN_API_KEY:
                env.setdefault("QWEN_API_KEY", _QWEN_API_KEY)
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "demo.cli", "-v", "ingest", _active_video_remote_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(VIDENGRAM_DIR),
                env=env,
                limit=2**20,  # 1 MB line limit (default 64 KB is too small for verbose output)
            )
            async for raw in proc.stdout:
                text = raw.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue

                # Structured events emitted by demo/cli.py for the UI
                seg_payload = _decode_cli_event(text, "SEGMENTS_JSON")
                if seg_payload is not None:
                    yield f"data: {json.dumps({'segments': seg_payload}, ensure_ascii=False)}\n\n"
                    continue

                cap_payload = _decode_cli_event(text, "CAPTION_JSON")
                if cap_payload is not None:
                    yield f"data: {json.dumps({'caption': cap_payload}, ensure_ascii=False)}\n\n"
                    continue

                # Regular log line
                print(f"[ingest] {text}")
                yield f"data: {json.dumps({'line': text}, ensure_ascii=False)}\n\n"

            rc = await proc.wait()
            print(f"[ingest] Process exited with returncode={rc}")
            current_group_id = _video_group_id(_active_video_remote_path)

            # Save history and generate thumbnail BEFORE yielding done,
            # so the frontend can fetch /videos immediately on receiving done.
            if rc == 0:
                await _generate_thumbnail(_active_video_local_path, current_group_id)
                filename = Path(_active_video_local_path).name
                _save_history_entry({
                    "group_id": current_group_id,
                    "video_path": _active_video_remote_path,
                    "video_local_path": _active_video_local_path,
                    "video_url": f"/uploaded/{filename}",
                    "filename": filename,
                })

            yield f"data: {json.dumps({'done': True, 'returncode': rc, 'group_id': current_group_id})}\n\n"

            # Trigger background graph builds (per-video + all-videos) after successful ingest
            if rc == 0:
                _graph_builder.trigger_build(
                    work_dir=_WORK_DIR,
                    group_id=current_group_id,
                    evermemos_url=EVERMEMOS_URL,
                    llm_base_url=_LLM_BASE_URL,
                    llm_api_key=_LLM_API_KEY,
                    llm_model=_LLM_MODEL,
                )
                # Delete stale all-videos cache and rebuild in parallel
                stale_all = _WORK_DIR / "graph_cache_all.json"
                try:
                    stale_all.unlink(missing_ok=True)
                except Exception:
                    pass
                _graph_builder.clear_all()
                all_history = _load_history()
                group_ids_and_names = [
                    (e["group_id"], e.get("filename", e["group_id"]))
                    for e in all_history
                ]
                if group_ids_and_names:
                    _graph_builder.trigger_build_all(
                        work_dir=_WORK_DIR,
                        group_ids_and_names=group_ids_and_names,
                        llm_base_url=_LLM_BASE_URL,
                        llm_api_key=_LLM_API_KEY,
                        llm_model=_LLM_MODEL,
                    )
        except Exception as e:
            print(f"[ingest] SSE stream error: {type(e).__name__}: {e}")
            try:
                proc.kill()
            except Exception:
                pass
            yield f"data: {json.dumps({'error': str(e), 'done': True, 'returncode': 1})}\n\n"

    return StreamingResponse(run(), media_type="text/event-stream")


# ── /memories — query best memory by timestamp ────────────────────────────
@app.get("/memories")
async def get_memories(ts: float = 0.0):
    """
    Return the pre-computed memory that best matches video time `ts`.
    Priority:
      1) Shortest-span memory covering `ts`
      2) Nearest memory (fallback)
    """
    group_id = _video_group_id(_active_video_remote_path)
    entries = await _ensure_cache(group_id)
    if not entries:
        return {"scene": "(No matching memory)", "dialogue": ""}

    best_content = None
    best_span = float("inf")

    for start_sec, end_sec, content in entries:
        if start_sec <= ts <= end_sec:
            span = end_sec - start_sec
            if span < best_span:
                best_span = span
                best_content = content

    if best_content is None:
        best_dist = float("inf")
        for start_sec, end_sec, content in entries:
            dist = min(abs(ts - start_sec), abs(ts - end_sec))
            if dist < best_dist:
                best_dist = dist
                best_content = content

    if not best_content:
        return {"scene": "(No matching memory)", "dialogue": ""}

    scene, dialogue = _extract_scene_dialogue(best_content)
    return {"scene": scene, "dialogue": dialogue}


@app.post("/reload_cache")
async def reload_cache():
    """Mark cache stale so the next request reloads from EverMemOS."""
    global _cache_dirty, _cache_empty_retries
    _cache_dirty = True
    _cache_empty_retries = 0
    return {"status": "ok"}


@app.get("/debug/evermemos")
async def debug_evermemos():
    """Debug: show raw EverMemOS search response and parsed items."""
    group_id = _video_group_id(_active_video_remote_path)
    search_url = f"{EVERMEMOS_URL}/api/v1/memories/search"
    cues_file = _WORK_DIR / f"{group_id}_cues.json"
    out = {
        "group_id": group_id,
        "evermemos_url": EVERMEMOS_URL,
        "cues_file": str(cues_file),
        "cues_file_exists": cues_file.exists(),
        "cache_size": len(_mem_cache) if _mem_cache is not None else None,
        "queries": [],
    }
    for params in [
        {"query": "video scene description dialogue events", "group_id": group_id, "top_k": 5},
    ]:
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                resp = await c.get(search_url, params=params)
            raw = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
            parsed = _parse_evermemos_items_with_meta(raw) if isinstance(raw, dict) else []
            out["queries"].append({
                "params": params,
                "status": resp.status_code,
                "raw_keys": list(raw.keys()) if isinstance(raw, dict) else str(raw)[:200],
                "result_keys": list(raw.get("result", {}).keys()) if isinstance(raw, dict) else [],
                "memories_count": len(raw.get("result", {}).get("memories", [])) if isinstance(raw, dict) else 0,
                "pending_count": len(raw.get("result", {}).get("pending_messages", [])) if isinstance(raw, dict) else 0,
                "parsed_count": len(parsed),
                "first_item": parsed[0] if parsed else None,
            })
        except Exception as e:
            out["queries"].append({"params": params, "error": str(e)})
    return out


# ── /segment_cues — full raw caption content for history feed rendering ──
@app.get("/segment_cues")
async def segment_cues():
    """
    Return all segment cues with full raw caption content (sorted by start_sec).
    Used by the frontend to rebuild the memory feed when switching to a historical video.
    [{start: float, end: float, content: str}, ...]
    """
    group_id = _video_group_id(_active_video_remote_path)
    cues_file = _WORK_DIR / f"{group_id}_cues.json"
    if not cues_file.exists():
        # Fall back to cache entries if no cues file
        entries = await _ensure_cache(group_id)
        cues = [
            {"start": float(s), "end": float(e), "content": c}
            for s, e, c in sorted(entries, key=lambda x: x[0])
        ]
        return {"group_id": group_id, "count": len(cues), "cues": cues}
    try:
        cues_data = json.loads(cues_file.read_text(encoding="utf-8"))
        cues = []
        for cue in cues_data:
            content = (cue.get("content") or "").strip()
            if not content:
                continue
            cues.append({
                "start": float(cue["start_sec"]),
                "end": float(cue["end_sec"]),
                "content": content,
            })
        # Sort by start time for chronological display
        cues.sort(key=lambda x: x["start"])
        return {"group_id": group_id, "count": len(cues), "cues": cues}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load segment cues: {e}")


# ── NEW: /subtitle_cues — build subtitle cues from timestamped memories ───
@app.get("/subtitle_cues")
async def subtitle_cues():
    """
    Return subtitle cues derived from memories:
      [{start: float, end: float, text: str}, ...]
    Only includes entries with non-empty dialogue.
    """
    group_id = _video_group_id(_active_video_remote_path)
    entries = await _ensure_cache(group_id)
    cues = []
    for start_sec, end_sec, content in entries:
        _, dialogue = _extract_scene_dialogue(content)
        dialogue = (dialogue or "").strip()
        if not dialogue:
            continue
        cues.append({"start": float(start_sec), "end": float(end_sec), "text": dialogue})

    # Sort shorter spans first so getSubtitleAt() returns the most-specific cue
    cues.sort(key=lambda x: (x["end"] - x["start"], x["start"]))
    return {"group_id": group_id, "count": len(cues), "cues": cues}


# ── /speech_cues — Whisper transcription cues (highest subtitle priority) ─
@app.get("/speech_cues")
async def speech_cues():
    """
    Return speech cues from the Whisper transcription saved by the pipeline.
    Loaded from {work_dir}/{group_id}_speech.json.

    [{start_sec: float, end_sec: float, text: str}, ...]
    Returns {"cues": []} if the file does not exist (transcription disabled or
    not yet run).
    """
    group_id = _video_group_id(_active_video_remote_path)
    speech_file = _WORK_DIR / f"{group_id}_speech.json"
    if not speech_file.exists():
        return {"cues": []}
    try:
        cues = json.loads(speech_file.read_text(encoding="utf-8"))
        return {"cues": cues}
    except Exception as e:
        print(f"[speech_cues] Failed to load {speech_file}: {e}")
        return {"cues": []}


# ── NEW: /memory_cues — scene + dialogue cues for frontend pre-loading ────
@app.get("/memory_cues")
async def memory_cues():
    """
    Return all timestamped memory cues with both scene and dialogue fields.
    Sorted by (span ascending, start ascending) so the frontend's linear scan
    always finds the most-specific (shortest) memory that covers a timestamp.

    [{start: float, end: float, scene: str, dialogue: str}, ...]
    """
    group_id = _video_group_id(_active_video_remote_path)
    entries = await _ensure_cache(group_id)
    cues = []
    for start_sec, end_sec, content in entries:
        scene, dialogue = _extract_scene_dialogue(content)
        scene = scene.strip()
        if not scene:
            continue
        cues.append({
            "start": float(start_sec),
            "end": float(end_sec),
            "scene": scene,
            "dialogue": dialogue.strip(),
        })
    # Shorter spans first → most-specific memory wins on first match
    cues.sort(key=lambda x: (x["end"] - x["start"], x["start"]))
    return {"group_id": group_id, "count": len(cues), "cues": cues}


# ── Store live frame analysis back to EverMemOS ───────────────────────────
def _fmt_ts(sec: float) -> str:
    """Format seconds as M:SS or H:MM:SS timecode."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _format_video_prefix(ts_sec: float, span_sec: float = 3.0) -> str:
    """Build a [Video M:SS - N:SS] prefix so /memories can parse it."""
    start = max(ts_sec - span_sec / 2, 0.0)
    end = max(ts_sec + span_sec / 2, 0.0)
    return f"[Video {_fmt_ts(start)} - {_fmt_ts(end)}]"


async def store_to_evermemos(ts_sec: float, scene: str, dialogue: str):
    """
    Store analysis result under the SAME group_id as the video,
    with timestamp prefix + Scene/Dialogue fields.
    """
    global _cache_dirty

    prefix = _format_video_prefix(ts_sec, span_sec=3.0)
    content_lines = [
        prefix,
        f"Scene: {(scene or '').strip()}",
        f"Dialogue: {(dialogue or '').strip()}",
    ]
    content = "\n".join(content_lines).strip() or "(Empty)"

    iso_time = datetime.now(tz=timezone.utc).isoformat()
    msg_id = f"frame_{int(ts_sec * 1000)}"

    group_id = _video_group_id(_active_video_remote_path)
    payload = {
        "message_id": msg_id,
        "create_time": iso_time,
        "sender": "videngram",
        "sender_name": "VidEngram",
        "role": "assistant",
        "content": content,
        "group_id": group_id,
        "group_name": f"VidEngram Live Frame Analysis: {Path(_active_video_remote_path).name}",
        "scene": "assistant",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as c:
            resp = await c.post(f"{EVERMEMOS_URL}/api/v1/memories", json=payload)
            if resp.status_code >= 400:
                print(f"[EverMemOS] Store failed {resp.status_code}: {resp.text[:200]}")
            else:
                _cache_dirty = True
    except Exception as e:
        print(f"[EverMemOS] Request error: {e}")


# ── /analyze — memory-augmented frame analysis (image-only) ───────────────
@app.post("/analyze")
async def analyze(data: dict):
    """
    Expected JSON body:
      - frame: base64 JPEG (with or without data:image/... prefix)
      - timestamp: seconds (float)
    SSE response:
      - data: {"token": "...", "ts": ...}
      - data: {"final": {"scene": "...", "dialogue": "..."}, "ts": ...}

    Note: This endpoint is image-only. It cannot reliably transcribe speech audio.
    Subtitles should come from /subtitle_cues (ingest memories) unless you add audio.
    """
    _ensure_llm_ready()

    frame_b64 = data["frame"]
    ts_sec = float(data.get("timestamp", 0.0))

    if frame_b64.startswith("data:image"):
        frame_b64 = frame_b64.split(",", 1)[-1]

    group_id = _video_group_id(_active_video_remote_path)

    async def stream_response():
        global_mems = await _retrieve_global_memories_for_connection(group_id)

        instruction = (
            "IMPORTANT: All output values must be written in ENGLISH regardless of the video language.\n\n"
            "You will be given (1) a video frame image and (2) a list of memories from different moments in the same video.\n"
            "Your task has TWO parts:\n\n"
            "PART 1 — Describe the current frame:\n"
            "  Write a concise visual description of what you see (scene), and note any on-screen dialogue text.\n\n"
            "PART 2 — Find genuine cross-segment connections:\n"
            "  Look at the provided video memories and identify REAL connections between the current frame\n"
            "  and other moments in the video. A connection can be:\n"
            "  - The same person/entity appearing at a different moment\n"
            "  - A topic, theme, or object that recurs or develops across the video\n"
            "  - An emotional or narrative arc linking two moments\n"
            "  - A contrast or turning point compared to an earlier scene\n"
            "  When you cite a connection, you MUST include the memory timestamp in [MM:SS] format.\n"
            "  You are encouraged to cite multiple memories if multiple genuine connections exist.\n"
            "  If you find NO genuine connection (the memories are unrelated to the current frame), set connections to empty string.\n"
            "  Do NOT force connections that do not exist.\n\n"
            "Return STRICT JSON and NOTHING ELSE (all values in English):\n"
            '{"scene":"<concise visual description in English>","dialogue":"<on-screen text only in English, else empty string>",'
            '"connections":"<English sentences explicitly stating connections to [MM:SS] moments, or empty string if none>"}'
        )

        if global_mems:
            mem_lines = [f"  {label} {snippet}" for label, snippet in global_mems]
            memory_text = (
                f"Global video memories ({len(global_mems)} sampled from entire video — use timestamps to cite connections):\n"
                + "\n".join(mem_lines)
            )
        else:
            memory_text = "Global video memories: (none available — video not yet ingested)"

        full_text = ""
        stream = await client.chat.completions.create(
            model=os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Omni-7B"),
            messages=[
                {"role": "system", "content": instruction},
                {"role": "system", "content": memory_text},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                    {"type": "text", "text": "Analyze this frame and find connections to other video moments."},
                ]},
            ],
            stream=True,
            max_tokens=600,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_text += token
                yield f"data: {json.dumps({'token': token, 'ts': ts_sec}, ensure_ascii=False)}\n\n"

        result = {"scene": full_text.strip(), "dialogue": "", "connections": ""}
        try:
            parsed = json.loads(full_text.strip())
            result = {
                "scene": (parsed.get("scene") or "").strip(),
                "dialogue": (parsed.get("dialogue") or "").strip(),
                "connections": (parsed.get("connections") or "").strip(),
            }
        except json.JSONDecodeError:
            m = re.search(r'\{[^{}]*"scene"[^{}]*\}', full_text, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group())
                    result = {
                        "scene": (parsed.get("scene") or "").strip(),
                        "dialogue": (parsed.get("dialogue") or "").strip(),
                        "connections": (parsed.get("connections") or "").strip(),
                    }
                except json.JSONDecodeError:
                    pass

        yield f"data: {json.dumps({'final': result, 'ts': ts_sec}, ensure_ascii=False)}\n\n"
        asyncio.create_task(store_to_evermemos(ts_sec, result["scene"], result["dialogue"]))

    return StreamingResponse(stream_response(), media_type="text/event-stream")


# ── /upload — receive a video file from the browser ───────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Save an uploaded video locally, then (when remote mode is active) SCP it to
    the analysis server so the segmenter/captioner pipeline can access it via SSH.
    Sets _active_video_remote_path to whichever path the pipeline must receive.
    Returns: {status, filename, video_url}
    """
    global _active_video_remote_path, _active_video_local_path, _mem_cache, _cache_dirty, _cache_empty_retries

    filename = Path(file.filename or "video.mp4").name  # strip path traversal
    local_save_path = UPLOAD_DIR / filename

    # ── 1. Write upload to local disk in 1 MB chunks ──────────────────────
    with open(local_save_path, "wb") as f_out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f_out.write(chunk)

    print(f"[upload] Saved locally: {local_save_path}")

    # ── 2. If remote mode is active, SCP file to the analysis server ───────
    if _REMOTE_HOST and _REMOTE_WORK_DIR:
        remote_upload_dir = f"{_REMOTE_WORK_DIR}/uploads"
        remote_path = f"{remote_upload_dir}/{filename}"

        # Create remote uploads directory (best-effort; ignore errors)
        mkdir_proc = await asyncio.create_subprocess_exec(
            "ssh", _REMOTE_HOST, f"mkdir -p {remote_upload_dir}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await mkdir_proc.wait()

        # SCP the local file to the remote server
        print(f"[upload] SCP → {_REMOTE_HOST}:{remote_path}")
        scp_proc = await asyncio.create_subprocess_exec(
            "scp", str(local_save_path), f"{_REMOTE_HOST}:{remote_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        rc = await scp_proc.wait()
        if rc != 0:
            err_bytes = await scp_proc.stderr.read()
            raise HTTPException(
                status_code=500,
                detail=f"SCP to remote server failed: {err_bytes.decode('utf-8', errors='replace').strip()}",
            )

        _active_video_remote_path = remote_path
        print(f"[upload] Active remote path: {_active_video_remote_path}")
    else:
        # Local-only mode (no REMOTE_HOST / REMOTE_WORK_DIR configured)
        _active_video_remote_path = str(local_save_path)
        print(f"[upload] Active local path: {_active_video_remote_path}")

    _active_video_local_path = str(local_save_path)

    # ── 3. Reset memory cache for the new video ────────────────────────────
    _mem_cache = None
    _cache_dirty = True
    _cache_empty_retries = 0

    return {"status": "ok", "filename": filename, "video_url": f"/uploaded/{filename}"}


# ── /qa — agentic Q&A over ingested video memories ────────────────────────
@app.post("/qa")
async def qa_endpoint(data: dict):
    """
    Run the VidEngram agent to answer a question about the ingested video.
    SSE response events:
      {"step": "Searching memories: ..."}   — agent intermediate steps
      {"text": "token..."}                  — streaming answer tokens (word-by-word)
      {"done": true}                        — completion signal
    """
    question = (data.get("question") or "").strip()
    search_scope = data.get("search_scope", "current")
    if not question:
        async def empty():
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    try:
        video_path = str(Path(_active_video_remote_path).resolve())
    except OSError:
        video_path = _active_video_remote_path

    async def stream():
        step_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        result_holder: dict = {}

        # Compute video duration upfront so it can be passed to the agent
        video_duration: Optional[float] = None
        try:
            from videngram.utils import get_video_duration as _get_dur
            video_duration = _get_dur(video_path)
        except Exception:
            pass

        def run_agent():
            try:
                # Lazy import to avoid circular deps at module load time
                import sys as _sys
                _videngram_dir_str = str(VIDENGRAM_DIR)
                if _videngram_dir_str not in _sys.path:
                    _sys.path.insert(0, _videngram_dir_str)
                from videngram.agent import VidEngramAgent
                from videngram.config import VidEngramConfig

                cfg = VidEngramConfig()
                agent = VidEngramAgent(cfg)

                def step_cb(desc: str):
                    loop.call_soon_threadsafe(step_queue.put_nowait, desc)

                response = agent.query(
                    question=question,
                    video_path=video_path,
                    video_duration=video_duration,
                    step_callback=step_cb,
                    search_scope=search_scope,
                )
                result_holder["answer"] = response.answer
                result_holder["sources"] = response.sources
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                # Sentinel to signal completion
                loop.call_soon_threadsafe(step_queue.put_nowait, None)

        # Run the synchronous agent in a thread pool
        agent_task = asyncio.create_task(asyncio.to_thread(run_agent))

        # Yield step events while agent is working
        while True:
            step = await step_queue.get()
            if step is None:
                break
            yield f"data: {json.dumps({'step': step}, ensure_ascii=False)}\n\n"

        await agent_task  # ensure thread is fully done

        if "error" in result_holder:
            err_msg = result_holder["error"]
            yield f"data: {json.dumps({'text': f'[Error: {err_msg}]'}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        answer = result_holder.get("answer", "")
        if not answer:
            yield f"data: {json.dumps({'done': True})}\n\n"
            return

        # Stream the answer word-by-word (20ms per word ≈ natural reading pace)
        words = answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.02)

        # Collect timestamped segments from retrieved sources
        # Only use [Video/Episode M:SS - N:SS] timestamps embedded in memory content.
        # The create_time fallback was removed because it consistently mapped to
        # incorrect positions regardless of actual segment location.
        segments = []
        seen_segments = set()
        for src in result_holder.get("sources", []):
            try:
                ts = src.timestamp_range
                if ts:
                    start_sec, end_sec = ts
                    # Drop segments whose start exceeds video duration
                    if video_duration is not None and start_sec >= video_duration:
                        continue
                    # Clamp end to video duration
                    if video_duration is not None and end_sec > video_duration:
                        end_sec = video_duration
                    key = (start_sec, end_sec)
                    if key not in seen_segments:
                        seen_segments.add(key)
                        segments.append({"startSec": start_sec, "endSec": end_sec})
            except Exception:
                pass
        done_payload: dict = {"done": True, "segments": segments}
        if video_duration is not None:
            done_payload["videoDuration"] = video_duration
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── /graph — relationship graph endpoints ─────────────────────────────────────
@app.get("/graph/status")
async def graph_status():
    """Return the current graph build status: idle | pending | ready | error.
    If a cache file exists from a previous ingest, auto-promote to 'ready'."""
    status = _graph_builder.get_status()
    # Promote from idle to ready when a per-video cache file exists
    if status == "idle":
        current_group_id = _video_group_id(_active_video_remote_path) if _active_video_remote_path else ""
        cache_file = _WORK_DIR / f"graph_cache_{current_group_id}.json" if current_group_id else None
        if cache_file and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                _graph_builder._graph_data = data
                _graph_builder._graph_status = "ready"
                status = "ready"
            except Exception:
                pass
        elif _active_video_remote_path:
            # No cache yet — trigger build from existing memories
            _graph_builder.trigger_build(
                work_dir=_WORK_DIR,
                group_id=_video_group_id(_active_video_remote_path),
                evermemos_url=EVERMEMOS_URL,
                llm_base_url=_LLM_BASE_URL,
                llm_api_key=_LLM_API_KEY,
                llm_model=_LLM_MODEL,
            )
            status = _graph_builder.get_status()
    return {"status": status}


@app.get("/graph/data")
async def graph_data():
    """Return the built relationship graph JSON."""
    data = _graph_builder.get_data()
    if data is None:
        # Try loading from per-video disk cache (e.g. server restart after ingest)
        current_group_id = _video_group_id(_active_video_remote_path) if _active_video_remote_path else ""
        cache_file = _WORK_DIR / f"graph_cache_{current_group_id}.json" if current_group_id else None
        if cache_file and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                _graph_builder._graph_data = data
                _graph_builder._graph_status = "ready"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load graph cache: {e}")
        else:
            raise HTTPException(status_code=404, detail="Graph not ready yet")
    return data


# ── /graph/all — all-videos relationship graph endpoints ──────────────────────

@app.get("/graph/all/status")
async def graph_all_status():
    """Return the all-videos graph build status: idle | pending | ready | error.
    Auto-loads from disk cache if available."""
    status = _graph_builder.get_all_status()
    if status == "idle":
        cache_file = _WORK_DIR / "graph_cache_all.json"
        if cache_file.exists():
            _graph_builder.load_all_cached(_WORK_DIR)
            status = _graph_builder.get_all_status()
        elif _load_history():
            # History exists but no cache — trigger build
            all_history = _load_history()
            group_ids_and_names = [
                (e["group_id"], e.get("filename", e["group_id"]))
                for e in all_history
            ]
            _graph_builder.trigger_build_all(
                work_dir=_WORK_DIR,
                group_ids_and_names=group_ids_and_names,
                llm_base_url=_LLM_BASE_URL,
                llm_api_key=_LLM_API_KEY,
                llm_model=_LLM_MODEL,
            )
            status = _graph_builder.get_all_status()
    return {"status": status}


@app.get("/graph/all/data")
async def graph_all_data():
    """Return the all-videos relationship graph JSON."""
    data = _graph_builder.get_all_data()
    if data is None:
        cache_file = _WORK_DIR / "graph_cache_all.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                _graph_builder._all_graph_data = data
                _graph_builder._all_graph_status = "ready"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load all-graph cache: {e}")
        else:
            raise HTTPException(status_code=404, detail="All-videos graph not ready yet")
    return data


# ── /videos — video history endpoints ─────────────────────────────────────

@app.get("/videos")
async def list_videos():
    """Return list of all successfully ingested videos."""
    return _load_history()


@app.get("/videos/{group_id}/thumbnail")
async def video_thumbnail(group_id: str):
    """Serve the thumbnail image for a video."""
    thumb_path = _WORK_DIR / f"{group_id}_thumb.jpg"
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(str(thumb_path), media_type="image/jpeg")


@app.post("/switch_video")
async def switch_video(data: dict):
    """Switch the active video context to a previously ingested video."""
    global _active_video_remote_path, _active_video_local_path, _mem_cache, _cache_dirty, _cache_empty_retries

    group_id = (data.get("group_id") or "").strip()
    if not group_id:
        raise HTTPException(status_code=400, detail="group_id is required")

    history = _load_history()
    entry = next((e for e in history if e.get("group_id") == group_id), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Video not found in history")

    # Switch active video paths
    _active_video_remote_path = entry["video_path"]
    _active_video_local_path = entry.get("video_local_path", entry["video_path"])

    # Reset memory cache so it rebuilds for the new video
    _mem_cache = None
    _cache_dirty = True
    _cache_empty_retries = 0

    # Load the pre-built graph for this video (fast — from disk cache)
    _graph_builder.load_cached(_WORK_DIR, group_id)

    print(f"[switch] Switched active video to {group_id} ({entry['filename']})")
    return {"status": "ok", "group_id": group_id}


# ── Static mounts (order matters) ─────────────────────────────────────────
# /uploaded must come before the /  catch-all
app.mount("/uploaded", StaticFiles(directory=str(UPLOAD_DIR)), name="uploaded")

if VIDEO_LOCAL_PATH.parent.exists():
    app.mount("/media", StaticFiles(directory=str(VIDEO_LOCAL_PATH.parent)), name="media")

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
