# VidEngram: Code Architecture & Utility Guide

**Version:** 0.1.0 | **Target:** Memory Genesis Competition 2026, Track 1 (Agent + Memory)
**Deadline:** March 16, 2026 | **Prize pool:** $80K+

---

## What VidEngram Does

VidEngram bridges two systems that don't natively talk to each other:

- **Qwen2.5-Omni-7B** (served via vLLM-Omni) — a multimodal model that can see video and hear audio simultaneously, but has no persistent memory
- **EverMemOS** — a structured long-term memory OS that achieves 93% on LoCoMo, but only accepts text

The core insight: instead of building multimodal memory from scratch, we use Qwen2.5-Omni as a *perceptual front-end* that converts video into rich structured text, then feed that text through a hippocampal-inspired consolidation pipeline before storing it in EverMemOS. An agentic ReAct loop then orchestrates retrieval, combining fast memory lookups with on-demand video re-analysis for answer grounding.

---

## Repository Structure

```
videngram/
├── videngram/                  # Core library (9 modules, ~2100 lines)
│   ├── __init__.py             # Public API exports
│   ├── config.py               # Dataclass configs with env-var overrides
│   ├── utils.py                # Data classes, timestamp mapping, ffmpeg helpers
│   ├── segmenter.py            # Video → temporal segments (HippoMM-inspired)
│   ├── captioner.py            # Segments → rich captions (Qwen2.5-Omni)
│   ├── consolidator.py         # Captions → deduplicated + summarized memories
│   ├── memory_writer.py        # Memories → EverMemOS (POST /api/v1/memories)
│   ├── memory_reader.py        # EverMemOS → retrieved memories (GET search)
│   ├── agent.py                # ReAct orchestrator with 5 tools
│   └── pipeline.py             # End-to-end ingest + query orchestration
├── demo/
│   ├── cli.py                  # Terminal interface (ingest / query / chat)
│   └── gradio_app.py           # Web UI for video upload + chat
├── examples/
│   └── vllm_omni_video_memory.py   # Standalone demo for vLLM-Omni contribution
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── test_videngram.py       # 38 unit tests (all pass without live services)
├── config/
│   └── default_config.yaml     # Full YAML config with all parameters
├── docker-compose.yml          # MongoDB + Elasticsearch + Milvus + Redis
├── scripts/start_services.sh   # One-command infrastructure launcher
├── pyproject.toml              # PEP 621 packaging
├── requirements.txt            # Pip dependencies
├── .env.template               # Environment variable template
└── README.md                   # Setup and usage documentation
```

---

## Module-by-Module Utility

### 1. `config.py` — Configuration Hub (145 lines)

**What it does:** Centralizes every tunable parameter into typed dataclasses with environment variable overrides. No magic strings scattered across the codebase.

**Why it matters:** The competition judges will modify parameters (different models, ports, thresholds). A clean config layer lets them do that from a single `.env` file or YAML without touching code.

**Key design decisions:**
- Qwen defaults to port **8091** (vLLM-Omni standard), not 8000
- EverMemOS uses **v1 API** endpoints (`/api/v1/memories`, `/api/v1/memories/search`)
- Agent planning LLM **falls back to Qwen** if no external LLM is configured
- `modalities: ["text"]` disables audio output generation for faster captioning

### 2. `utils.py` — Data Classes & Helpers (190 lines)

**What it does:** Defines the data contracts that flow between modules:
- `VideoSegment` — a temporal slice of the source video
- `Caption` — structured text produced by Qwen2.5-Omni for a segment
- `ConsolidatedMemory` — post-consolidation unit ready for EverMemOS
- `MemoryResult` — a single retrieval result from EverMemOS
- `AgentAction` / `AgentResponse` — the agent's reasoning trace

**Critical utility — Timestamp Mapping:**
```
video_sec_to_datetime(90.0, time_scale_factor=60)
# → "2025-01-01T01:30:00+00:00"
```
EverMemOS expects real datetimes for temporal reasoning ("what happened before X?"). We scale video seconds by 60x so that events 1 second apart in the video are 1 minute apart in virtual time — giving EverMemOS enough temporal resolution to reason about ordering.

### 3. `segmenter.py` — Temporal Pattern Separation (242 lines)

**What it does:** Splits a video into semantically meaningful segments.

**Two strategies:**
- **Fixed:** Uniform windows (e.g., every 30 seconds). Simple and predictable.
- **Adaptive** (default): Uses ffmpeg's scene detection filter + silence detection to find natural boundaries. Mirrors HippoMM's temporal pattern separation (§3.1).

**Why adaptive matters:** A 30-second fixed window might split mid-sentence or mid-action. Adaptive segmentation finds the natural "paragraph breaks" in the video — scene transitions, speaker pauses — so each segment is a coherent unit of meaning.

**Robustness:** Falls back to fixed windows if ffmpeg scene detection fails. Enforces min/max duration constraints to prevent too-short or too-long segments.

### 4. `captioner.py` — Perceptual Encoding (240 lines)

**What it does:** Sends extracted video clips to Qwen2.5-Omni via vLLM-Omni's OpenAI-compatible API and gets back structured captions covering scene, people, dialogue, sounds, text, objects, emotion, and temporal progression.

**Key advantage over HippoMM:** Qwen2.5-Omni processes video + audio in a single model pass. HippoMM needed three separate models (ImageBind for visual embedding, Whisper for audio transcription, QwenVL for description) plus a fusion step. Single-pass is more efficient and avoids alignment errors.

**Features:**
- Sync and async captioning (bounded concurrency with `asyncio.Semaphore`)
- `analyze_clip()` method for on-demand queries — the agent uses this to "look at" specific video moments

**API usage:**
```python
# vLLM-Omni OpenAI-compatible endpoint
client.chat.completions.create(
    model="Qwen/Qwen2.5-Omni-7B",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"file://{clip}"}},
        {"type": "text", "text": "Describe this segment..."},
    ]}],
    extra_body={"modalities": ["text"]},  # text-only output
)
```

### 5. `consolidator.py` — Memory Consolidation (306 lines)

**What it does:** This is a core novelty component. Instead of dumping raw captions into EverMemOS (which would be naive RAG), we perform three stages of consolidation:

1. **Filter** — Remove near-duplicate segments using word-overlap similarity (Jaccard). If two consecutive segments describe nearly the same thing (e.g., a static scene), merge them.

2. **Episode Summaries** — Group related segments (using a configurable window) and use an LLM to synthesize them into coherent episode narratives. This creates a hierarchical memory: you can retrieve either the granular segment or the episode overview.

3. **Entity Profiles** — Scan across all episodes to extract recurring people, topics, or concepts into profile entries. These go into EverMemOS as `profile` memory type, enabling "who is X?" queries.

**Why this matters for the competition:** This transforms raw video captions into the kind of structured, hierarchical memory that EverMemOS is designed to reason over. It's the difference between "searching subtitles" and "understanding the video's narrative."

### 6. `memory_writer.py` — EverMemOS Ingestion (153 lines)

**What it does:** Writes consolidated memories to EverMemOS via `POST /api/v1/memories`.

**Key payload mapping:**
| VidEngram field | EverMemOS field | Purpose |
|---|---|---|
| `memory_id` | `message_id` | Unique per memory |
| Virtual datetime | `create_time` | Temporal ordering |
| `"video_{type}"` | `sender` | Source identification |
| Caption text | `content` | The actual memory |
| Video hash | `group_id` | Scopes memories to this video |
| `"assistant"` | `scene` | Optimal for episode extraction |

**Robustness:** Rate-limiting between writes (`delay_between`), configurable indexing wait time, per-message error handling.

### 7. `memory_reader.py` — Memory Retrieval (214 lines)

**What it does:** Retrieves memories from EverMemOS using multiple strategies:

| Mode | EverMemOS method | HippoMM equivalent | When to use |
|---|---|---|---|
| `search_episodes(rrf)` | `rrf` | Fast Retrieval (Φ_fast) | Most queries |
| `search_episodes(bm25)` | `bm25` | — | Keyword-heavy queries |
| `search_episodes(embedding)` | `embedding` | — | Semantic similarity |
| `search_profiles()` | `rrf` on profiles | — | "Who is X?" queries |
| `search_agentic()` | `agentic` | Detailed Recall (Ψ_detailed) | Complex multi-hop |
| `multi_source_search()` | All types | — | Comprehensive retrieval |

All searches are scoped to the video's `group_id`, so memories from different videos don't interfere.

### 8. `agent.py` — ReAct Orchestrator (417 lines)

**What it does:** This is the *most novel component* for the competition. Instead of a simple retrieve→answer pipeline, the agent implements a ReAct reasoning loop:

```
User Question → THINK → ACTION(tool) → OBSERVE → THINK → ... → ANSWER
```

**Five tools available to the agent:**

| Tool | Purpose | When agent uses it |
|---|---|---|
| `search_episodes(query)` | Fast memory search | First attempt for any factual question |
| `search_profiles(query)` | Entity lookup | "Who is...", "Tell me about..." |
| `search_deep(query)` | Multi-hop agentic search | Complex reasoning, cross-references |
| `look_at_video(start, end, q)` | Extract + re-analyze clip | Visual verification, counting, reading text |
| `get_timeline(start, end)` | Chronological event listing | "What happened between X and Y?" |

**Why this is truly agentic:**
- The agent *plans* which tools to use based on query complexity
- It *escalates*: starts with fast search, uses deep search or video grounding only if needed
- It *iterates*: if first retrieval is insufficient, it tries different approaches
- It *grounds*: can send specific video clips back to Qwen2.5-Omni for fresh analysis, verifying or enriching its answer

**Fallback safety:** If the ReAct loop hits max iterations or the LLM errors, it falls back to a simple retrieve-and-answer mode.

### 9. `pipeline.py` — Orchestration (192 lines)

**What it does:** Provides the two-method API that users actually interact with:

```python
pipe = VidEngramPipeline()
stats = pipe.ingest("video.mp4")         # segment → caption → consolidate → store
response = pipe.query("What happened?", "video.mp4")  # ReAct agent loop
print(response.answer)
```

Tracks ingestion stats, manages multi-turn conversation history, and provides pretty-printed summaries.

---

## How the Components Connect

```
┌─────────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌──────────┐
│   VIDEO.mp4 │───▶│ Segmenter│───▶│  Captioner   │───▶│ Consolidator │───▶│  Writer  │
│  (user file)│    │(ffmpeg)  │    │(Qwen2.5-Omni)│    │(dedup+sum+   │    │(EverMemOS│
│             │    │          │    │(vLLM-Omni    │    │ profile)     │    │ v1 API)  │
└─────────────┘    │ adaptive │    │ port 8091)   │    │              │    │          │
                   │ or fixed │    │              │    │              │    │          │
                   └──────────┘    └──────────────┘    └──────────────┘    └──────────┘
                                                                                │
     ┌──────────────────────────────────────────────────────────────────────────┘
     │                              QUERY PATH
     ▼
┌──────────┐    ┌───────────┐    ┌─────────────────────────────────────────────────┐
│  User    │◀──▶│  Agent    │◀──▶│                   TOOLS                          │
│  Question│    │  (ReAct)  │    │  search_episodes ──▶ Reader (rrf/bm25/embedding) │
│          │    │           │    │  search_profiles ──▶ Reader (profile type)        │
│          │    │  THINK    │    │  search_deep     ──▶ Reader (agentic mode)        │
│          │    │  ACTION   │    │  look_at_video   ──▶ Captioner.analyze_clip()     │
│          │    │  OBSERVE  │    │  get_timeline    ──▶ Reader + time filter          │
│          │    │  ANSWER   │    │                                                   │
└──────────┘    └───────────┘    └─────────────────────────────────────────────────┘
```

---

## External API Contracts

### vLLM-Omni (Qwen2.5-Omni-7B)

```
Server:   vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
Endpoint: POST http://localhost:8091/v1/chat/completions
Auth:     api_key="EMPTY" (local serving)
Input:    {"type": "video_url", "video_url": {"url": "file:///path/to/clip.mp4"}}
Output:   modalities=["text"] for text-only (no audio generation)
```

### EverMemOS

```
Server:   uv run python src/run.py --port 8001
Store:    POST http://localhost:8001/api/v1/memories
          Body: {message_id, create_time, sender, role, content, group_id, scene}
Search:   POST http://localhost:8001/api/v1/memories/search
          Body: {query, group_id, memory_types, retrieve_method, top_k}
          Note: Uses POST (not GET) because GET-with-body is non-standard
                and can be silently dropped by proxies (RFC 9110 §9.3.1)
Health:   GET  http://localhost:8001/health
```

**Memory types:** `episodic_memory`, `profile`, `semantic_memory`
**Retrieve methods:** `rrf`, `bm25`, `embedding`, `agentic`

---

## Competition Differentiators (Why We Win)

### 1. Not Naive Caption-RAG
Most video+memory systems do: caption every frame → embed → vector search → answer. That's just RAG over subtitles. VidEngram adds hippocampal consolidation (dedup → episodes → profiles) to create hierarchical, reasoned memory.

### 2. Agentic, Not Static
The agent decides *at query time* what tools to use. A simple question gets fast search. A complex question triggers multi-hop retrieval. A visual question triggers actual video re-analysis. This adaptivity is the definition of agentic behavior.

### 3. Context Grounding
When memory alone isn't enough, the agent can extract a specific video clip and send it to Qwen2.5-Omni for fresh analysis. This is inspired by hippomm2's `feature/qwen-context-grounding` branch — the answer is grounded in actual video evidence, not just stored text.

### 4. Unified Audio-Visual
Qwen2.5-Omni processes video+audio in one pass. No ImageBind + Whisper + fusion pipeline. Fewer moving parts, fewer failure modes, better understanding.

### 5. Temporal Reasoning via Virtual Timestamps
The timestamp scaling trick (1 video-sec = 60 virtual-sec) gives EverMemOS enough temporal spread to reason about "before", "after", "during" — temporal questions that flat embedding search can't handle.

---

## Testing

```bash
# Run all 44 unit tests (no live services needed)
pytest tests/ -v

# Tests cover:
#   Config defaults and env overrides (6 tests)
#   Config validation (3 tests)
#   Data classes and timestamp math (10 tests)
#   Segmenter boundary logic (4 tests)
#   Consolidator dedup logic (2 tests)
#   Memory writer payload shape + role field (3 tests)
#   Memory reader POST + response parsing (6 tests)
#   Agent parameter parsing (5 tests)
#   Pipeline cleanup + bounded cache (2 tests)
#   Integration smoke tests (3 tests)
```

---

## Evaluation

VidEngram includes an evaluation framework for benchmarking against QA pairs:

```bash
# Generate a QA template for a video
python -m scripts.evaluate --video lecture.mp4 --generate-qa
# → Creates lecture_qa.json with 7 template questions across 5 categories

# Fill in reference answers, then evaluate
python -m scripts.evaluate --video lecture.mp4 --qa lecture_qa.json --output results.json

# With LLM-as-judge automated scoring
python -m scripts.evaluate --video lecture.mp4 --qa lecture_qa.json --llm-judge
```

**Metrics tracked:**
- Latency per query
- Tools used per query (measures agentic behavior)
- Timestamp citation rate (measures grounding)
- LLM-as-judge scores: relevance, completeness, grounding (0-5 scale)
- Category breakdown: factual, temporal, entity, multi-hop, visual

---

## Robustness Features

**Config validation** — Call `config.validate()` to catch issues before running:
```python
cfg = VidEngramConfig()
issues = cfg.validate()
# Returns: ["[CRITICAL] ffmpeg not found", "[WARNING] Qwen on port 8000..."]
```

**HTTP retry logic** — All HTTP calls use `requests.Session` with:
- Connection pooling (reuses TCP connections)
- Automatic retry on 502/503/504 (3 retries, exponential backoff)

**Temp file cleanup** — Extracted clips don't accumulate forever:
```python
pipe.cleanup()             # Clean all temp files
pipe.cleanup("video.mp4")  # Clean specific video's temp files
```

**Bounded caches** — Pipeline ingestion stats use LRU eviction (max 50 entries)

---

## Repo Strategy: Standalone vs vLLM-Omni Branch

**Recommendation: Standalone repo + lightweight vLLM-Omni example contribution.**

| Approach | Pros | Cons |
|---|---|---|
| **Standalone repo** (recommended) | Clear competition submission; judges see your full system; separation of concerns | Need to maintain independently |
| **vLLM-Omni branch** | Visibility in vLLM-Omni ecosystem; cross-pollination | Mixes application layer with serving layer; dilutes competition novelty |
| **Both** (hybrid) | Best of both worlds | Slightly more maintenance |

The `examples/vllm_omni_video_memory.py` file is a self-contained 170-line script that could be contributed to vLLM-Omni's `examples/` directory as a PR, showcasing real-world multimodal memory workflows. The full VidEngram system lives in its own repo for competition submission.

---

## Quick Start

```bash
# 1. Infrastructure
docker compose up -d

# 2. EverMemOS
cd EverMemOS && uv sync && uv run python src/run.py --port 8001

# 3. vLLM-Omni (requires GPU with ≥24GB VRAM)
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --dtype bfloat16

# 4. VidEngram
pip install -e .
python -m demo.cli ingest path/to/video.mp4
python -m demo.cli chat path/to/video.mp4
```
