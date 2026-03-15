# VidEngram: Hippocampal-Inspired Video Memory with EverMemOS

> **Memory Genesis Competition 2026 — Track 1: Agent + Memory**

VidEngram is built on two core pillars: **Qwen2.5-Omni** for native multimodal video+audio understanding, and **EverMemOS** for structured long-term memory with hierarchical storage, hybrid retrieval, and temporal reasoning. Inspired by HippoMM's hippocampal memory formation principles, VidEngram doesn't just caption and retrieve — it *understands*, *consolidates*, and *reasons* about video content through EverMemOS's three-layer episodic memory architecture (segment → episode → entity profile).

## Demo Video

<!-- YouTube link coming soon -->

## Key Features

| Feature | Description |
|---|---|
| **Native multimodal understanding** | Qwen2.5-Omni processes video frames and audio in a single pass — no separate vision/audio pipelines |
| **Hippocampal memory consolidation** | Three-layer hierarchy: segment → episode summary → entity profile |
| **ReAct agentic Q&A** | 6 tools, autonomous dispatch, self-reflection on answer quality |
| **Forced timestamp citations** | All answers include `[Video M:SS - N:SS]` anchors grounded in actual video time |
| **Streaming ingest architecture** | Speech and segment memories written concurrently as captions arrive |
| **Web UI + knowledge graph** | Real-time subtitle overlay, scene navigation, D3.js entity relationship graph |

## Architecture

```
Video Input
    │
    ▼
┌──────────────────────────┐
│  1. ASR Transcriber      │  Whisper API — extracts speech segments with timestamps
└──────────┬───────────────┘
           │ transcript
           ▼
┌──────────────────────────┐
│  2. Temporal Segmenter   │  HippoMM-inspired pattern separation
│     (ASR-guided +        │  ASR boundaries + ffmpeg scene/silence detection
│      scene + silence)    │  Parallel clip extraction with min/max duration constraints
└──────────┬───────────────┘
           │ segments
           ▼
┌──────────────────────────┐
│  3. Captioner            │  Qwen2.5-Omni-7B via vLLM
│     (video+audio→text)   │  9-field structured output:
│                          │  SCENE / PEOPLE / ACTIONS / DIALOGUE /
│                          │  SOUNDS / TEXT / OBJECTS / EMOTION / TEMPORAL
└──────────┬───────────────┘
           │ captions (streamed)
           ▼
┌──────────────────────────┐
│  4. Consolidator         │  HippoMM-inspired memory consolidation
│     - Dedup filtering    │  Jaccard similarity > 0.85 → merge
│     - Episode summaries  │  Related segments grouped into narrative episodes
│     - Entity profiles    │  Cross-episode entity resolution and merging
└──────────┬───────────────┘
           │ consolidated memories
           ▼
┌──────────────────────────┐
│  5. EverMemOS Writer     │  POST /api/v1/memories
│     (structured storage) │  Concurrent writes (3 workers), streaming segment memories
│                          │  MongoDB + Elasticsearch + Milvus indexing
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  6. Agentic Query Agent  │  ReAct reasoning loop
│     ┌─ search_episodes   │  Fast hybrid retrieval (BM25 + vector)
│     ├─ search_profiles   │  Entity / speaker profile lookup
│     ├─ search_deep       │  LLM-guided multi-hop retrieval
│     ├─ look_at_video     │  Extract clip + re-analyze with Qwen2.5-Omni
│     ├─ search_speech     │  BM25 search over Whisper speech transcripts
│     └─ get_timeline      │  Chronological event listing for a time range
└──────────────────────────┘
```

## Key Novelties

1. **Hippocampal Memory Pipeline** — Not naive caption→RAG. Three-stage consolidation (dedup, episodes, profiles) creates a hierarchical memory structure that enables multi-hop reasoning.

2. **EverMemOS as the Memory Backbone** — All consolidated memories are stored in EverMemOS's native three-layer hierarchy (MemCell → Episode → Entity Profile). Retrieval combines BM25 keyword search, dense vector search, and LLM-guided reranking in a single hybrid (RRF) call — capabilities provided entirely by EverMemOS with no custom retrieval code needed.

3. **Agentic ReAct Orchestrator** — The query agent *plans* which tools to use, *executes* searches and video analysis, *observes* results, and *iterates*. Six tools cover fast retrieval, profile lookup, multi-hop retrieval, speech search, video grounding, and timeline queries.

4. **Context Grounding** — When memory alone isn't enough, the agent extracts the specific video clip and re-analyzes it with Qwen2.5-Omni, verifying and enriching its answer with fresh multimodal evidence.

## Quick Start

> For the complete step-by-step setup guide, see [E2E_SETUP.md](E2E_SETUP.md).

### Prerequisites

- Python 3.10+, `uv`, Docker & Docker Compose, ffmpeg
- GPU with ≥30GB VRAM (Qwen2.5-Omni + Embedding + Reranker), or 2 GPUs to split

### Model checkpoints

Download to a local directory on your GPU server:

| Model | Size | Purpose |
|---|---|---|
| [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | ~14GB bf16 | Video captioning + video grounding |
| [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | ~8GB bf16 | EverMemOS vector embeddings |
| [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | ~8GB bf16 | EverMemOS search reranking |

### Deployment topology

VidEngram splits compute across two machines:

```
┌─────────────────────────────┐     ┌──────────────────────────────────────┐
│       Local machine         │     │          Remote GPU server           │
│                             │     │                                      │
│  EverMemOS  (port 8001)     │     │  Qwen2.5-Omni-7B     (port 8091)    │
│  MongoDB / ES / Milvus /    │     │  Qwen3-Embedding-4B  (port 8000)    │
│    Redis  (Docker)          │     │  Qwen3-Reranker-4B   (port 12000)   │
│                             │     │                                      │
│  VidEngram backend          │     │  Videos SCP'd here; ffmpeg/Qwen     │
│    (port 7860)   ───────────┼────▶│  run here via SSH                   │
└─────────────────────────────┘     └──────────────────────────────────────┘
```

Use SSH port forwarding to connect local services to the remote GPU server:

```bash
ssh -L 8091:localhost:8091 -L 8000:localhost:8000 -L 12000:localhost:12000 user@remote-gpu-server
```

### Startup order

| # | Service | Command | Port |
|---|---------|---------|------|
| 1 | Docker infra | `cd EverMemOS && docker compose up -d` | 27017, 19200, 19530, 6379 |
| 2 | vLLM servers **(remote)** | `bash serve_all.sh` | 8091, 8000, 12000 |
| 3 | EverMemOS | `cd EverMemOS && uv run python src/run.py --port 8001` | 8001 |
| 4 | VidEngram backend | `uvicorn backend.server:app --host 0.0.0.0 --port 7860` | 7860 |

Then open **http://localhost:7860** in your browser.

For CLI usage:

```bash
python -m demo.cli ingest path/to/video.mp4
python -m demo.cli query  path/to/video.mp4 "What was the main argument?"
python -m demo.cli chat   path/to/video.mp4
```

For full setup details including `.env` configuration, EverMemOS setup, GPU layout tips, and troubleshooting, see **[E2E_SETUP.md](E2E_SETUP.md)**.

## Python API

```python
from videngram import VidEngramPipeline

pipeline = VidEngramPipeline()

# Ingest
stats = pipeline.ingest("lecture.mp4")
print(f"Processed {stats['segments']} segments, wrote {stats['memories_total']} memories")
# e.g. memories_segments / memories_episodes / memories_profiles / memories_speech

# Query
response = pipeline.query("What examples did the speaker use?", "lecture.mp4")
print(response.answer)                        # includes [Video M:SS - N:SS] citations
print(f"Agent used {len(response.actions)} tool calls")
print(f"Grounded in {len(response.grounded_clips)} video clips")
```

## Project Structure

```
videngram/
├── videngram/
│   ├── __init__.py
│   ├── config.py           # Dataclass configs (Qwen, EverMemOS, Segmenter, Consolidator, Agent, ...)
│   ├── utils.py            # Data classes (VideoSegment, Caption, ConsolidatedMemory,
│   │                       #   MemoryResult, AgentAction, AgentResponse) + timestamp helpers
│   ├── pipeline.py         # End-to-end orchestration (ingest + query)
│   ├── segmenter.py        # HippoMM-inspired temporal pattern separation
│   ├── captioner.py        # Qwen2.5-Omni structured captioning (9 fields, local + external API)
│   ├── transcriber.py      # Speech transcription (Whisper-compatible ASR)
│   ├── consolidator.py     # Memory consolidation (dedup → episodes → entity profiles)
│   ├── memory_writer.py    # EverMemOS ingestion adapter (streaming, concurrent writes)
│   ├── memory_reader.py    # EverMemOS retrieval — fast (RRF/BM25/embedding) + agentic
│   ├── agent.py            # ReAct agent with 6 tools
│   └── visualizer.py       # t-SNE memory embedding visualization
├── backend/
│   ├── server.py           # FastAPI backend + serves frontend/index.html (port 7860)
│   └── graph_builder.py    # Async entity relationship graph extraction (background task)
├── frontend/
│   └── index.html          # Single-file Web UI — subtitles, scene nav, D3.js graph (no build step)
├── demo/
│   └── cli.py              # CLI: ingest / query / chat
├── tests/
│   ├── test_videngram.py           # Unit tests (pytest, no live services needed)
│   └── test_runtime_validation.py  # End-to-end execution path validation (mocked externals)
├── config/
│   └── default_config.yaml # Default YAML configuration
├── serve_all.sh            # vLLM model server launcher (Omni + Embedding + Reranker)
├── docker-compose.yml      # MongoDB, Elasticsearch, Milvus, Redis (local dev)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Package metadata (Python 3.10+)
├── .env.template           # Environment variable template
└── E2E_SETUP.md            # Full deployment guide
```

## How It Works: HippoMM → VidEngram

| HippoMM Concept | VidEngram Implementation |
|---|---|
| Perceptual Encoding | `captioner.py`: Qwen2.5-Omni 9-field structured captions (unified AV) |
| Pattern Separation | `segmenter.py`: ASR-guided + scene/silence detection; dedup filtering in `consolidator.py` |
| Memory Consolidation | `consolidator.py`: episode summaries + entity profile construction |
| Pattern Completion | `agent.py`: ReAct agent tool dispatch |
| Fast Recall (Φ_fast) | `memory_reader.py`: RRF / BM25 / embedding hybrid search |
| Detailed Recall (Ψ_detailed) | `memory_reader.py`: LLM-guided multi-hop retrieval + video grounding via `look_at_video` |
| Episodic Memory | `memory_writer.py`: EverMemOS MemCell → Episode → Profile hierarchy |

## Dependencies

| Component | Version | Purpose |
|---|---|---|
| Qwen2.5-Omni-7B | — | Multimodal video+audio→text understanding and video grounding |
| Qwen3-Embedding-4B | — | EverMemOS vector embeddings |
| Qwen3-Reranker-4B | — | EverMemOS search reranking |
| vLLM | ≥0.16.0 | Model serving with OpenAI-compatible API |
| EverMemOS | ≥1.2.0 | Structured long-term memory system |
| FastAPI / uvicorn | — | Backend web server |
| MongoDB | 7.0 | EverMemOS document storage |
| Elasticsearch | 8.15 | BM25 keyword search |
| Milvus | 2.4 | Vector similarity search |
| Redis | 7.x | Caching layer |
| ffmpeg | — | Video segmentation and clip extraction |

## Citation

If you use VidEngram, please cite:

```bibtex
@misc{videngram2026,
  title={VidEngram: Hippocampal-Inspired Video Memory with EverMemOS},
  year={2026},
  note={Memory Genesis Competition 2026, Track 1: Agent + Memory}
}
```

Also cite the foundational works:
- **HippoMM**: Lin et al., "HippoMM: Hippocampal-inspired Multimodal Memory for Long Audiovisual Event Understanding", arXiv:2504.10739
- **EverMemOS**: EverMind AI, https://github.com/SuanAI/EverMemOS
- **Qwen2.5-Omni**: Qwen Team, Alibaba Cloud

## License

MIT
