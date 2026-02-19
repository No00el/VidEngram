# VidEngram: Hippocampal-Inspired Video Memory with EverMemOS

> **Memory Genesis Competition 2026 — Track 1: Agent + Memory**

VidEngram bridges **Qwen2.5-Omni** multimodal understanding with **EverMemOS** structured long-term memory to enable agentic comprehension of long videos. Inspired by HippoMM's hippocampal memory formation principles, VidEngram doesn't just caption and retrieve — it *understands*, *consolidates*, and *reasons* about video content through structured episodic memory.

## Architecture

```
Video Input
    │
    ▼
┌──────────────────────────┐
│  1. Temporal Segmenter   │  HippoMM-inspired pattern separation
│     (scene + silence)    │  ffmpeg scene detection + silencedetect
└──────────┬───────────────┘
           │ segments
           ▼
┌──────────────────────────┐
│  2. Captioner            │  Qwen2.5-Omni-7B via vLLM-Omni
│     (video+audio→text)   │  Structured: scene/people/dialogue/sounds/emotion
└──────────┬───────────────┘
           │ captions
           ▼
┌──────────────────────────┐
│  3. Consolidator         │  HippoMM-inspired memory consolidation
│     - Dedup filtering    │  Pattern separation → encoding → replay
│     - Episode summaries  │  Creates hierarchical memory structure
│     - Entity profiles    │
└──────────┬───────────────┘
           │ consolidated memories
           ▼
┌──────────────────────────┐
│  4. EverMemOS Writer     │  POST /api/v1/memories
│     (structured storage) │  MemCell → Episode → Profile extraction
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│  5. Agentic Query Agent  │  ReAct reasoning loop
│     ┌─ search_episodes   │  5 tools for query answering
│     ├─ search_profiles   │  Adaptive strategy routing
│     ├─ search_deep       │  Self-reflection on answer quality
│     ├─ look_at_video     │  Context grounding via clip re-analysis
│     └─ get_timeline      │  Temporal event listing
└──────────────────────────┘
```

## Key Novelties (Competition Differentiators)

1. **Hippocampal Memory Pipeline** — Not naive caption→RAG. Three-stage consolidation (dedup, episodes, profiles) creates a hierarchical memory structure that enables multi-hop reasoning.

2. **Agentic ReAct Orchestrator** — The query agent *plans* which tools to use, *executes* searches and video analysis, *observes* results, and *iterates*. It self-decides between fast retrieval vs deep retrieval vs direct video grounding.

3. **Context Grounding** — When memory alone isn't enough, the agent extracts the specific video clip and re-analyzes it with Qwen2.5-Omni, verifying/enriching its answer with fresh multimodal evidence.

4. **Temporal Reasoning via Virtual Timestamps** — Video seconds are mapped to virtual calendar datetimes so EverMemOS's temporal reasoning engine can understand "before/after/during" relationships in video events.

5. **Unified Audio-Visual Understanding** — Qwen2.5-Omni processes video and audio in a single pass, eliminating the error-prone fusion of separate vision/audio pipelines.

## Quick Start

### Prerequisites
- Python 3.10+
- Docker 20.10+
- GPU with ≥24GB VRAM (for Qwen2.5-Omni-7B)
- ffmpeg installed system-wide

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/videngram.git
cd videngram

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.template .env
# Edit .env with your API keys
```

### 2. Start Infrastructure

```bash
# Start Docker services (MongoDB, Elasticsearch, Milvus, Redis)
docker compose up -d

# Clone and start EverMemOS (in a separate terminal)
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS
uv sync
cp env.template .env  # Edit with your LLM API key
uv run python src/run.py --port 8001

# Start Qwen2.5-Omni-7B via vLLM-Omni (in another terminal)
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --dtype bfloat16
```

### 3. Ingest and Query

```bash
# Ingest a video
python -m demo.cli ingest path/to/video.mp4

# Interactive chat
python -m demo.cli chat path/to/video.mp4

# Single query
python -m demo.cli query path/to/video.mp4 "What was the main argument?"

# Web UI
python -m demo.gradio_app
```

## Python API

```python
from videngram.pipeline import VidEngramPipeline

pipeline = VidEngramPipeline()

# Ingest
stats = pipeline.ingest("lecture.mp4")
print(f"Created {stats['memories_total']} memories")

# Query
response = pipeline.query("What examples did the speaker use?", "lecture.mp4")
print(response.answer)
print(f"Agent used {len(response.actions)} tool calls")
print(f"Grounded in {len(response.grounded_clips)} video clips")
```

## Project Structure

```
videngram/
├── videngram/
│   ├── __init__.py
│   ├── config.py           # Dataclass configs (Qwen, EverMemOS, Segmenter, Agent)
│   ├── utils.py            # Data classes, timestamp mapping, ffmpeg helpers
│   ├── segmenter.py        # HippoMM-inspired temporal pattern separation
│   ├── captioner.py        # Qwen2.5-Omni structured captioning
│   ├── consolidator.py     # Memory consolidation (dedup + episodes + profiles)
│   ├── memory_writer.py    # EverMemOS ingestion adapter
│   ├── memory_reader.py    # EverMemOS retrieval (rrf/bm25/embedding/agentic)
│   ├── agent.py            # ReAct agent with 5 tools
│   └── pipeline.py         # End-to-end orchestration
├── demo/
│   ├── cli.py              # CLI demo (ingest/query/chat)
│   └── gradio_app.py       # Gradio web UI
├── scripts/
│   └── start_services.sh   # Infrastructure launcher
├── config/
│   └── default_config.yaml # YAML config (HippoMM-style)
├── docker-compose.yml      # MongoDB, Elasticsearch, Milvus, Redis
├── requirements.txt
├── .env.template
└── README.md
```

## How It Works: HippoMM → VidEngram

| HippoMM Concept | VidEngram Implementation |
|---|---|
| Temporal Pattern Separation | `segmenter.py`: scene-change + silence detection via ffmpeg |
| Perceptual Encoding | `captioner.py`: Qwen2.5-Omni structured captions (unified AV) |
| Memory Consolidation | `consolidator.py`: dedup → episodes → profiles |
| Short-Term → Long-Term | `memory_writer.py`: EverMemOS MemCell → Episode → Profile |
| Fast Retrieval (Φ_fast) | `memory_reader.py`: RRF/BM25/embedding search |
| Detailed Recall (Ψ_detailed) | `memory_reader.py`: Agentic multi-hop retrieval |
| ImageBind cross-modal features | Replaced by Qwen2.5-Omni (processes video+audio natively) |
| ThetaEvent semantic summaries | `consolidator.py`: LLM-generated episode summaries |
| Query answering | `agent.py`: ReAct loop with 5 tools + context grounding |

## Dependencies

| Component | Version | Purpose |
|---|---|---|
| Qwen2.5-Omni-7B | — | Multimodal understanding (video+audio→text) |
| vLLM-Omni | ≥0.14.0 | Model serving with OpenAI-compatible API |
| EverMemOS | ≥1.2.0 | Structured long-term memory system |
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
- **EverMemOS**: EverMind AI, https://github.com/EverMind-AI/EverMemOS
- **Qwen2.5-Omni**: Qwen Team, Alibaba Cloud

## License

MIT
