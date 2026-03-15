# VidEngram End-to-End Setup Guide

## Prerequisites

- Python 3.10+
- [vLLM](https://github.com/vllm-project/vllm) v0.16+ (with Qwen2.5-Omni support)
- ffmpeg and ffprobe
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) (for EverMemOS)
- At least 1 GPU with ~30GB VRAM (embedding + reranker + Omni), or 2 GPUs to split
- An OpenAI-compatible LLM API (OpenAI, LiteLLM proxy, etc.) for agent reasoning

### Model checkpoints

Download these to a local directory (referred to as `$CKPT_DIR` below):

| Model | Size | Purpose |
|-------|------|---------|
| [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | ~14GB bf16 | Video captioning + video grounding |
| [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) | ~8GB bf16 | EverMemOS vector embeddings |
| [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) | ~8GB bf16 | EverMemOS reranking |

## Deployment topology

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

The backend on your local machine connects to the remote vLLM servers. Set
`QWEN_BASE_URL` in `.env` to the remote server's address (see Step 6), or use
SSH port forwarding:

```bash
ssh -L 8091:localhost:8091 -L 8000:localhost:8000 -L 12000:localhost:12000 user@remote-gpu-server
```

## Architecture overview

```
Video --> Segmenter (ffmpeg, on remote) --> Captioner (Qwen2.5-Omni, on remote)
      --> Consolidator (GPT-4o-mini or other text LLM)
      --> MemoryWriter --> EverMemOS
                              |
                              v
                    MongoDB / Elasticsearch / Milvus / Redis

Query --> Backend (FastAPI :7860, local)
            |
            v
          Agent (GPT-4o-mini ReAct loop)
            |-- search_episodes   --> EverMemOS
            |-- search_profiles   --> EverMemOS
            |-- search_deep       --> EverMemOS
            |-- look_at_video     --> Qwen2.5-Omni (video grounding, on remote)
            |-- get_timeline      --> EverMemOS
            v
          Answer with timestamps + sources
```

Qwen2.5-Omni handles video/audio understanding only. A separate text LLM (e.g.
GPT-4o-mini) handles all text reasoning - ReAct agent planning, episode summaries,
and profile extraction. This split is intentional: Qwen doesn't follow structured
ReAct format reliably, while instruction-tuned text LLMs excel at it.

## Step 1: Clone EverMemOS

```bash
cd /path/to/videngram   # your project root
git clone https://github.com/SuanAI/EverMemOS.git
cd EverMemOS
uv sync
```

## Step 2: Start Docker infrastructure

EverMemOS needs MongoDB, Elasticsearch, Milvus, and Redis:

```bash
cd EverMemOS
docker compose up -d
```

Wait ~30 seconds, then verify:

```bash
curl -s http://localhost:19200/_cluster/health | python3 -m json.tool   # Elasticsearch
docker exec memsys-mongodb mongosh -u admin -p memsys123 --quiet --eval 'db.runCommand("ping")'
curl -s http://localhost:19530/v1/vector/collections                    # Milvus
docker exec memsys-redis redis-cli ping                                 # Redis
```

## Step 3: Configure EverMemOS .env

```bash
cd EverMemOS
cp env.template .env
```

Edit `.env` and set at minimum:

```bash
# LLM for internal memory extraction (any OpenAI-compatible endpoint)
LLM_BASE_URL=https://api.openai.com/v1       # or your LiteLLM proxy
LLM_API_KEY=<your-api-key>
LLM_MODEL=gpt-4o-mini

# Embedding + Reranker are on the remote GPU server.
# Requires SSH port forwarding (see Deployment topology) or direct IP.
EMBEDDING_BASE_URL=http://localhost:8000/v1
# Or: EMBEDDING_BASE_URL=http://<remote-server-ip>:8000/v1
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
RERANKER_BASE_URL=http://localhost:12000/v1
# Or: RERANKER_BASE_URL=http://<remote-server-ip>:12000/v1
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

See `env.template` for all available options.

## Step 4: Start vLLM model servers (on remote GPU server)

> **Run all commands in this step on the remote GPU server**, not your local machine.

Edit `serve_all.sh` to match your setup:

- Set `CKPT_DIR` to where your model checkpoints are
- Adjust `CUDA_VISIBLE_DEVICES` for your GPU layout
- Adjust `--gpu-memory-utilization` based on available VRAM

Then launch:

```bash
cd /path/to/videngram   # project root on the remote server
conda activate vllm
bash serve_all.sh
```

This starts three servers:

| Model | Default Port | GPU | Notes |
|-------|-------------|-----|-------|
| Qwen2.5-Omni-7B | 8091 | GPU 0 (dedicated) | ~14GB VRAM, needs `--allowed-local-media-path` |
| Qwen3-Embedding-4B | 8000 | GPU 2 (shared) | ~8GB VRAM |
| Qwen3-Reranker-4B | 12000 | GPU 2 (shared) | ~8GB VRAM, can share GPU with embedding |

Wait a few minutes for models to load, then verify on the remote server:

```bash
curl -s http://localhost:8091/v1/models | python3 -m json.tool
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:12000/v1/models | python3 -m json.tool
```

**GPU layout tips:**

- If you have 1 large GPU (80GB+), all 3 models can share it. Set
  `--gpu-memory-utilization` to ~0.15/0.15/0.55 respectively.
- If you have 2+ GPUs, put Omni on one and embedding + reranker on another.
  They're small enough to share at 40% utilization each.
- If GPU 0 is shared with other processes, lower `--gpu-memory-utilization` for Omni.

## Step 5: Start EverMemOS

```bash
# In a new terminal
cd EverMemOS
uv run python src/run.py --port 8001
```

Verify:

```bash
curl -s http://localhost:8001/health
```

## Step 6: Configure VidEngram .env

```bash
cd /path/to/videngram
cp .env.template .env
```

Edit `.env`:

```bash
# Qwen2.5-Omni for video captioning (vLLM on remote GPU server)
# If using SSH port forwarding (ssh -L 8091:localhost:8091 ...):
QWEN_BASE_URL=http://localhost:8091/v1
# Or connect directly to the remote server:
# QWEN_BASE_URL=http://<remote-server-ip>:8091/v1
QWEN_MODEL=Qwen/Qwen2.5-Omni-7B
QWEN_API_KEY=EMPTY

# EverMemOS server
EVERMEMOS_BASE_URL=http://localhost:8001

# LLM for EverMemOS internal memory extraction
# Default: local Qwen (no extra cost, lower quality)
LLM_API_KEY=EMPTY
LLM_MODEL=Qwen/Qwen2.5-Omni-7B
LLM_BASE_URL=http://localhost:8091/v1
# Recommended: use an external text LLM for better extraction quality
# LLM_BASE_URL=https://api.openai.com/v1
# LLM_API_KEY=<your-api-key>
# LLM_MODEL=gpt-4o-mini

# Text LLM for agent reasoning + consolidation (any OpenAI-compatible endpoint)
PLANNING_LLM_BASE_URL=https://api.openai.com/v1   # or your LiteLLM proxy URL + /v1
PLANNING_LLM_MODEL=gpt-4o-mini
PLANNING_LLM_API_KEY=<your-api-key>

# Working directory for temp files (clips, segments, etc.)
VIDENGRAM_WORK_DIR=/tmp/videngram
```

### Optional: Speech transcription (strongly recommended)

```bash
# Whisper-compatible ASR API
TRANSCRIBER_BASE_URL=https://api.openai.com/v1
TRANSCRIBER_API_KEY=<your-api-key>
TRANSCRIBER_MODEL=whisper-1
```

Without `TRANSCRIBER_API_KEY`, ASR is skipped entirely:
- Video segmentation falls back to silence/scene detection only
- Subtitles are disabled
- Qwen self-transcribes dialogue (lower accuracy)
- No speech memories are stored in EverMemOS

### Optional: Remote processing via SSH

```bash
# When both are set, uploaded videos are SCP'd to the remote server and all
# ffmpeg/ffprobe/captioner/transcriber operations run via SSH on that server.
# Leave unset to run everything locally (requires local ffmpeg).
REMOTE_HOST=user@your-remote-host
REMOTE_WORK_DIR=/home/user/videngram/work
```

The `PLANNING_LLM_*` variables control which LLM the agent and consolidator use for
text-only tasks. If unset, they fall back to Qwen2.5-Omni (not recommended).

## Step 7: Install VidEngram dependencies

```bash
# Step 1: Create a virtual environment (choose one)
python -m venv .venv   # standard library
# or
uv venv                # faster, requires uv

# Step 2: Activate and install
source .venv/bin/activate
pip install -e .
```

This installs the `videngram` package along with all dependencies including the
FastAPI backend (`fastapi`, `uvicorn`, `python-multipart`, `aiofiles`).

## Step 8: Run VidEngram backend

```bash
# In a new terminal
cd /path/to/videngram
source .venv/bin/activate
uvicorn backend.server:app --host 0.0.0.0 --port 7860
```

The backend serves a web UI at `http://localhost:7860` and exposes REST + SSE
endpoints for video ingestion, memory querying, and analysis.

### Health check

```bash
python -c "
from videngram.config import VidEngramConfig
cfg = VidEngramConfig()
issues = cfg.validate()
for i in issues:
    print(i)
if not issues:
    print('All checks passed')
"
```

### Key API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (frontend/index.html) |
| `/upload` | POST | Upload a video file (optionally SCP to remote) |
| `/ingest` | POST | Ingest a video into EverMemOS (streams progress via SSE) |
| `/qa` | POST | Agentic Q&A over ingested video (streams steps + answer via SSE) |
| `/analyze` | POST | Memory-augmented frame analysis (streams tokens via SSE) |
| `/memories` | GET | Get memory matching a specific video timestamp |
| `/memory_cues` | GET | All timestamped memories (scene + dialogue) |
| `/subtitle_cues` | GET | Subtitle cues derived from memory timestamps |
| `/speech_cues` | GET | Whisper transcription cues |
| `/segment_cues` | GET | Raw segment captions |
| `/videos` | GET | List all ingested videos |
| `/switch_video` | POST | Switch active video context |
| `/graph/data` | GET | Relationship graph for current video |

All streaming endpoints (`/ingest`, `/qa`, `/analyze`) return
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

### Alternative: CLI usage

For scripting or quick testing without the backend:

```bash
# Ingest a video
python -m demo.cli ingest /path/to/video.mp4

# Single query
python -m demo.cli query /path/to/video.mp4 "What happens in this video?"

# Interactive chat
python -m demo.cli chat /path/to/video.mp4

# Ingest then chat in one step
python -m demo.cli chat --ingest-first /path/to/video.mp4
```

Use `-v` before the subcommand for debug logging:

```bash
python -m demo.cli -v query /path/to/video.mp4 "What happened?"
```

## Data management

### Clear all memories

```bash
cd EverMemOS
uv run python src/bootstrap.py demo/tools/clear_all_data.py
```

### Delete memories for a specific video

```bash
# Find the video's group_id
python -c "
from videngram.memory_writer import MemoryWriter
print(MemoryWriter._video_group_id('/path/to/video.mp4'))
"

# Delete by group_id
curl -X DELETE "http://localhost:8001/api/v1/memories" \
  -H "Content-Type: application/json" \
  -d '{"group_id": "<group_id_from_above>"}'
```

## Troubleshooting

**vLLM OOM or crash**
- Check `logs/vllm_*.log` for errors
- Lower `--gpu-memory-utilization` in `serve_all.sh`
- Use `nvidia-smi` to check what else is using GPU memory

**Port already in use**
- Check with `lsof -i :<port>` and kill the conflicting process
- Or change the port in `serve_all.sh` and the corresponding `.env` files

**Captioning error "Cannot load local files"**
- Qwen vLLM needs `--allowed-local-media-path` pointing to your `VIDENGRAM_WORK_DIR`
- Already set in `serve_all.sh` - make sure the path matches your `.env`

**EverMemOS 405 on search**
- The search endpoint uses GET with query params, not POST
- This is already handled in `memory_reader.py`

**Agent gives hallucinated or empty answers**
- Make sure `PLANNING_LLM_*` is set in `.env` to use a text LLM like GPT
- Without it, Qwen handles ReAct reasoning and doesn't follow the format reliably

**EverMemOS can't connect to backends**
- Make sure Docker containers are healthy: `docker compose ps`
- Elasticsearch needs ~30s to fully initialize

**No subtitles or speech memories**
- Set `TRANSCRIBER_API_KEY` in `.env` to enable Whisper ASR
- Without it, speech is not transcribed and subtitle cues will be empty

**Stop everything**

```bash
# On the remote GPU server:
pkill -f "vllm serve"

# On your local machine:
cd EverMemOS && docker compose down            # Docker infra
# EverMemOS: Ctrl+C in its terminal
# Backend: Ctrl+C in its terminal
```

## Startup order summary

| Order | Service | Command | Ports |
|-------|---------|---------|-------|
| 1 | Docker infra | `cd EverMemOS && docker compose up -d` | 27017, 19200, 19530, 6379 |
| 2 | vLLM servers **(remote)** | `bash serve_all.sh` | 8091, 8000, 12000 |
| 3 | EverMemOS | `cd EverMemOS && uv run python src/run.py --port 8001` | 8001 |
| 4 | VidEngram backend | `cd /path/to/videngram && source .venv/bin/activate && uvicorn backend.server:app --host 0.0.0.0 --port 7860` | 7860 |

## Code changes from original template

### Modifications to existing files

| File | Change | Why |
|------|--------|-----|
| `videngram/memory_writer.py` | Accept HTTP 202 as success | EverMemOS returns 202 for async processing |
| `videngram/memory_reader.py` | GET instead of POST for search | EverMemOS search endpoint uses GET with query params |
| `videngram/memory_reader.py` | Handle `pending_messages` in response parser | Fallback when memories haven't been fully processed |
| `videngram/agent.py` | Prioritize ACTION over ANSWER in parser | Prevents premature termination when LLM outputs both |
| `videngram/agent.py` | Conditional `extra_body` for Qwen only | Non-Qwen LLMs don't need `modalities: ["text"]` |
| `videngram/consolidator.py` | Use external text LLM for summaries + profiles | Better quality than Qwen for text-only tasks |
| `serve_all.sh` | Split models across GPUs, set `--allowed-local-media-path` | Avoid OOM; required for local file access |

### New components

| File | Purpose |
|------|---------|
| `backend/server.py` | FastAPI backend with REST + SSE endpoints; manages video state, memory cache, video history, and graph building |
| `backend/graph_builder.py` | Async background task that extracts entity relationship graphs from EverMemOS memories |
| `frontend/index.html` | Static single-page web UI served directly by the backend; no build step required |
