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

## Architecture overview

```
Video --> Segmenter (ffmpeg) --> Captioner (Qwen2.5-Omni)
      --> Consolidator (GPT-4o-mini or other text LLM)
      --> MemoryWriter --> EverMemOS
                              |
                              v
                    MongoDB / Elasticsearch / Milvus / Redis

Query --> Agent (GPT-4o-mini ReAct loop)
            |-- search_episodes   --> EverMemOS
            |-- search_profiles   --> EverMemOS
            |-- search_deep       --> EverMemOS
            |-- look_at_video     --> Qwen2.5-Omni (video grounding)
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

# Local vLLM embedding + reranker
EMBEDDING_BASE_URL=http://localhost:8000/v1
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
RERANKER_BASE_URL=http://localhost:12000/v1
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

See `env.template` for all available options.

## Step 4: Start vLLM model servers

Edit `serve_all.sh` to match your setup:

- Set `CKPT_DIR` to where your model checkpoints are
- Adjust `CUDA_VISIBLE_DEVICES` for your GPU layout
- Adjust `--gpu-memory-utilization` based on available VRAM

Then launch:

```bash
bash serve_all.sh
```

This starts three servers:

| Model | Default Port | Notes |
|-------|-------------|-------|
| Qwen3-Embedding-4B | 8000 | ~8GB VRAM |
| Qwen3-Reranker-4B | 12000 | ~8GB VRAM, can share GPU with embedding |
| Qwen2.5-Omni-7B | 8091 | ~14GB VRAM, needs `--allowed-local-media-path` |

Wait a few minutes for models to load, then verify:

```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
curl -s http://localhost:12000/v1/models | python3 -m json.tool
curl -s http://localhost:8091/v1/models | python3 -m json.tool
```

**GPU layout tips:**

- If you have 1 large GPU (80GB+), all 3 models can share it. Set
  `--gpu-memory-utilization` to ~0.15/0.15/0.55 respectively.
- If you have 2+ GPUs, put Omni on one and embedding + reranker on another.
  They're small enough to share at 40% utilization each.
- If GPU 0 is shared with other processes, lower `--gpu-memory-utilization` for Omni.

## Step 5: Start EverMemOS

```bash
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
# Qwen2.5-Omni for video captioning (local vLLM)
QWEN_BASE_URL=http://localhost:8091/v1
QWEN_MODEL=Qwen/Qwen2.5-Omni-7B
QWEN_API_KEY=EMPTY

# EverMemOS server
EVERMEMOS_BASE_URL=http://localhost:8001

# LLM for EverMemOS internal memory extraction
LLM_API_KEY=<your-api-key>
LLM_MODEL=gpt-4o-mini

# Text LLM for agent reasoning + consolidation (any OpenAI-compatible endpoint)
PLANNING_LLM_BASE_URL=https://api.openai.com/v1   # or your LiteLLM proxy URL + /v1
PLANNING_LLM_MODEL=gpt-4o-mini
PLANNING_LLM_API_KEY=<your-api-key>

# Working directory for temp files (clips, segments, etc.)
VIDENGRAM_WORK_DIR=/tmp/videngram
```

The `PLANNING_LLM_*` variables control which LLM the agent and consolidator use for
text-only tasks. If unset, they fall back to Qwen2.5-Omni (not recommended).

## Step 7: Install VidEngram dependencies

```bash
pip install -e .
# or if not using editable install:
pip install openai requests python-dotenv
```

## Step 8: Run VidEngram

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

### Ingest a video

```bash
python -m demo.cli ingest /path/to/video.mp4
```

Expected output: segments created, captions generated (Qwen), episodes + profiles
consolidated (GPT), memories written to EverMemOS.

### Query

```bash
python -m demo.cli query /path/to/video.mp4 "What happens in this video?"
```

The agent uses GPT-4o-mini to reason through the ReAct loop, calling tools like
`search_episodes`, `search_profiles`, or `look_at_video`, then produces a final
answer with timestamps and sources.

### Interactive chat

```bash
python -m demo.cli chat /path/to/video.mp4
```

If the video hasn't been ingested yet:

```bash
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

**Stop everything**

```bash
pkill -f "vllm serve"                          # vLLM servers
cd EverMemOS && docker compose down            # Docker infra
# EverMemOS: Ctrl+C in its terminal
```

## Startup order summary

| Order | Service | Command | Ports |
|-------|---------|---------|-------|
| 1 | Docker infra | `cd EverMemOS && docker compose up -d` | 27017, 19200, 19530, 6379 |
| 2 | vLLM servers | `bash serve_all.sh` | 8000, 8091, 12000 |
| 3 | EverMemOS | `cd EverMemOS && uv run python src/run.py --port 8001` | 8001 |
| 4 | VidEngram | `python -m demo.cli ...` | - |

## Code changes from original template

| File | Change | Why |
|------|--------|-----|
| `videngram/memory_writer.py` | Accept HTTP 202 as success | EverMemOS returns 202 for async processing |
| `videngram/memory_reader.py` | GET instead of POST for search | EverMemOS search endpoint uses GET with query params |
| `videngram/memory_reader.py` | Handle `pending_messages` in response parser | Fallback when memories haven't been fully processed |
| `videngram/agent.py` | Prioritize ACTION over ANSWER in parser | Prevents premature termination when LLM outputs both |
| `videngram/agent.py` | Conditional `extra_body` for Qwen only | Non-Qwen LLMs don't need `modalities: ["text"]` |
| `videngram/consolidator.py` | Use external text LLM for summaries + profiles | Better quality than Qwen for text-only tasks |
| `serve_all.sh` | Split models across GPUs, set `--allowed-local-media-path` | Avoid OOM; required for local file access |
