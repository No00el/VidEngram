#!/bin/bash
# VidEngram: Start all vLLM model servers
# Usage: bash serve_all.sh
#
# Serves 3 models across GPUs:
#   - Qwen2.5-Omni-7B      (GPU 0, port 8091) - video captioning + agent reasoning
#   - Qwen3-Embedding-4B    (GPU 2, port 8000) - EverMemOS embedding
#   - Qwen3-Reranker-4B     (GPU 2, port 12000) - EverMemOS reranking

set -e

CKPT_DIR="${CKPT_DIR:-/path/to/your/checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "$LOG_DIR"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate vllm

echo "========================================"
echo "  VidEngram vLLM Model Server Launcher"
echo "========================================"
echo ""

# GPU 0: Qwen2.5-Omni-7B (needs most memory for multimodal + KV cache)
# GPU 2: Embedding + Reranker (both small 4B models, share GPU 2)

# --- Embedding model (GPU 2, port 8000) ---
echo "[1/3] Starting Qwen3-Embedding-4B on GPU 2, port 8000..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$CKPT_DIR/Qwen3-Embedding-4B" \
    --served-model-name Qwen/Qwen3-Embedding-4B \
    --dtype=half \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --enforce-eager \
    --gpu-memory-utilization 0.4 \
    --pooler-config '{"pooling_type":"LAST", "normalize":true}' \
    > "$LOG_DIR/vllm_embedding.log" 2>&1 &
EMBED_PID=$!
echo "  PID: $EMBED_PID | Log: $LOG_DIR/vllm_embedding.log"

# --- Reranker model (GPU 2, port 12000) ---
echo "[2/3] Starting Qwen3-Reranker-4B on GPU 2, port 12000..."
CUDA_VISIBLE_DEVICES=2 vllm serve "$CKPT_DIR/Qwen3-Reranker-4B" \
    --served-model-name Qwen/Qwen3-Reranker-4B \
    --dtype=half \
    --port 12000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --enforce-eager \
    --gpu-memory-utilization 0.4 \
    > "$LOG_DIR/vllm_reranker.log" 2>&1 &
RERANK_PID=$!
echo "  PID: $RERANK_PID | Log: $LOG_DIR/vllm_reranker.log"

# --- Qwen2.5-Omni-7B (GPU 0, port 8091) ---
echo "[3/3] Starting Qwen2.5-Omni-7B on GPU 0, port 8091..."
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT_DIR/Qwen2.5-Omni-7B" \
    --served-model-name Qwen/Qwen2.5-Omni-7B \
    --dtype=half \
    --port 8091 \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 \
    --allowed-local-media-path "${VIDENGRAM_WORK_DIR:-/tmp/videngram}" \
    > "$LOG_DIR/vllm_omni.log" 2>&1 &
OMNI_PID=$!
echo "  PID: $OMNI_PID | Log: $LOG_DIR/vllm_omni.log"

echo ""
echo "All servers launching. PIDs: embed=$EMBED_PID rerank=$RERANK_PID omni=$OMNI_PID"
echo ""
echo "Check status:"
echo "  tail -f $LOG_DIR/vllm_embedding.log"
echo "  tail -f $LOG_DIR/vllm_reranker.log"
echo "  tail -f $LOG_DIR/vllm_omni.log"
echo ""
echo "Verify (wait a few minutes for models to load):"
echo "  curl -s http://localhost:8002/v1/models"
echo "  curl -s http://localhost:12000/v1/models"
echo "  curl -s http://localhost:8091/v1/models"
echo ""
echo "Stop all: kill $EMBED_PID $RERANK_PID $OMNI_PID"
