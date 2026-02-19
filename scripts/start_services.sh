#!/bin/bash
# VidEngram: Start all services
# Usage: ./scripts/start_services.sh

set -e

echo "═══════════════════════════════════════════════"
echo "  VidEngram Service Launcher"
echo "═══════════════════════════════════════════════"

# ── Step 1: Docker infrastructure ──────────────────
echo ""
echo "Step 1/3: Starting Docker infrastructure..."
echo "  (MongoDB, Elasticsearch, Milvus, Redis)"
docker compose up -d
echo "  Waiting 15s for services to initialize..."
sleep 15

# Verify
echo "  Checking services..."
curl -s http://localhost:9200/_cluster/health > /dev/null && echo "  ✓ Elasticsearch" || echo "  ✗ Elasticsearch"
curl -s http://localhost:27017 > /dev/null 2>&1 && echo "  ✓ MongoDB" || echo "  ✗ MongoDB"
echo ""

# ── Step 2: EverMemOS API Server ───────────────────
echo "Step 2/3: Starting EverMemOS API server..."
echo "  (Run this in a separate terminal if not backgrounding)"

if [ -d "EverMemOS" ]; then
    cd EverMemOS
    nohup uv run python src/run.py --port 8001 > ../logs/evermemos.log 2>&1 &
    EVERMEMOS_PID=$!
    echo "  EverMemOS PID: $EVERMEMOS_PID"
    cd ..
    sleep 5
    curl -s http://localhost:8001/health > /dev/null && echo "  ✓ EverMemOS" || echo "  ✗ EverMemOS (check logs/evermemos.log)"
else
    echo "  ⚠ EverMemOS directory not found."
    echo "  Clone it: git clone https://github.com/EverMind-AI/EverMemOS.git"
    echo "  Then: cd EverMemOS && uv sync && cp env.template .env"
fi
echo ""

# ── Step 3: vLLM-Omni (Qwen2.5-Omni-7B) ──────────
echo "Step 3/3: Starting vLLM-Omni with Qwen2.5-Omni-7B..."
echo "  (Requires GPU with ~24GB VRAM)"
echo ""
echo "  Run in a separate terminal:"
echo "    vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --dtype bfloat16"
echo ""
echo "  Or use the Docker image:"
echo "    docker run --gpus all -p 8091:8091 vllm-omni:latest \\"
echo "      vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091"
echo ""

echo "═══════════════════════════════════════════════"
echo "  Once all services are running:"
echo "    python -m demo.cli ingest path/to/video.mp4"
echo "    python -m demo.cli chat path/to/video.mp4"
echo "═══════════════════════════════════════════════"
