"""
vLLM-Omni + EverMemOS: Video Memory Pipeline Example
=====================================================

This example shows how to use vLLM-Omni's OpenAI-compatible API with
Qwen2.5-Omni-7B to build a video memory system backed by EverMemOS.

This is a self-contained script that can be contributed to vLLM-Omni's
`examples/` directory to demonstrate real-world multimodal memory workflows.

Prerequisites:
    1. vLLM-Omni serving Qwen2.5-Omni-7B:
       vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --dtype bfloat16

    2. EverMemOS running (with Docker infra):
       cd EverMemOS && docker compose up -d && uv run python src/run.py --port 8001

Usage:
    python vllm_omni_video_memory.py path/to/video.mp4 "What happened at the start?"
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────

VLLM_OMNI_URL = "http://localhost:8091/v1"
VLLM_OMNI_MODEL = "Qwen/Qwen2.5-Omni-7B"
EVERMEMOS_URL = "http://localhost:8001"
SEGMENT_SECONDS = 30
TIME_SCALE = 60  # 1 video-second = 60 virtual-seconds
BASE_DT = "2025-01-01T00:00:00+00:00"


def get_duration(video_path: str) -> float:
    """Get video duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", video_path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(json.loads(out.stdout)["format"]["duration"])


def extract_clip(video_path: str, start: float, end: float, out: str):
    """Extract video clip with ffmpeg."""
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(start), "-i", video_path,
        "-t", str(end - start), "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-loglevel", "error", out,
    ], check=True)


def caption_clip(client: OpenAI, clip_path: str, seg_index: int, total: int) -> str:
    """Send video clip to vLLM-Omni for structured captioning.

    Uses the OpenAI-compatible /v1/chat/completions endpoint with
    video_url content type and modalities=["text"] for text-only output.
    """
    response = client.chat.completions.create(
        model=VLLM_OMNI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise video analyst. Describe the scene, "
                    "people, dialogue, sounds, and any text visible on screen. "
                    "Be factual and concise."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"file://{clip_path}"}},
                    {"type": "text", "text": f"Segment {seg_index+1}/{total}. Describe this clip."},
                ],
            },
        ],
        max_tokens=1024,
        temperature=0.3,
        extra_body={"modalities": ["text"]},
    )
    return response.choices[0].message.content.strip()


def video_sec_to_dt(sec: float) -> str:
    """Map video timestamp to virtual datetime for EverMemOS temporal reasoning."""
    base = datetime.fromisoformat(BASE_DT)
    return (base + timedelta(seconds=sec * TIME_SCALE)).isoformat()


def store_memory(caption: str, seg_id: str, start_sec: float, video_name: str):
    """Store caption as memory in EverMemOS via POST /api/v1/memories."""
    payload = {
        "message_id": seg_id,
        "create_time": video_sec_to_dt(start_sec),
        "sender": "video_observer",
        "content": f"[Video {start_sec/60:.1f}-{(start_sec+SEGMENT_SECONDS)/60:.1f} min] {caption}",
        "group_id": f"vid_{video_name}",
        "scene": "assistant",
    }
    resp = requests.post(f"{EVERMEMOS_URL}/api/v1/memories", json=payload, timeout=30)
    return resp.status_code in (200, 201)


def query_memory(question: str, video_name: str) -> list[dict]:
    """Search EverMemOS for relevant memories via GET /api/v1/memories/search."""
    payload = {
        "query": question,
        "group_id": f"vid_{video_name}",
        "memory_types": ["episodic_memory", "profile", "semantic_memory"],
        "retrieve_method": "rrf",
        "top_k": 5,
    }
    resp = requests.get(f"{EVERMEMOS_URL}/api/v1/memories/search", json=payload, timeout=15)
    if resp.status_code == 200:
        data = resp.json()
        return data.get("results", data.get("data", []))
    return []


def answer_question(client: OpenAI, question: str, memories: list[dict]) -> str:
    """Generate answer from retrieved memories using vLLM-Omni (text mode)."""
    context = "\n".join(
        f"- {m.get('content', m.get('text', str(m)))[:300]}"
        for m in memories[:5]
    )
    response = client.chat.completions.create(
        model=VLLM_OMNI_MODEL,
        messages=[
            {"role": "system", "content": "Answer based on video memories. Cite timestamps."},
            {"role": "user", "content": f"Question: {question}\n\nMemories:\n{context}"},
        ],
        max_tokens=512,
        temperature=0.3,
        extra_body={"modalities": ["text"]},
    )
    return response.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(
        description="vLLM-Omni + EverMemOS video memory demo"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("question", nargs="?", help="Question about the video")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    args = parser.parse_args()

    video_path = str(Path(args.video).resolve())
    video_name = Path(video_path).stem
    client = OpenAI(base_url=VLLM_OMNI_URL, api_key="EMPTY")

    # ── Ingest ──
    if not args.skip_ingest:
        print(f"Ingesting: {video_path}")
        duration = get_duration(video_path)
        n_segments = int(duration // SEGMENT_SECONDS) + (1 if duration % SEGMENT_SECONDS > 5 else 0)
        print(f"  Duration: {duration:.0f}s → {n_segments} segments")

        tmp_dir = Path(f"/tmp/vllm_omni_demo/{video_name}")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_segments):
            start = i * SEGMENT_SECONDS
            end = min(start + SEGMENT_SECONDS, duration)
            clip = str(tmp_dir / f"seg_{i:03d}.mp4")

            print(f"  [{i+1}/{n_segments}] Extracting {start:.0f}s-{end:.0f}s...")
            extract_clip(video_path, start, end, clip)

            print(f"  [{i+1}/{n_segments}] Captioning via vLLM-Omni...")
            caption = caption_clip(client, clip, i, n_segments)

            seg_id = f"seg_{video_name}_{i:03d}"
            ok = store_memory(caption, seg_id, start, video_name)
            status = "✓" if ok else "✗"
            print(f"  [{i+1}/{n_segments}] {status} Stored to EverMemOS")

        print(f"\nWaiting 5s for indexing...")
        time.sleep(5)
        print("Ingestion complete.\n")

    # ── Query ──
    if args.question:
        print(f"Q: {args.question}")
        memories = query_memory(args.question, video_name)
        print(f"  Retrieved {len(memories)} memories")
        answer = answer_question(client, args.question, memories)
        print(f"\nA: {answer}")
    elif not args.skip_ingest:
        # Interactive mode
        print("Ask questions (type 'quit' to exit):\n")
        while True:
            q = input("Q: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            memories = query_memory(q, video_name)
            answer = answer_question(client, q, memories)
            print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
