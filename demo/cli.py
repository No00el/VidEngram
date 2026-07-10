#!/usr/bin/env python3
"""
VidEngram CLI Demo
Interactive command-line interface for video ingestion and querying.

Usage:
    # Ingest a video
    python -m demo.cli ingest path/to/video.mp4

    # Batch-ingest a directory (or a .txt manifest of video paths)
    python -m demo.cli batch path/to/videos/

    # Query interactively
    python -m demo.cli chat path/to/video.mp4

    # Single query
    python -m demo.cli query path/to/video.mp4 "What happened in the first 5 minutes?"
"""
import argparse
import base64
import logging
import sys

from videngram.config import VidEngramConfig
from videngram.pipeline import VidEngramPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress HTTP client libraries that log full request/response bodies at DEBUG
    for noisy in ("httpx", "httpcore", "openai", "openai._base_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_ingest(args):
    """Ingest a video into VidEngram memory."""
    import json as _json

    def _encode_event(obj: dict) -> str:
        return base64.b64encode(_json.dumps(obj, ensure_ascii=False).encode("utf-8")).decode("ascii")

    def on_caption_ready(event_type, seg_or_segs, data):
        """Emit structured JSON events so the demo web UI can sync video playback."""
        if event_type == "segments":
            segs_list = seg_or_segs
            seg_data = [
                {"index": i, "start_sec": s.start_sec, "end_sec": s.end_sec}
                for i, s in enumerate(segs_list)
            ]
            payload = {"count": len(seg_data), "segments": seg_data}
            print(f"SEGMENTS_JSON_B64: {_encode_event(payload)}", flush=True)
        elif event_type == "caption":
            index, cap = data
            seg = seg_or_segs
            payload = {
                "index": index,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "text": cap.raw_text,
            }
            print(f"CAPTION_JSON_B64: {_encode_event(payload)}", flush=True)

    pipeline = VidEngramPipeline()
    print(f"\n🎬 Ingesting: {args.video}")
    print("This may take a while depending on video length...\n")

    stats = pipeline.ingest(
        args.video,
        parallel_caption=args.parallel,
        on_caption_ready=on_caption_ready,
    )
    print("\n✅ Ingestion complete!")


def cmd_batch(args):
    """Batch-ingest every video in a directory (or listed in a .txt manifest)."""
    from pathlib import Path

    VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    root = Path(args.path)
    if root.is_dir():
        videos = sorted(p for p in root.rglob(args.glob) if p.suffix.lower() in VIDEO_EXTS)
    elif root.suffix.lower() == ".txt":
        # One video path per line; blank lines and '#' comments are ignored.
        videos = [Path(ln.strip()) for ln in root.read_text().splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    else:
        videos = [root]

    if not videos:
        print(f"No videos found under {args.path!r}")
        sys.exit(1)

    pipeline = VidEngramPipeline()
    print(f"\n📦 Batch ingest: {len(videos)} video(s)\n")
    succeeded, failed = 0, []
    for i, video in enumerate(videos):
        print(f"[{i + 1}/{len(videos)}] 🎬 {video}")
        try:
            pipeline.ingest(str(video), parallel_caption=args.parallel)
            succeeded += 1
            print("    ✅ done\n")
        except Exception as e:  # one bad video should not abort the whole batch
            failed.append((str(video), str(e)))
            print(f"    ❌ failed: {e}\n")

    print("=" * 60)
    print(f"Batch complete: {succeeded} succeeded, {len(failed)} failed")
    for v, e in failed:
        print(f"  ✗ {v}: {e}")


def cmd_query(args):
    """Single query against an ingested video."""
    pipeline = VidEngramPipeline()
    print(f"\n🔍 Querying: '{args.question}'")
    print(f"   Video: {args.video}\n")

    response = pipeline.query(args.question, args.video, multi_turn=False)

    print("=" * 60)
    print(f"Answer: {response.answer}")
    print("=" * 60)

    if response.sources:
        print(f"\n📚 Sources ({len(response.sources)}):")
        for i, s in enumerate(response.sources[:5]):
            print(f"  [{i+1}] {s.content[:150]}...")

    if response.actions:
        print(f"\n🔧 Agent actions ({len(response.actions)}):")
        for a in response.actions:
            print(f"  → {a.tool}({a.input_params})")

    if response.grounded_clips:
        print(f"\n🎥 Grounded clips: {response.grounded_clips}")


def cmd_chat(args):
    """Interactive chat with video memory."""
    pipeline = VidEngramPipeline()

    if args.ingest_first:
        print(f"\n🎬 Ingesting video first: {args.video}")
        pipeline.ingest(args.video, parallel_caption=args.parallel)

    print(f"\n💬 VidEngram Chat — Video: {args.video}")
    print("   Type your questions. Commands: /clear, /stats, /quit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question == "/quit":
            print("Goodbye!")
            break
        if question == "/clear":
            pipeline.clear_history()
            print("  [Chat history cleared]")
            continue
        if question == "/stats":
            stats = pipeline._ingested_videos.get(str(args.video), {})
            if stats:
                pipeline._print_stats(stats)
            else:
                print("  [No ingestion stats available]")
            continue

        response = pipeline.query(question, args.video)

        print(f"\nVidEngram: {response.answer}")

        if response.actions:
            tools_used = ", ".join(a.tool for a in response.actions)
            print(f"  [Tools: {tools_used}]")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="VidEngram — Video Memory Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a video into memory")
    p_ingest.add_argument("video", help="Path to video file")
    p_ingest.add_argument("--parallel", action="store_true", help="Parallel captioning")

    # batch
    p_batch = sub.add_parser("batch", help="Batch-ingest a directory or manifest of videos")
    p_batch.add_argument("path", help="Directory of videos, or a .txt manifest (one path per line)")
    p_batch.add_argument("--glob", default="*",
                         help="Filename pattern when PATH is a directory (default: all videos)")
    p_batch.add_argument("--parallel", action="store_true", help="Parallel captioning")

    # query
    p_query = sub.add_parser("query", help="Ask a single question")
    p_query.add_argument("video", help="Path to video file")
    p_query.add_argument("question", help="Question to ask")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat with video")
    p_chat.add_argument("video", help="Path to video file")
    p_chat.add_argument("--ingest-first", action="store_true",
                        help="Ingest video before chatting")
    p_chat.add_argument("--parallel", action="store_true", help="Parallel captioning")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"ingest": cmd_ingest, "batch": cmd_batch, "query": cmd_query, "chat": cmd_chat}[args.command](args)


if __name__ == "__main__":
    main()
