#!/usr/bin/env python3
"""
VidEngram Gradio Demo
Web UI for video ingestion and interactive Q&A.

Usage:
    python -m demo.gradio_app
"""
import logging
import tempfile
from pathlib import Path

import gradio as gr

from videngram.config import VidEngramConfig
from videngram.pipeline import VidEngramPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

pipeline = VidEngramPipeline()


def ingest_video(video_file, progress=gr.Progress()):
    """Handle video upload and ingestion."""
    if video_file is None:
        return "⚠️ Please upload a video file first."

    progress(0.1, desc="Starting ingestion...")
    try:
        stats = pipeline.ingest(video_file, parallel_caption=False)
        progress(1.0, desc="Done!")

        return (
            f"✅ **Ingestion Complete**\n\n"
            f"- **Segments:** {stats.get('segments', 0)}\n"
            f"- **Memories:** {stats.get('memories_total', 0)} "
            f"({stats.get('memories_segments', 0)} segments, "
            f"{stats.get('memories_episodes', 0)} episodes, "
            f"{stats.get('memories_profiles', 0)} profiles)\n"
            f"- **Time:** {stats.get('total_time', 0):.1f}s\n\n"
            f"Ready to answer questions!"
        )
    except Exception as e:
        return f"❌ **Error:** {e}"


def ask_question(video_file, question, history):
    """Handle a question about the video."""
    if video_file is None:
        return history + [[question, "⚠️ Please upload and ingest a video first."]]
    if not question.strip():
        return history

    try:
        response = pipeline.query(question, video_file)

        answer = response.answer
        if response.actions:
            tools = ", ".join(a.tool for a in response.actions)
            answer += f"\n\n*Tools used: {tools}*"

        return history + [[question, answer]]
    except Exception as e:
        return history + [[question, f"❌ Error: {e}"]]


def clear_chat():
    pipeline.clear_history()
    return []


# Build the Gradio interface
with gr.Blocks(
    title="VidEngram — Video Memory Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# 🎬 VidEngram\n"
        "### Hippocampal-inspired Video Memory with EverMemOS\n"
        "Upload a video → Ingest into memory → Ask questions interactively"
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            ingest_btn = gr.Button("🧠 Ingest into Memory", variant="primary")
            ingest_output = gr.Markdown(label="Ingestion Status")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=400)
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What happened in the first 5 minutes?",
                    scale=5,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat")

    # Sample questions
    gr.Markdown("### 💡 Sample Questions")
    gr.Examples(
        examples=[
            "What was the main topic discussed?",
            "Who were the key speakers and what were their roles?",
            "What happened in the second half of the video?",
            "Were there any recurring themes or callbacks?",
            "Summarize the video in 3 sentences.",
        ],
        inputs=msg_input,
    )

    # Wire up events
    ingest_btn.click(ingest_video, [video_input], [ingest_output])
    send_btn.click(ask_question, [video_input, msg_input, chatbot], [chatbot])
    msg_input.submit(ask_question, [video_input, msg_input, chatbot], [chatbot])
    send_btn.click(lambda: "", None, [msg_input])
    msg_input.submit(lambda: "", None, [msg_input])
    clear_btn.click(clear_chat, None, [chatbot])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
