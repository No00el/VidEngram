#!/usr/bin/env python3
"""
VidEngram Evaluation Script
============================
Benchmark VidEngram against a set of video QA pairs.

This demonstrates quantitative evaluation for the Memory Genesis Competition.
Supports both automated scoring (LLM-as-judge) and manual review.

Usage:
    # Run with a QA file
    python -m scripts.evaluate --video lecture.mp4 --qa qa_pairs.json

    # Run with inline questions
    python -m scripts.evaluate --video lecture.mp4 \\
        --questions "What topic is discussed?" "Who is the speaker?"

    # Generate a QA template from an ingested video
    python -m scripts.evaluate --video lecture.mp4 --generate-qa

QA File Format (qa_pairs.json):
    [
        {
            "question": "What is discussed in the first 5 minutes?",
            "reference_answer": "The speaker introduces machine learning basics.",
            "category": "factual",
            "difficulty": "easy"
        },
        ...
    ]
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from videngram.config import VidEngramConfig
from videngram.pipeline import VidEngramPipeline

logger = logging.getLogger("videngram.eval")


@dataclass
class QAPair:
    question: str
    reference_answer: str = ""
    category: str = "general"   # factual, temporal, entity, multi-hop, visual
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvalResult:
    question: str
    reference_answer: str
    predicted_answer: str
    category: str
    difficulty: str
    num_tools_used: int = 0
    tools_used: list = field(default_factory=list)
    has_timestamp_citation: bool = False
    latency_sec: float = 0.0
    # LLM-as-judge scores (0-5)
    relevance_score: Optional[float] = None
    completeness_score: Optional[float] = None
    grounding_score: Optional[float] = None


@dataclass
class EvalSummary:
    video_path: str
    total_questions: int
    avg_latency_sec: float
    avg_tools_per_query: float
    timestamp_citation_rate: float
    category_breakdown: dict = field(default_factory=dict)
    results: list = field(default_factory=list)
    # If LLM scoring was used
    avg_relevance: Optional[float] = None
    avg_completeness: Optional[float] = None
    avg_grounding: Optional[float] = None


def evaluate(
    pipeline: VidEngramPipeline,
    video_path: str,
    qa_pairs: list[QAPair],
    use_llm_judge: bool = False,
) -> EvalSummary:
    """Run evaluation on a set of QA pairs.
    
    Args:
        pipeline: Initialized VidEngramPipeline (video should already be ingested)
        video_path: Path to the video
        qa_pairs: List of QA pairs to evaluate
        use_llm_judge: Whether to use LLM-as-judge for scoring
    
    Returns:
        EvalSummary with per-question results and aggregate metrics
    """
    results = []

    for i, qa in enumerate(qa_pairs):
        print(f"\n  [{i+1}/{len(qa_pairs)}] Q: {qa.question[:80]}...")

        t0 = time.time()
        pipeline.clear_history()
        response = pipeline.query(qa.question, video_path, multi_turn=False)
        latency = time.time() - t0

        # Check for timestamp citations in the answer
        import re
        has_ts = bool(re.search(r'\d+:\d+|\d+\.\d+\s*min', response.answer))

        tools = [a.tool for a in response.actions]

        result = EvalResult(
            question=qa.question,
            reference_answer=qa.reference_answer,
            predicted_answer=response.answer,
            category=qa.category,
            difficulty=qa.difficulty,
            num_tools_used=len(response.actions),
            tools_used=tools,
            has_timestamp_citation=has_ts,
            latency_sec=latency,
        )

        # LLM-as-judge scoring
        if use_llm_judge and qa.reference_answer:
            scores = _llm_judge_score(
                qa.question, qa.reference_answer, response.answer, pipeline
            )
            result.relevance_score = scores.get("relevance", None)
            result.completeness_score = scores.get("completeness", None)
            result.grounding_score = scores.get("grounding", None)

        results.append(result)
        tools_str = ", ".join(tools) if tools else "none"
        print(f"       A: {response.answer[:100]}...")
        print(f"       [{latency:.1f}s, tools: {tools_str}, ts_cited: {has_ts}]")

    # Aggregate metrics
    avg_latency = sum(r.latency_sec for r in results) / len(results)
    avg_tools = sum(r.num_tools_used for r in results) / len(results)
    ts_rate = sum(1 for r in results if r.has_timestamp_citation) / len(results)

    # Category breakdown
    categories = {}
    for r in results:
        cat = r.category
        if cat not in categories:
            categories[cat] = {"count": 0, "ts_cited": 0, "avg_latency": 0}
        categories[cat]["count"] += 1
        categories[cat]["ts_cited"] += int(r.has_timestamp_citation)
        categories[cat]["avg_latency"] += r.latency_sec
    for cat in categories:
        categories[cat]["avg_latency"] /= categories[cat]["count"]
        categories[cat]["ts_citation_rate"] = (
            categories[cat]["ts_cited"] / categories[cat]["count"]
        )

    summary = EvalSummary(
        video_path=video_path,
        total_questions=len(results),
        avg_latency_sec=avg_latency,
        avg_tools_per_query=avg_tools,
        timestamp_citation_rate=ts_rate,
        category_breakdown=categories,
        results=[asdict(r) for r in results],
    )

    # LLM judge aggregates
    relevance_scores = [r.relevance_score for r in results if r.relevance_score is not None]
    completeness_scores = [r.completeness_score for r in results if r.completeness_score is not None]
    grounding_scores = [r.grounding_score for r in results if r.grounding_score is not None]
    if relevance_scores:
        summary.avg_relevance = sum(relevance_scores) / len(relevance_scores)
    if completeness_scores:
        summary.avg_completeness = sum(completeness_scores) / len(completeness_scores)
    if grounding_scores:
        summary.avg_grounding = sum(grounding_scores) / len(grounding_scores)

    return summary


def _llm_judge_score(
    question: str,
    reference: str,
    predicted: str,
    pipeline: VidEngramPipeline,
) -> dict:
    """Use the planning LLM as a judge to score answer quality.
    
    Returns dict with relevance, completeness, grounding scores (0-5).
    """
    from openai import OpenAI

    agent_cfg = pipeline.config.agent
    client = OpenAI(
        base_url=agent_cfg.planning_llm_base_url,
        api_key=agent_cfg.planning_llm_api_key,
    )

    judge_prompt = f"""Score the predicted answer against the reference answer.
    
Question: {question}
Reference Answer: {reference}
Predicted Answer: {predicted}

Score each dimension 0-5:
- relevance: Does the predicted answer address the question?
- completeness: Does it cover the key points from the reference?
- grounding: Does it cite specific timestamps or evidence?

Respond with ONLY a JSON object: {{"relevance": N, "completeness": N, "grounding": N}}"""

    try:
        response = client.chat.completions.create(
            model=agent_cfg.planning_llm_model,
            messages=[
                {"role": "system", "content": "You are an evaluation judge. Respond only with JSON."},
                {"role": "user", "content": judge_prompt},
            ],
            max_tokens=100,
            temperature=0.1,
            extra_body={"modalities": ["text"]},
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"LLM judge failed: {e}")
        return {}


def generate_qa_template(video_path: str) -> list[dict]:
    """Generate a sample QA template covering different question categories."""
    name = Path(video_path).stem
    return [
        {
            "question": f"What is the main topic discussed in {name}?",
            "reference_answer": "",
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "question": "Who are the main speakers or people shown?",
            "reference_answer": "",
            "category": "entity",
            "difficulty": "easy",
        },
        {
            "question": "What happens in the first 2 minutes?",
            "reference_answer": "",
            "category": "temporal",
            "difficulty": "easy",
        },
        {
            "question": "What is the relationship between the topics discussed in the first and second halves?",
            "reference_answer": "",
            "category": "multi-hop",
            "difficulty": "hard",
        },
        {
            "question": "Are there any text or signs visible on screen? What do they say?",
            "reference_answer": "",
            "category": "visual",
            "difficulty": "medium",
        },
        {
            "question": "What is the emotional tone of the video?",
            "reference_answer": "",
            "category": "factual",
            "difficulty": "medium",
        },
        {
            "question": "Summarize the key takeaways from this video.",
            "reference_answer": "",
            "category": "multi-hop",
            "difficulty": "hard",
        },
    ]


def print_summary(summary: EvalSummary):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print("  VidEngram Evaluation Summary")
    print("=" * 60)
    print(f"  Video:                  {Path(summary.video_path).name}")
    print(f"  Questions:              {summary.total_questions}")
    print(f"  Avg latency:            {summary.avg_latency_sec:.1f}s")
    print(f"  Avg tools per query:    {summary.avg_tools_per_query:.1f}")
    print(f"  Timestamp citation:     {summary.timestamp_citation_rate:.0%}")

    if summary.avg_relevance is not None:
        print(f"\n  LLM Judge Scores (0-5):")
        print(f"    Relevance:            {summary.avg_relevance:.1f}")
        print(f"    Completeness:         {summary.avg_completeness:.1f}")
        print(f"    Grounding:            {summary.avg_grounding:.1f}")

    if summary.category_breakdown:
        print(f"\n  Category Breakdown:")
        for cat, stats in summary.category_breakdown.items():
            print(f"    {cat:12s}: {stats['count']} questions, "
                  f"ts_cite={stats['ts_citation_rate']:.0%}, "
                  f"avg_lat={stats['avg_latency']:.1f}s")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VidEngram Evaluation")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--qa", help="Path to QA pairs JSON file")
    parser.add_argument("--questions", nargs="+", help="Inline questions")
    parser.add_argument("--generate-qa", action="store_true",
                        help="Generate a QA template file")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Use LLM-as-judge for scoring")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip video ingestion")
    parser.add_argument("--output", help="Save results to JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.generate_qa:
        template = generate_qa_template(args.video)
        out = Path(args.video).stem + "_qa.json"
        with open(out, "w") as f:
            json.dump(template, f, indent=2)
        print(f"Generated QA template: {out}")
        print("Fill in the reference_answer fields, then run evaluation.")
        return

    # Build QA pairs
    qa_pairs = []
    if args.qa:
        with open(args.qa) as f:
            raw = json.load(f)
        qa_pairs = [QAPair(**item) for item in raw]
    elif args.questions:
        qa_pairs = [QAPair(question=q) for q in args.questions]
    else:
        print("Error: Provide --qa file or --questions")
        sys.exit(1)

    # Initialize pipeline and ingest
    pipeline = VidEngramPipeline()
    if not args.skip_ingest:
        print(f"Ingesting video: {args.video}")
        pipeline.ingest(args.video)

    # Run evaluation
    print(f"\nEvaluating {len(qa_pairs)} questions...")
    summary = evaluate(
        pipeline, args.video, qa_pairs,
        use_llm_judge=args.llm_judge,
    )

    print_summary(summary)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
