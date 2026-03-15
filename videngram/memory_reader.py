"""
VidEngram Memory Reader
Retrieves memories from EverMemOS via lightweight and agentic modes.

Maps to HippoMM's dual-pathway retrieval:
- Lightweight (rrf/bm25/embedding) → HippoMM's Fast Retrieval (Φ_fast)
- Agentic (LLM-guided multi-round) → HippoMM's Detailed Recall (Ψ_detailed)

The agent orchestrator decides which mode to use based on query complexity.

Note on HTTP method: EverMemOS /api/v1/memories/search uses GET with query params.
"""
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import requests

from .config import VidEngramConfig
from .utils import MemoryResult, _fmt_time, create_http_session
from .memory_writer import MemoryWriter

logger = logging.getLogger("videngram.memory_reader")


class MemoryReader:
    """Reads memories from EverMemOS with multiple retrieval strategies."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.evermemos
        self._work_dir = config.work_dir
        self._session = create_http_session()

    def search_episodes(
        self,
        query: str,
        video_path: str,
        mode: str = "hybrid",
        top_k: int = 5,
        omit_group_id: bool = False,
    ) -> list[MemoryResult]:
        """Search episode memories (captions + speech transcripts).

        Both Qwen visual captions and Whisper speech transcripts are stored under
        memory_type='episodic_memory' in EverMemOS (distinguished by user_id sender).
        A single call retrieves both, ranked by relevance score.
        """
        results = self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="episodic_memory",
            retrieval_mode=mode,
            top_k=top_k,
            omit_group_id=omit_group_id,
        )
        return self._rescue_timestamps(results, video_path)

    def search_speech_bm25(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
        omit_group_id: bool = False,
    ) -> list[MemoryResult]:
        """BM25 keyword search over all episodic memories (captions + speech).

        Best for: "when did speaker say X", exact word lookup, keyword queries.
        Results get timestamp rescue from local cues.json for accuracy.
        """
        combined = self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="episodic_memory",
            retrieval_mode="bm25",
            top_k=top_k,
            omit_group_id=omit_group_id,
        )
        return self._rescue_timestamps(combined, video_path)

    def search_profiles(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
        omit_group_id: bool = False,
    ) -> list[MemoryResult]:
        """Search entity profiles (EverMemOS memory_type='profile').

        Returns profiles of people, topics, or concepts from the video.
        """
        return self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="profile",
            retrieval_mode="hybrid",
            top_k=top_k,
            omit_group_id=omit_group_id,
        )

    def search_semantic(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
    ) -> list[MemoryResult]:
        """Search semantic memories (facts, preferences extracted by EverMemOS)."""
        return self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="semantic_memory",
            retrieval_mode="embedding",
            top_k=top_k,
        )

    def search_agentic(
        self,
        query: str,
        video_path: str,
        top_k: int = 20,
        omit_group_id: bool = False,
    ) -> list[MemoryResult]:
        """LLM-guided agentic retrieval for complex multi-hop queries.

        This is the "big gun" — it uses the LLM to expand the query,
        perform multiple rounds of retrieval, and intelligently fuse results.
        Slower (~2-10s) but much more thorough for complex questions.

        Mirrors HippoMM's Detailed Recall pathway (Ψ_detailed).
        """
        params: dict = {
            "query": query,
            "memory_types": ["episodic_memory"],
            "retrieve_method": "agentic",
            "top_k": top_k,
        }
        if omit_group_id:
            # Cross-video: search each known video's group_id, merge by score
            group_ids = self._all_known_group_ids()
            if not group_ids:
                group_ids = [MemoryWriter._video_group_id(video_path)]
            all_results: list[MemoryResult] = []
            seen: set[str] = set()
            for gid in group_ids:
                p = {**params, "group_id": gid}
                try:
                    resp = self._session.get(self.cfg.search_url, params=p, timeout=60)
                    if resp.status_code == 200:
                        for r in self._parse_results(resp.json()):
                            if r.content not in seen:
                                seen.add(r.content)
                                all_results.append(r)
                except requests.RequestException:
                    pass
            all_results.sort(key=lambda r: r.score, reverse=True)
            return all_results[:top_k]

        params["group_id"] = MemoryWriter._video_group_id(video_path)

        try:
            resp = self._session.get(
                self.cfg.search_url,
                params=params,
                timeout=60,
            )
            if resp.status_code == 200:
                return self._parse_results(resp.json())
            else:
                logger.warning(f"Agentic retrieval failed: {resp.status_code}")
                # Fallback to lightweight
                return self.search_episodes(query, video_path, mode="hybrid", top_k=top_k)
        except requests.RequestException as e:
            logger.error(f"Agentic retrieval error: {e}")
            return []

    def multi_source_search(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
    ) -> dict[str, list[MemoryResult]]:
        """Search across ALL memory sources in parallel.

        Returns results grouped by source type. The agent can then
        synthesize across sources for comprehensive answers.
        """
        results = {}
        results["episodes"] = self.search_episodes(
            query, video_path, top_k=top_k
        )
        results["profiles"] = self.search_profiles(
            query, video_path, top_k=3
        )
        results["semantic"] = self.search_semantic(
            query, video_path, top_k=top_k
        )
        return results

    # ── Cross-Video Source Tagging ────────────────────────────────────

    @staticmethod
    def tag_cross_video_content(
        results: list[MemoryResult], current_group_id: str
    ) -> list[MemoryResult]:
        """For memories from other videos, strip the [Video M:SS] timestamp prefix.

        Cross-video memories should contribute knowledge to the answer but
        must not carry timestamps — timestamps from another video's timeline
        are meaningless and would confuse the agent and the frontend seeker.
        """
        tagged = []
        for r in results:
            src_gid = r.metadata.get("source_group_id", "")
            if src_gid and src_gid != current_group_id:
                # Strip the [Video/Episode M:SS - N:SS] prefix entirely
                new_content = re.sub(
                    r'^\[(?:Video|Episode)(?:\s+analysis)?\s+\d+:\d{2}(?::\d{2})?\s*-\s*\d+:\d{2}(?::\d{2})?\]\s*',
                    '',
                    r.content,
                ).strip()
                tagged.append(MemoryResult(
                    content=new_content,
                    score=r.score,
                    memory_type=r.memory_type,
                    metadata=r.metadata,
                ))
            else:
                tagged.append(r)
        return tagged

    # ── Timestamp Rescue ──────────────────────────────────────────────

    def _rescue_timestamps(
        self, results: list[MemoryResult], video_path: str
    ) -> list[MemoryResult]:
        """Replace EverMemOS timestamps with accurate ones from local cues.json.

        EverMemOS processes stored content through its own LLM and may corrupt
        or lose the [Video M:SS - N:SS] timestamp prefix we embedded.
        We recover accurate timestamps by:
          1. Reading start_sec directly from result metadata (written by MemoryWriter)
          2. Finding the matching entry in the local cues.json file for start + end
          3. Rewriting the [Video ...] prefix in the result content

        Cross-video memories (source_group_id != current video) are skipped —
        their cues belong to a different video's cues.json.
        """
        cues = self._load_cues(video_path)
        if not cues:
            return results

        current_gid = MemoryWriter._video_group_id(video_path)
        rescued = []
        for r in results:
            # Skip rescue for memories from other videos — we'd use wrong cues
            if r.metadata.get("source_group_id", current_gid) != current_gid:
                rescued.append(r)
                continue
            start_sec = r.metadata.get("start_sec")
            if start_sec is not None:
                try:
                    start_sec = float(start_sec)
                except (ValueError, TypeError):
                    start_sec = None
            if start_sec is not None:
                cue = self._find_cue_by_start(cues, start_sec)
                if cue:
                    body = re.sub(
                        r'^\[(?:Video|Episode)[^\]]*\]\s*', '', r.content
                    ).strip()
                    ts_prefix = (
                        f"[Video {_fmt_time(cue['start_sec'])}"
                        f" - {_fmt_time(cue['end_sec'])}]"
                    )
                    rescued.append(MemoryResult(
                        content=f"{ts_prefix} {body}",
                        score=r.score,
                        memory_type=r.memory_type,
                        metadata={
                            **r.metadata,
                            "start_sec": cue["start_sec"],
                            "end_sec": cue["end_sec"],
                        },
                    ))
                    continue
            rescued.append(r)
        return rescued

    def _load_cues(self, video_path: str) -> list[dict]:
        """Load cues.json for the given video (segment captions with exact timestamps)."""
        group_id = MemoryWriter._video_group_id(video_path)
        cues_file = self._work_dir / f"{group_id}_cues.json"
        try:
            return json.loads(cues_file.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _find_cue_by_start(
        self, cues: list[dict], start_sec: float, tolerance: float = 2.0
    ) -> Optional[dict]:
        """Find the cue whose start_sec is closest to start_sec within tolerance seconds."""
        best: Optional[dict] = None
        best_dist = tolerance
        for cue in cues:
            dist = abs(cue.get("start_sec", 0.0) - start_sec)
            if dist < best_dist:
                best_dist = dist
                best = cue
        return best

    # ── Internal ──────────────────────────────────────────────────────

    def _all_known_group_ids(self) -> list[str]:
        """Return all group_ids found in work_dir (from *_cues.json filenames)."""
        cues_files = list(self._work_dir.glob("*_cues.json"))
        group_ids = []
        for f in cues_files:
            gid = f.name.removesuffix("_cues.json")
            if gid:
                group_ids.append(gid)
        return group_ids

    def _retrieve_lightweight(
        self,
        query: str,
        video_path: str,
        data_source: str,
        retrieval_mode: str = "hybrid",
        top_k: int = 5,
        omit_group_id: bool = False,
    ) -> list[MemoryResult]:
        """Call EverMemOS GET /api/v1/memories/search with query params."""
        if omit_group_id:
            # Cross-video: search each known video's group_id, merge by score
            group_ids = self._all_known_group_ids()
            if not group_ids:
                group_ids = [MemoryWriter._video_group_id(video_path)]
            all_results: list[MemoryResult] = []
            seen: set[str] = set()
            for gid in group_ids:
                p: dict = {
                    "query": query,
                    "memory_types": [data_source],
                    "retrieve_method": retrieval_mode,
                    "top_k": top_k,
                    "group_id": gid,
                }
                try:
                    resp = self._session.get(self.cfg.search_url, params=p, timeout=45)
                    if resp.status_code == 200:
                        for r in self._parse_results(resp.json()):
                            if r.content not in seen:
                                seen.add(r.content)
                                all_results.append(r)
                except requests.RequestException:
                    pass
            all_results.sort(key=lambda r: r.score, reverse=True)
            return all_results[:top_k]

        params: dict = {
            "query": query,
            "memory_types": [data_source],
            "retrieve_method": retrieval_mode,
            "top_k": top_k,
            "group_id": MemoryWriter._video_group_id(video_path),
        }

        try:
            resp = self._session.get(
                self.cfg.search_url,
                params=params,
                timeout=45,
            )
            if resp.status_code == 200:
                return self._parse_results(resp.json())
            else:
                logger.warning(
                    f"Lightweight retrieval failed ({data_source}/{retrieval_mode}): "
                    f"{resp.status_code}"
                )
                return []
        except requests.RequestException as e:
            logger.error(f"Retrieval error: {e}")
            return []

    @staticmethod
    def _parse_results(response_data: dict) -> list[MemoryResult]:
        """Parse EverMemOS response into MemoryResult objects.

        EverMemOS search returns:
          {
            "result": {
              "memories": [ { "<group_id>": [ {memory_obj}, ... ] } ],
              "scores":   [ { "<group_id>": [ score, ... ] } ]
            }
          }
        Each element in "memories" is a dict keyed by group_id (or memory_type in
        older versions); each element in "scores" is the parallel score array.
        This parser handles all observed formats.
        """
        results = []

        items: list = []
        score_groups: list = []
        detected_format = "unknown"

        if isinstance(response_data, list):
            items = response_data
            detected_format = "bare_list"
        elif isinstance(response_data, dict):
            if "results" in response_data:
                items = response_data["results"]
                detected_format = "results"
            elif "result" in response_data:
                inner = response_data["result"]
                if isinstance(inner, dict):
                    items = inner.get("memories", inner.get("data", []))
                    score_groups = inner.get("scores", [])
                    detected_format = "result.memories" if "memories" in inner else "result.data"
                    if not items and "pending_messages" in inner:
                        items = inner["pending_messages"]
                        detected_format = "result.pending_messages"
                elif isinstance(inner, list):
                    items = inner
                    detected_format = "result_list"
            elif "data" in response_data:
                items = response_data["data"]
                detected_format = "data"
            elif "memories" in response_data:
                items = response_data["memories"]
                score_groups = response_data.get("scores", [])
                detected_format = "memories"
            else:
                logger.warning(
                    f"_parse_results: unrecognised response format — "
                    f"top-level keys: {list(response_data.keys())[:10]}"
                )
        logger.debug(f"_parse_results: format={detected_format}, {len(items)} group(s)")

        def _extract_item(d: dict, score: float = 0.0) -> MemoryResult:
            """Extract a MemoryResult from a single memory object dict."""
            content = (
                d.get("episode") or
                d.get("content") or
                d.get("text") or
                d.get("memory") or
                d.get("summary") or
                d.get("atomic_fact") or
                ""
            )
            if isinstance(content, list):
                content = "; ".join(str(x) for x in content if x)
            content = str(content).strip()

            meta = dict(d.get("metadata", {}) or {})
            for time_key in ("create_time", "timestamp", "created_at", "message_create_time"):
                val = d.get(time_key)
                if val and "create_time" not in meta:
                    meta["create_time"] = val
                    break

            item_score = float(d.get("score", d.get("relevance_score", score)))
            return MemoryResult(
                content=content,
                score=item_score,
                memory_type=d.get("memory_type", d.get("type", "")),
                metadata=meta,
            )

        for group_idx, item in enumerate(items):
            # Fetch parallel scores for this group if present
            group_scores: dict = {}
            if group_idx < len(score_groups) and isinstance(score_groups[group_idx], dict):
                group_scores = score_groups[group_idx]

            if isinstance(item, dict):
                # Handle old format where items have an explicit "memories" key
                memories_list = item.get("memories")
                if memories_list and isinstance(memories_list, list):
                    for sub in memories_list:
                        if isinstance(sub, dict):
                            results.append(_extract_item(sub))
                    continue

                # Current EverMemOS format: { "<group_id_or_type>": [memory_obj, ...] }
                # Each value that is a list of dicts contains the actual memory objects.
                unpacked = False
                for key, val in item.items():
                    if isinstance(val, list):
                        key_scores = group_scores.get(key, [])
                        for mem_idx, sub in enumerate(val):
                            if isinstance(sub, dict):
                                score = (
                                    float(key_scores[mem_idx])
                                    if mem_idx < len(key_scores)
                                    else 0.0
                                )
                                r = _extract_item(sub, score=score)
                                if key and "source_group_id" not in r.metadata:
                                    r.metadata["source_group_id"] = key
                                results.append(r)
                                unpacked = True
                if not unpacked:
                    # Fallback: treat the dict itself as a flat memory object
                    results.append(_extract_item(item))

            elif isinstance(item, str):
                results.append(MemoryResult(content=item))

        return results
