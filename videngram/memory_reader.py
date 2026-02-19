"""
VidEngram Memory Reader
Retrieves memories from EverMemOS via lightweight and agentic modes.

Maps to HippoMM's dual-pathway retrieval:
- Lightweight (rrf/bm25/embedding) → HippoMM's Fast Retrieval (Φ_fast)
- Agentic (LLM-guided multi-round) → HippoMM's Detailed Recall (Ψ_detailed)

The agent orchestrator decides which mode to use based on query complexity.

Note on HTTP method: EverMemOS /api/v1/memories/search uses GET with query params.
"""
import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import VidEngramConfig
from .utils import MemoryResult
from .memory_writer import MemoryWriter

logger = logging.getLogger("videngram.memory_reader")


def _create_session(retries: int = 3, backoff: float = 0.5) -> requests.Session:
    """Create an HTTP session with retry logic and connection pooling."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class MemoryReader:
    """Reads memories from EverMemOS with multiple retrieval strategies."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.evermemos
        self._session = _create_session()

    def search_episodes(
        self,
        query: str,
        video_path: str,
        mode: str = "rrf",
        top_k: int = 5,
    ) -> list[MemoryResult]:
        """Search episode memories (EverMemOS memory_type='episodic_memory').

        This retrieves both segment-level memories and episode summaries.
        The 'rrf' mode is recommended as it combines BM25 keyword matching
        with embedding-based semantic search via reciprocal rank fusion.
        """
        return self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="episodic_memory",
            retrieval_mode=mode,
            top_k=top_k,
        )

    def search_profiles(
        self,
        query: str,
        video_path: str,
        top_k: int = 5,
    ) -> list[MemoryResult]:
        """Search entity profiles (EverMemOS memory_type='profile').

        Returns profiles of people, topics, or concepts from the video.
        """
        return self._retrieve_lightweight(
            query=query,
            video_path=video_path,
            data_source="profile",
            retrieval_mode="rrf",
            top_k=top_k,
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
    ) -> list[MemoryResult]:
        """LLM-guided agentic retrieval for complex multi-hop queries.

        This is the "big gun" — it uses the LLM to expand the query,
        perform multiple rounds of retrieval, and intelligently fuse results.
        Slower (~2-10s) but much more thorough for complex questions.

        Mirrors HippoMM's Detailed Recall pathway (Ψ_detailed).
        """
        group_id = MemoryWriter._video_group_id(video_path)

        params = {
            "query": query,
            "group_id": group_id,
            "memory_type": "episodic_memory",
            "top_k": top_k,
        }

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
                return self.search_episodes(query, video_path, mode="rrf", top_k=top_k)
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

    # ── Internal ──────────────────────────────────────────────────────

    def _retrieve_lightweight(
        self,
        query: str,
        video_path: str,
        data_source: str,
        retrieval_mode: str = "rrf",
        top_k: int = 5,
    ) -> list[MemoryResult]:
        """Call EverMemOS GET /api/v1/memories/search with query params."""
        group_id = MemoryWriter._video_group_id(video_path)

        params = {
            "query": query,
            "group_id": group_id,
            "memory_type": data_source,
            "top_k": top_k,
        }

        try:
            resp = self._session.get(
                self.cfg.search_url,
                params=params,
                timeout=15,
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
        
        Handles multiple response formats observed across EverMemOS versions:
          - {"results": [...]}           — v1 standard
          - {"data": [...]}              — alternative format
          - {"result": {"memories": [...]}} — retrieval endpoint format
          - [...]                        — bare list
        """
        results = []

        # Determine which format we got
        items = []
        if isinstance(response_data, list):
            items = response_data
        elif isinstance(response_data, dict):
            if "results" in response_data:
                items = response_data["results"]
            elif "result" in response_data:
                inner = response_data["result"]
                if isinstance(inner, dict):
                    items = inner.get("memories", inner.get("data", []))
                    # Also include pending_messages if no processed memories yet
                    if not items and "pending_messages" in inner:
                        items = inner["pending_messages"]
                elif isinstance(inner, list):
                    items = inner
            elif "data" in response_data:
                items = response_data["data"]
            elif "memories" in response_data:
                items = response_data["memories"]

        for item in items:
            if isinstance(item, dict):
                # Handle nested memory format (some EverMemOS versions wrap in groups)
                memories_list = item.get("memories", None)
                if memories_list and isinstance(memories_list, list):
                    for sub in memories_list:
                        if isinstance(sub, dict):
                            results.append(MemoryResult(
                                content=sub.get("memory", sub.get("content", sub.get("text", ""))),
                                score=float(sub.get("score", sub.get("relevance_score", 0.0))),
                                memory_type=sub.get("memory_type", sub.get("type", "")),
                                metadata=sub.get("metadata", {}),
                            ))
                else:
                    results.append(MemoryResult(
                        content=item.get("content", item.get("text", item.get("memory", ""))),
                        score=float(item.get("score", item.get("relevance_score", 0.0))),
                        memory_type=item.get("memory_type", item.get("type", "")),
                        metadata=item.get("metadata", {}),
                    ))
            elif isinstance(item, str):
                results.append(MemoryResult(content=item))

        return results
