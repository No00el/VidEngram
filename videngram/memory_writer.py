"""
VidEngram Memory Writer
Adapts consolidated memories for EverMemOS ingestion via POST /api/v1/memories.

Key design decisions:
- scene="assistant" (1-on-1 mode) — better episode extraction than group_chat
  for video content where there's one "observer" describing events
- group_id = video hash — scopes all memories to this specific video
- create_time = virtual datetime mapped from video timestamp — preserves
  temporal ordering for EverMemOS's temporal reasoning
- sender = "video_{type}" — consistent sender for MemCell extraction
"""
import logging
import time
import hashlib
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import VidEngramConfig
from .utils import ConsolidatedMemory, video_sec_to_datetime, generate_id

logger = logging.getLogger("videngram.memory_writer")


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


class MemoryWriter:
    """Writes consolidated memories to EverMemOS."""

    def __init__(self, config: VidEngramConfig):
        self.cfg = config.evermemos
        self.time_cfg = {
            "base_datetime": config.base_datetime,
            "time_scale_factor": config.time_scale_factor,
        }
        self._session = _create_session()

    def check_health(self) -> bool:
        """Verify EverMemOS is running."""
        try:
            resp = self._session.get(self.cfg.health_url, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def write_memories(
        self,
        memories: list[ConsolidatedMemory],
        video_path: str,
        delay_between: float = 0.3,
        wait_for_indexing: float = 10.0,
    ) -> dict:
        """Write all consolidated memories to EverMemOS.

        Args:
            memories: List of consolidated memories to store
            video_path: Source video path (used for group_id)
            delay_between: Seconds between API calls (avoid overwhelming)
            wait_for_indexing: Seconds to wait after all writes for ES/Milvus indexing

        Returns:
            Stats dict with counts of successful/failed writes
        """
        group_id = self._video_group_id(video_path)
        group_name = Path(video_path).stem

        stats = {"total": len(memories), "success": 0, "failed": 0}
        logger.info(
            f"Writing {len(memories)} memories to EverMemOS "
            f"(group={group_id}, video={group_name})"
        )

        for i, mem in enumerate(memories):
            success = self._write_single(mem, group_id, group_name, i)
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1

            if delay_between > 0 and i < len(memories) - 1:
                time.sleep(delay_between)

        # Wait for EverMemOS indexing (Elasticsearch + Milvus)
        if wait_for_indexing > 0:
            logger.info(f"Waiting {wait_for_indexing}s for indexing to complete...")
            time.sleep(wait_for_indexing)

        logger.info(
            f"Write complete: {stats['success']}/{stats['total']} succeeded, "
            f"{stats['failed']} failed"
        )
        return stats

    def _write_single(
        self,
        memory: ConsolidatedMemory,
        group_id: str,
        group_name: str,
        index: int,
    ) -> bool:
        """Write a single memory to EverMemOS POST /api/v1/memories.

        Maps ConsolidatedMemory fields to EverMemOS's v1 schema:
        - message_id: unique ID per memory
        - create_time: virtual datetime from video timestamp
        - sender: identifies the source type
        - content: the enriched text content
        - group_id: video-scoped group
        - scene: "assistant" for optimal MemCell extraction
        
        Consistent with EverMemOS docs:
        curl -X POST http://localhost:8001/api/v1/memories -d '{...}'
        """
        # Map video seconds to virtual datetime for temporal ordering
        create_time = video_sec_to_datetime(
            memory.start_sec, **self.time_cfg
        )

        payload = {
            "message_id": memory.memory_id,
            "create_time": create_time,
            "sender": f"video_{memory.memory_type}",
            "role": "user",  # Added in EverMemOS v1.2.0 changelog
            "content": memory.content,
            # Video-scoped group for memory isolation
            "group_id": group_id,
            "group_name": group_name,
            "scene": "assistant",
        }

        try:
            resp = self._session.post(
                self.cfg.memorize_url,
                json=payload,
                timeout=30,
            )
            if resp.status_code in (200, 201, 202):
                logger.debug(f"  [{index}] ✓ Wrote {memory.memory_id}")
                return True
            else:
                logger.warning(
                    f"  [{index}] ✗ Failed {memory.memory_id}: "
                    f"{resp.status_code} {resp.text[:200]}"
                )
                return False
        except requests.RequestException as e:
            logger.error(f"  [{index}] ✗ Request error for {memory.memory_id}: {e}")
            return False

    @staticmethod
    def _video_group_id(video_path: str) -> str:
        """Generate a stable group_id from video path.
        All memories for the same video share this group_id."""
        name = Path(video_path).stem
        h = hashlib.md5(video_path.encode()).hexdigest()[:8]
        return f"vid_{name}_{h}"
