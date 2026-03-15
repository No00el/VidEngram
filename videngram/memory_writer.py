"""
VidEngram Memory Writer
Adapts consolidated memories for EverMemOS ingestion via POST /api/v1/memories.

Key design decisions:
- scene="assistant" (1-on-1 mode) — better episode extraction than group_chat
  for video content where there's one "observer" describing events
- group_id = video hash — scopes all memories to this specific video
- start_sec stored in metadata — used by MemoryReader for timestamp rescue
- sender = "video_{type}" — consistent sender for MemCell extraction
"""
import logging
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import VidEngramConfig
from .utils import ConsolidatedMemory, generate_id

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
        self._session = _create_session()       # used by check_health (single thread)
        self._thread_local = threading.local()  # per-worker sessions for concurrent writes

    def _get_session(self) -> requests.Session:
        """Return a per-thread Session so concurrent workers don't share state."""
        if not hasattr(self._thread_local, "session"):
            self._thread_local.session = _create_session()
        return self._thread_local.session

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
        wait_for_indexing: float = 15.0,
    ) -> dict:
        """Write all consolidated memories to EverMemOS concurrently.

        Uses 3 concurrent workers to eliminate the old per-request delay while
        keeping concurrency bounded enough to avoid EverMemOS clustering race
        conditions on the same group_id.

        Temporal ordering is preserved via each memory's create_time field
        (virtual datetime mapped from video timestamp), not by insertion order.

        Args:
            memories: List of consolidated memories to store
            video_path: Source video path (used for group_id)
            wait_for_indexing: Seconds to wait after all writes for ES/Milvus indexing

        Returns:
            Stats dict with counts of successful/failed writes
        """
        group_id = self._video_group_id(video_path)
        group_name = Path(video_path).stem

        stats = {"total": len(memories), "success": 0, "failed": 0}
        logger.info(
            f"Writing {len(memories)} memories to EverMemOS "
            f"(group={group_id}, video={group_name}, workers=3)"
        )

        with ThreadPoolExecutor(max_workers=3) as pool:
            future_to_idx = {
                pool.submit(self._write_single, mem, group_id, group_name, i): i
                for i, mem in enumerate(memories)
            }
            for future in as_completed(future_to_idx):
                if future.result():
                    stats["success"] += 1
                else:
                    stats["failed"] += 1

        # Wait for EverMemOS background indexing (Elasticsearch + Milvus).
        # Increased from 10s to 15s because all memories are submitted nearly
        # simultaneously, giving EverMemOS more concurrent background tasks.
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
        # Map video timestamp to a virtual ISO 8601 datetime for temporal ordering.
        # Base epoch (2026-01-01 UTC) + start_sec gives EverMemOS a meaningful create_time.
        base_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        create_time = (base_dt + timedelta(seconds=memory.start_sec)).isoformat()

        payload = {
            "message_id": memory.memory_id,
            "user_id": "videngram",
            "sender": f"video_{memory.memory_type}",
            "role": "user",  # Added in EverMemOS v1.2.0 changelog
            "content": memory.content,
            "create_time": create_time,
            # Video-scoped group for memory isolation
            "group_id": group_id,
            "group_name": group_name,
            "scene": "assistant",
            # start_sec stored for timestamp rescue in MemoryReader
            "metadata": {"start_sec": memory.start_sec, "end_sec": memory.end_sec},
        }

        try:
            resp = self._get_session().post(
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
