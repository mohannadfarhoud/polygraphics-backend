"""Broadcast remote-worker WebSocket notifications when a job becomes ``QUEUED``.

Thread-safe: ``schedule_worker_job_notice`` may be called from JobManager (sync, any thread).
"""

from __future__ import annotations

import asyncio
import json
import logging
from starlette.websockets import WebSocket

logger = logging.getLogger(__name__)

_hub: WorkerHub | None = None


def init_hub(hub: WorkerHub) -> None:
    global _hub
    _hub = hub


def schedule_worker_job_notice(job_id: str) -> None:
    if _hub is None:
        return
    _hub.schedule_notify_job(job_id)


class WorkerHub:
    """Connected GPU workers (each tab/process is one WebSocket)."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def register(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def unregister(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast_job_assigned(self, job_id: str) -> None:
        msg = json.dumps({"type": "job_assigned", "job_id": job_id})
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(msg)
            except Exception as exc:
                logger.debug("worker ws send failed: %s", exc)
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)

    def schedule_notify_job(self, job_id: str) -> None:
        loop = self._loop
        if loop is None:
            return

        async def _run() -> None:
            await self.broadcast_job_assigned(job_id)

        try:
            asyncio.run_coroutine_threadsafe(_run(), loop)
        except Exception as exc:
            logger.debug("schedule worker ws notify failed: %s", exc)
