from __future__ import annotations

from enum import Enum
from typing import Protocol


class JobStatus(str, Enum):
    """PENDING = images saved; call POST /jobs/{id}/start to run the pipeline."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobRepository(Protocol):
    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        ...


class WebSocketNotifier(Protocol):
    def notify_job_update(
        self,
        job_id: str,
        status: JobStatus,
        *,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        ...


class NoopJobRepository:
    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        print(f"[DB] {job_id} => {status} model={model_url} fmt={model_format} error={error}")


class NoopWebSocketNotifier:
    def notify_job_update(
        self,
        job_id: str,
        status: JobStatus,
        *,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        print(f"[WS] {job_id} => {status} model={model_url} fmt={model_format} error={error}")

