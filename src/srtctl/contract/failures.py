# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure codes and the structured failure record shared by ``srtctl apply`` and the in-job coordinator."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

RECORD_SCHEMA_VERSION = 1


class FailureCode(str, Enum):
    """Stable top-level failure categories."""

    INVOCATION_INVALID = "invocation_invalid"
    RECIPE_INVALID = "recipe_invalid"
    RECIPE_SCHEMA_UNSUPPORTED = "recipe_schema_unsupported"
    CLUSTER_CONFIG_INVALID = "cluster_config_invalid"
    SETUP_INCOMPLETE = "setup_incomplete"
    PREFLIGHT_FAILED = "preflight_failed"
    SUBMISSION_FAILED = "submission_failed"
    SBATCH_REJECTED = "sbatch_rejected"
    JOB_PREPARATION_FAILED = "job_preparation_failed"
    METADATA_WRITE_FAILED = "metadata_write_failed"
    SBATCH_ARCHIVE_FAILED = "sbatch_archive_failed"
    INFRA_SERVICE_FAILED = "infra_service_failed"
    WORKER_FAILED = "worker_failed"
    HEALTH_CHECK_TIMEOUT = "health_check_timeout"
    BENCHMARK_FAILED = "benchmark_failed"
    TELEMETRY_REQUIRED_FAILED = "telemetry_required_failed"
    TERMINATED = "terminated"
    UNKNOWN = "unknown"


class Cause(BaseModel):
    """The underlying failure a record wraps, with only the fields that were observed."""

    model_config = ConfigDict(frozen=True)

    producer: str | None = None
    code: str | None = None
    return_code: int | None = None
    message: str | None = None


class SrtFailure(RuntimeError):
    """A failure with a stable code, serializable as one failure record."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        subcode: str | None = None,
        cause: Cause | None = None,
        issues: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.subcode = subcode
        self.cause = cause
        self.issues = issues
        self.identity: dict[str, Any] = {}
        self.reported = False

    def record(self) -> dict[str, Any]:
        """The failure as a JSON-ready dict; unobserved values are omitted."""
        record: dict[str, Any] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "status": "error",
            **{key: value for key, value in self.identity.items() if value is not None},
            "code": self.code.value,
        }
        if self.subcode is not None:
            record["subcode"] = self.subcode
        record["error"] = self.message
        if self.cause is not None:
            record["cause"] = self.cause.model_dump(exclude_none=True)
        if self.issues is not None:
            record["issues"] = self.issues
        return record

    def human_line(self) -> str:
        """One-line summary for the terminal."""
        label = self.code.value if self.subcode is None else f"{self.code.value} / {self.subcode}"
        line = f"Error [{label}]: {self.message}"
        if "slurm_job_id" in self.identity:
            line += f" (job {self.identity['slurm_job_id']} accepted, output {self.identity.get('output_dir')})"
        return line


def as_failure(exc: BaseException) -> SrtFailure:
    """Return ``exc`` if it is an ``SrtFailure``, else wrap it as ``UNKNOWN``."""
    if isinstance(exc, SrtFailure):
        return exc
    return SrtFailure(FailureCode.UNKNOWN, str(exc) or type(exc).__name__)
