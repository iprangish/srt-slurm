# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator-side failure records: ``logs/failures.jsonl``, the sweep-log human line and the optional stdout copy."""

import json
import logging
import os
from pathlib import Path

from srtctl.contract.failures import Cause, FailureCode, SrtFailure
from srtctl.core.launcher import get_launcher
from srtctl.core.processes import ProcessRegistry
from srtctl.core.schema import ReportingConfig, ReportingFailuresConfig

logger = logging.getLogger(__name__)

FAILURES_FILE = "failures.jsonl"


class FailureLog:
    """Writes failure records for one job attempt."""

    def __init__(
        self, log_dir: Path, *, slurm_job_id: str, restart_count: int, file: bool = True, stdout: bool = False
    ) -> None:
        self._path = log_dir / FAILURES_FILE
        self._identity = {"slurm_job_id": slurm_job_id, "restart_count": restart_count}
        self._file = file
        self._stdout = stdout
        self._written_processes: set[str] = set()
        self._enabled = True
        self.written = False
        self.registry: ProcessRegistry | None = None

    @classmethod
    def disabled(cls) -> "FailureLog":
        """A log that writes nothing."""
        log = cls(Path(), slurm_job_id="", restart_count=0, file=False)
        log._enabled = False
        return log

    @classmethod
    def from_env(cls, config_path: Path) -> "FailureLog | None":
        """The log for the running job, or None outside a job."""
        job_id = get_launcher().job_id()
        if job_id is None:
            return None
        output_dir = os.environ.get("SRTCTL_OUTPUT_DIR")
        log_dir = (Path(output_dir) if output_dir else config_path.parent) / "logs"
        return cls(log_dir, slurm_job_id=job_id, restart_count=int(os.environ.get("SLURM_RESTART_COUNT", "0")))

    def configure(self, reporting: ReportingConfig | None) -> None:
        """Apply the recipe's ``reporting.failures`` toggles; defaults when unset."""
        toggles = (reporting.failures if reporting else None) or ReportingFailuresConfig()
        self._file, self._stdout = toggles.file, toggles.stdout

    def rotate(self) -> None:
        """Move a previous attempt's file to the next free ``failures.<N>.jsonl``."""
        if not self._enabled or not self._path.exists():
            return
        n = 1
        while (rotated := self._path.with_name(f"failures.{n}.jsonl")).exists():
            n += 1
        try:
            self._path.rename(rotated)
        except OSError as e:
            logger.warning("Could not rotate %s: %s", self._path, e)

    def write(self, failure: SrtFailure) -> None:
        """Log the human line and append the record; never raises."""
        if not self._enabled:
            return
        failure.identity = dict(self._identity)
        logger.error(failure.human_line())
        self.written = True
        line = json.dumps(failure.record())
        if self._file:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.warning("Could not write failure record to %s: %s", self._path, e)
        if self._stdout:
            print(f"SRT_FAILURE {line}", flush=True)

    def flush_process_failures(self) -> bool:
        """Write a record per newly failed process; return whether any process failed."""
        if not self._enabled or self.registry is None:
            return False
        for name, exit_code, detail in self.registry.failures(skip=self._written_processes):
            self._written_processes.add(name)
            cause = Cause(producer=name, return_code=exit_code, message=detail or None)
            self.write(SrtFailure(FailureCode.WORKER_FAILED, f"Process {name} failed", cause=cause))
        return bool(self._written_processes)

    def write_unless_process_failed(self, failure: SrtFailure) -> None:
        """Write ``failure`` only when no process failure explains it."""
        if not self.flush_process_failures():
            self.write(failure)
