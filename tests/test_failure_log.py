# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the coordinator failure log and its process-registry inputs."""

import json
import logging
import signal
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from srtctl.contract.failures import FailureCode, SrtFailure
from srtctl.core.config import resolve_config_with_defaults
from srtctl.core.failure_log import FailureLog
from srtctl.core.processes import ManagedProcess, ProcessRegistry, setup_signal_handlers
from srtctl.core.schema import ReportingConfig


def _failure() -> SrtFailure:
    return SrtFailure(FailureCode.BENCHMARK_FAILED, "benchmark exited 1")


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_write_appends_record_and_human_line(tmp_path, caplog):
    log = FailureLog(tmp_path, slurm_job_id="42", restart_count=1)
    failure = _failure()
    with caplog.at_level(logging.ERROR, logger="srtctl.core.failure_log"):
        log.write(failure)

    assert "Error [benchmark_failed]" in caplog.text
    assert "accepted" not in caplog.text
    assert _lines(tmp_path / "failures.jsonl") == [failure.record()]
    assert failure.record()["slurm_job_id"] == "42"
    assert failure.record()["restart_count"] == 1
    assert log.written


def test_file_created_lazily_and_stdout_copy_is_opt_in(tmp_path, capsys):
    log = FailureLog(tmp_path, slurm_job_id="42", restart_count=0)
    assert not (tmp_path / "failures.jsonl").exists()
    assert not log.written
    log.write(_failure())
    assert (tmp_path / "failures.jsonl").exists()
    assert "SRT_FAILURE {" not in capsys.readouterr().out

    out_dir = tmp_path / "out"
    FailureLog(out_dir, slurm_job_id="42", restart_count=0, stdout=True).write(_failure())
    assert "SRT_FAILURE {" in capsys.readouterr().out

    no_file = tmp_path / "nofile"
    FailureLog(no_file, slurm_job_id="42", restart_count=0, file=False).write(_failure())
    assert not no_file.exists()


def test_disabled_writes_nothing(tmp_path, capsys):
    FailureLog.disabled().write(_failure())
    assert capsys.readouterr().out == ""


def test_rotate_renames_previous_attempt(tmp_path):
    (tmp_path / "failures.jsonl").write_text("new\n")
    (tmp_path / "failures.1.jsonl").write_text("old\n")
    FailureLog(tmp_path, slurm_job_id="42", restart_count=2).rotate()

    assert not (tmp_path / "failures.jsonl").exists()
    assert (tmp_path / "failures.2.jsonl").read_text() == "new\n"
    assert (tmp_path / "failures.1.jsonl").read_text() == "old\n"


def _registry(tmp_path: Path) -> ProcessRegistry:
    registry = ProcessRegistry("1")
    for name in ("decode_0", "prefill_0"):
        registry.add_process(ManagedProcess(name=name, popen=SimpleNamespace(pid=1, poll=lambda: 1), log_file=None))
    return registry


def test_process_failures_written_once_in_detection_order(tmp_path):
    registry = _registry(tmp_path)
    registry.record_failure("decode_0", "Rank0 Task exit code: 1")
    registry.record_failure("prefill_0")
    log = FailureLog(tmp_path, slurm_job_id="42", restart_count=0)
    log.registry = registry
    timeout = SrtFailure(FailureCode.HEALTH_CHECK_TIMEOUT, "timed out")

    log.write_unless_process_failed(timeout)
    records = _lines(tmp_path / "failures.jsonl")
    assert [(r["code"], r["cause"]["producer"]) for r in records] == [
        ("worker_failed", "decode_0"),
        ("worker_failed", "prefill_0"),
    ]
    assert records[0]["cause"]["message"] == "Rank0 Task exit code: 1"

    log.write_unless_process_failed(timeout)
    assert len(_lines(tmp_path / "failures.jsonl")) == 2


def test_write_unless_process_failed_writes_failure_without_process_failures(tmp_path):
    log = FailureLog(tmp_path, slurm_job_id="42", restart_count=0)
    log.registry = ProcessRegistry("1")
    log.write_unless_process_failed(_failure())
    assert [r["code"] for r in _lines(tmp_path / "failures.jsonl")] == ["benchmark_failed"]


def test_registry_failures_detail_prefers_fatal_line_then_tail(tmp_path):
    registry = ProcessRegistry("1")
    log_file = tmp_path / "w.log"
    log_file.write_text("\n".join(f"line {i}" for i in range(5)))
    for name, path in (("a", log_file), ("b", log_file), ("c", None)):
        registry.add_process(ManagedProcess(name=name, popen=SimpleNamespace(pid=1, poll=lambda: 3), log_file=path))
    registry.record_failure("a", "fatal marker")
    registry.record_failure("b")
    registry.record_failure("c")

    assert registry.failures(tail_lines=2) == [
        ("a", 3, "fatal marker"),
        ("b", 3, "line 3\nline 4"),
        ("c", 3, ""),
    ]


@pytest.fixture
def restore_signal_handlers():
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_signal_hook_runs_once_before_cleanup(restore_signal_handlers):
    order: list[str] = []
    registry = ProcessRegistry("1")
    registry.cleanup = lambda: order.append("cleanup")  # type: ignore[method-assign]
    setup_signal_handlers(threading.Event(), registry, on_signal=lambda name: order.append(f"hook {name}"))

    for _ in range(2):
        with pytest.raises(SystemExit):
            signal.raise_signal(signal.SIGTERM)

    assert order == ["hook SIGTERM", "cleanup", "cleanup"]


def test_from_env_without_job_id_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SRTCTL_JOB_ID", raising=False)
    assert FailureLog.from_env(tmp_path / "recipe.yaml") is None


def test_from_env_resolves_log_dir_and_restart_count(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_RESTART_COUNT", "3")
    monkeypatch.delenv("SRTCTL_OUTPUT_DIR", raising=False)
    log = FailureLog.from_env(tmp_path / "recipe.yaml")
    assert log is not None
    log.write(_failure())
    record = _lines(tmp_path / "logs" / "failures.jsonl")[0]
    assert (record["slurm_job_id"], record["restart_count"]) == ("42", 3)

    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(tmp_path / "out"))
    FailureLog.from_env(tmp_path / "recipe.yaml").write(_failure())  # type: ignore[union-attr]
    assert (tmp_path / "out" / "logs" / "failures.jsonl").exists()


def _reporting(recipe: dict, cluster: dict):
    resolved = resolve_config_with_defaults({"schema": 2, "name": "t", **recipe}, cluster)
    return ReportingConfig.Schema().load(resolved["reporting"]) if "reporting" in resolved else None


def test_reporting_failures_defaults_and_cluster_block_rule():
    cluster = {"reporting": {"failures": {"stdout": True}}}

    inherited = _reporting({}, cluster)
    assert inherited is not None and inherited.failures is not None
    assert (inherited.failures.file, inherited.failures.stdout) == (True, True)

    own = _reporting({"reporting": {"status": {"endpoint": "http://x"}}}, cluster)
    assert own is not None and own.failures is None
    assert ReportingConfig.Schema().load({"failures": {}}).failures.file is True
    assert ReportingConfig.Schema().load({"failures": {}}).failures.stdout is False
