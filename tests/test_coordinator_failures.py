# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failure records the in-job coordinator writes at its existing failure points."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli import do_sweep
from srtctl.cli.do_sweep import SweepOrchestrator
from srtctl.cli.submit import generate_minimal_sbatch_script
from srtctl.contract.failures import FailureCode, SrtFailure
from srtctl.core.failure_log import FailureLog
from srtctl.core.processes import ManagedProcess, ProcessRegistry
from srtctl.core.runtime import Nodes, RuntimeContext
from srtctl.core.schema import (
    BenchmarkConfig,
    HostSetupConfig,
    ModelConfig,
    ResourceConfig,
    RoleConfig,
    SrtConfig,
    TelemetryConfig,
    TelemetryExporterConfig,
)
from srtctl.core.status import JobStage, JobStatus
from srtctl.mock import MockOptions, run_mock_sweep

MOCK_RECIPE = {
    "schema": 2,
    "name": "coordinator-failures",
    "model": {"path": "hf:fake/mock-model", "container": "nvcr.io/fake:latest", "precision": "fp8"},
    "resources": {"gpu_type": "h100", "gpus_per_node": 8},
    "roles": {"agg": {"nodes": 1, "workers": 1}},
    "benchmark": {"type": "custom", "command": "echo fake-benchmark"},
}


def _config(**fields) -> SrtConfig:
    fields.setdefault("benchmark", BenchmarkConfig(type="custom", command="echo load", concurrencies=[1]))
    return SrtConfig(
        name="coordinator-failures",
        model=ModelConfig(path="/models/test", container="test.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="gb200", gpus_per_node=4),
        roles={"prefill": RoleConfig(nodes=1), "decode": RoleConfig(nodes=1)},
        **fields,
    )


def _runtime(tmp_path: Path) -> RuntimeContext:
    return RuntimeContext(
        job_id="12345",
        run_name="test-run",
        nodes=Nodes(head="node0", bench="node0", infra="node0", worker=("node1", "node2")),
        head_node_ip="10.0.0.1",
        infra_node_ip="10.0.0.1",
        log_dir=tmp_path,
        model_path=Path("/models/test"),
        container_image=Path("/img.sqsh"),
        gpus_per_node=4,
        network_interface=None,
        container_mounts={},
        environment={},
    )


def _orchestrator(tmp_path: Path, config: SrtConfig | None = None, registry: ProcessRegistry | None = None):
    failures = FailureLog(tmp_path, slurm_job_id="12345", restart_count=0)
    failures.registry = registry or ProcessRegistry("12345")
    return SweepOrchestrator(config=config or _config(), runtime=_runtime(tmp_path), failures=failures)


def _failed_registry() -> ProcessRegistry:
    registry = ProcessRegistry("12345")
    registry.add_process(ManagedProcess(name="decode_0", popen=SimpleNamespace(pid=1, poll=lambda: 1), log_file=None))
    registry.record_failure("decode_0")
    return registry


def _records(log_dir: Path) -> list[dict]:
    path = log_dir / "failures.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _codes(log_dir: Path) -> list[str]:
    return [record["code"] for record in _records(log_dir)]


def _run_benchmark(orch: SweepOrchestrator, *, healthy: bool = True, script_exit: int = 0, reporter=None) -> int:
    with (
        patch.object(orch, "_wait_for_service_ready", return_value=healthy),
        patch.object(orch, "start_tachometer", return_value=[]),
        patch.object(orch, "_run_benchmark_script", return_value=script_exit),
    ):
        return orch.run_benchmark(orch.failures.registry, threading.Event(), reporter)


def test_health_gate_timeout_records_health_check_timeout(tmp_path):
    orch = _orchestrator(tmp_path)
    reporter = MagicMock()

    assert _run_benchmark(orch, healthy=False, reporter=reporter) == 1

    (record,) = _records(tmp_path)
    assert record["code"] == "health_check_timeout"
    assert "cause" not in record
    reporter.report.assert_called_once_with(JobStatus.FAILED, JobStage.BENCHMARK, "Workers failed health check")


def test_worker_death_during_health_wait_records_worker_failed(tmp_path):
    orch = _orchestrator(tmp_path, registry=_failed_registry())

    assert _run_benchmark(orch, healthy=False) == 1

    assert _codes(tmp_path) == ["worker_failed"]


def test_benchmark_nonzero_records_benchmark_failed(tmp_path):
    orch = _orchestrator(tmp_path)

    assert _run_benchmark(orch, script_exit=7) == 7

    (record,) = _records(tmp_path)
    assert record["code"] == "benchmark_failed"
    assert record["cause"] == {"producer": "benchmark", "return_code": 7}


def test_benchmark_stopped_by_worker_death_records_worker_failed(tmp_path):
    orch = _orchestrator(tmp_path, registry=_failed_registry())

    assert _run_benchmark(orch, script_exit=1) == 1

    assert _codes(tmp_path) == ["worker_failed"]


def test_terminal_service_failure_after_worker_death_records_worker_failed(tmp_path):
    orch = _orchestrator(tmp_path, _config(benchmark=BenchmarkConfig(type="manual")), registry=_failed_registry())
    orch.terminal_processes["client"] = [SimpleNamespace(name="client", is_running=False, exit_code=2)]

    with patch.object(orch, "_public_api_node", return_value="node0"):
        assert _run_benchmark(orch) == 2

    assert _codes(tmp_path) == ["worker_failed"]


def test_eval_only_health_timeout_records_health_check_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ONLY", "true")
    orch = _orchestrator(tmp_path)

    with (
        patch.object(orch, "_public_api_node", return_value="node0"),
        patch.object(orch, "_wait_for_service_ready", return_value=False),
    ):
        assert orch._run_post_eval(threading.Event()) == 1

    assert _codes(tmp_path) == ["health_check_timeout"]


def _srun_exits(returncode: int):
    proc = MagicMock()
    proc.wait.return_value = returncode
    proc.returncode = returncode
    proc.poll.return_value = None
    return patch("srtctl.cli.do_sweep.start_srun_process", return_value=proc)


def _host_setup(tmp_path):
    orch = _orchestrator(tmp_path, _config(host_setup=HostSetupConfig(commands=["true"])))
    with _srun_exits(1):
        orch._run_host_setup()


def _clock_sync(tmp_path):
    exporter = TelemetryExporterConfig(container_image="dcgm-exporter", port=9401)
    telemetry = TelemetryConfig(enabled=True, dcgm_exporter=exporter, required=True)
    with _srun_exits(1):
        _orchestrator(tmp_path, _config(telemetry=telemetry))._check_clock_sync()


def _sequential_start(tmp_path):
    with patch("srtctl.cli.mixins.worker_stage.wait_for_health", return_value=False):
        _orchestrator(tmp_path)._wait_for_worker_ready(SimpleNamespace(node="node1", sys_port=8081))


@pytest.mark.parametrize(
    ("site", "match", "code"),
    [
        (_host_setup, "host_setup failed on", FailureCode.INFRA_SERVICE_FAILED),
        (_clock_sync, "clock_sync_check: node0", FailureCode.TELEMETRY_REQUIRED_FAILED),
        (_sequential_start, "did not become healthy", FailureCode.HEALTH_CHECK_TIMEOUT),
    ],
    ids=["host_setup", "clock_sync", "sequential_start"],
)
def test_raise_site_codes(tmp_path, site, match, code):
    with pytest.raises(RuntimeError, match=match) as raised:
        site(tmp_path)

    assert isinstance(raised.value, SrtFailure)
    assert raised.value.code is code


def _mock_sweep(tmp_path: Path, recipe_fields: dict | None = None) -> tuple[int, Path]:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump({**MOCK_RECIPE, **(recipe_fields or {})}))
    output_dir = tmp_path / "outputs" / "4242"
    exit_code = run_mock_sweep(
        config_path=recipe,
        output_dir=output_dir,
        job_id="4242",
        options=MockOptions(child_duration_s=0.05, phase_pause_s=0.01),
    )
    return exit_code, output_dir / "logs"


def test_finalizer_flip_records_telemetry_required_failed(tmp_path):
    with patch.object(SweepOrchestrator, "finalize_power_telemetry", return_value=1):
        exit_code, log_dir = _mock_sweep(tmp_path)

    assert exit_code == 1
    assert _codes(log_dir) == ["telemetry_required_failed"]


def test_required_power_startup_records_telemetry_required_failed(tmp_path):
    with patch.object(SweepOrchestrator, "power_telemetry_blocks_benchmark", return_value=True):
        exit_code, log_dir = _mock_sweep(tmp_path)

    assert exit_code == 1
    assert _codes(log_dir) == ["telemetry_required_failed"]


def test_unclassified_nonzero_exit_records_unknown(tmp_path):
    with patch.object(SweepOrchestrator, "run_benchmark", return_value=3):
        exit_code, log_dir = _mock_sweep(tmp_path)

    assert exit_code == 3
    (record,) = _records(log_dir)
    assert (record["code"], record["error"]) == ("unknown", "Sweep failed with exit code 3")


def test_successful_run_writes_nothing(tmp_path):
    exit_code, log_dir = _mock_sweep(tmp_path)

    assert exit_code == 0
    assert not (log_dir / "failures.jsonl").exists()


def test_raised_failure_in_run_reaches_the_file(tmp_path):
    with patch.object(SweepOrchestrator, "_run_host_commands", return_value=["node0"]):
        exit_code, log_dir = _mock_sweep(tmp_path, {"host_setup": {"commands": ["true"]}})

    assert exit_code == 1
    (record,) = _records(log_dir)
    assert (record["code"], record["cause"]) == ("infra_service_failed", {"producer": "host_setup"})


def test_run_eval_failure_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_EVAL", "true")
    real_post_eval = SweepOrchestrator._run_post_eval

    def eval_server_down(self, stop_event):
        with patch("srtctl.cli.do_sweep.wait_for_port", return_value=False):
            assert real_post_eval(self, stop_event) == 1
        return 1

    with patch.object(SweepOrchestrator, "_run_post_eval", eval_server_down):
        exit_code, log_dir = _mock_sweep(tmp_path)

    assert exit_code == 0
    assert not (log_dir / "failures.jsonl").exists()


def test_signal_records_terminated_once_with_subcode(tmp_path, restore_signal_handlers):
    record_signal = SweepOrchestrator._record_signal

    def record_then_signal_again(self, sig):
        record_signal(self, sig)
        signal.raise_signal(signal.SIGINT)  # a forwarded copy arriving mid-shutdown

    with (
        patch.object(SweepOrchestrator, "_print_connection_info", lambda self: signal.raise_signal(signal.SIGTERM)),
        patch.object(SweepOrchestrator, "_record_signal", record_then_signal_again),
        pytest.raises(SystemExit),
    ):
        _mock_sweep(tmp_path)

    (record,) = _records(tmp_path / "outputs" / "4242" / "logs")
    assert (record["code"], record["subcode"], record["error"]) == ("terminated", "SIGTERM", "Received SIGTERM")
    assert "cause" not in record


def test_signal_during_eval_is_not_followed_by_a_telemetry_record(tmp_path, monkeypatch, restore_signal_handlers):
    monkeypatch.setenv("RUN_EVAL", "true")
    with (
        patch.object(SweepOrchestrator, "_run_post_eval", lambda self, stop: signal.raise_signal(signal.SIGTERM)),
        patch.object(SweepOrchestrator, "finalize_power_telemetry", return_value=1),
        pytest.raises(SystemExit),
    ):
        _mock_sweep(tmp_path)

    assert _codes(tmp_path / "outputs" / "4242" / "logs") == ["terminated"]


def _main(tmp_path: Path, monkeypatch, config_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "failures.jsonl").write_text("previous attempt\n")
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    monkeypatch.setenv("SRTCTL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["do_sweep", str(config_path)])
    with patch.object(do_sweep, "setup_logging"), pytest.raises(SystemExit) as exited:
        do_sweep.main()
    assert exited.value.code == 1
    assert (log_dir / "failures.1.jsonl").read_text() == "previous attempt\n"
    return log_dir


def test_early_exit_missing_config_rotates_then_records(tmp_path, monkeypatch):
    log_dir = _main(tmp_path, monkeypatch, tmp_path / "missing.yaml")

    (record,) = _records(log_dir)
    assert (record["code"], record["slurm_job_id"]) == ("job_preparation_failed", "777")


@pytest.mark.parametrize(
    ("recipe_text", "code"),
    [("schema: 2\nname: incomplete\n", "recipe_invalid"), ("name: pre-2.0\n", "recipe_schema_unsupported")],
)
def test_early_exit_load_failure_keeps_its_code(tmp_path, monkeypatch, recipe_text, code):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(recipe_text)

    log_dir = _main(tmp_path, monkeypatch, recipe)

    assert _codes(log_dir) == [code]


def test_failure_before_run_records_unknown(tmp_path, monkeypatch):
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(yaml.dump(MOCK_RECIPE))

    with patch.object(do_sweep.RuntimeContext, "from_config", side_effect=ValueError("nodelist has 0 nodes")):
        log_dir = _main(tmp_path, monkeypatch, recipe)

    (record,) = _records(log_dir)
    assert (record["code"], record["error"]) == ("unknown", "nodelist has 0 nodes")


_STUB_COORDINATOR = """\
import pathlib, signal, sys, time
marker = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, lambda *_: (marker.write_text("SIGTERM"), sys.exit(7)))
marker.write_text("started")
time.sleep(30)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_batch_script_forwards_sigterm_to_the_coordinator(tmp_path):
    """scancel signals only the batch shell; the coordinator must still get SIGTERM to record it."""
    marker = tmp_path / "marker"
    (tmp_path / "coordinator.py").write_text(_STUB_COORDINATOR)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {
        "uv": f'[ "$1" = run ] && exec {sys.executable} {tmp_path}/coordinator.py {marker}\necho uv-stub',
        "flock": "true",
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    config = SrtConfig(
        name="test",
        model=ModelConfig(path="/model", container="/container.sqsh", precision="fp8"),
        resources=ResourceConfig(gpu_type="h100", gpus_per_node=8),
        roles={"agg": RoleConfig(nodes=1)},
    )
    with patch(
        "srtctl.cli.submit.get_srtslurm_setting", lambda key, *_: str(tmp_path) if key == "srtctl_root" else None
    ):
        script = generate_minimal_sbatch_script(config, tmp_path / "cfg.yaml", output_dir=tmp_path / "out")
    (tmp_path / "job.sh").write_text(script)

    job = subprocess.Popen(
        ["bash", str(tmp_path / "job.sh")],
        env={**os.environ, "SLURM_JOB_ID": "1"},
        stdout=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while not (marker.exists() and marker.read_text()) and time.monotonic() < deadline:
        time.sleep(0.05)
    job.send_signal(signal.SIGTERM)
    out, _ = job.communicate(timeout=10)

    assert marker.read_text() == "SIGTERM"
    assert job.returncode == 7
    assert "Sweep failed (exit code: 7)" in out
