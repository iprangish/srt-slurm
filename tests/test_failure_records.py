# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the failure record contract."""

from srtctl.contract.failures import Cause, FailureCode, SrtFailure, as_failure
from srtctl.version import version_info


def test_record_has_fixed_fields_and_omits_unobserved():
    failure = SrtFailure(
        FailureCode.HEALTH_CHECK_TIMEOUT, "Server did not become healthy", cause=Cause(producer="frontend")
    )
    failure.identity = {"slurm_job_id": "42", "restart_count": 0, "variant": None}

    assert failure.record() == {
        "schema_version": 1,
        "status": "error",
        "slurm_job_id": "42",
        "restart_count": 0,
        "code": "health_check_timeout",
        "error": "Server did not become healthy",
        "cause": {"producer": "frontend"},
    }


def test_human_line_shows_code_subcode_and_accepted_job():
    failure = SrtFailure(FailureCode.SBATCH_REJECTED, "Slurm rejected the job", subcode="ESLURM_INVALID_TIME_LIMIT")
    assert failure.human_line() == "Error [sbatch_rejected / ESLURM_INVALID_TIME_LIMIT]: Slurm rejected the job"

    failure.identity = {"slurm_job_id": "42", "output_dir": "/o/42"}
    assert failure.human_line() == (
        "Error [sbatch_rejected / ESLURM_INVALID_TIME_LIMIT]: Slurm rejected the job (job 42 accepted, output /o/42)"
    )


def test_as_failure_wraps_unknown_and_passes_failures_through():
    failure = SrtFailure(FailureCode.WORKER_FAILED, "boom")
    assert as_failure(failure) is failure

    wrapped = as_failure(ValueError("bad"))
    assert wrapped.code is FailureCode.UNKNOWN
    assert str(wrapped) == "bad"
    assert as_failure(KeyError()).args[0] == "KeyError"


def test_version_info_reports_record_schema_version():
    assert version_info().record_schema_version == 1
    assert "records v1" in str(version_info())


def test_human_line_without_output_dir_has_no_job_suffix():
    failure = SrtFailure(FailureCode.BENCHMARK_FAILED, "boom")
    failure.identity = {"slurm_job_id": "42", "restart_count": 0}
    assert failure.human_line() == "Error [benchmark_failed]: boom"
