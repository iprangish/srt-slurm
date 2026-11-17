# Failure Records

When a job fails, the in-job coordinator (`do_sweep`) writes one structured record per failure, using the same schema and codes as the records `srtctl apply --json` prints at submission time (see [Failure records](cli.md#failure-records) in the CLI guide). Use them to learn why a job failed without reading the logs.

## Where the records go

| Destination | Default | Content |
| --- | --- | --- |
| `<job_output_dir>/logs/failures.jsonl` | on | One JSON object per line. Created on the first failure; a successful run leaves no file. |
| Sweep log | always | A human line `Error [code]: message`, or `Error [code / subcode]: message`, at ERROR level. |
| Sweep log, `SRT_FAILURE ` prefix | off | The same JSON record on a line of its own, for log scrapers. |

Writing never raises: if the file cannot be written, a warning is logged and the job continues.

Each record has `schema_version`, `status` (`error`), `code`, optional `subcode`, `error`, and optional `cause` and `issues`, as described in the CLI guide. The coordinator identifies the job attempt with `slurm_job_id` and `restart_count` (`SLURM_RESTART_COUNT`) instead of `output_dir`. Values that were not observed are omitted.

## Ordering

Records are appended in the order the failures were observed, so the first record is the primary cause. Later records are consequences, such as a worker dying after the first failure or the shutdown signal that ended the job.

## Coordinator codes

| Code | Written when |
| --- | --- |
| `job_preparation_failed` | The recipe file passed to the coordinator does not exist. |
| `recipe_invalid` | The recipe fails to load or validate inside the job. |
| `infra_service_failed` | `host_setup` commands fail on a node (`cause.producer: host_setup`), or model staging fails (`cause.producer: stage_model`). |
| `telemetry_required_failed` | `telemetry.required` is set and clock-sync, power telemetry or its finalization fails. |
| `worker_failed` | A managed process (worker, frontend or service) exits abnormally. One record per process; `cause.producer` is the process name and `cause.return_code` its exit code. |
| `health_check_timeout` | The server did not become healthy, unless a process failure already explains it. |
| `benchmark_failed` | The benchmark or eval-only run exited non-zero (`cause.producer` names it, with `cause.return_code`), unless a process failure explains it. |
| `terminated` | The coordinator received a shutdown signal; `subcode` is `SIGTERM` or `SIGINT`. Any process failure that preceded the signal is recorded first. |
| `unknown` | An unclassified exception, or a non-zero exit with no earlier record. |

A failure explained by an already-written record is not written twice.

## Requeued jobs

A requeued job starts with an existing `failures.jsonl`. The coordinator first renames it to the next free `failures.<N>.jsonl`, oldest = 1, so each attempt has its own file and `restart_count` in the records tells them apart.

## Configuration

```yaml
reporting:
  failures:
    file: true     # write logs/failures.jsonl (default true)
    stdout: false  # also print SRT_FAILURE lines to the sweep log (default false)
```

A `reporting` block in `srtslurm.yaml` is applied only when the recipe has no `reporting` key at all. A recipe that sets any `reporting` key replaces the cluster block entirely, so repeat `reporting.failures` there if you want non-default values. See the [generated schema reference](schema-reference.md) for the fields.

## Combining with Slurm accounting

The record explains the cause; Slurm reports the outcome.

1. Take the job state from `sacct` (under `launcher: docker`, from the exit code of `srtctl apply`).
2. If the job failed, read `logs/failures.jsonl` for the cause; the first line is the primary one.
3. A failed job with no file is unclassified, not successful. It may be one of the gaps below.

## Known gaps

- A coordinator that is killed without a chance to run cleanup (SIGKILL, node loss, out-of-memory kill) leaves no record.
- Failures in the batch script before the Python coordinator starts, such as a missing Python environment, leave no record.
