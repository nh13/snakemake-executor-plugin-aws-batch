"""Emit the snakesee remote-job-state contract from AWS Batch job status.

snakesee (a terminal UI for monitoring Snakemake workflows) cannot query AWS
Batch itself — it is a passive reader of the workflow's event stream. This module
lets the executor *push* the rich state it already learns from ``describe_jobs``
(queue vs. run, the external job id/ARN, the true execution-window timestamps,
exit code, failure reason) to snakesee, by attaching a structured payload to an
ordinary log record under a well-known key.

The snakesee logger plugin (``snakemake-logger-plugin-snakesee``) recognises that
key and translates the payload into an enriched snakesee event. Executors that
don't care about snakesee simply never call this; snakesee degrades gracefully
when fields are absent. The payload shape is the versioned contract documented in
snakesee's design spec.
"""

from typing import Any, Optional

# Log-record attribute / key the snakesee logger plugin looks for.
WIRE_KEY = "snakesee_remote"

# Wire-contract version this executor emits.
SCHEMA_VERSION = 1

# AWS Batch status string -> normalized snakesee phase.
_STATUS_TO_PHASE = {
    "SUBMITTED": "queued",
    "PENDING": "queued",
    "RUNNABLE": "queued",
    "STARTING": "queued",
    "RUNNING": "running",
    "SUCCEEDED": "succeeded",
    "FAILED": "failed",
}


def phase_for_status(batch_status: Optional[str]) -> Optional[str]:
    """Map an AWS Batch status string to a normalized snakesee phase, or None."""
    if batch_status is None:
        return None
    return _STATUS_TO_PHASE.get(batch_status)


def _epoch_seconds(millis: Any) -> Optional[float]:
    """Convert an AWS Batch millisecond timestamp to epoch seconds, or None."""
    if millis is None:
        return None
    try:
        return float(millis) / 1000.0
    except (TypeError, ValueError):
        return None


def build_payload(
    snakemake_jobid: Optional[int],
    external_jobid: Optional[str],
    job_info: dict,
    region: Optional[str] = None,
    termination: Optional[dict] = None,
) -> Optional[dict]:
    """Build the snakesee remote-state payload from a describe_jobs entry.

    Args:
        snakemake_jobid: Snakemake's internal integer job id (for correlation).
        external_jobid: The AWS Batch job id/ARN.
        job_info: A single entry from ``describe_jobs()["jobs"]``.
        region: AWS region, used by snakesee to build console deep links.
        termination: Optional ``{termination_category, termination_source,
            termination_confidence}`` classification for a failed job.

    Returns:
        The payload dict, or None if the Batch status can't be mapped to a phase,
        or there is no usable integer Snakemake job id to correlate on. (Group
        jobs have a UUID string id rather than an int and are skipped here;
        Snakemake reports group progress through a separate channel.)
    """
    phase = phase_for_status(job_info.get("status"))
    # snakesee correlates on Snakemake's integer job id. bool is an int subclass
    # we explicitly reject; group jobs (UUID strings) are skipped cleanly.
    if (
        phase is None
        or not isinstance(snakemake_jobid, int)
        or isinstance(snakemake_jobid, bool)
    ):
        return None

    container = job_info.get("container") or {}
    attempts = job_info.get("attempts") or []

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "terminal" if phase in ("succeeded", "failed") else "state",
        "jobid": int(snakemake_jobid),
        "executor": "aws-batch",
        "phase": phase,
        "remote_status": job_info.get("status"),
    }

    # Optional fields — include only when present so snakesee degrades cleanly.
    if external_jobid is not None:
        payload["external_jobid"] = external_jobid
    if region is not None:
        payload["region"] = region

    queued_at = _epoch_seconds(job_info.get("createdAt"))
    started_at = _epoch_seconds(job_info.get("startedAt"))
    stopped_at = _epoch_seconds(job_info.get("stoppedAt"))
    if queued_at is not None:
        payload["queued_at"] = queued_at
    if started_at is not None:
        payload["started_at"] = started_at
    if stopped_at is not None:
        payload["stopped_at"] = stopped_at

    job_queue = job_info.get("jobQueue")
    if job_queue is not None:
        payload["queue"] = job_queue
    log_stream = container.get("logStreamName")
    if log_stream is not None:
        payload["log_stream"] = log_stream
    if attempts:
        # AWS populates attempts[] while the job is still running, so this is the
        # attempt count *so far*, not necessarily the final retry count.
        payload["attempt"] = len(attempts)
    exit_code = container.get("exitCode")
    if exit_code is not None:
        payload["exit_code"] = exit_code
    status_reason = job_info.get("statusReason")
    if status_reason is not None:
        payload["status_reason"] = status_reason

    # Merge the termination classification (only its known keys, only when set).
    if termination:
        for key in (
            "termination_category",
            "termination_source",
            "termination_confidence",
        ):
            value = termination.get(key)
            if value is not None:
                payload[key] = value

    return payload


def emit(logger: Any, payload: Optional[dict]) -> None:
    """Attach a remote-state payload to a log record for snakesee to consume.

    No-op when payload is None. Uses INFO level with the payload under the
    well-known ``extra`` key; the message text is informational only.

    Note: this relies on ``logger`` being a stdlib ``logging.Logger`` that
    accepts ``extra=`` (true under Snakemake 9+, where the snakesee logger plugin
    lives). On older Snakemake the call may raise; callers treat emission as a
    best-effort side channel, so a failure degrades to "no snakesee events"
    rather than disrupting job execution.
    """
    if not payload:
        return
    logger.info(
        "snakesee remote job %s -> %s",
        payload.get("external_jobid", payload.get("jobid")),
        payload.get("phase"),
        extra={WIRE_KEY: payload},
    )
