"""Classify why an AWS Batch job died, for the snakesee remote-state contract.

snakesee renders *why* a job terminated but cannot determine it — only the
backend can, and even then with varying certainty. This module classifies a
failed Batch job into a normalized ``{category, source, confidence}`` triple,
using the most authoritative signal available:

1. **High confidence (``aws_instance_state``):** resolve the job's EC2 instance
   (``container.containerInstanceArn`` -> ECS container instance -> instance id)
   and read ``ec2:DescribeInstances`` ``StateReason.Code``. A value of
   ``Server.SpotInstanceTermination`` is AWS's own structured signal that the
   instance was reclaimed as a Spot interruption.
2. **Low confidence (``status_reason``):** fall back to string patterns in the
   job's / container's status reason (Spot, OOM, timeout) when the instance
   lookup isn't available (e.g. the terminated instance's metadata has aged out,
   or the necessary IAM permissions aren't granted).

Everything is best-effort: any AWS error degrades to a lower tier or to no
classification rather than raising, so job monitoring is never disrupted.
"""

from typing import Any, Optional

# Classification value set (mirrors snakesee.remote_termination; the packages
# can't share code, so the contract strings are duplicated by design).
TERM_SPOT = "spot"
TERM_OOM = "oom"
TERM_TIMEOUT = "timeout"

SOURCE_AWS_INSTANCE_STATE = "aws_instance_state"
SOURCE_STATUS_REASON = "status_reason"

CONFIDENCE_HIGH = "high"
CONFIDENCE_LOW = "low"

# AWS's StateReason.Code for an instance reclaimed as a Spot interruption.
SPOT_TERMINATION_CODE = "Server.SpotInstanceTermination"


def classify_termination(
    job_info: dict,
    ec2_client: Any = None,
    ecs_client: Any = None,
) -> Optional[dict]:
    """Classify a failed job's termination into a contract triple, or None.

    Args:
        job_info: A single ``describe_jobs()["jobs"]`` entry for a failed job.
        ec2_client: A boto3 EC2 client (or None to skip the high-confidence tier).
        ecs_client: A boto3 ECS client (or None to skip the high-confidence tier).

    Returns:
        ``{"termination_category", "termination_source", "termination_confidence"}``
        or None when the cause can't be determined.
    """
    # Tier 1: authoritative instance state reason.
    instance_id = _resolve_instance_id(job_info, ecs_client)
    if instance_id and ec2_client is not None:
        if (
            _instance_state_reason_code(instance_id, ec2_client)
            == SPOT_TERMINATION_CODE
        ):
            return _triple(TERM_SPOT, SOURCE_AWS_INSTANCE_STATE, CONFIDENCE_HIGH)

    # Tier 2: string heuristics over the status reason / container reason.
    return _classify_from_reason(job_info)


def _triple(category: str, source: str, confidence: str) -> dict:
    return {
        "termination_category": category,
        "termination_source": source,
        "termination_confidence": confidence,
    }


def _resolve_instance_id(job_info: dict, ecs_client: Any) -> Optional[str]:
    """Resolve the EC2 instance id backing the job, via its ECS container instance."""
    container = job_info.get("container") or {}
    arn = container.get("containerInstanceArn")
    if not arn or ecs_client is None:
        return None
    cluster = _cluster_from_container_instance_arn(arn)
    if cluster is None:
        # Old short-ARN format doesn't embed the cluster, and Batch uses managed
        # clusters (e.g. "AWSBatch-<ce>-<uuid>"), never "default" — so we can't
        # guess it. Degrade to the string tier rather than risk a wrong cluster.
        return None
    try:
        resp = ecs_client.describe_container_instances(
            cluster=cluster, containerInstances=[arn]
        )
        instances = resp.get("containerInstances", [])
        if instances:
            return instances[0].get("ec2InstanceId")
    except Exception:
        return None
    return None


def _cluster_from_container_instance_arn(arn: str) -> Optional[str]:
    """Extract the ECS cluster name from a container-instance ARN.

    New-format ARNs are ``arn:aws:ecs:<region>:<acct>:container-instance/<cluster>/<id>``;
    old short ARNs omit the cluster, in which case None is returned.
    """
    tail = arn.split(":container-instance/")[-1]
    parts = tail.split("/")
    if len(parts) >= 2:
        return parts[0]
    return None


def _instance_state_reason_code(instance_id: str, ec2_client: Any) -> Optional[str]:
    """Return the EC2 instance's StateReason.Code, or None on any failure."""
    try:
        resp = ec2_client.describe_instances(InstanceIds=[instance_id])
        for reservation in resp.get("Reservations", []):
            for instance in reservation.get("Instances", []):
                return (instance.get("StateReason") or {}).get("Code")
    except Exception:
        return None
    return None


def _classify_from_reason(job_info: dict) -> Optional[dict]:
    """Low-confidence classification from the status-reason / container-reason text.

    The status reason and container reason are matched as a single lowercased
    blob (so a phrase can span the two fields). Spot is checked first as it is the
    most specific. The OOM patterns deliberately require an explicit
    out-of-memory phrasing — a bare "memory" mention also appears in scheduling/
    sizing failures (e.g. "insufficient memory available to schedule"), which are
    not OOM kills.
    """
    container = job_info.get("container") or {}
    text = " ".join(
        s for s in (job_info.get("statusReason"), container.get("reason")) if s
    ).lower()
    if not text:
        return None
    if "spot" in text:
        return _triple(TERM_SPOT, SOURCE_STATUS_REASON, CONFIDENCE_LOW)
    if (
        "outofmemory" in text
        or "out of memory" in text
        or "oomkilled" in text
        or "memory usage" in text  # ECS OOM: "Container killed due to memory usage"
    ):
        return _triple(TERM_OOM, SOURCE_STATUS_REASON, CONFIDENCE_LOW)
    if "timeout" in text or "timed out" in text:
        return _triple(TERM_TIMEOUT, SOURCE_STATUS_REASON, CONFIDENCE_LOW)
    return None
