"""Unit tests for the job-termination classifier (mocked boto, no AWS creds)."""

from unittest.mock import MagicMock

from snakemake_executor_plugin_aws_batch import termination

CI_ARN = "arn:aws:ecs:us-east-1:123456789012:container-instance/my-cluster/abc123"


def _ecs_returning(instance_id):
    ecs = MagicMock()
    ecs.describe_container_instances.return_value = {
        "containerInstances": [{"ec2InstanceId": instance_id}]
    }
    return ecs


def _ec2_with_state_code(code):
    ec2 = MagicMock()
    ec2.describe_instances.return_value = {
        "Reservations": [{"Instances": [{"StateReason": {"Code": code}}]}]
    }
    return ec2


class TestHighConfidenceSpot:
    def test_instance_state_spot_termination_is_high_confidence(self):
        job_info = {
            "container": {"containerInstanceArn": CI_ARN},
            "statusReason": "Host EC2 ...",
        }
        result = termination.classify_termination(
            job_info,
            ec2_client=_ec2_with_state_code(termination.SPOT_TERMINATION_CODE),
            ecs_client=_ecs_returning("i-abc"),
        )
        assert result == {
            "termination_category": "spot",
            "termination_source": "aws_instance_state",
            "termination_confidence": "high",
        }

    def test_non_spot_state_code_falls_through_to_reason(self):
        # Instance terminated for a non-spot reason; no "spot" text -> no classification.
        job_info = {
            "container": {"containerInstanceArn": CI_ARN},
            "statusReason": "host failed",
        }
        result = termination.classify_termination(
            job_info,
            ec2_client=_ec2_with_state_code("Server.InternalError"),
            ecs_client=_ecs_returning("i-abc"),
        )
        assert result is None


class TestClientDegradation:
    def test_no_clients_uses_string_tier(self):
        job_info = {
            "statusReason": "Spot interruption: capacity reclaimed",
            "container": {},
        }
        result = termination.classify_termination(job_info)
        assert result["termination_category"] == "spot"
        assert result["termination_source"] == "status_reason"
        assert result["termination_confidence"] == "low"

    def test_ecs_error_degrades_to_string_tier(self):
        ecs = MagicMock()
        ecs.describe_container_instances.side_effect = Exception("AccessDenied")
        job_info = {
            "container": {"containerInstanceArn": CI_ARN},
            "statusReason": "Spot interruption",
        }
        result = termination.classify_termination(
            job_info, ec2_client=_ec2_with_state_code("x"), ecs_client=ecs
        )
        assert result["termination_category"] == "spot"
        assert result["termination_source"] == "status_reason"

    def test_aged_out_instance_degrades(self):
        ec2 = MagicMock()
        ec2.describe_instances.side_effect = Exception("InvalidInstanceID.NotFound")
        job_info = {
            "container": {"containerInstanceArn": CI_ARN},
            "statusReason": "host failed",
        }
        # No spot text + instance gone -> no classification (None), not a crash.
        result = termination.classify_termination(
            job_info, ec2_client=ec2, ecs_client=_ecs_returning("i-abc")
        )
        assert result is None


class TestStringTier:
    def test_oom_from_container_reason(self):
        job_info = {"container": {"reason": "OutOfMemoryError: Container killed"}}
        assert (
            termination.classify_termination(job_info)["termination_category"] == "oom"
        )

    def test_timeout(self):
        job_info = {"statusReason": "Job attempt timed out", "container": {}}
        assert (
            termination.classify_termination(job_info)["termination_category"]
            == "timeout"
        )

    def test_unrecognized_reason_returns_none(self):
        assert (
            termination.classify_termination(
                {"statusReason": "exited", "container": {}}
            )
            is None
        )

    def test_bare_memory_mention_is_not_oom(self):
        # A scheduling/sizing failure mentioning "memory" must NOT classify as OOM.
        job_info = {
            "statusReason": "insufficient memory available to schedule the job",
            "container": {},
        }
        assert termination.classify_termination(job_info) is None

    def test_explicit_oom_phrasings(self):
        for reason in (
            "OutOfMemoryError",
            "Container killed due to memory usage",
            "OOMKilled",
        ):
            job_info = {"container": {"reason": reason}}
            assert (
                termination.classify_termination(job_info)["termination_category"]
                == "oom"
            )

    def test_empty_returns_none(self):
        assert termination.classify_termination({}) is None


class TestInstanceStateLookup:
    def test_empty_reservations_returns_none(self):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {"Reservations": []}
        assert termination._instance_state_reason_code("i-abc", ec2) is None

    def test_missing_state_reason_returns_none(self):
        ec2 = MagicMock()
        ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [{}]}]  # no StateReason key
        }
        assert termination._instance_state_reason_code("i-abc", ec2) is None


class TestClusterFromArn:
    def test_new_format_extracts_cluster(self):
        assert termination._cluster_from_container_instance_arn(CI_ARN) == "my-cluster"

    def test_old_short_arn_has_no_cluster(self):
        old = "arn:aws:ecs:us-east-1:1:container-instance/abc123"
        assert termination._cluster_from_container_instance_arn(old) is None

    def test_short_arn_skips_high_confidence_tier(self):
        # Without a resolvable cluster, the instance lookup is skipped (clean degrade).
        old = "arn:aws:ecs:us-east-1:1:container-instance/abc123"
        job_info = {
            "container": {"containerInstanceArn": old},
            "statusReason": "host failed",
        }
        result = termination.classify_termination(
            job_info,
            ec2_client=_ec2_with_state_code(termination.SPOT_TERMINATION_CODE),
            ecs_client=MagicMock(),
        )
        assert result is None  # ecs never consulted, no spot text
