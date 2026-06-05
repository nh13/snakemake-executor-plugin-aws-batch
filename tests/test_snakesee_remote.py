"""Unit tests for the snakesee remote-job-state emission helpers.

All tests use synthetic describe_jobs entries — no AWS credentials required.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from snakemake_executor_plugin_aws_batch import snakesee_remote


def _job_info(status, **overrides):
    base = {
        "status": status,
        "jobName": "snakejob-align-7",
        "jobQueue": "arn:aws:batch:us-east-1:1:job-queue/gv-spot",
        "createdAt": 100000,  # ms -> 100.0s
        "container": {},
    }
    base.update(overrides)
    return base


class TestPhaseForStatus:
    def test_queue_states_map_to_queued(self):
        for s in ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING"):
            assert snakesee_remote.phase_for_status(s) == "queued"

    def test_running_succeeded_failed(self):
        assert snakesee_remote.phase_for_status("RUNNING") == "running"
        assert snakesee_remote.phase_for_status("SUCCEEDED") == "succeeded"
        assert snakesee_remote.phase_for_status("FAILED") == "failed"

    def test_unknown_status_none(self):
        assert snakesee_remote.phase_for_status("WEIRD") is None
        assert snakesee_remote.phase_for_status(None) is None


class TestBuildPayload:
    def test_running_payload(self):
        info = _job_info(
            "RUNNING",
            startedAt=142000,
            container={"logStreamName": "JobDef/default/abc"},
        )
        payload = snakesee_remote.build_payload(
            snakemake_jobid=7,
            external_jobid="arn:aws:batch:us-east-1:1:job/abc",
            job_info=info,
            region="us-east-1",
        )
        assert payload is not None
        assert payload["schema_version"] == 1
        assert payload["kind"] == "state"
        assert payload["jobid"] == 7
        assert payload["executor"] == "aws-batch"
        assert payload["phase"] == "running"
        assert payload["remote_status"] == "RUNNING"
        assert payload["external_jobid"] == "arn:aws:batch:us-east-1:1:job/abc"
        assert payload["region"] == "us-east-1"
        assert payload["queued_at"] == 100.0  # createdAt 100000 ms
        assert payload["started_at"] == 142.0  # startedAt 142000 ms
        assert payload["queue"] == "arn:aws:batch:us-east-1:1:job-queue/gv-spot"
        assert payload["log_stream"] == "JobDef/default/abc"

    def test_terminal_succeeded_payload(self):
        info = _job_info(
            "SUCCEEDED",
            startedAt=142000,
            stoppedAt=200000,
            container={"exitCode": 0},
            attempts=[{"x": 1}],
        )
        payload = snakesee_remote.build_payload(7, "abc", info, region="us-east-1")
        assert payload is not None
        assert payload["kind"] == "terminal"
        assert payload["phase"] == "succeeded"
        assert payload["stopped_at"] == 200.0
        assert payload["exit_code"] == 0
        assert payload["attempt"] == 1

    def test_failed_payload_includes_reason(self):
        info = _job_info(
            "FAILED",
            stoppedAt=200000,
            statusReason="Essential container in task exited",
            container={"exitCode": 137},
        )
        payload = snakesee_remote.build_payload(7, "abc", info)
        assert payload is not None
        assert payload["phase"] == "failed"
        assert payload["exit_code"] == 137
        assert payload["status_reason"] == "Essential container in task exited"

    def test_none_when_status_unmappable(self):
        assert snakesee_remote.build_payload(7, "abc", _job_info("WEIRD")) is None

    def test_none_when_no_snakemake_jobid(self):
        assert snakesee_remote.build_payload(None, "abc", _job_info("RUNNING")) is None

    def test_none_for_group_job_uuid_id(self):
        # Group jobs have a UUID string id, not an int — skip cleanly (no raise).
        uuid_id = "3a7c1e2f-0000-4444-8888-abcdef012345"
        assert (
            snakesee_remote.build_payload(uuid_id, "abc", _job_info("RUNNING")) is None
        )

    def test_termination_merged_when_provided(self):
        info = _job_info("FAILED", stoppedAt=200000, container={"exitCode": 137})
        term = {
            "termination_category": "spot",
            "termination_source": "aws_instance_state",
            "termination_confidence": "high",
        }
        payload = snakesee_remote.build_payload(7, "abc", info, termination=term)
        assert payload["termination_category"] == "spot"
        assert payload["termination_source"] == "aws_instance_state"
        assert payload["termination_confidence"] == "high"

    def test_no_termination_keys_when_absent(self):
        info = _job_info("FAILED", stoppedAt=200000)
        payload = snakesee_remote.build_payload(7, "abc", info)
        assert "termination_category" not in payload

    def test_cost_estimate_merged_when_provided(self):
        info = _job_info("SUCCEEDED", stoppedAt=200000)
        payload = snakesee_remote.build_payload(7, "abc", info, cost_estimate=0.1234)
        assert payload["cost_estimate"] == 0.1234

    def test_no_cost_key_when_absent(self):
        info = _job_info("SUCCEEDED", stoppedAt=200000)
        assert "cost_estimate" not in snakesee_remote.build_payload(7, "abc", info)

    def test_optional_fields_omitted_when_absent(self):
        # A bare queued job with no timestamps/queue still produces a minimal payload.
        info = {"status": "SUBMITTED", "container": {}}
        payload = snakesee_remote.build_payload(7, None, info)
        assert payload is not None
        assert payload["phase"] == "queued"
        assert "external_jobid" not in payload
        assert "started_at" not in payload
        assert "queue" not in payload


class TestEmit:
    def test_emit_attaches_payload_under_wire_key(self):
        logger = MagicMock()
        payload = {"schema_version": 1, "phase": "running", "jobid": 7}
        snakesee_remote.emit(logger, payload)
        logger.info.assert_called_once()
        _, kwargs = logger.info.call_args
        assert kwargs["extra"][snakesee_remote.WIRE_KEY] is payload

    def test_emit_noop_on_none(self):
        logger = MagicMock()
        snakesee_remote.emit(logger, None)
        logger.info.assert_not_called()

    def test_emit_through_real_stdlib_logger_does_not_raise(self):
        # The "doesn't disrupt non-snakesee users" guarantee: a plain stdlib
        # Logger must accept the message + extra payload without raising, and the
        # payload must land on the record under the wire key.
        import logging

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        logger = logging.getLogger("test.snakesee.emit")
        logger.setLevel(logging.INFO)
        handler = _Capture()
        logger.addHandler(handler)
        try:
            snakesee_remote.emit(
                logger, {"jobid": 7, "phase": "running", "external_jobid": "abc"}
            )
        finally:
            logger.removeHandler(handler)
        assert len(records) == 1
        assert getattr(records[0], snakesee_remote.WIRE_KEY)["phase"] == "running"


class TestEmitDedup:
    """The executor emits once per phase transition (via job.aux)."""

    def _executor_with_logger(self):
        from snakemake_executor_plugin_aws_batch import Executor

        ex = Executor.__new__(Executor)
        ex.logger = MagicMock()
        ex.settings = SimpleNamespace(region="us-east-1")
        return ex

    def _submitted_job(self, jobid=7):
        return SimpleNamespace(
            job=SimpleNamespace(jobid=jobid),
            external_jobid="arn:aws:batch:us-east-1:1:job/abc",
            aux={},
        )

    def test_repeated_same_phase_emits_once(self):
        ex = self._executor_with_logger()
        job = self._submitted_job()
        info = _job_info("RUNNING", startedAt=142000)
        ex._emit_snakesee_state(job, info)
        ex._emit_snakesee_state(job, info)  # same phase again -> no second emit
        assert ex.logger.info.call_count == 1
        assert job.aux["_snakesee_phase"] == "running"

    def test_phase_change_emits_again(self):
        ex = self._executor_with_logger()
        job = self._submitted_job()
        ex._emit_snakesee_state(job, _job_info("RUNNABLE"))  # queued
        ex._emit_snakesee_state(job, _job_info("RUNNING", startedAt=142000))  # running
        ex._emit_snakesee_state(
            job, _job_info("SUCCEEDED", startedAt=142000, stoppedAt=200000)
        )  # terminal
        assert ex.logger.info.call_count == 3

    def test_aux_none_is_safe(self):
        ex = self._executor_with_logger()
        job = SimpleNamespace(
            job=SimpleNamespace(jobid=7), external_jobid="abc", aux=None
        )
        ex._emit_snakesee_state(
            job, _job_info("RUNNING", startedAt=142000)
        )  # must not raise
        ex.logger.info.assert_not_called()

    def test_failed_phase_classifies_and_includes_termination(self):
        # On a FAILED phase, _emit_snakesee_state classifies termination and the
        # emitted payload carries the classification. Inject EC2/ECS mocks so the
        # high-confidence instance-state tier fires.
        ex = self._executor_with_logger()
        ex._aws_clients = {
            "ecs": MagicMock(
                describe_container_instances=MagicMock(
                    return_value={"containerInstances": [{"ec2InstanceId": "i-abc"}]}
                )
            ),
            "ec2": MagicMock(
                describe_instances=MagicMock(
                    return_value={
                        "Reservations": [
                            {
                                "Instances": [
                                    {
                                        "StateReason": {
                                            "Code": "Server.SpotInstanceTermination"
                                        }
                                    }
                                ]
                            }
                        ]
                    }
                )
            ),
        }
        job = self._submitted_job()
        info = _job_info(
            "FAILED",
            stoppedAt=200000,
            container={
                "exitCode": 137,
                "containerInstanceArn": "arn:aws:ecs:us-east-1:1:container-instance/c/abc",
            },
        )
        ex._emit_snakesee_state(job, info)
        ex.logger.info.assert_called_once()
        _, kwargs = ex.logger.info.call_args
        payload = kwargs["extra"][snakesee_remote.WIRE_KEY]
        assert payload["termination_category"] == "spot"
        assert payload["termination_confidence"] == "high"

    def test_running_phase_does_not_classify(self):
        # Classification only runs on the failed phase; a running emit has no
        # termination keys and never touches the AWS clients.
        ex = self._executor_with_logger()
        ex._classify_termination = MagicMock()  # would raise if called incorrectly
        job = self._submitted_job()
        ex._emit_snakesee_state(job, _job_info("RUNNING", startedAt=142000))
        ex._classify_termination.assert_not_called()

    def test_classification_cached_across_retries(self):
        # If emission is retried for the same failed job, classification (and its
        # AWS calls) runs at most once — the result is cached on job.aux.
        ex = self._executor_with_logger()
        ex._classify_termination = MagicMock(
            return_value={"termination_category": "spot"}
        )
        job = self._submitted_job()
        info = _job_info("FAILED", stoppedAt=200000)
        ex._emit_snakesee_state(job, info)
        # Force a retry by clearing the phase dedup marker (simulating a prior
        # emit failure) while leaving the cached classification in place.
        job.aux.pop("_snakesee_phase", None)
        ex._emit_snakesee_state(job, info)
        assert ex._classify_termination.call_count == 1

    def test_aws_client_lazily_created_and_cached(self):
        ex = self._executor_with_logger()
        with patch("boto3.client", return_value=MagicMock()) as mk:
            a = ex._aws_client("ec2")
            b = ex._aws_client("ec2")
        assert a is b
        mk.assert_called_once_with("ec2", region_name="us-east-1")

    def test_get_job_status_emits_terminal_before_returning(self):
        # Drive the real _get_job_status control flow: a terminal poll must emit a
        # terminal-kind event AND return the exit code. Guards against a refactor
        # that returns before emitting.
        ex = self._executor_with_logger()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_jobs.return_value = {
            "jobs": [
                _job_info(
                    "SUCCEEDED",
                    startedAt=142000,
                    stoppedAt=200000,
                    jobDefinition="arn:def",
                    container={"exitCode": 0},
                )
            ]
        }
        job = self._submitted_job()
        status_code, msg = ex._get_job_status(job)
        assert (status_code, msg) == (0, None)
        # Emission happened before the return, carrying the terminal payload.
        ex.logger.info.assert_called_once()
        _, kwargs = ex.logger.info.call_args
        payload = kwargs["extra"][snakesee_remote.WIRE_KEY]
        assert payload["kind"] == "terminal"
        assert payload["phase"] == "succeeded"
