"""Tests for batched status polling, CloudWatch log tails, and stuck-job warnings."""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from snakemake_executor_plugin_aws_batch import Executor, RUNNABLE_STUCK_SECONDS


class _AsyncNullContext:
    """Stand-in for status_rate_limiter (an async context manager)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _executor(**settings):
    ex = Executor.__new__(Executor)
    ex.logger = MagicMock()
    ex.settings = SimpleNamespace(region="us-east-1", job_queue="arn:q", **settings)
    return ex


def _job(jid="job-1"):
    return SimpleNamespace(external_jobid=jid, aux={})


class TestDescribeJobsBulk:
    def test_maps_ids_to_job_info(self):
        ex = _executor()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_jobs.return_value = {
            "jobs": [
                {"jobId": "a", "status": "RUNNING"},
                {"jobId": "b", "status": "SUCCEEDED"},
            ]
        }
        result = ex._describe_jobs_bulk(["a", "b"])
        assert set(result) == {"a", "b"}
        assert result["b"]["status"] == "SUCCEEDED"
        ex.batch_client.describe_jobs.assert_called_once_with(jobs=["a", "b"])

    def test_chunks_in_hundreds(self):
        ex = _executor()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_jobs.return_value = {"jobs": []}
        ex._describe_jobs_bulk([str(i) for i in range(250)])
        # 250 ids -> 3 calls (100, 100, 50).
        assert ex.batch_client.describe_jobs.call_count == 3
        sizes = [
            len(c.kwargs["jobs"]) for c in ex.batch_client.describe_jobs.call_args_list
        ]
        assert sizes == [100, 100, 50]

    def test_chunk_error_is_skipped(self):
        ex = _executor()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_jobs.side_effect = Exception("throttled")
        # Error logged, empty map returned, no raise.
        assert ex._describe_jobs_bulk(["a"]) == {}
        ex.logger.error.assert_called()

    def test_empty_input(self):
        ex = _executor()
        ex.batch_client = MagicMock()
        assert ex._describe_jobs_bulk([]) == {}
        ex.batch_client.describe_jobs.assert_not_called()


class TestCheckActiveJobs:
    """The async generator's yield-vs-report contract (the load-bearing behavior)."""

    def _run(self, ex, jobs):
        async def collect():
            return [j async for j in ex.check_active_jobs(jobs)]

        return asyncio.run(collect())

    def _executor_for_loop(self, bulk_map):
        ex = _executor()
        ex.status_rate_limiter = _AsyncNullContext()
        ex._describe_jobs_bulk = MagicMock(return_value=bulk_map)
        ex.report_job_success = MagicMock()
        ex.report_job_error = MagicMock()
        ex.cleanup_job_resources = MagicMock()
        return ex

    def test_running_job_is_yielded_not_reported(self):
        ex = self._executor_for_loop(
            {"j1": {"jobId": "j1", "status": "RUNNING", "container": {}}}
        )
        job = _job("j1")
        result = self._run(ex, [job])
        assert result == [job]  # still active
        ex.report_job_success.assert_not_called()
        ex.report_job_error.assert_not_called()

    def test_job_absent_from_map_is_yielded(self):
        # describe returned nothing for this id -> keep monitoring, never dropped.
        ex = self._executor_for_loop({})
        job = _job("j1")
        assert self._run(ex, [job]) == [job]
        ex.report_job_success.assert_not_called()

    def test_succeeded_job_reported_and_not_yielded(self):
        ex = self._executor_for_loop(
            {"j1": {"jobId": "j1", "status": "SUCCEEDED", "container": {}}}
        )
        job = _job("j1")
        assert self._run(ex, [job]) == []
        ex.report_job_success.assert_called_once_with(job)
        ex.cleanup_job_resources.assert_called_once_with(job)

    def test_failed_job_reported_as_error(self):
        ex = self._executor_for_loop(
            {
                "j1": {
                    "jobId": "j1",
                    "status": "FAILED",
                    "container": {"exitCode": 1},
                    "statusReason": "boom",
                }
            }
        )
        job = _job("j1")
        assert self._run(ex, [job]) == []
        ex.report_job_error.assert_called_once()
        ex.report_job_success.assert_not_called()


class TestFailureLogTail:
    def test_returns_joined_log_lines(self):
        ex = _executor()
        ex._aws_clients = {
            "logs": MagicMock(
                get_log_events=MagicMock(
                    return_value={
                        "events": [{"message": "line1"}, {"message": "line2"}]
                    }
                )
            )
        }
        job_info = {"container": {"logStreamName": "JobDef/default/abc"}}
        assert ex._failure_log_tail(job_info) == "line1\nline2"

    def test_uses_custom_log_group_when_configured(self):
        logs = MagicMock(
            get_log_events=MagicMock(return_value={"events": [{"message": "x"}]})
        )
        ex = _executor()
        ex._aws_clients = {"logs": logs}
        job_info = {
            "container": {
                "logStreamName": "s",
                "logConfiguration": {"options": {"awslogs-group": "/custom/group"}},
            }
        }
        ex._failure_log_tail(job_info)
        assert logs.get_log_events.call_args.kwargs["logGroupName"] == "/custom/group"

    def test_none_without_stream(self):
        ex = _executor()
        assert ex._failure_log_tail({"container": {}}) is None

    def test_degrades_on_error(self):
        ex = _executor()
        ex._aws_clients = {
            "logs": MagicMock(
                get_log_events=MagicMock(side_effect=Exception("AccessDenied"))
            )
        }
        job_info = {"container": {"logStreamName": "s"}}
        assert ex._failure_log_tail(job_info) is None


class TestMaybeWarnStuck:
    def test_warns_after_threshold_once(self):
        ex = _executor(runnable_stuck_seconds=300)
        ex._diagnose_queue_capacity = MagicMock(return_value="Likely cause: ...")
        job = _job()
        job.aux["_stuck_since"] = time.time() - 1000  # well past threshold
        ex._maybe_warn_stuck(job, "RUNNABLE")
        ex._maybe_warn_stuck(job, "RUNNABLE")  # second call: already warned
        assert ex.logger.warning.call_count == 1
        assert job.aux["_stuck_warned"] is True

    def test_not_warned_before_threshold(self):
        ex = _executor(runnable_stuck_seconds=300)
        job = _job()  # _stuck_since set to now on first call
        ex._maybe_warn_stuck(job, "RUNNABLE")
        ex.logger.warning.assert_not_called()

    def test_disabled_when_zero(self):
        ex = _executor(runnable_stuck_seconds=0)
        job = _job()
        job.aux["_stuck_since"] = time.time() - 10000
        ex._maybe_warn_stuck(job, "RUNNABLE")
        ex.logger.warning.assert_not_called()

    def test_running_clears_stuck_state(self):
        ex = _executor(runnable_stuck_seconds=300)
        job = _job()
        job.aux["_stuck_since"] = 123.0
        ex._maybe_warn_stuck(job, "RUNNING")
        assert "_stuck_since" not in job.aux

    def test_default_threshold_when_setting_absent(self):
        ex = _executor()  # no runnable_stuck_seconds on settings
        assert getattr(ex.settings, "runnable_stuck_seconds", None) is None
        job = _job()
        job.aux["_stuck_since"] = time.time() - (RUNNABLE_STUCK_SECONDS + 10)
        ex._diagnose_queue_capacity = MagicMock(return_value="diag")
        ex._maybe_warn_stuck(job, "RUNNABLE")
        ex.logger.warning.assert_called_once()


class TestDiagnoseQueueCapacity:
    def _ex_with_queue(self, queue, compute_envs=None):
        ex = _executor()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_job_queues.return_value = {
            "jobQueues": [queue] if queue else []
        }
        ex.batch_client.describe_compute_environments.return_value = {
            "computeEnvironments": compute_envs or []
        }
        return ex

    def test_disabled_queue(self):
        ex = self._ex_with_queue({"state": "DISABLED", "computeEnvironmentOrder": []})
        assert "DISABLED" in ex._diagnose_queue_capacity()

    def test_disabled_compute_environment(self):
        ex = self._ex_with_queue(
            {
                "state": "ENABLED",
                "computeEnvironmentOrder": [{"computeEnvironment": "ce1"}],
            },
            compute_envs=[
                {
                    "computeEnvironmentName": "ce1",
                    "state": "DISABLED",
                    "status": "VALID",
                }
            ],
        )
        msg = ex._diagnose_queue_capacity()
        assert "ce1" in msg and "DISABLED" in msg

    def test_maxvcpus_zero(self):
        ex = self._ex_with_queue(
            {
                "state": "ENABLED",
                "computeEnvironmentOrder": [{"computeEnvironment": "ce1"}],
            },
            compute_envs=[
                {
                    "computeEnvironmentName": "ce1",
                    "state": "ENABLED",
                    "status": "VALID",
                    "computeResources": {"maxvCpus": 0},
                }
            ],
        )
        assert "maxvCpus=0" in ex._diagnose_queue_capacity()

    def test_healthy_environment(self):
        ex = self._ex_with_queue(
            {
                "state": "ENABLED",
                "computeEnvironmentOrder": [{"computeEnvironment": "ce1"}],
            },
            compute_envs=[
                {
                    "computeEnvironmentName": "ce1",
                    "state": "ENABLED",
                    "status": "VALID",
                    "computeResources": {"maxvCpus": 256},
                }
            ],
        )
        assert "healthy" in ex._diagnose_queue_capacity()

    def test_queue_not_found(self):
        ex = self._ex_with_queue(None)
        assert "not found" in ex._diagnose_queue_capacity()

    def test_diagnose_swallows_errors(self):
        ex = _executor()
        ex.batch_client = MagicMock()
        ex.batch_client.describe_job_queues.side_effect = Exception("boom")
        assert "could not diagnose" in ex._diagnose_queue_capacity()
