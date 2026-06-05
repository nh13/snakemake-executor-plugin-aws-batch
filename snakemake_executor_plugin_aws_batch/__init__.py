__author__ = "Jake VanCampen, Johannes Köster"
__copyright__ = "Copyright 2025, Snakemake community"
__email__ = "jake.vancampen7@gmail.com"
__license__ = "MIT"

import time
from dataclasses import dataclass, field
from pprint import pformat
from typing import List, AsyncGenerator, Optional
from snakemake_executor_plugin_aws_batch.batch_client import BatchClient
from snakemake_executor_plugin_aws_batch.batch_job_builder import BatchJobBuilder
from snakemake_executor_plugin_aws_batch import pricing
from snakemake_executor_plugin_aws_batch import snakesee_remote
from snakemake_executor_plugin_aws_batch import termination
from snakemake_interface_executor_plugins.executors.base import SubmittedJobInfo
from snakemake_interface_executor_plugins.executors.remote import RemoteExecutor
from snakemake_interface_executor_plugins.settings import (
    ExecutorSettingsBase,
    CommonSettings,
)
from snakemake_interface_executor_plugins.jobs import (
    JobExecutorInterface,
)
from snakemake_interface_common.exceptions import WorkflowError

# How long a job may sit in a waiting state (SUBMITTED/PENDING/RUNNABLE) before we
# diagnose the queue/compute environment and warn that it may be stuck.
RUNNABLE_STUCK_SECONDS = 300

# Batch queue/compute-environment `status` values that definitively prevent jobs
# from running. CREATING/UPDATING are transient and must NOT fail the preflight
# check (a queue mid-update is recoverable), so only these abort.
FATAL_BATCH_STATUSES = frozenset({"INVALID", "DELETING", "DELETED"})


def _ms_to_seconds(millis) -> Optional[float]:
    """Convert an AWS millisecond epoch timestamp to seconds, or None."""
    if millis is None:
        return None
    try:
        return float(millis) / 1000.0
    except (TypeError, ValueError):
        return None


def _container_vcpus(container: dict) -> Optional[float]:
    """Extract a Batch container's allocated vCPUs (resourceRequirements or legacy)."""
    for requirement in container.get("resourceRequirements") or []:
        if requirement.get("type") == "VCPU":
            try:
                return float(requirement.get("value"))
            except (TypeError, ValueError):
                return None
    vcpus = container.get("vcpus")
    try:
        return float(vcpus) if vcpus is not None else None
    except (TypeError, ValueError):
        return None


# Optional:
# Define additional settings for your executor.
# They will occur in the Snakemake CLI as --<executor-name>-<param-name>
# Omit this class if you don't need any.
# Make sure that all defined fields are Optional and specify a default value
# of None or anything else that makes sense in your case.
@dataclass
class ExecutorSettings(ExecutorSettingsBase):
    region: Optional[str] = field(
        default=None,
        metadata={
            "help": "AWS Region",
            "env_var": False,
            "required": True,
        },
    )
    job_queue: Optional[str] = field(
        default=None,
        metadata={
            "help": "The AWS Batch task queue ARN used for running tasks",
            "env_var": True,
            "required": True,
        },
    )
    job_role: Optional[str] = field(
        default=None,
        metadata={
            "help": "The AWS job role ARN that is used for running the tasks",
            "env_var": True,
            "required": True,
        },
    )
    tags: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The tags that should be applied to all of the batch tasks,"
                "of the form KEY=VALUE"
            ),
            "env_var": False,
            "required": False,
        },
    )
    task_timeout: Optional[int] = field(
        default=300,
        metadata={
            "help": (
                "Task timeout (seconds) will force AWS Batch to terminate "
                "a Batch task if it fails to finish within the timeout, minimum 60"
            )
        },
    )
    runnable_stuck_seconds: Optional[int] = field(
        default=RUNNABLE_STUCK_SECONDS,
        metadata={
            "help": (
                "Warn if a job stays in a waiting state (SUBMITTED/PENDING/RUNNABLE) "
                "this many seconds without starting, with a diagnosis of the job "
                "queue / compute environment. Set to 0 to disable the warning."
            ),
            "env_var": False,
            "required": False,
        },
    )
    estimate_cost: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Estimate each job's cost from its instance type, run duration and "
                "the AWS price list (on-demand) or spot price history (spot), and "
                "surface it for monitoring. This is a list/market-price ESTIMATE, "
                "not the billed amount. Requires pricing:GetProducts and "
                "ec2:DescribeSpotPriceHistory; degrades silently without them."
            ),
            "env_var": False,
            "required": False,
        },
    )


# Required:
# Specify common settings shared by various executors.
common_settings = CommonSettings(
    # define whether your executor plugin executes locally
    # or remotely. In virtually all cases, it will be remote execution
    # (cluster, cloud, etc.). Only Snakemake's standard execution
    # plugins (snakemake-executor-plugin-dryrun, snakemake-executor-plugin-local)
    # are expected to specify False here.
    non_local_exec=True,
    # whether the executor implies to not have a shared file system
    implies_no_shared_fs=True,
    # whether to deploy workflow sources to default storage provider before execution
    job_deploy_sources=True,
    # whether arguments for setting the storage provider shall be passed to jobs
    pass_default_storage_provider_args=True,
    # whether arguments for setting default resources shall be passed to jobs
    pass_default_resources_args=True,
    # whether environment variables shall be passed to jobs (if False, use
    # self.envvars() to obtain a dict of environment variables and their values
    # and pass them e.g. as secrets to the execution backend)
    pass_envvar_declarations_to_cmd=False,
    # whether the default storage provider shall be deployed before the job is run on
    # the remote node. Set to False so workers do NOT `pip install
    # snakemake-storage-plugin-s3` at startup: that pulls unpinned versions whose
    # newer PyPI metadata requires snakemake-interface-storage-plugins >= 4 and
    # breaks snakemake 8.x. Users must pre-install a compatible plugin version
    # in the container image.
    auto_deploy_default_storage_provider=False,
    # specify initial amount of seconds to sleep before checking for job status
    init_seconds_before_status_checks=0,
)


# Required:
# Implementation of your executor
class Executor(RemoteExecutor):
    def __post_init__(self):
        # snakemake/snakemake:latest container image
        self.container_image = self.workflow.remote_execution_settings.container_image

        self.next_seconds_between_status_checks = 5

        self.settings = self.workflow.executor_settings
        self.logger.debug(f"ExecutorSettings: {pformat(self.settings, indent=2)}")

        try:
            self.batch_client = BatchClient(region_name=self.settings.region)
        except Exception as e:
            raise WorkflowError(f"Failed to initialize AWS Batch client: {e}") from e

        # Fail fast on a definitively misconfigured queue/compute environment/role
        # before submitting any jobs (degrades to a no-op if state is uncertain).
        self._preflight_validate()

    def run_job(self, job: JobExecutorInterface):
        # Implement here how to run a job.
        # You can access the job's resources, etc.
        # via the job object.
        # After submitting the job, you have to call
        # self.report_job_submission(job_info).
        # with job_info being of type
        # snakemake_interface_executor_plugins.executors.base.SubmittedJobInfo.
        # If required, make sure to pass the job's id to the job_info object, as keyword
        # argument 'external_job_id'.

        try:
            job_definition = BatchJobBuilder(
                logger=self.logger,
                job=job,
                envvars=self.envvars(),
                container_image=self.container_image,
                settings=self.settings,
                job_command=self.format_job_exec(job),
                batch_client=self.batch_client,
            )
            job_info = job_definition.submit()
            log_info = {
                "job_name": job_info["jobName"],
                "jobId": job_info["jobId"],
                "job_queue": job_definition.job_queue,
            }
            self.logger.debug(f"AWS Batch job submitted: {log_info}")
        except Exception as e:
            raise WorkflowError(f"Failed to submit AWS Batch job: {e}") from e

        self.report_job_submission(
            SubmittedJobInfo(
                job=job, external_jobid=job_info["jobId"], aux=dict(job_info)
            )
        )

    async def check_active_jobs(
        self, active_jobs: List[SubmittedJobInfo]
    ) -> AsyncGenerator[SubmittedJobInfo, None]:
        # Check the status of active jobs.

        # You have to iterate over the given list active_jobs.
        # If you provided it above, each will have its external_jobid set according
        # to the information you provided at submission time.
        # For jobs that have finished successfully, you have to call
        # self.report_job_success(active_job).
        # For jobs that have errored, you have to call
        # self.report_job_error(active_job).
        # This will also take care of providing a proper error message.
        # Usually there is no need to perform additional logging here.
        # Jobs that are still running have to be yielded.
        #
        # For queries to the remote middleware, please use
        # self.status_rate_limiter like this:
        #
        # async with self.status_rate_limiter:
        #    # query remote middleware here
        #
        # To modify the time until the next call of this method,
        # you can set self.next_sleep_seconds here.
        self.logger.debug(f"Monitoring {len(active_jobs)} active Batch jobs")

        # Fetch the status of every active job in as few API calls as possible:
        # describe_jobs accepts up to 100 ids per call, so this is one call per 100
        # jobs instead of one per job, which matters (and avoids throttling) on
        # large workflows.
        async with self.status_rate_limiter:
            info_by_id = self._describe_jobs_bulk(
                [job.external_jobid for job in active_jobs]
            )

        for job in active_jobs:
            job_info = info_by_id.get(job.external_jobid)
            if job_info is None:
                # No status this round (job not yet visible, or a describe error);
                # keep monitoring it.
                yield job
                continue

            status_code, msg = self._interpret_job_status(job, job_info)
            if status_code is not None:
                if status_code == 0:
                    self.report_job_success(job)
                else:
                    message = f"AWS Batch job failed. Code: {status_code}, Msg: {msg}."
                    self.report_job_error(job, msg=message)
                self.cleanup_job_resources(job)
            else:
                yield job

    def _describe_jobs_bulk(self, external_jobids: List[str]) -> dict:
        """Describe many Batch jobs, mapping external_jobid -> job_info.

        Chunks the ids into batches of 100 (the describe_jobs limit). A failed
        chunk is logged and skipped; its jobs are simply absent from the map and
        get monitored again next poll.
        """
        info_by_id: dict = {}
        for start in range(0, len(external_jobids), 100):
            chunk = external_jobids[start : start + 100]
            try:
                response = self.batch_client.describe_jobs(jobs=chunk)
                for job_info in response.get("jobs", []):
                    jid = job_info.get("jobId")
                    if jid is not None:
                        info_by_id[jid] = job_info
            except Exception as e:
                self.logger.error(f"Error describing Batch jobs: {e}")
        # describe_jobs silently omits unknown ids (it doesn't error on them); a
        # persistently missing id (e.g. a job aged out of Batch's describe
        # retention) would otherwise be re-yielded forever with no signal.
        missing = [jid for jid in external_jobids if jid not in info_by_id]
        if missing:
            self.logger.debug(
                f"No Batch status returned for {len(missing)} job(s): {missing}"
            )
        return info_by_id

    def _interpret_job_status(
        self, job: SubmittedJobInfo, job_info: dict
    ) -> tuple[Optional[int], Optional[str]]:
        """Interpret a describe_jobs entry into (exit_code, message).

        Returns (None, None) while the job is still active. Also emits the
        snakesee remote-state event, warns about stuck jobs, and (on failure)
        appends a CloudWatch log tail to the message. Never raises.
        """
        try:
            job_status = job_info.get("status", "UNKNOWN")
            # push the job_definition_arn to the aux dict for use in cleanup
            job.aux["job_definition_arn"] = job_info.get("jobDefinition", None)
            exit_code = job_info.get("container", {}).get("exitCode", None)

            # Surface the rich Batch state (queue/run/terminal, timestamps, ids) to
            # snakesee via the logger plugin. Emit once per phase transition so the
            # event stream isn't spammed on every poll.
            self._emit_snakesee_state(job, job_info)

            # Warn if the job has been waiting (not running) for too long.
            self._maybe_warn_stuck(job, job_status)

            if job_status == "SUCCEEDED":
                return 0, None
            elif job_status == "FAILED":
                reason = job_info.get("statusReason", "Unknown reason")
                retry_history = self._attempt_history(job_info)
                if retry_history:
                    reason = f"{reason}\nRetry history: {retry_history}"
                log_tail = self._failure_log_tail(job_info)
                if log_tail:
                    reason = f"{reason}\n--- last CloudWatch log lines ---\n{log_tail}"
                return exit_code or 1, reason
            else:
                self.logger.debug(
                    {
                        "job_name": job_info.get("jobName", "unknown"),
                        "job_id": job.external_jobid,
                        "status": job_status,
                    }
                )
                return None, None
        except Exception as e:  # pragma: no cover - defensive
            self.logger.error(f"Error interpreting job status: {e}")
            return None, None

    def _get_job_status(
        self, job: SubmittedJobInfo
    ) -> tuple[Optional[int], Optional[str]]:
        """Poll a single Batch job's status (describe + interpret).

        Retained for direct/single-job use; the polling loop uses the bulk path.

        Returns:
            tuple: (exit_code, failure_message)
        """
        try:
            response = self.batch_client.describe_jobs(jobs=[job.external_jobid])
            jobs = response.get("jobs", [])
            if not jobs:
                return None, f"No job found with ID {job.external_jobid}"
            return self._interpret_job_status(job, jobs[0])
        except Exception as e:
            self.logger.error(f"Error getting job status: {e}")
            return None, str(e)

    def _failure_log_tail(self, job_info: dict, max_lines: int = 20) -> Optional[str]:
        """Return the tail of a failed job's CloudWatch log stream, or None.

        Best-effort: the actual stderr lives in CloudWatch (default log group
        ``/aws/batch/job``), so surfacing the last lines turns a terse
        ``statusReason`` into an actionable error. Requires ``logs:GetLogEvents``;
        any failure (missing permission, no stream) degrades to None.
        """
        container = job_info.get("container") or {}
        log_stream = container.get("logStreamName")
        if not log_stream:
            return None
        # Use the job's configured awslogs group if present, else the Batch default.
        log_options = (container.get("logConfiguration") or {}).get("options") or {}
        log_group = log_options.get("awslogs-group", "/aws/batch/job")
        try:
            logs_client = self._aws_client("logs")
            response = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=log_stream,
                limit=max_lines,
                startFromHead=False,
            )
            messages = [e.get("message", "") for e in response.get("events", [])]
            return "\n".join(m for m in messages if m) or None
        except Exception as e:
            self.logger.debug(f"could not fetch CloudWatch log tail: {e}")
            return None

    @staticmethod
    def _attempt_history(job_info: dict) -> Optional[str]:
        """Summarize per-attempt failure reasons for a retried job, or None.

        A job that failed after retries shows only its final ``statusReason``;
        the earlier attempts' reasons (e.g. a Spot reclamation on attempt 1) live
        in ``attempts[]``. Surfacing them makes the retry story legible. Returns
        None for single-attempt jobs.
        """
        attempts = job_info.get("attempts") or []
        if len(attempts) <= 1:
            return None
        parts = []
        for i, attempt in enumerate(attempts, start=1):
            container = attempt.get("container") or {}
            why = attempt.get("statusReason") or container.get("reason") or "unknown"
            parts.append(f"attempt {i}: {why}")
        return "; ".join(parts)

    def _maybe_warn_stuck(self, job: SubmittedJobInfo, job_status: str) -> None:
        """Warn once if a job has waited (not running) beyond the stuck threshold.

        A job stuck in SUBMITTED/PENDING/RUNNABLE usually means the compute
        environment is disabled, at ``maxvCpus``, or has no matching instances —
        a common silent failure mode. After the configured threshold
        (``runnable_stuck_seconds``, default ``RUNNABLE_STUCK_SECONDS``; set to 0
        to disable) we diagnose the queue/compute environment and log a single
        actionable warning.
        """
        if job.aux is None:
            return
        threshold = getattr(self.settings, "runnable_stuck_seconds", None)
        if threshold is None:
            threshold = RUNNABLE_STUCK_SECONDS
        if threshold <= 0:
            return  # warning disabled
        waiting_states = ("SUBMITTED", "PENDING", "RUNNABLE")
        if job_status not in waiting_states:
            job.aux.pop("_stuck_since", None)
            return
        if job.aux.get("_stuck_warned"):
            return
        now = time.time()
        since = job.aux.setdefault("_stuck_since", now)
        if now - since < threshold:
            return
        self.logger.warning(
            f"AWS Batch job {job.external_jobid} has been {job_status} for "
            f"{int(now - since)}s without starting. {self._diagnose_queue_capacity()}"
        )
        job.aux["_stuck_warned"] = True

    def _queue_problems(self) -> Optional[List[str]]:
        """Return definitive job-queue / compute-environment misconfigurations.

        Returns an empty list when everything looks healthy, a non-empty list of
        problem descriptions when the queue or a compute environment is in a state
        that would prevent jobs from ever starting (disabled/invalid, maxvCpus=0),
        or None when the state can't be determined (no queue configured, or an API
        error — never block the workflow on a transient failure).
        """
        queue_arn = getattr(self.settings, "job_queue", None)
        if not queue_arn:
            return None
        try:
            queues = self.batch_client.describe_job_queues(jobQueues=[queue_arn]).get(
                "jobQueues", []
            )
            if not queues:
                return ["job queue not found"]
            jq = queues[0]
            problems: List[str] = []
            if jq.get("state") != "ENABLED":
                problems.append(f"job queue is {jq.get('state')} (not ENABLED)")
            if jq.get("status") in FATAL_BATCH_STATUSES:
                problems.append(f"job queue status is {jq.get('status')}")
            ce_arns = [
                o.get("computeEnvironment")
                for o in jq.get("computeEnvironmentOrder", [])
            ]
            ce_arns = [c for c in ce_arns if c]
            if ce_arns:
                ces = self.batch_client.describe_compute_environments(
                    computeEnvironments=ce_arns
                ).get("computeEnvironments", [])
                for ce in ces:
                    name = ce.get("computeEnvironmentName", "?")
                    if ce.get("state") != "ENABLED":
                        problems.append(
                            f"compute environment {name} is {ce.get('state')}"
                        )
                    if ce.get("status") in FATAL_BATCH_STATUSES:
                        problems.append(
                            f"compute environment {name} status is {ce.get('status')}"
                        )
                    if (ce.get("computeResources") or {}).get("maxvCpus") == 0:
                        problems.append(f"compute environment {name} has maxvCpus=0")
            return problems
        except Exception as e:
            self.logger.debug(f"could not check job queue / compute environment: {e}")
            return None

    def _diagnose_queue_capacity(self) -> str:
        """Diagnose why jobs may not be starting, from the queue/compute env state."""
        problems = self._queue_problems()
        if problems is None:
            return "(could not diagnose queue capacity)"
        if problems:
            return "Likely cause: " + "; ".join(problems) + "."
        return (
            "Compute environment looks healthy; capacity may be saturated or no "
            "matching instance type is currently available."
        )

    def _preflight_validate(self) -> None:
        """Fail fast on a definitively misconfigured queue / compute environment / role.

        Best-effort about *uncertainty*: a transient API error or missing
        describe permission degrades to a warning and the workflow proceeds. Only
        a confirmed-bad configuration (disabled/invalid queue or compute
        environment, maxvCpus=0, or a non-existent job role) raises, before any
        job is submitted.
        """
        problems = self._queue_problems()
        if problems:
            raise WorkflowError(
                "AWS Batch preflight check failed — jobs would never start: "
                + "; ".join(problems)
                + ". Check the configured --aws-batch-job-queue and its compute "
                "environment(s)."
            )
        self._validate_job_role()

    def _validate_job_role(self) -> None:
        """Verify the configured job role exists (best-effort; needs iam:GetRole).

        A confirmed-missing role (``NoSuchEntity``) fails fast; anything else
        (most importantly a missing ``iam:GetRole`` permission) degrades silently.
        """
        from botocore.exceptions import ClientError

        role_arn = getattr(self.settings, "job_role", None)
        if not role_arn or "/" not in role_arn:
            return
        # GetRole takes the bare role name, not the IAM path: for
        # arn:aws:iam::<acct>:role/<path>/<name> the name is the final segment.
        role_name = role_arn.rsplit("/", 1)[-1]
        try:
            self._aws_client("iam").get_role(RoleName=role_name)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchEntity":
                raise WorkflowError(
                    f"Configured AWS Batch job role does not exist: {role_arn}"
                ) from e
            self.logger.debug(
                f"could not verify job role (likely missing iam:GetRole): {e}"
            )
        except Exception as e:
            self.logger.debug(f"could not verify job role: {e}")

    def _emit_snakesee_state(self, job: SubmittedJobInfo, job_info: dict) -> None:
        """Emit a snakesee remote-state event when the job's phase changes.

        De-duplicated per job: the normalized phase last emitted is stashed in
        ``job.aux`` so each queue->run->terminal transition is reported once even
        though ``check_active_jobs`` polls repeatedly. Best-effort: any failure
        here must never disrupt job monitoring.
        """
        try:
            if job.aux is None:
                return
            phase = snakesee_remote.phase_for_status(job_info.get("status"))
            if phase is None or job.aux.get("_snakesee_phase") == phase:
                return
            snakemake_jobid = getattr(getattr(job, "job", None), "jobid", None)
            region = getattr(self.settings, "region", None)
            # Classify why the job died (only on the terminal failed phase). Cache
            # the result on the job so that if emission is retried on a later poll
            # (e.g. a transient logging failure) we don't re-issue the AWS calls.
            term = None
            if phase == "failed":
                if "_snakesee_termination" not in job.aux:
                    job.aux["_snakesee_termination"] = self._classify_termination(
                        job_info
                    )
                term = job.aux["_snakesee_termination"]
            # Estimate cost once the job reaches a terminal phase (succeeded or
            # failed — both consumed compute). Cached on the job to avoid repeat
            # AWS calls if emission is retried.
            cost = None
            if phase in ("succeeded", "failed"):
                if "_snakesee_cost" not in job.aux:
                    job.aux["_snakesee_cost"] = self._estimate_cost(job_info)
                cost = job.aux["_snakesee_cost"]
            payload = snakesee_remote.build_payload(
                snakemake_jobid=snakemake_jobid,
                external_jobid=job.external_jobid,
                job_info=job_info,
                region=region,
                termination=term,
                cost_estimate=cost,
            )
            if payload is not None:
                snakesee_remote.emit(self.logger, payload)
                job.aux["_snakesee_phase"] = phase
        except (
            Exception
        ) as e:  # pragma: no cover - defensive; monitoring must not break
            self.logger.debug(f"snakesee remote-state emit skipped: {e}")

    def _classify_termination(self, job_info: dict) -> Optional[dict]:
        """Best-effort classification of a failed job's termination cause.

        Uses EC2/ECS lookups for a high-confidence Spot-interruption signal,
        falling back to status-reason string heuristics. Never raises.

        Note: the high-confidence tier calls ``ecs:DescribeContainerInstances``
        and ``ec2:DescribeInstances`` with the executor's own credentials. If
        those permissions are absent the calls are swallowed and classification
        degrades to the low-confidence string tier.
        """
        try:
            return termination.classify_termination(
                job_info,
                ec2_client=self._aws_client("ec2"),
                ecs_client=self._aws_client("ecs"),
            )
        except Exception as e:  # pragma: no cover - defensive
            self.logger.debug(f"termination classification skipped: {e}")
            return None

    def _estimate_cost(self, job_info: dict) -> Optional[float]:
        """Best-effort estimated USD cost for a finished job, or None.

        Resolves the job's EC2 instance (type, vCPUs, spot/on-demand, AZ), prices
        it (cached on-demand list price or spot market price), and apportions the
        cost over the run window by the job's vCPU share. Opt-in via
        ``estimate_cost``; never raises.
        """
        if not getattr(self.settings, "estimate_cost", False):
            return None
        try:
            details = self._resolve_instance_details(job_info)
            if not details or not details.get("instance_type"):
                return None
            price = pricing.price_per_hour(
                details["instance_type"],
                getattr(self.settings, "region", None) or "",
                details.get("az"),
                details.get("lifecycle"),
                pricing_client=self._pricing_client(),
                ec2_client=self._aws_client("ec2"),
                cache=self._price_cache(),
            )
            container = job_info.get("container") or {}
            return pricing.compute_cost(
                price,
                _ms_to_seconds(job_info.get("startedAt")),
                _ms_to_seconds(job_info.get("stoppedAt")),
                job_vcpus=_container_vcpus(container),
                instance_vcpus=details.get("vcpus"),
            )
        except Exception as e:  # pragma: no cover - defensive
            self.logger.debug(f"cost estimation skipped: {e}")
            return None

    def _resolve_instance_details(self, job_info: dict) -> Optional[dict]:
        """Resolve {instance_type, vcpus, lifecycle, az} for a job's EC2 instance."""
        instance_id = termination._resolve_instance_id(
            job_info, self._aws_client("ecs")
        )
        if not instance_id:
            return None
        try:
            response = self._aws_client("ec2").describe_instances(
                InstanceIds=[instance_id]
            )
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    cpu = instance.get("CpuOptions") or {}
                    vcpus = None
                    if cpu.get("CoreCount") and cpu.get("ThreadsPerCore"):
                        vcpus = cpu["CoreCount"] * cpu["ThreadsPerCore"]
                    else:
                        # Without instance vCPUs the cost can't be apportioned and
                        # defaults to the whole-instance price (an over-count when
                        # the instance was shared) — log it so that's diagnosable.
                        self.logger.debug(
                            "instance vCPUs unresolved; cost estimate may over-count"
                        )
                    return {
                        "instance_type": instance.get("InstanceType"),
                        "vcpus": vcpus,
                        # InstanceLifecycle is "spot"/"scheduled"; absent => on-demand.
                        "lifecycle": instance.get("InstanceLifecycle"),
                        "az": (instance.get("Placement") or {}).get("AvailabilityZone"),
                    }
        except Exception as e:  # pragma: no cover - defensive
            self.logger.debug(f"could not resolve instance details: {e}")
        return None

    def _aws_client(self, service: str):
        """Lazily create and cache a boto3 client for a service (e.g. ec2/ecs/iam/logs)."""
        cache = self.__dict__.setdefault("_aws_clients", {})
        if service not in cache:
            import boto3

            cache[service] = boto3.client(
                service, region_name=getattr(self.settings, "region", None)
            )
        return cache[service]

    def _pricing_client(self):
        """Lazily create the AWS Price List client (only in us-east-1 / ap-south-1)."""
        cache = self.__dict__.setdefault("_aws_clients", {})
        if "pricing" not in cache:
            import boto3

            cache["pricing"] = boto3.client("pricing", region_name="us-east-1")
        return cache["pricing"]

    def _price_cache(self) -> pricing.PriceCache:
        """Lazily create the persistent price cache."""
        if not hasattr(self, "_price_cache_obj"):
            self._price_cache_obj = pricing.PriceCache(
                path=pricing.default_cache_path()
            )
        return self._price_cache_obj

    def _terminate_job(self, job: SubmittedJobInfo):
        """terminate job from submitted job info"""
        try:
            self.logger.debug(f"terminating job {job.external_jobid}")
            self.batch_client.terminate_job(
                jobId=job.external_jobid,
                reason="terminated by snakemake",
            )
        except Exception as e:
            self.logger.info(
                f"failed to terminate Batch job: {job.external_jobid} with error: {e}"
            )

    def _deregister_job(self, job: SubmittedJobInfo):
        """deregister batch job definition"""
        try:
            job_def_arn = job.aux.get("job_definition_arn")
            if job_def_arn is not None:
                self.logger.debug(f"de-registering Batch job definition {job_def_arn}")
                self.batch_client.deregister_job_definition(jobDefinition=job_def_arn)
        except Exception as e:
            # AWS expires job definitions after 6mo
            # so failing to delete them isn't fatal
            self.logger.info(
                "failed to deregister Batch job definition "
                f"{job_def_arn} with error {e}"
            )

    def cleanup_job_resources(self, job: SubmittedJobInfo):
        """Terminate and deregister job resources"""
        self._terminate_job(job)
        self._deregister_job(job)

    def cancel_jobs(self, active_jobs: List[SubmittedJobInfo]):
        # Cancel all active jobs.
        # This method is called when Snakemake is interrupted.
        # perform additional steps on shutdown if necessary
        # deregister everything from AWS so the environment is clean
        self.logger.info("shutting down...")
        # cleanup jobs
        for j in active_jobs:
            self.cleanup_job_resources(j)
