"""Unit tests for BatchJobBuilder.submit() tag handling.

These tests focus solely on tag propagation to submit_job and do not require
AWS credentials or a live Snakemake workflow.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from snakemake_executor_plugin_aws_batch.batch_job_builder import (
    SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR,
    BatchJobBuilder,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_builder(tags=None) -> BatchJobBuilder:
    """Return a BatchJobBuilder with minimal mocks.

    The batch_client is fully mocked so no AWS calls are made.  build_job_definition
    is patched in each test that exercises submit() so we only test the tag-assembly
    logic in isolation.
    """
    settings = SimpleNamespace(
        job_queue="test-queue",
        job_role="arn:aws:iam::123456789:role/test-role",
        tags=tags,
        task_timeout=300,
    )

    batch_client = MagicMock()
    # _get_platform_from_queue is called during __init__; short-circuit it.
    batch_client.describe_job_queues.return_value = {"jobQueues": []}

    logger = MagicMock()
    job = MagicMock()
    job.name = "test_rule"
    job.threads = 1
    job.resources = {"_cores": 1, "mem_mb": 1024}

    builder = BatchJobBuilder(
        logger=logger,
        job=job,
        envvars={},
        container_image="test-image:latest",
        settings=settings,
        job_command="snakemake ...",
        batch_client=batch_client,
    )
    return builder


def _fake_job_def():
    """Return a minimal job-definition response for build_job_definition mocking."""
    return {"jobDefinitionName": "snakejob-def-test", "revision": 1}


# ---------------------------------------------------------------------------
# Tests for _build_job_tags
# ---------------------------------------------------------------------------

class TestBuildJobTags:
    def test_none_settings_tags_returns_empty(self):
        builder = _make_builder(tags=None)
        assert builder._build_job_tags() == {}

    def test_empty_dict_settings_tags_returns_empty(self):
        builder = _make_builder(tags={})
        assert builder._build_job_tags() == {}

    def test_settings_tags_included(self):
        builder = _make_builder(tags={"Env": "prod", "Project": "fgumi"})
        result = builder._build_job_tags()
        assert result == {"Env": "prod", "Project": "fgumi"}

    def test_settings_tags_not_mutated(self):
        """_build_job_tags must return a copy, not mutate settings.tags."""
        original = {"Env": "prod"}
        builder = _make_builder(tags=original)
        result = builder._build_job_tags()
        result["Extra"] = "value"
        assert "Extra" not in original

    def test_env_var_tags_parsed_and_merged(self):
        builder = _make_builder(tags={"Env": "prod"})
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Team=data,Cost=low"}):
            result = builder._build_job_tags()
        assert result == {"Env": "prod", "Team": "data", "Cost": "low"}

    def test_env_var_tags_override_settings_tags_on_conflict(self):
        builder = _make_builder(tags={"Env": "prod", "Team": "bio"})
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Team=data"}):
            result = builder._build_job_tags()
        assert result["Team"] == "data"
        assert result["Env"] == "prod"

    def test_env_var_only_no_settings_tags(self):
        builder = _make_builder(tags=None)
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Owner=alice"}):
            result = builder._build_job_tags()
        assert result == {"Owner": "alice"}

    def test_env_var_with_value_containing_equals(self):
        """A VALUE that itself contains '=' should be handled (key=rest of string)."""
        builder = _make_builder(tags=None)
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Url=http://x=1"}):
            result = builder._build_job_tags()
        assert result == {"Url": "http://x=1"}

    def test_empty_env_var_ignored(self):
        builder = _make_builder(tags={"Env": "prod"})
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: ""}):
            result = builder._build_job_tags()
        assert result == {"Env": "prod"}

    def test_absent_env_var_ignored(self):
        builder = _make_builder(tags={"Env": "prod"})
        env = {k: v for k, v in os.environ.items() if k != SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR}
        with patch.dict(os.environ, env, clear=True):
            result = builder._build_job_tags()
        assert result == {"Env": "prod"}


# ---------------------------------------------------------------------------
# Tests for submit() — tags propagation to batch_client.submit_job
# ---------------------------------------------------------------------------

class TestSubmitTagPropagation:
    def _run_submit(self, builder: BatchJobBuilder):
        """Patch build_job_definition and submit_job, then call submit()."""
        builder.batch_client.submit_job.return_value = {
            "jobName": "snakejob-test",
            "jobId": "abc-123",
            "jobQueue": "test-queue",
        }
        with patch.object(builder, "build_job_definition", return_value=(_fake_job_def(), "snakejob-test")):
            return builder.submit(), builder.batch_client.submit_job.call_args

    def test_tags_from_settings_passed_to_submit_job(self):
        builder = _make_builder(tags={"Env": "prod"})
        _, call_args = self._run_submit(builder)
        assert call_args.kwargs.get("tags") == {"Env": "prod"} or \
               call_args[1].get("tags") == {"Env": "prod"} or \
               ("tags" in call_args[0][0] if call_args[0] else False) or \
               call_args.kwargs.get("tags") == {"Env": "prod"}
        # Normalise: extract the tags kwarg regardless of how mock recorded it
        submitted_tags = _extract_tags(call_args)
        assert submitted_tags == {"Env": "prod"}

    def test_env_var_tags_passed_to_submit_job(self):
        builder = _make_builder(tags=None)
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Team=data"}):
            _, call_args = self._run_submit(builder)
        assert _extract_tags(call_args) == {"Team": "data"}

    def test_merged_tags_passed_to_submit_job(self):
        builder = _make_builder(tags={"Env": "prod"})
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Team=data"}):
            _, call_args = self._run_submit(builder)
        assert _extract_tags(call_args) == {"Env": "prod", "Team": "data"}

    def test_no_tags_key_in_job_params_when_empty(self):
        """When tags is empty, 'tags' should not appear in submit_job call."""
        builder = _make_builder(tags=None)
        env = {k: v for k, v in os.environ.items() if k != SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR}
        with patch.dict(os.environ, env, clear=True):
            _, call_args = self._run_submit(builder)
        assert _extract_tags(call_args) is None

    def test_env_var_overrides_settings_in_submit_job(self):
        builder = _make_builder(tags={"Team": "bio"})
        with patch.dict(os.environ, {SNAKEMAKE_AWS_BATCH_JOB_TAGS_ENV_VAR: "Team=data"}):
            _, call_args = self._run_submit(builder)
        assert _extract_tags(call_args) == {"Team": "data"}


def _extract_tags(call_args) -> dict | None:
    """Extract the 'tags' value from a mock call_args, or None if not present."""
    # call_args is a unittest.mock.call object; kwargs is the preferred accessor
    if call_args is None:
        return None
    kwargs = call_args.kwargs if hasattr(call_args, "kwargs") else call_args[1]
    return kwargs.get("tags")
