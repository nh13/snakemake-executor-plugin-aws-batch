"""A rule queue map: rule names mapped to the AWS Batch job queues their jobs run on.

A deployment that routes a workflow's rules to different queues (for example GPU rules
to a GPU queue) without editing the workflow passes the map with
``--aws-batch-rule-queues`` (or ``SNAKEMAKE_AWS_BATCH_RULE_QUEUES``). Unlike
``--set-resources``, which replaces a workflow profile's whole ``set-resources`` when
given on the command line, the map leaves every other resource alone.

The map is a JSON object of rule name to job queue (a name or a job queue ARN), given
inline or as the path of a file holding it. A job's queue is the map's entry for its
rule, else the rule's ``batch_queue`` resource, else ``--aws-batch-job-queue``. A group
job runs as one AWS Batch job, so its rules must all map to the same queue, or none of
them be in the map.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from snakemake_interface_common.exceptions import WorkflowError
from snakemake_interface_executor_plugins.jobs import (
    GroupJobExecutorInterface,
    JobExecutorInterface,
)

# A job queue name, as CreateJobQueue accepts it.
_QUEUE_NAME = r"[A-Za-z0-9_-]{1,128}"

# A job queue as SubmitJob's jobQueue takes it: its name, or its ARN.
_QUEUE_ARN = r"arn:aws(?:-[a-z]+)*:batch:[a-z0-9-]+:[0-9]{12}:job-queue/" + _QUEUE_NAME
JOB_QUEUE = re.compile(rf"{_QUEUE_NAME}|{_QUEUE_ARN}")


def _unique_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """A JSON object hook that rejects a key given twice (JSON keeps the last)."""
    obj: Dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"rule {key!r} is in the map twice")
        obj[key] = value
    return obj


class RuleQueues:
    """Rule names to the job queues their jobs run on."""

    def __init__(self, mapping: Mapping[str, Any], source: Optional[str] = None):
        # Where the map was read from, for error messages.
        self.source = source
        self._entries: Dict[str, str] = {}
        for rule, queue in mapping.items():
            if not isinstance(rule, str) or not rule.isidentifier():
                raise ValueError(f"rule queue map key {rule!r} is not a rule name")
            if not isinstance(queue, str) or not JOB_QUEUE.fullmatch(queue):
                raise ValueError(
                    f"rule queue map entry {rule!r}: {queue!r} is not a job queue name "
                    "or job queue ARN"
                )
            self._entries[rule] = queue

    @classmethod
    def from_json(cls, text: str, source: Optional[str] = None) -> "RuleQueues":
        """Load a JSON object of rule name -> job queue."""
        data = json.loads(text, object_pairs_hook=_unique_keys)
        if not isinstance(data, dict):
            raise ValueError("a rule queue map must be a JSON object")
        return cls(data, source=source)

    @classmethod
    def from_setting(cls, value: str) -> "RuleQueues":
        """Load the map from a setting's value: the JSON object itself when its first
        non-blank character is ``{``, else the path of a file holding it."""
        if value.lstrip().startswith("{"):
            return cls.from_json(value)
        return cls.from_json(Path(value).read_text(), source=value)

    def queue_of(self, job: JobExecutorInterface) -> Optional[str]:
        """The queue the map gives ``job``, or ``None`` when its rule is not in the map.

        A group job takes the queue its rules map to. Raises ``WorkflowError`` when its
        rules map to different queues, or only some of them are in the map: the group
        runs as one AWS Batch job, on one queue.
        """
        if not isinstance(job, GroupJobExecutorInterface):
            return self._entries.get(job.name)
        rules = sorted({member.name for member in job.jobs})
        queues = {rule: self._entries.get(rule) for rule in rules}
        if len(set(queues.values())) <= 1:
            return next(iter(queues.values()), None)
        listed = ", ".join(
            f"{rule} -> {queue if queue is not None else '(not in the map)'}"
            for rule, queue in queues.items()
        )
        raise WorkflowError(
            f"{job.name}: the rules of this group job do not share one queue in the "
            f"rule queue map ({listed}). A group runs as one AWS Batch job, so map all "
            "of its rules to the same queue, or leave all of them out of the map"
        )

    def __len__(self) -> int:
        """The number of rules in the map."""
        return len(self._entries)
