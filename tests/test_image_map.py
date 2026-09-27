import json

import pytest
from snakemake_interface_common.exceptions import WorkflowError

from snakemake_executor_plugin_aws_batch import resolve_container_image
from snakemake_executor_plugin_aws_batch.image_map import ImageMap, normalize

DIGEST = "sha256:" + "a" * 64
PINNED = f"123456789012.dkr.ecr.us-east-1.amazonaws.com/workers@{DIGEST}"


@pytest.mark.parametrize(
    "ref, want",
    [
        ("ubuntu", "docker.io/library/ubuntu:latest"),
        ("ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
        ("biocontainers/samtools:1.21", "docker.io/biocontainers/samtools:1.21"),
        ("index.docker.io/library/ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
        (
            "quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0",
            "quay.io/biocontainers/fastqc:0.12.1--hdfd78af_0",
        ),
        ("localhost:5000/coord", "localhost:5000/coord:latest"),
        (f"quay.io/x/y@{DIGEST.replace('a', 'A')}", f"quay.io/x/y@{DIGEST}"),
        (f"quay.io/x/y:1@{DIGEST}", f"quay.io/x/y:1@{DIGEST}"),
    ],
)
def test_normalize_reads_references_like_docker(ref, want):
    assert normalize(ref) == want


def test_a_reference_resolves_as_written_or_normalized():
    image_map = ImageMap({"ubuntu:24.04": PINNED, PINNED: PINNED})
    assert image_map.resolve("ubuntu:24.04") == PINNED
    assert image_map.resolve("docker.io/library/ubuntu:24.04") == PINNED
    # An identity entry lets an already-pinned image through.
    assert image_map.resolve(PINNED) == PINNED
    assert image_map.resolve("ubuntu:22.04") is None


def test_a_spelling_the_map_names_wins_over_a_normalized_one():
    image_map = ImageMap({"ubuntu": "a", "docker.io/library/ubuntu:latest": "b"})
    assert image_map.resolve("ubuntu") == "a"
    assert image_map.resolve("docker.io/library/ubuntu:latest") == "b"


def test_the_map_is_read_from_a_json_object(tmp_path):
    path = tmp_path / "images.json"
    path.write_text(json.dumps({"quay.io/b/fastqc:1": PINNED}))
    assert ImageMap.from_file(str(path)).resolve("quay.io/b/fastqc:1") == PINNED
    path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        ImageMap.from_file(str(path))
    with pytest.raises(ValueError, match="not a reference"):
        ImageMap({"x": ""})


def test_jobs_resolve_through_the_map_strictly():
    image_map = ImageMap({"quay.io/b/fastqc:1": PINNED})
    assert resolve_container_image(None, "rule a", "anything:1") == "anything:1"
    assert resolve_container_image(image_map, "rule a", "quay.io/b/fastqc:1") == PINNED
    missing = r"rule b: container image 'quay.io/b/seqkit:2'"
    with pytest.raises(WorkflowError, match=missing):
        resolve_container_image(image_map, "rule b", "quay.io/b/seqkit:2")
