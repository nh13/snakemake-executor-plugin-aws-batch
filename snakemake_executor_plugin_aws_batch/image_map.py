"""A container image map: the images a workflow may run, mapped to the images to run.

A deployment that mirrors or rebuilds every image a workflow declares (e.g. into a
private registry, pinned by digest) passes the map with
``--aws-batch-container-image-map`` (or ``SNAKEMAKE_AWS_BATCH_CONTAINER_IMAGE_MAP``).
Every job's image, the global ``--container-image`` or a rule's
``aws_batch_container_image`` resource, is then looked up in it, and a job whose image
is not in the map fails instead of pulling an image the deployment did not vet.

The map is a JSON object of image reference to image reference. A reference is looked up
as written, then in its normalized form (``ubuntu`` and
``docker.io/library/ubuntu:latest`` are the same image), so the map may use either
spelling.
"""

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

DOCKER_HUB = "docker.io"


def normalize(ref: str) -> str:
    """The canonical spelling of an image reference, as Docker reads it.

    The first path component is a registry only when it contains a ``.`` or ``:`` or is
    ``localhost``; otherwise the image is on Docker Hub, and a one-part name is under
    ``library/``. A reference with neither tag nor digest means ``:latest``.
    """
    ref = ref.strip()
    name, _, digest = ref.partition("@")
    tag = None
    if ":" in name.rsplit("/", 1)[-1]:
        name, tag = name.rsplit(":", 1)
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry, repository = first, rest
    else:
        registry, repository = DOCKER_HUB, name
    if registry in ("index.docker.io", "registry-1.docker.io"):
        registry = DOCKER_HUB
    if registry == DOCKER_HUB and "/" not in repository:
        repository = f"library/{repository}"
    if not tag and not digest:
        tag = "latest"
    normalized = f"{registry}/{repository}"
    if tag:
        normalized += f":{tag}"
    if digest:
        normalized += f"@{digest.lower()}"
    return normalized


class ImageMap:
    """Image references to the images to run in their place."""

    def __init__(self, mapping: Mapping[str, str]):
        self._mapping: Dict[str, str] = {}
        for ref, image in mapping.items():
            if not isinstance(ref, str) or not isinstance(image, str) or not image:
                raise ValueError(
                    f"image map entry {ref!r}: {image!r} is not a reference"
                )
            self._mapping[ref] = image
        # Normalized spellings, unless the map names that spelling itself.
        for ref, image in list(self._mapping.items()):
            self._mapping.setdefault(normalize(ref), image)

    @classmethod
    def from_file(cls, path: str) -> "ImageMap":
        """Load a JSON object of reference -> reference."""
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: an image map is a JSON object")
        return cls(data)

    def resolve(self, ref: str) -> Optional[str]:
        """The image to run for ``ref``, or ``None`` if the map does not have it."""
        return self._mapping.get(ref) or self._mapping.get(normalize(ref))

    def __len__(self) -> int:
        return len(self._mapping)
