"""Workspace-owned policy assets; reward-model caches remain independent."""
import os
import json
import re
import tempfile
from pathlib import Path

from .io import atomic_json, sha256_file, stable_hash


MODELS_ROOT = Path(__file__).resolve().parents[3] / "models"
OPENPI_CACHE_ROOT = MODELS_ROOT / ".cache/openpi"
SOURCE_FILE = ".model-source.json"


def local_model_directory(value: str | Path) -> Path:
    """Resolve the workspace rename at IO boundaries, not in experiment identities."""
    path = Path(value).expanduser().absolute()
    old_root = MODELS_ROOT.parent / "weights"
    if not path.exists() and path.is_relative_to(old_root):
        return MODELS_ROOT / path.relative_to(old_root)
    return path


def repository_directory(repo: str) -> Path:
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(repo)
    return MODELS_ROOT / repo.replace("/", "-")


def deployment_digest(root: Path) -> str:
    return stable_hash({str(path.relative_to(root)): sha256_file(path)
                        for path in sorted(root.rglob("*")) if path.is_file()
                        and not any(part.startswith(".") for part in path.relative_to(root).parts)})


def repository_metadata(repo: str) -> dict | None:
    root = repository_directory(repo)
    if not root.exists():
        return None
    try:
        metadata = json.loads((root / SOURCE_FILE).read_text())
        if (metadata["repo_id"] != repo or not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"])
                or not re.fullmatch(r"[0-9a-f]{64}", metadata["sha256"])):
            raise ValueError("invalid source identity")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid model source metadata: {root}") from exc
    return metadata


def download_repository(repo: str, revision: str, *, allow_patterns=None) -> Path:
    """Publish a complete direct download without replacing an existing model."""
    from filelock import FileLock
    from huggingface_hub import snapshot_download

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model download requires an immutable repository revision")
    destination = repository_directory(repo)
    locks = MODELS_ROOT / ".cache/locks"
    locks.mkdir(parents=True, exist_ok=True)
    with FileLock(str(locks / f"{destination.name}.lock")):
        metadata = repository_metadata(repo)
        if metadata is not None:
            if metadata["revision"] != revision or metadata.get("allow_patterns") != allow_patterns:
                raise ValueError(f"Model source revision/download scope differs: {destination}")
            if deployment_digest(destination) != metadata["sha256"]:
                raise ValueError(f"Downloaded model files changed: {destination}")
            return destination
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=MODELS_ROOT) as temporary:
            staging = Path(temporary) / "model"
            snapshot_download(repo, revision=revision, local_dir=staging, allow_patterns=allow_patterns)
            atomic_json(staging / SOURCE_FILE, {"repo_id": repo, "revision": revision,
                        "allow_patterns": allow_patterns, "sha256": deployment_digest(staging)})
            staging.rename(destination)
    return destination


def configure_openpi_cache() -> None:
    # OpenPI's downloader also supplies the tokenizer during inference.
    os.environ["OPENPI_DATA_HOME"] = str(OPENPI_CACHE_ROOT)
