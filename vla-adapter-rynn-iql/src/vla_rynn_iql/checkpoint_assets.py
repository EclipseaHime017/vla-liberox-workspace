"""Immutable VLA base resolution shared by training and simulation."""
from pathlib import Path
import re
import shutil
import tempfile

from .io import sha256_file, stable_hash
from .model_storage import download_repository, repository_metadata


def checkpoint_digest(root: Path, hash_file=sha256_file) -> str:
    return stable_hash({str(path.relative_to(root)): hash_file(path)
        for path in sorted(root.rglob("*")) if path.is_file()
        and path.suffix in {".json", ".safetensors", ".pt", ".bin", ".py"}})


def resolve_revision(checkpoint: str, revision: str | None = None, hash_file=sha256_file) -> str:
    local = Path(checkpoint).expanduser()
    if local.is_dir():
        actual = checkpoint_digest(local, hash_file)
        if revision is not None and actual != revision:
            raise ValueError("Local base checkpoint changed; select the model again")
        return actual
    if checkpoint.startswith(("/", "./", "../", "~/")):
        raise FileNotFoundError(f"Base checkpoint directory is missing: {checkpoint}")
    if revision is None:
        from huggingface_hub import HfApi
        metadata = repository_metadata(checkpoint)
        revision = metadata["revision"] if metadata else HfApi().model_info(checkpoint, timeout=10).sha
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Cannot resolve an immutable base checkpoint revision")
    return revision


def checkpoint_view(checkpoint: str, revision: str):
    if not revision:
        raise ValueError("Base checkpoint revision is unverified; select the model again")
    source = Path(checkpoint).expanduser()
    local = source.is_dir()
    if not local:
        resolve_revision(checkpoint, revision)
        source = download_repository(checkpoint, revision)
    holder = tempfile.TemporaryDirectory(prefix="liberox-checkpoint-")
    destination = Path(holder.name)
    try:
        for path in source.rglob("*"):
            if not path.is_file() or (not local and any(part.startswith(".") for part in path.relative_to(source).parts)):
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Upstream rewrites config/code. Never let it modify base assets.
            if local or path.suffix in {".json", ".py"}:
                shutil.copyfile(path, target)
            else:
                target.symlink_to(path.resolve())
        if local and checkpoint_digest(destination) != revision:
            raise ValueError("Local base checkpoint changed while loading; select the model again")
        return holder
    except BaseException:
        holder.cleanup()
        raise
