"""Load every base component from one immutable HF snapshot without mutating it."""
from pathlib import Path
import shutil
import tempfile

from .catalog import PolicyEntry, checkpoint_digest


def checkpoint_view(entry: PolicyEntry):
    if not entry.base_revision:
        raise ValueError("Base checkpoint revision is unverified; select the model again")
    source = Path(entry.base_checkpoint).expanduser()
    local = source.is_dir()
    if not local:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(entry.base_checkpoint, revision=entry.base_revision))
    holder = tempfile.TemporaryDirectory(prefix="liberox-checkpoint-")
    destination = Path(holder.name)
    try:
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            if local or path.suffix in {".json", ".py"}:
                # The upstream loader rewrites auto_map and syncs Python code.
                shutil.copyfile(path, target)
            else:
                target.symlink_to(path.resolve())
        if local and checkpoint_digest(destination) != entry.base_revision:
            raise ValueError("Local base checkpoint changed while loading; select the model again")
        return holder
    except BaseException:
        holder.cleanup()
        raise
