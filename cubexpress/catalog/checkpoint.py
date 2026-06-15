"""checkpoint: save/resume progress for large multi-rt discovery runs.

Discovering 250k rts can take a long time; a crash midway would lose everything.
A checkpoint records, as it goes, which rts are already resolved (and their
results) to a JSONL file. Re-running with the same checkpoint path skips the
done rts and resumes with the rest.

Format: JSON Lines. The FIRST line is a header {"signature": ...} identifying
the rt list (so resuming with a different list is detected, not silently mixed).
Each subsequent line is one resolved rt: {"gid": int, "imgs": [...]}. Appending
a line per resolved rt means a crash loses at most the line being written.
"""

from __future__ import annotations

import hashlib
import json
import os

from cubexpress.geo.transform import RasterTransform


def rts_signature(rts: list[RasterTransform]) -> str:
    """A stable hash of the rt list, to detect resuming with a different list.

    Built from each rt's defining fields in order. If the user resumes a
    checkpoint with a different (or reordered) rt list, the signature won't
    match and we refuse to mix incompatible results.

    Args:
        rts: the rt list being discovered.

    Returns:
        A short hex signature.
    """
    h = hashlib.sha256()
    for rt in rts:
        key = f"{rt.crs}|{rt.translate_x}|{rt.translate_y}|{rt.scale_x}|{rt.scale_y}|{rt.width}|{rt.height}"
        h.update(key.encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def load_checkpoint(path: str, signature: str) -> dict[int, list[dict]]:
    """Load resolved rts from a checkpoint file, if it exists and matches.

    Args:
        path: checkpoint file path.
        signature: the current rt list's signature; must match the file's header.

    Returns:
        {gid: imgs} for rts already resolved. Empty dict if the file does not
        exist.

    Raises:
        ValueError: if the file exists but its signature does not match (i.e.
            the checkpoint was made for a different rt list).
    """
    if not os.path.exists(path):
        return {}

    resolved: dict[int, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if not first.strip():
            return {}  # empty file, start fresh
        header = json.loads(first)
        if header.get("signature") != signature:
            raise ValueError(
                f"checkpoint {path!r} was made for a different rt list "
                f"(signature {header.get('signature')!r} != {signature!r}). "
                f"Use a different checkpoint path, or delete the old file to "
                f"start fresh."
            )
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            resolved[rec["gid"]] = rec["imgs"]
    return resolved


def init_checkpoint(path: str, signature: str) -> None:
    """Create a fresh checkpoint file with the signature header.

    Only writes the header if the file does not already exist (so resuming does
    not clobber prior progress).

    Args:
        path: checkpoint file path.
        signature: the rt list's signature to record in the header.
    """
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"signature": signature}) + "\n")


def append_checkpoint(path: str, gid: int, imgs: list[dict]) -> None:
    """Append one resolved rt to the checkpoint file.

    Args:
        path: checkpoint file path (must already have a header).
        gid: the resolved rt's global index.
        imgs: its discovered images.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"gid": gid, "imgs": imgs}) + "\n")
