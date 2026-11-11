"""Overlay FlashInfer main's Prims-TS sparse MLA onto the image's FlashInfer 0.7.2/0.7.3.

The opt-in `prims_ts` DSA backend needs FlashInfer's `flashinfer/attention/prims_ts/` from main
784bfaf5 (flashinfer-ai/flashinfer #7410 and #7984; #8186 for the SM107 sparse MLA); the 0.7.2
line ships an older copy that fails at startup, and v0.7.3 does not contain these updates either
(24 release-side commits over 0.7.2, verified by blob-sha comparison of the prims_ts tree). This
replaces only that directory's Python files (CuTe DSL; no csrc, AOT or jit-cache module changes).
Everything else in FlashInfer, including the DynBlock-rebuilt modules, stays at its release.

Idempotent: a marker records the overlaid tree, so a rerun is a no-op unless a reinstall
replaced the files. Needs network access only when it overlays; K31_FLASHINFER_ARCHIVE_URL
overrides the download for throttled networks.
"""

import hashlib
import io
import json
import os
import sysconfig
import tarfile
import urllib.request
from importlib.metadata import version
from pathlib import Path

FLASHINFER_COMMIT = "784bfaf5bda80259d11a41955ab8c979914197f1"
# 0.7.3 carries only release-side commits over 0.7.2; the supporting modules the
# overlay touches are unchanged between the two.
FLASHINFER_BASE_VERSIONS = frozenset({"0.7.2", "0.7.3"})
SUBDIR = "flashinfer/attention/prims_ts"

SITE = Path(sysconfig.get_paths()["purelib"])
TARGET = SITE / SUBDIR
MARKER = TARGET / "k31-prims-ts.json"


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    installed = version("flashinfer-python")
    if installed not in FLASHINFER_BASE_VERSIONS:
        raise SystemExit(
            f"flashinfer-python {installed} is installed; the Prims-TS overlay is validated on "
            f"{sorted(FLASHINFER_BASE_VERSIONS)} only"
        )
    if MARKER.exists():
        marker = json.loads(MARKER.read_text())
        if marker.get("commit") == FLASHINFER_COMMIT and marker.get(
            "tree_sha256"
        ) == tree_sha256(TARGET):
            print(f"Prims-TS already overlaid from flashinfer {FLASHINFER_COMMIT[:8]}")
            return

    # codeload serves the whole-repo tarball; mirror via K31_FLASHINFER_ARCHIVE_URL
    # when github is throttled (any URL returning the same tar.gz works).
    url = os.environ.get(
        "K31_FLASHINFER_ARCHIVE_URL",
        f"https://codeload.github.com/flashinfer-ai/flashinfer/tar.gz/{FLASHINFER_COMMIT}",
    )
    with urllib.request.urlopen(url, timeout=600) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz")
    prefix = f"flashinfer-{FLASHINFER_COMMIT}/{SUBDIR}/"
    members = [
        m
        for m in archive.getmembers()
        if m.isfile() and m.name.startswith(prefix) and m.name.endswith(".py")
    ]
    if not members:
        raise SystemExit(f"{SUBDIR} not found in flashinfer {FLASHINFER_COMMIT}")

    for path in TARGET.rglob("*.py"):
        path.unlink()
    for member in members:
        dest = TARGET / member.name[len(prefix) :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.extractfile(member).read())

    MARKER.write_text(
        json.dumps({"commit": FLASHINFER_COMMIT, "tree_sha256": tree_sha256(TARGET)})
    )
    print(
        f"overlaid {len(members)} Prims-TS files from flashinfer {FLASHINFER_COMMIT[:8]}"
    )


if __name__ == "__main__":
    main()
