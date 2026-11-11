"""k31-dev image self-check. The image is all-in-one but source-free, so:
--build-time checks only the baked dependency floor (Dockerfile); a full run
additionally requires the cloned, editable-installed sglang and probes GPUs
plus the SM90 MXFP4/Marlin paths K3.1 needs.
"""

import argparse
import importlib
import sys


def _check(cond: bool, ok: str, fail: str) -> None:
    print(("PASS " if cond else "FAIL ") + (ok if cond else fail))
    if not cond:
        sys.exit(1)


def run_build_time() -> None:
    import importlib.util

    import torch

    # Source-free invariant: sglang arrives via the runtime clone and its
    # editable install; the baked floor must not contain it.
    _check(
        importlib.util.find_spec("sglang") is None,
        "sglang absent from the dependency floor",
        "sglang present at build; the base editable uninstall is broken",
    )
    importlib.import_module("pytest")
    from packaging.version import Version

    print(f"torch {torch.__version__} / cuda {torch.version.cuda}")
    _check(True, "dependency floor importable (torch, pytest)", "")
    # Aligned with the pyproject pin; a drift here means the Dockerfile's
    # jit-cache comment was not respected.
    import flashinfer

    _check(
        Version(flashinfer.__version__).major == 0 and Version(
            flashinfer.__version__
        ).minor
        == 7,
        f"flashinfer {flashinfer.__version__}",
        f"flashinfer {flashinfer.__version__} (expected 0.7.x pinned by python/pyproject.toml)",
    )
    import subprocess

    subprocess.run(
        ["/opt/sgl-eval-venv/bin/python", "-c", "import sgl_eval"],
        check=True,
    )
    _check(True, "sgl-eval venv (/opt/sgl-eval-venv) importable", "")


def run_runtime() -> None:
    import importlib.util

    import torch

    if importlib.util.find_spec("sglang") is None:
        _check(
            False,
            "",
            'sglang not installed; clone the fork and pip install -e "python[test]"',
        )
    _check(True, "sglang installed (clone + editable install)", "")
    run_build_time()
    n = torch.cuda.device_count()
    cap = torch.cuda.get_device_capability(0) if n else None
    _check(n > 0, f"{n} GPU(s) visible, sm={cap}", "no CUDA device visible")
    # SM90 MXFP4 candidates: Marlin (default on Hopper) and FlashInfer's
    # interleave helpers (opt-in cutlass_sm90 path, FlashInfer PR #3084).
    try:
        from flashinfer.fused_moe import (
            interleave_moe_scales_for_sm90_mixed_gemm as _s,
            interleave_moe_weights_for_sm90_mixed_gemm as _w,
        )

        _check(callable(_s) and callable(_w), "FlashInfer SM90 MXFP4 helpers", "")
    except (ImportError, AttributeError):
        print("WARN FlashInfer SM90 MXFP4 helpers missing (marlin still available)")
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (  # noqa: F401
        prepare_moe_mxfp4_layer_for_marlin,
    )

    _check(True, "Marlin fp4 weight-prep importable", "")
    import subprocess

    # Built by the editable install (setup.py auto-discovers the Rust workspace
    # when cargo is on PATH); a silent fallback only costs radix-cache perf.
    tree = subprocess.run(
        [sys.executable, "-c", "from sglang.srt.mem_cache.rust_tree_core import extension"],
        capture_output=True,
        text=True,
    )
    _check(
        tree.returncode == 0,
        "Rust TreeCore extension loads",
        f"Rust TreeCore missing:\n{tree.stderr[-800:]}\n"
        're-run pip install -e "python[test]" with cargo on PATH',
    )
    if n >= 2:
        cuda_p2p = torch.cuda.can_device_access_peer(0, 1)
        print(f"P2P GPU0<->GPU1: {cuda_p2p} (NVLink expected on the H200 node)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build-time", action="store_true")
    a = p.parse_args()
    (run_build_time if a.build_time else run_runtime)()
    print("dev_verify: OK")
