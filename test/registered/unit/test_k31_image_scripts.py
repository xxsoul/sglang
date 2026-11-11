"""k31-image build scripts must resolve env-authored URLs past empty ARGs.

BuildKit injects an unset declared --build-arg into the RUN environment as an
empty string, so os.environ.get(name, default) returns "" and the image build
died with ValueError: unknown url type: '' (seen 2026-11-11 at the Prims-TS
overlay). The resolvers must fall back on truthiness for empty strings while
still honoring an explicit mirror URL.
"""

import importlib.util
import unittest
from pathlib import Path

K31_IMAGE = Path(__file__).resolve().parents[3] / "docker" / "k31-image"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, K31_IMAGE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestK31ImageEnvFallbacks(unittest.TestCase):
    def test_resolve_archive_url_fallbacks(self):
        resolve = _load_script("overlay_flashinfer_prims_ts").resolve_archive_url
        self.assertEqual(
            resolve({"K31_FLASHINFER_ARCHIVE_URL": "https://mirror/x.tar.gz"}),
            "https://mirror/x.tar.gz",
        )
        for env in ({}, {"K31_FLASHINFER_ARCHIVE_URL": ""}):
            with self.subTest(env=env):
                self.assertEqual(
                    resolve(env),
                    "https://codeload.github.com/flashinfer-ai/flashinfer/"
                    "tar.gz/784bfaf5bda80259d11a41955ab8c979914197f1",
                )

    def test_resolve_rustup_init_url_fallbacks(self):
        resolve = _load_script("build_dynamo").resolve_rustup_init_url
        self.assertEqual(
            resolve({"K31_RUSTUP_INIT_URL": "https://rsproxy.cn/rustup-init.sh"}),
            "https://rsproxy.cn/rustup-init.sh",
        )
        for env in ({}, {"K31_RUSTUP_INIT_URL": ""}):
            with self.subTest(env=env):
                self.assertEqual(resolve(env), "https://sh.rustup.rs")

    def test_resolve_dynamo_repo_fallbacks(self):
        resolve = _load_script("build_dynamo").resolve_dynamo_repo
        self.assertEqual(
            resolve({"K31_DYNAMO_REPO": "https://ghproxy.net/https://x/dynamo"}),
            "https://ghproxy.net/https://x/dynamo",
        )
        for env in ({}, {"K31_DYNAMO_REPO": ""}):
            with self.subTest(env=env):
                self.assertEqual(
                    resolve(env), "https://github.com/ai-dynamo/dynamo"
                )


if __name__ == "__main__":
    unittest.main()
