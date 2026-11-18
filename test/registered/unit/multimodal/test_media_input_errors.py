"""Client media problems surface as ValueError (HTTP 400), not server errors."""

import io
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import pybase64
import requests
from PIL import Image

from sglang.srt.environ import envs
from sglang.srt.multimodal.cache.identity import snapshot_media
from sglang.srt.multimodal.media_artifacts.base import MediaArtifactCacheMixin
from sglang.srt.multimodal.media_io import (
    CLIENT_MEDIA_EXCEPTIONS,
    configure_media_url_security,
    get_image_bytes,
    get_video_bytes,
    load_audio,
    load_image,
    load_video,
    media_url_timeout,
    read_local_media_file,
)
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin
from sglang.srt.multimodal.processors.mimo_v2 import MiMoProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _image_bytes(fmt: str, size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 10, 10)).save(buf, format=fmt)
    return buf.getvalue()


class _KimiStub(KimiGridMMDataMixin, BaseMultimodalProcessor):
    gpu_image_decode = False
    uses_wide_image_identity = True


class _PolicyTestCase(CustomTestCase):
    def setUp(self):
        self.addCleanup(configure_media_url_security, [], max_file_size_mb=64)


class TestLegacyMediaURLTimeout(_PolicyTestCase):
    def test_default_is_five_seconds(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_MEDIA_URL_TIMEOUT", None)
            os.environ.pop("REQUEST_TIMEOUT", None)
            self.assertEqual(envs.SGLANG_MEDIA_URL_TIMEOUT.get(), 5.0)

    def test_legacy_request_timeout_is_honored(self):
        with patch.dict(os.environ, {"REQUEST_TIMEOUT": "12"}):
            os.environ.pop("SGLANG_MEDIA_URL_TIMEOUT", None)
            self.assertEqual(envs.SGLANG_MEDIA_URL_TIMEOUT.get(), 12.0)

    def test_new_variable_wins(self):
        with patch.dict(
            os.environ, {"REQUEST_TIMEOUT": "12", "SGLANG_MEDIA_URL_TIMEOUT": "30.5"}
        ):
            self.assertEqual(envs.SGLANG_MEDIA_URL_TIMEOUT.get(), 30.5)

    def test_image_download_uses_it(self):
        with (
            patch.dict(os.environ, {"SGLANG_MEDIA_URL_TIMEOUT": "7"}),
            patch(
                "sglang.srt.multimodal.media_io.download_remote_media",
                return_value=b"x",
            ) as download,
        ):
            get_image_bytes("https://example.com/a.png")
        download.assert_called_once_with("https://example.com/a.png", timeout=7.0)


class TestLegacyLocalMediaFiles(_PolicyTestCase):
    def setUp(self):
        super().setUp()
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "a.png")
        with open(self.path, "wb") as f:
            f.write(_image_bytes("PNG"))

    def test_reads_regular_file(self):
        self.assertEqual(get_image_bytes(self.path), _image_bytes("PNG"))
        self.assertEqual(get_image_bytes("file://" + self.path), _image_bytes("PNG"))

    def test_missing_file_is_client_error(self):
        with self.assertRaises(ValueError):
            read_local_media_file(os.path.join(self.dir.name, "missing.png"))

    def test_non_regular_file_is_rejected(self):
        with self.assertRaises(ValueError):
            read_local_media_file(self.dir.name)
        if os.path.exists("/dev/zero"):
            with self.assertRaises(ValueError):
                get_image_bytes("file:///dev/zero")

    def test_can_be_disabled(self):
        with patch.dict(os.environ, {"SGLANG_MEDIA_ALLOW_LOCAL_FILES": "0"}):
            with self.assertRaises(ValueError):
                get_image_bytes(self.path)
            with self.assertRaises(ValueError):
                snapshot_media(self.path)
            with self.assertRaisesRegex(ValueError, "not allowed"):
                snapshot_media("file://" + self.path)

    def test_raw_base64_starting_with_slash_is_not_a_path(self):
        jpeg = _image_bytes("JPEG")
        encoded = pybase64.b64encode(jpeg).decode()
        self.assertTrue(encoded.startswith("/9j/"))
        self.assertEqual(get_image_bytes(encoded), jpeg)


class TestMediaURLTimeout(_PolicyTestCase):
    def test_server_arg_wins_over_request_timeout(self):
        with patch.dict(os.environ, {"REQUEST_TIMEOUT": "12"}):
            configure_media_url_security([], timeout=None)
            self.assertEqual(media_url_timeout(3), 12.0)
            configure_media_url_security([], timeout=30.5)
            self.assertEqual(media_url_timeout(3), 30.5)

    def test_every_loader_uses_it(self):
        configure_media_url_security([], timeout=7)
        url = "https://example.com/a.png"
        for target, call in (
            ("sglang.srt.multimodal.media_io", lambda: get_image_bytes(url)),
            ("sglang.srt.multimodal.media_io", lambda: get_video_bytes(url)),
            (
                "sglang.srt.multimodal.processors.mimo_v2",
                lambda: MiMoProcessor.fetch_image(url),
            ),
        ):
            with (
                self.subTest(target=target),
                patch(
                    f"{target}.download_remote_media",
                    return_value=_image_bytes("PNG"),
                ) as download,
            ):
                call()
            download.assert_called_once_with(url, timeout=7.0)


class TestLocalMediaFiles(_PolicyTestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.path = os.path.join(self.dir, "a.png")
        with open(self.path, "wb") as f:
            f.write(_image_bytes("PNG"))
        self.missing = os.path.join(self.dir, "missing.png")
        self.fifo = os.path.join(self.dir, "fifo.png")
        os.mkfifo(self.fifo)
        self.addCleanup(self._release_fifo_readers)

    def _release_fifo_readers(self):
        # A regression blocks a reader in open(); a writer lets it finish.
        try:
            os.close(os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK))
        except OSError:
            pass

    def _call(self, fn, *args, timeout=5):
        outcome = {}

        def run():
            try:
                outcome["value"] = fn(*args)
            except BaseException as e:
                outcome["error"] = e

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout)
        self.assertFalse(thread.is_alive(), f"{fn} did not return in {timeout}s")
        if "error" in outcome:
            raise outcome["error"]
        return outcome["value"]

    def test_fifo_is_rejected_without_blocking(self):
        image, _ = load_image("file://" + self.path, gpu_image_decode=False)
        self.assertEqual(image.size, (8, 8))
        for fn, value in (
            (get_image_bytes, self.fifo),
            (load_image, "file://" + self.fifo),
            (snapshot_media, self.fifo),
            (load_video, self.fifo),
            (load_audio, "file://" + self.fifo),
        ):
            with (
                self.subTest(fn=fn, value=value),
                self.assertRaisesRegex(ValueError, "not a regular file"),
            ):
                self._call(fn, value)

    @unittest.skipUnless(os.path.exists("/dev/zero"), "needs /dev/zero")
    def test_device_and_directory_are_rejected(self):
        for fn, value in (
            (get_video_bytes, "file:///dev/zero"),
            (load_audio, "/dev/zero"),
            (load_video, self.dir),
            (load_audio, self.dir),
            (lambda v: _KimiStub._load_single_item(v, Modality.IMAGE), self.dir),
        ):
            with (
                self.subTest(fn=fn, value=value),
                self.assertRaisesRegex(ValueError, "not a regular file|Is a directory"),
            ):
                self._call(fn, value)

    def test_size_is_capped(self):
        configure_media_url_security([], max_file_size_mb=1)
        large = os.path.join(self.dir, "large.mp4")
        with open(large, "wb") as f:
            f.write(b"\0" * (1024 * 1024 + 1))
        for fn in (get_image_bytes, load_video):
            with (
                self.subTest(fn=fn),
                self.assertRaisesRegex(ValueError, "byte limit"),
            ):
                fn(large)

    def test_missing_absolute_path_says_not_found(self):
        for fn in (get_image_bytes, load_video, load_audio):
            with (
                self.subTest(fn=fn),
                self.assertRaisesRegex(ValueError, "(?i)no such file"),
            ):
                fn(self.missing)

    def test_disallowed_rejects_every_path_alike(self):
        configure_media_url_security([], allow_local_files=False)
        cwd = os.getcwd()
        os.chdir(self.dir)
        self.addCleanup(os.chdir, cwd)
        loaders = (
            get_image_bytes,
            snapshot_media,
            lambda v: _KimiStub._load_single_item(v, Modality.IMAGE),
            load_video,
            get_video_bytes,
            load_audio,
        )
        # Existing and missing files fail alike, so the error is no oracle.
        for value in (
            self.path,
            self.missing,
            "file://" + self.path,
            "file://" + self.missing,
            "a.png",
            "missing.png",
        ):
            for fn in loaders:
                if fn in (get_image_bytes, get_video_bytes, load_video) and (
                    value in ("a.png", "missing.png")
                ):
                    continue  # relative names without file:// are base64 here
                with (
                    self.subTest(fn=fn, value=value),
                    self.assertRaisesRegex(ValueError, "not allowed"),
                ):
                    fn(value)

    def test_raw_base64_starting_with_slash_is_not_a_path(self):
        jpeg = _image_bytes("JPEG")
        encoded = pybase64.b64encode(jpeg).decode()
        self.assertTrue(encoded.startswith("/9j/"))
        self.assertEqual(get_image_bytes(encoded), jpeg)


class TestLegacyImageDecodeErrors(CustomTestCase):
    def test_decompression_bomb_is_client_error(self):
        self.assertIn(Image.DecompressionBombError, CLIENT_MEDIA_EXCEPTIONS)
        data = _image_bytes("PNG", size=(64, 64))
        with patch.object(Image, "MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(ValueError):
                load_image(data, gpu_image_decode=False)

    def test_common_formats_decode(self):
        for fmt in ("PNG", "JPEG", "GIF", "TIFF", "BMP", "WEBP"):
            with self.subTest(fmt=fmt):
                image, _ = load_image(_image_bytes(fmt), gpu_image_decode=False)
                self.assertEqual(image.size, (8, 8))

    def test_heic_decodes(self):
        try:
            import pillow_heif
        except ImportError:
            self.skipTest("pillow-heif is not installed")
        heif = pillow_heif.from_pillow(Image.new("RGB", (8, 8), (0, 200, 0)))
        buf = io.BytesIO()
        heif.save(buf, format="HEIF")
        image, _ = load_image(buf.getvalue(), gpu_image_decode=False)
        self.assertEqual(image.size, (8, 8))


class TestSnapshotMediaErrors(CustomTestCase):
    def test_download_failures_are_client_errors(self):
        for error in (
            requests.exceptions.Timeout("slow"),
            requests.exceptions.ConnectionError("down"),
            requests.exceptions.HTTPError("404"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch(
                    "sglang.srt.multimodal.media_io.get_image_bytes",
                    side_effect=error,
                ),
            ):
                with self.assertRaises(ValueError):
                    snapshot_media("https://example.com/" + "a" * 200)


class TestImageDecodeErrors(CustomTestCase):
    def test_decompression_bomb_on_the_artifact_path(self):
        processor = MediaArtifactCacheMixin()
        processor.gpu_image_decode = False
        snapshot = snapshot_media(_image_bytes("PNG", size=(64, 64)))
        with (
            patch.object(Image, "MAX_IMAGE_PIXELS", 100),
            self.assertRaises(ValueError) as ctx,
        ):
            processor.decode_media_snapshot(snapshot, Modality.IMAGE)
        self.assertIsInstance(ctx.exception.__cause__, Image.DecompressionBombError)


class TestKimiURLErrors(CustomTestCase):
    def test_download_failure_is_a_client_error(self):
        with (
            patch(
                "sglang.srt.multimodal.media_io.get_image_bytes",
                side_effect=requests.exceptions.Timeout("slow"),
            ),
            self.assertRaises(ValueError),
        ):
            _KimiStub._load_single_item("https://example.com/a.png", Modality.IMAGE)


if __name__ == "__main__":
    unittest.main()
