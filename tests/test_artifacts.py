import hashlib
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from vcc_h1_eval.artifacts import download, install_asset, sha256, verify_asset
from vcc_h1_eval.paths import BenchmarkPaths


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    requests = 0

    def do_GET(self):
        type(self).requests += 1
        if self.path == "/fail":
            self.send_response(503)
            self.end_headers()
            return
        start = 0
        status = 200
        if value := self.headers.get("Range"):
            start = int(value.removeprefix("bytes=").removesuffix("-"))
            status = 206
        body = self.payload[start:]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        if status == 206:
            self.send_header(
                "Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}"
            )
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@pytest.fixture
def range_server():
    RangeHandler.payload = bytes(range(256)) * 4096
    RangeHandler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join()


def source_spec(server, payload: bytes) -> dict:
    return {
        "url": f"http://127.0.0.1:{server.server_port}/artifact",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_download_resumes_and_reuses_verified_file(tmp_path: Path, range_server) -> None:
    destination = tmp_path / "artifact.bin"
    partial = tmp_path / "artifact.bin.part"
    partial.write_bytes(RangeHandler.payload[:12345])
    spec = source_spec(range_server, RangeHandler.payload)

    download(spec, destination)
    assert destination.read_bytes() == RangeHandler.payload
    assert not partial.exists()
    requests = RangeHandler.requests

    download(spec, destination)
    assert RangeHandler.requests == requests


def test_download_rejects_wrong_checksum(tmp_path: Path, range_server) -> None:
    spec = source_spec(range_server, RangeHandler.payload)
    spec["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        download(spec, tmp_path / "artifact.bin")


def test_download_surfaces_http_failure(tmp_path: Path, range_server) -> None:
    spec = source_spec(range_server, RangeHandler.payload)
    spec["url"] = f"http://127.0.0.1:{range_server.server_port}/fail"
    with pytest.raises(httpx.HTTPStatusError):
        download(spec, tmp_path / "artifact.bin")


def test_asset_installs_portably_and_rejects_escape(tmp_path: Path) -> None:
    payload = b"reference"
    archive = tmp_path / "asset.zip"
    manifest = {
        "benchmark_version": "test",
        "files": {"reference_cache/value.bin": hashlib.sha256(payload).hexdigest()},
    }
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("reference_cache/value.bin", payload)
        bundle.writestr("asset_manifest.json", json.dumps(manifest))

    paths = BenchmarkPaths.resolve(tmp_path / "data")
    paths.root.mkdir()
    install_asset(paths, archive)
    verify_asset(paths.benchmark_dir)
    assert sha256(paths.benchmark_dir / "reference_cache/value.bin") == manifest["files"][
        "reference_cache/value.bin"
    ]

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as bundle:
        bundle.writestr("../outside", b"bad")
        bundle.writestr("asset_manifest.json", json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="unsafe archive member"):
        install_asset(paths, bad)
