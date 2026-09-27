import base64
import os
import subprocess
import pytest
from app import app, validate_audio_type

@pytest.fixture(scope="session")
def test_opus_file(tmp_path_factory):
    fn = tmp_path_factory.mktemp("audio") / "test_opus.ogg"
    # Generate a 1-second sine wave in opus format
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:a", "libopus", str(fn)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(fn)

def test_validate_audio_type_mimetypes():
    # Explicit requirement: audio/ogg; codecs=opus
    valid, ext = validate_audio_type(content_type="audio/ogg; codecs=opus")
    assert valid is True
    assert ext == "ogg"

    valid, ext = validate_audio_type(content_type="audio/ogg;codecs=opus")
    assert valid is True

    valid, ext = validate_audio_type(content_type="audio/ogg")
    assert valid is True
    assert ext == "ogg"

    valid, ext = validate_audio_type(content_type="audio/opus")
    assert valid is True

    valid, ext = validate_audio_type(content_type="audio/wav")
    assert valid is True
    assert ext == "wav"

    valid, ext = validate_audio_type(content_type="audio/mpeg")
    assert valid is True
    assert ext == "mp3"

    # Invalid MIME types
    valid, _ = validate_audio_type(content_type="application/pdf")
    assert valid is False

    valid, _ = validate_audio_type(content_type="image/png")
    assert valid is False

    valid, _ = validate_audio_type(content_type="audio/ogg; codecs=unknown")
    assert valid is False

def test_validate_audio_type_extensions():
    assert validate_audio_type(filename="voice.ogg")[0] is True
    assert validate_audio_type(filename="voice.opus")[0] is True
    assert validate_audio_type(filename="voice.wav")[0] is True
    assert validate_audio_type(filename="voice.mp3")[0] is True
    assert validate_audio_type(filename="document.pdf")[0] is False

def test_validate_audio_type_magic_bytes():
    assert validate_audio_type(raw_bytes=b"OggS\x00\x02")[0] is True
    assert validate_audio_type(raw_bytes=b"%PDF-1.4")[0] is False

def test_health_endpoint():
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "online"
    assert "audio/ogg; codecs=opus" in data["supported_mimetypes"]

def test_transcribe_multipart_opus(test_opus_file):
    client = app.test_client()
    with open(test_opus_file, "rb") as f:
        res = client.post(
            "/api/transcribe",
            data={"file": (f, "sample.ogg", "audio/ogg; codecs=opus")},
            content_type="multipart/form-data"
        )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "text" in data
    assert "duration" in data
    assert "segments" in data

def test_transcribe_raw_binary_opus(test_opus_file):
    client = app.test_client()
    with open(test_opus_file, "rb") as f:
        audio_bytes = f.read()

    res = client.post(
        "/api/transcribe",
        data=audio_bytes,
        headers={"Content-Type": "audio/ogg; codecs=opus"}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "text" in data

def test_transcribe_json_base64(test_opus_file):
    client = app.test_client()
    with open(test_opus_file, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    res = client.post(
        "/api/transcribe",
        json={"audio": b64, "mimetype": "audio/ogg; codecs=opus"}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "text" in data

def test_transcribe_invalid_mimetype():
    client = app.test_client()
    res = client.post(
        "/api/transcribe",
        data=b"fake pdf content",
        headers={"Content-Type": "application/pdf"}
    )
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data


@pytest.fixture(scope="session")
def test_mkv_file(tmp_path_factory):
    fn = tmp_path_factory.mktemp("video") / "test_sample.mkv"
    # Generate a 1-second video with audio sine wave in MKV container
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
        "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-c:v", "libx264", "-c:a", "aac",
        "-shortest", str(fn)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(fn)


def test_validate_mkv():
    valid, ext = validate_audio_type(content_type="video/x-matroska")
    assert valid is True
    assert ext == "mkv"

    valid, ext = validate_audio_type(filename="meeting.mkv")
    assert valid is True
    assert ext == "mkv"

    valid, ext = validate_audio_type(raw_bytes=b"\x1a\x45\xdf\xa3\x93\x42")
    assert valid is True
    assert ext == "mkv"


def test_transcribe_mkv_multipart(test_mkv_file):
    client = app.test_client()
    with open(test_mkv_file, "rb") as f:
        res = client.post(
            "/api/transcribe",
            data={"file": (f, "test.mkv", "video/x-matroska")},
            content_type="multipart/form-data"
        )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "text" in data
    assert "duration" in data


def test_transcribe_local_file_endpoint(test_mkv_file):
    client = app.test_client()
    res = client.post(
        "/api/transcribe-file",
        json={"filepath": test_mkv_file, "sync": True}
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "text" in data
    assert "segments" in data


def test_request_with_retry_handles_429(monkeypatch):
    from unittest.mock import MagicMock
    from client import request_with_retry

    calls = 0
    def mock_request(method, url, **kwargs):
        nonlocal calls
        calls += 1
        resp = MagicMock()
        if calls == 1:
            resp.status_code = 429
            resp.headers = {"Retry-After": "0.01"}
            return resp
        resp.status_code = 200
        resp.json.return_value = {"status": "ok"}
        return resp

    import requests
    monkeypatch.setattr(requests, "request", mock_request)
    resp = request_with_retry("GET", "http://example.com/api/test", initial_backoff=0.01)
    assert calls == 2
    assert resp.status_code == 200


