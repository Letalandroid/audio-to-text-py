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
