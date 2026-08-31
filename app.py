import io
import os
import subprocess
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_file

from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"ogg", "wav", "mp3"}

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", None)
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", 600))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()

model = None
model_lock = threading.Lock()


def get_model():
    global model
    with model_lock:
        if model is None:
            device = "cuda" if os.environ.get("WHISPER_DEVICE") == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_timestamp(seconds):
    millis = int(round(seconds * 1000))
    h, rem = divmod(millis, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{m:02d}:{s:02d},{ms:03d}"


def build_txt(segments):
    lines = []
    for seg in segments:
        start = format_timestamp(seg["start"])
        lines.append(f"[{start}] {seg['text'].strip()}")
    return "\n\n".join(lines)


def get_duration(filepath):
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def update_job(job_id, **kwargs):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job.update(kwargs)


def transcribe_job(job_id, filepath):
    try:
        update_job(job_id, status="processing", progress=0)

        total_duration = get_duration(filepath)
        whisper = get_model()
        kwargs = {}
        if WHISPER_LANGUAGE:
            kwargs["language"] = WHISPER_LANGUAGE

        segments_iter, _info = whisper.transcribe(
            filepath, chunk_length=10, vad_filter=True, **kwargs
        )

        segments = []
        last_end = 0.0
        for seg in segments_iter:
            segments.append(
                {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            )
            last_end = max(last_end, seg.end)
            if total_duration:
                pct = min(99.0, last_end / total_duration * 100)
                update_job(job_id, progress=round(pct, 1))

        txt = build_txt(segments)

        update_job(
            job_id,
            status="done",
            progress=100,
            segments=segments,
            txt=txt,
        )
    except Exception as exc:
        update_job(job_id, status="error", error=str(exc))
    finally:
        try:
            os.remove(filepath)
        except OSError:
            pass


def janitor():
    while True:
        time.sleep(30)
        now = time.time()
        with jobs_lock:
            expired = [
                job_id
                for job_id, job in jobs.items()
                if now - job["created"] > JOB_TTL_SECONDS
            ]
            for job_id in expired:
                jobs.pop(job_id, None)


threading.Thread(target=janitor, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No se envió ningún archivo"}), 400
    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "error": "Formato no permitido. Usa archivos .ogg, .wav o .mp3."
                }
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]
    ext = file.filename.rsplit(".", 1)[1].lower()
    audio_path = os.path.join(UPLOAD_FOLDER, f"{job_id}.{ext}")

    file.save(audio_path)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "status": "queued",
            "progress": 0,
            "error": None,
            "segments": None,
            "txt": None,
            "created": time.time(),
        }

    thread = threading.Thread(
        target=transcribe_job, args=(job_id, audio_path), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.get("/api/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Trabajo no encontrado"}), 404
        return jsonify({k: v for k, v in job.items()})


@app.get("/api/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job["status"] != "done":
            return jsonify({"error": "Transcripción no disponible"}), 404
        filename = job["filename"]
        txt = job["txt"]

    base = os.path.splitext(filename)[0]
    return send_file(
        io.BytesIO(txt.encode("utf-8")),
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name=f"{base}.txt",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
