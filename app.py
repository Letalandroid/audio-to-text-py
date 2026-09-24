import base64
import io
import os
import subprocess
import threading
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from faster_whisper import WhisperModel

# Cargar variables de entorno desde .env si existe
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

# Configuración mediante variables de entorno
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", None)
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", 600))
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", 200))

# Formatos y MIME Types soportados
ALLOWED_EXTENSIONS = {"ogg", "opus", "wav", "mp3"}

SUPPORTED_MIME_TYPES = [
    "audio/ogg; codecs=opus",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "video/ogg",
    "application/ogg",
]

ALLOWED_BASE_MIMES = {
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "video/ogg",
    "application/ogg",
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024

jobs = {}
jobs_lock = threading.Lock()

model = None
model_lock = threading.Lock()


def get_model():
    """Inicializa y cachea el modelo WhisperModel en memoria."""
    global model
    with model_lock:
        if model is None:
            model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        return model


def parse_content_type(ct: str):
    """Separa el base MIME type y sus parámetros (ej. codecs=opus)."""
    if not ct:
        return "", {}
    parts = [p.strip() for p in ct.split(";")]
    base = parts[0].lower()
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().lower()] = v.strip().lower().strip("\"'")
    return base, params


def validate_audio_type(content_type: str = "", filename: str = "", raw_bytes: bytes = None) -> tuple[bool, str]:
    """
    Valida si el content_type, extensión de archivo o magic bytes son soportados.
    Soporta explícitamente:
    - MIME Type: 'audio/ogg; codecs=opus'
    - MIME Type: 'audio/ogg', 'audio/opus', 'audio/wav', 'audio/mpeg'
    - Extensiones: .ogg, .opus, .wav, .mp3
    - Magic bytes: OggS, RIFF WAVE, ID3 / MP3 sync
    Retorna (es_valido: bool, extension_resuelta: str).
    """
    # 1. Validación por Content-Type
    if content_type:
        base, params = parse_content_type(content_type)
        if base in ALLOWED_BASE_MIMES:
            if "codecs" in params:
                codecs = params["codecs"]
                if "opus" in codecs or "vorbis" in codecs:
                    return True, "ogg"
                elif base == "audio/ogg":
                    return False, ""
            if "opus" in base or "ogg" in base:
                return True, "ogg"
            elif "wav" in base:
                return True, "wav"
            elif "mpeg" in base or "mp3" in base:
                return True, "mp3"
            return True, "ogg"

    # 2. Validación por extensión de archivo
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return True, "ogg" if ext == "opus" else ext

    # 3. Validación por magic bytes (inspección de cabecera binaria)
    if raw_bytes and len(raw_bytes) >= 4:
        if raw_bytes[:4] == b"OggS":
            return True, "ogg"
        if raw_bytes[:4] == b"RIFF" and len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WAVE":
            return True, "wav"
        if raw_bytes[:3] == b"ID3" or raw_bytes[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return True, "mp3"

    return False, ""


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


def transcribe_audio_file(filepath: str, language: str = None) -> dict:
    """
    Transcribe un archivo de audio de forma síncrona usando faster-whisper.
    Retorna diccionario con texto, segmentos, idioma detectado y duración.
    """
    total_duration = get_duration(filepath)
    whisper = get_model()
    kwargs = {}
    lang = language or WHISPER_LANGUAGE
    if lang:
        kwargs["language"] = lang

    segments_iter, info = whisper.transcribe(
        filepath, chunk_length=10, vad_filter=True, **kwargs
    )

    segments = []
    text_parts = []
    for seg in segments_iter:
        clean_text = seg.text.strip()
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": clean_text,
        })
        if clean_text:
            text_parts.append(clean_text)

    full_text = " ".join(text_parts).strip()
    detected_lang = getattr(info, "language", lang)
    duration = getattr(info, "duration", total_duration)

    return {
        "text": full_text,
        "segments": segments,
        "language": detected_lang,
        "duration": round(duration, 2) if duration is not None else total_duration,
    }


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


@app.get("/health")
@app.get("/api/health")
@app.get("/api/info")
def health():
    return jsonify({
        "status": "online",
        "service": "audio-to-text",
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
        "language": WHISPER_LANGUAGE,
        "supported_mimetypes": [
            "audio/ogg; codecs=opus",
            "audio/ogg",
            "audio/opus",
            "audio/wav",
            "audio/mpeg",
        ],
        "supported_extensions": [".ogg", ".opus", ".wav", ".mp3"],
    })


@app.post("/api/transcribe")
def transcribe_sync():
    """
    Endpoint directo para peticiones cURL / API:
    Recibe el audio y devuelve inmediatamente la transcripción en JSON.
    Soporta:
    1. multipart/form-data: campo 'file' o 'audio'
    2. Raw binary stream con Content-Type: 'audio/ogg; codecs=opus', 'audio/ogg', etc.
    3. JSON con audio en base64: {"audio": "<b64>", "mimetype": "audio/ogg; codecs=opus"}
    """
    req_ct = request.content_type or ""
    language = request.args.get("language") or request.form.get("language") or os.environ.get("WHISPER_LANGUAGE")
    temp_path = None

    try:
        # Caso 1: multipart/form-data
        if "multipart/form-data" in req_ct:
            file = request.files.get("file") or request.files.get("audio")
            if file is None or file.filename == "":
                return jsonify({"error": "No se envió ningún archivo. Usa el campo 'file' o 'audio'."}), 400

            file_mimetype = file.content_type or ""
            is_valid, ext = validate_audio_type(content_type=file_mimetype, filename=file.filename)

            # Si no se validó por cabecera o extensión, inspeccionar magic bytes
            sample_bytes = file.read(64)
            file.seek(0)
            if not is_valid:
                is_valid, ext = validate_audio_type(content_type=file_mimetype, filename=file.filename, raw_bytes=sample_bytes)

            if not is_valid:
                return jsonify({
                    "error": "Formato de audio no soportado.",
                    "detail": f"Tipo recibido: '{file_mimetype}', archivo: '{file.filename}'.",
                    "supported_mimetypes": [
                        "audio/ogg; codecs=opus",
                        "audio/ogg",
                        "audio/opus",
                        "audio/wav",
                        "audio/mpeg",
                    ],
                    "supported_extensions": [".ogg", ".opus", ".wav", ".mp3"],
                }), 400

            job_id = uuid.uuid4().hex[:12]
            temp_path = os.path.join(UPLOAD_FOLDER, f"sync_{job_id}.{ext}")
            file.save(temp_path)

        # Caso 2: application/json con base64
        elif "application/json" in req_ct:
            data = request.get_json(silent=True) or {}
            audio_b64 = data.get("audio") or data.get("data") or data.get("file")
            if not audio_b64:
                return jsonify({"error": "Falta el campo 'audio' o 'data' con contenido base64."}), 400

            audio_mimetype = data.get("mimetype") or data.get("content_type") or ""
            if "," in audio_b64 and audio_b64.startswith("data:"):
                prefix, audio_b64 = audio_b64.split(",", 1)
                if not audio_mimetype and ":" in prefix and ";" in prefix:
                    audio_mimetype = prefix.split(":", 1)[1].split(";", 1)[0]

            try:
                raw_bytes = base64.b64decode(audio_b64)
            except Exception as e:
                return jsonify({"error": f"Error decodificando base64: {str(e)}"}), 400

            is_valid, ext = validate_audio_type(content_type=audio_mimetype, raw_bytes=raw_bytes)
            if not is_valid:
                return jsonify({
                    "error": "Formato de audio no soportado.",
                    "detail": f"Tipo recibido: '{audio_mimetype}'.",
                    "supported_mimetypes": [
                        "audio/ogg; codecs=opus",
                        "audio/ogg",
                        "audio/opus",
                        "audio/wav",
                        "audio/mpeg",
                    ],
                    "supported_extensions": [".ogg", ".opus", ".wav", ".mp3"],
                }), 400

            job_id = uuid.uuid4().hex[:12]
            temp_path = os.path.join(UPLOAD_FOLDER, f"sync_{job_id}.{ext}")
            with open(temp_path, "wb") as f:
                f.write(raw_bytes)

        # Caso 3: Raw binary stream en el body
        else:
            raw_bytes = request.get_data()
            if not raw_bytes:
                return jsonify({"error": "Cuerpo de solicitud vacío. Envía audio en form-data, JSON base64 o binario directo."}), 400

            is_valid, ext = validate_audio_type(content_type=req_ct, raw_bytes=raw_bytes)
            if not is_valid:
                return jsonify({
                    "error": "Formato de audio no soportado.",
                    "detail": f"Content-Type recibido: '{req_ct}'.",
                    "supported_mimetypes": [
                        "audio/ogg; codecs=opus",
                        "audio/ogg",
                        "audio/opus",
                        "audio/wav",
                        "audio/mpeg",
                    ],
                    "supported_extensions": [".ogg", ".opus", ".wav", ".mp3"],
                }), 400

            job_id = uuid.uuid4().hex[:12]
            temp_path = os.path.join(UPLOAD_FOLDER, f"sync_{job_id}.{ext}")
            with open(temp_path, "wb") as f:
                f.write(raw_bytes)

        # Transcripción síncrona
        result = transcribe_audio_file(temp_path, language=language)
        return jsonify({
            "status": "success",
            "text": result["text"],
            "language": result["language"],
            "duration": result["duration"],
            "segments": result["segments"],
        }), 200

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.post("/api/upload")
def upload():
    """
    Endpoint para carga asíncrona (usado por la interfaz web) o síncrona si sync=true.
    """
    file = request.files.get("file") or request.files.get("audio")
    if file is None or file.filename == "":
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file_mimetype = file.content_type or ""
    sample_bytes = file.read(64)
    file.seek(0)
    is_valid, ext = validate_audio_type(content_type=file_mimetype, filename=file.filename, raw_bytes=sample_bytes)

    if not is_valid:
        return (
            jsonify({
                "error": "Formato no permitido. Usa archivos .ogg, .opus, .wav o .mp3 (incluyendo audio/ogg; codecs=opus)."
            }),
            400,
        )

    # Si se pide síncrono por parámetro ?sync=true
    if request.args.get("sync") == "true" or request.form.get("sync") == "true":
        job_id = uuid.uuid4().hex[:12]
        temp_path = os.path.join(UPLOAD_FOLDER, f"sync_{job_id}.{ext}")
        file.save(temp_path)
        try:
            result = transcribe_audio_file(temp_path)
            return jsonify({
                "status": "success",
                "text": result["text"],
                "language": result["language"],
                "duration": result["duration"],
                "segments": result["segments"],
            }), 200
        except Exception as exc:
            return jsonify({"status": "error", "error": str(exc)}), 500
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    job_id = uuid.uuid4().hex[:12]
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
    app.run(host=HOST, port=PORT, debug=False)
