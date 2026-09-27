"""
Cliente HTTP para el servicio audio-to-text con manejo robusto de:
- HTTP 429 (Too Many Requests / Rate Limiting) con Retry-After y Backoff Exponencial
- Timeouts en requests
- Reintentos configurables
"""

import os
import sys
import time
import requests


def request_with_retry(
    method: str,
    url: str,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    timeout: float = 60.0,
    **kwargs,
) -> requests.Response:
    """
    Ejecuta peticiones HTTP con reintentos automáticos ante:
    - Status 429 (Too Many Requests / Rate Limit)
    - Timeouts (requests.exceptions.Timeout)
    - Errores de conexión transitorios (requests.exceptions.ConnectionError)
    """
    backoff = initial_backoff
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, timeout=timeout, **kwargs)

            # Manejo explícito de 429 Too Many Requests
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.strip().isdigit():
                    wait_time = float(retry_after.strip())
                else:
                    wait_time = backoff

                print(
                    f"[WARN] HTTP 429 (Rate Limit / Too Many Requests) en '{url}'. "
                    f"Intento {attempt}/{max_retries}. Esperando {wait_time:.1f}s antes de reintentar..."
                )
                time.sleep(wait_time)
                backoff *= 2.0
                continue

            # Si el servidor responde 503 o 504 (temporarily unavailable / gateway timeout)
            if resp.status_code in (502, 503, 504):
                print(
                    f"[WARN] HTTP {resp.status_code} en '{url}'. "
                    f"Intento {attempt}/{max_retries}. Esperando {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= 2.0
                continue

            return resp

        except requests.exceptions.Timeout as exc:
            last_exception = exc
            print(
                f"[WARN] Timeout al conectar con '{url}' ({timeout}s). "
                f"Intento {attempt}/{max_retries}. Reintentando en {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= 2.0

        except requests.exceptions.ConnectionError as exc:
            last_exception = exc
            print(
                f"[WARN] Error de conexión con '{url}'. "
                f"Intento {attempt}/{max_retries}. Reintentando en {backoff:.1f}s..."
            )
            time.sleep(backoff)
            backoff *= 2.0

    if last_exception:
        raise last_exception
    raise RuntimeError(f"Excedido el número máximo de reintentos ({max_retries}) hacia {url}")


def submit_local_file_transcription(
    filepath: str,
    base_url: str = "http://localhost:5000",
    language: str = "es",
    poll_interval: float = 3.0,
    timeout: float = 120.0,
) -> dict:
    """
    Envía un archivo local para transcribir usando el endpoint /api/transcribe-file
    y monitorea el progreso respetando los límites de velocidad (evita 429) y timeouts.
    """
    abs_path = os.path.abspath(filepath)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Archivo no encontrado: {abs_path}")

    url = f"{base_url.rstrip('/')}/api/transcribe-file"
    print(f"Enviando solicitud de transcripción para: {abs_path}")

    res = request_with_retry(
        "POST",
        url,
        json={"filepath": abs_path, "language": language},
        timeout=timeout,
    )

    if res.status_code not in (200, 202):
        raise RuntimeError(f"Error al enviar archivo ({res.status_code}): {res.text}")

    data = res.json()
    job_id = data.get("job_id")
    if not job_id:
        return data  # Transcripción síncrona retornada directamente

    print(f"Trabajo encolado con ID: {job_id}. Monitoreando estado...")

    status_url = f"{base_url.rstrip('/')}/api/status/{job_id}"
    while True:
        status_res = request_with_retry("GET", status_url, timeout=30.0)
        job_info = status_res.json()
        status = job_info.get("status")
        progress = job_info.get("progress", 0)

        print(f"Estado: {status} | Progreso: {progress}%", end="\r", flush=True)

        if status == "done":
            print(f"\n¡Transcripción completada exitosamente!")
            return job_info
        elif status == "error":
            print()
            raise RuntimeError(f"Error en la transcripción: {job_info.get('error')}")

        time.sleep(poll_interval)
