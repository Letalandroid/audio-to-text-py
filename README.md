# 🎙️ Audio a Texto (Speech-to-Text API)

Web + API REST para transcribir audios (`.ogg`, `.opus`, `.wav`, `.mp3`) a texto usando **faster-whisper** (Whisper de OpenAI optimizado en CPU con CTranslate2). Soporta peticiones síncronas directas vía `curl` en formato JSON y subidas desde la interfaz web con barra de progreso en vivo.

## ✨ Características

- 🎯 **Endpoint síncrono para cURL / APIs (`/api/transcribe`)**: Envía el archivo y recibe la transcripción en JSON directamente.
- 🎵 **Soporte completo de formatos y MIME Types**:
  - `audio/ogg; codecs=opus` (formato estándar de notas de voz de WhatsApp / Evolution API)
  - `audio/ogg`, `audio/opus`
  - `audio/wav`, `audio/x-wav`, `audio/wave`
  - `audio/mpeg`, `audio/mp3`
- ⚙️ **Configuración flexible con variables de entorno (`.env`)**.
- 📤 Subida por interfaz web (drag & drop) con procesamiento asíncrono y barra de progreso.
- 🧹 **Almacenamiento 100% temporal**: los archivos de audio se eliminan automáticamente tras procesarse.
- 🐧 Compatible con Linux, Raspberry Pi 4 (ARM64) y servidores x86_64.

---

## ⚙️ Configuración (.env)

Copia `.env.example` a `.env` y ajusta las variables según tus necesidades:

```bash
cp .env.example .env
```

| Variable | Descripción | Default |
| --- | --- | --- |
| `HOST` | Host para escuchar peticiones | `0.0.0.0` |
| `PORT` | Puerto del servidor HTTP | `5000` |
| `WHISPER_MODEL` | Tamaño del modelo (`tiny`, `base`, `small`, `medium`, `large-v3`) | `base` |
| `WHISPER_LANGUAGE` | Idioma forzado (ej: `es`); si se omite, se autodetecta | — |
| `WHISPER_DEVICE` | Dispositivo de cómputo (`cpu` o `cuda`) | `cpu` |
| `WHISPER_COMPUTE_TYPE` | Tipo de cómputo (`int8` recomendado en CPU, `float16` en GPU) | `int8` |
| `JOB_TTL_SECONDS` | Segundos para purgar trabajos de la interfaz web en memoria | `600` |
| `MAX_CONTENT_LENGTH_MB`| Límite máximo de subida en megabytes | `200` |

---

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Letalandroid/audio-to-text-py.git
cd audio-to-text-py

# 2. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Asegurarse de tener ffmpeg instalado
# Debian / Ubuntu / Raspberry Pi OS:
sudo apt update && sudo apt install -y ffmpeg
```

---

## 🔌 Uso de la API con cURL

### 1. Petición directa (Multipart Form Data)

Ideal para enviar archivos de audio desde la terminal:

```bash
curl -X POST http://localhost:5000/api/transcribe \
  -F "file=@audio.ogg;type=audio/ogg; codecs=opus"
```

O simplemente:

```bash
curl -X POST http://localhost:5000/api/transcribe \
  -F "file=@audio.ogg"
```

### 2. Petición directa enviando audio binario en el cuerpo

Útil para webhooks o clientes que envían el flujo de audio en bruto con el MIME Type especificado:

```bash
curl -X POST http://localhost:5000/api/transcribe \
  -H "Content-Type: audio/ogg; codecs=opus" \
  --data-binary @audio.ogg
```

### 3. Petición con Base64 en JSON

Ideal para integraciones como Evolution API o n8n:

```bash
curl -X POST http://localhost:5000/api/transcribe \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "GkXfo59ChoEBQveBAULygQ8UA...",
    "mimetype": "audio/ogg; codecs=opus"
  }'
```

### 4. Forzar idioma en la petición

Puedes pasar el parámetro `language` (ej. `es`, `en`):

```bash
curl -X POST "http://localhost:5000/api/transcribe?language=es" \
  -F "file=@audio.ogg"
```

---

## 📥 Respuesta JSON

```json
{
  "status": "success",
  "text": "Hola, esto es una prueba de transcripción de audio a texto.",
  "language": "es",
  "duration": 3.45,
  "segments": [
    {
      "start": 0.0,
      "end": 3.45,
      "text": "Hola, esto es una prueba de transcripción de audio a texto."
    }
  ]
}
```

Si el formato o MIME Type no es soportado, devuelve HTTP 400 con los detalles:

```json
{
  "error": "Formato de audio no soportado.",
  "detail": "Tipo recibido: 'application/pdf', archivo: 'document.pdf'.",
  "supported_mimetypes": [
    "audio/ogg; codecs=opus",
    "audio/ogg",
    "audio/opus",
    "audio/wav",
    "audio/mpeg"
  ],
  "supported_extensions": [".ogg", ".opus", ".wav", ".mp3"]
}
```

---

## 🛠️ Servicio Systemd (Raspberry Pi / Linux)

Para ejecutar el servicio automáticamente en el arranque:

1. Crea el archivo de servicio `/etc/systemd/system/audio-to-text.service`:

```ini
[Unit]
Description=Servicio Speech to Text (faster-whisper)
After=network.target

[Service]
User=lta
WorkingDirectory=/home/lta/projects/audio-to-text-py
Environment="PATH=/home/lta/projects/audio-to-text-py/.venv/bin"
ExecStart=/home/lta/projects/audio-to-text-py/.venv/bin/python app.py

Restart=always
RestartSec=5
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

2. Habilita y arranca el servicio:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now audio-to-text.service
sudo systemctl status audio-to-text.service
```

---

## 📄 Licencia

[MIT](LICENSE)
