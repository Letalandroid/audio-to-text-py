# 🎙️ Audio a Texto

Web + API para transcribir audios (`.ogg`, `.wav`, `.mp3`) a texto usando **faster-whisper** (Whisper de OpenAI optimizado en CPU). Subes el archivo desde la web, una API lo recibe y la transcripción se procesa en segundo plano con una **barra de progreso en vivo**. Al terminar puedes ver la transcripción como chat y **descargar el resultado en `.txt`**.

## ✨ Características

- 📤 Subida de audio por **arrastrar y soltar** o selección de archivo.
- 🔁 Procesamiento **asíncrono**: la API devuelve un `job_id` y el estado se consulta por polling.
- 📊 **Barra de progreso** con porcentaje en tiempo real.
- 💬 Visualización de la transcripción estilo **chat** (con timestamps) o vista de texto plano.
- ⬇️ Descarga del resultado como archivo `.txt`.
- 📱 Diseño **responsive**.
- 🐍 Entorno virtual (`venv`) incluido.

## 🚀 Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/Letalandroid/audio-to-text-py.git
cd audio-to-text-py

# 2. Crear y activar el entorno virtual
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. (Opcional) Instalar ffmpeg si no está disponible
# Ubuntu/Debian: sudo apt install ffmpeg
```

## ▶️ Uso

```bash
python app.py
```

Abre http://localhost:5000 en tu navegador, sube un audio y espera a que termine la transcripción.

La primera ejecución descarga el modelo Whisper (por defecto `small`) desde Hugging Face; las siguientes son inmediatas.

## ⚙️ Configuración (variables de entorno)

| Variable           | Descripción                               | Default  |
| ------------------ | ----------------------------------------- | -------- |
| `WHISPER_MODEL`    | Tamaño del modelo (tiny/base/small/medium) | `small`  |
| `WHISPER_LANGUAGE` | Idioma forzado (p. ej. `es`); si se omite, se detecta solo | — |
| `PORT`             | Puerto del servidor                        | `5000`   |

Ejemplo:

```bash
WHISPER_MODEL=base WHISPER_LANGUAGE=es python app.py
```

## 🔌 API

| Método | Ruta                  | Descripción                                        |
| ------ | --------------------- | -------------------------------------------------- |
| `POST` | `/api/upload`         | Sube el audio (multipart, campo `file`) → `job_id` |
| `GET`  | `/api/status/<id>`    | Estado del trabajo (progreso, segmentos, texto)    |
| `GET`  | `/api/download/<id>`  | Descarga la transcripción como `.txt`              |

Ejemplo con `curl`:

```bash
curl -F "file=@audio.ogg" http://localhost:5000/api/upload
```

## 🗂️ Estructura

```
audio-to-text/
├── app.py                 # API + servidor Flask
├── templates/index.html   # Interfaz web
├── static/
│   ├── css/style.css      # Estilos responsive
│   └── js/app.js          # Subida, polling y render
├── uploads/               # Audios temporales (gitignored)
├── transcriptions/        # Archivos .txt generados (gitignored)
├── requirements.txt
└── README.md
```

## 📄 Licencia

[MIT](LICENSE)
