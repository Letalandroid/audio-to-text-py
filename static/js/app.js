(function () {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const fileInfo = document.getElementById("file-info");
  const fileName = document.getElementById("file-name");
  const fileSize = document.getElementById("file-size");
  const progressWrap = document.getElementById("progress-wrap");
  const progressBar = document.getElementById("progress-bar");
  const progressValue = document.getElementById("progress-value");
  const progressStatus = document.getElementById("progress-status");
  const errorBox = document.getElementById("error-box");
  const resultCard = document.getElementById("result-card");
  const chatView = document.getElementById("chat-view");
  const rawView = document.getElementById("raw-view");
  const rawText = document.getElementById("raw-text");
  const toggleBtn = document.getElementById("toggle-view");
  const downloadBtn = document.getElementById("download-btn");

  let currentJobId = null;
  let currentSegments = [];
  let currentTxt = "";
  let currentTxtFile = "";
  let isChatView = true;
  let polling = false;

  function humanSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function fmtTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    if (h) return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
    return `${m}:${String(s).padStart(2, "0")}`;
  }

  function resetUI() {
    errorBox.hidden = true;
    errorBox.textContent = "";
    progressBar.className = "progress-bar";
    progressWrap.hidden = false;
    progressBar.style.width = "0%";
    progressValue.textContent = "0%";
    progressStatus.textContent = "Enviando archivo…";
    resultCard.hidden = true;
    chatView.innerHTML = "";
  }

  function showError(msg) {
    errorBox.textContent = msg;
    errorBox.hidden = false;
    progressStatus.textContent = "Error";
    progressBar.classList.add("error");
  }

  function startPolling(jobId) {
    currentJobId = jobId;
    polling = true;

    const tick = async () => {
      if (!polling) return;
      try {
        const res = await fetch(`/api/status/${jobId}`);
        if (!res.ok) throw new Error("No se pudo consultar el estado");
        const job = await res.json();
        updateProgress(job);

        if (job.status === "done") {
          polling = false;
          currentSegments = job.segments || [];
          currentTxt = job.txt || "";
          currentTxtFile = job.txt_file || "";
          showResult();
          return;
        }
        if (job.status === "error") {
          polling = false;
          showError(job.error || "Ocurrió un error durante la transcripción.");
          return;
        }
        setTimeout(tick, 700);
      } catch (err) {
        polling = false;
        showError(err.message);
      }
    };

    tick();
  }

  function updateProgress(job) {
    const pct = Math.max(0, Math.min(100, Math.round(job.progress || 0)));
    progressBar.style.width = pct + "%";
    progressValue.textContent = pct + "%";

    if (job.status === "queued") {
      progressStatus.textContent = "En cola, preparando modelo…";
    } else if (job.status === "processing") {
      progressStatus.textContent = "Transcribiendo audio…";
    }
  }

  function renderChat() {
    chatView.innerHTML = "";
    currentSegments.forEach((seg, i) => {
      const div = document.createElement("div");
      div.className = "bubble " + (i % 2 === 0 ? "other" : "me");
      const text = document.createElement("span");
      text.textContent = seg.text;
      const time = document.createElement("span");
      time.className = "bubble-time";
      time.textContent = fmtTime(seg.start);
      div.appendChild(text);
      div.appendChild(time);
      chatView.appendChild(div);
    });
  }

  function showResult() {
    progressBar.classList.add("done");
    progressStatus.textContent = "¡Completado!";
    progressValue.textContent = "100%";

    resultCard.hidden = false;
    rawText.value = currentTxt;
    downloadBtn.href = `/api/download/${currentJobId}`;
    downloadBtn.setAttribute("download", "");

    setView("chat");
    chatView.scrollTop = chatView.scrollHeight;
  }

  function setView(view) {
    isChatView = view === "chat";
    chatView.hidden = !isChatView;
    rawView.hidden = isChatView;
    toggleBtn.textContent = isChatView ? "Ver texto" : "Ver chat";
    if (!isChatView) {
      rawText.focus();
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["ogg", "wav", "mp3"].includes(ext)) {
      showError("Formato no permitido. Sube un archivo .ogg, .wav o .mp3.");
      return;
    }

    resetUI();
    fileName.textContent = file.name;
    fileSize.textContent = humanSize(file.size);
    fileInfo.hidden = false;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Error al subir el archivo");
      progressStatus.textContent = "Transcribiendo audio…";
      startPolling(data.job_id);
    } catch (err) {
      showError(err.message);
    }
  }

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );

  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );

  dropzone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });

  toggleBtn.addEventListener("click", () => setView(isChatView ? "raw" : "chat"));
})();
