// =============================================================================
// RECONOCIMIENTO DE NÚMEROS (0-9) — FRONTEND PARA PYTHON OPENCV
// =============================================================================
// Envía frames de cámara y dibujos al backend Flask/Python para clasificación
// con OpenCV (KNN). Muestra resultados en tiempo real.
// =============================================================================

const API_URL = 'http://localhost:5000';

// =============================================================================
// REFERENCIAS AL DOM
// =============================================================================
const video         = document.getElementById('video');
const canvasRaw     = document.getElementById('canvasRaw');
const canvasProc    = document.getElementById('canvasProc');
const canvasDigit   = document.getElementById('canvasDigit');
const drawCanvas    = document.getElementById('drawCanvas');
const ctxRaw        = canvasRaw.getContext('2d');
const ctxProc       = canvasProc.getContext('2d');
const ctxDigit      = canvasDigit.getContext('2d');
const ctxDraw       = drawCanvas.getContext('2d');
const info          = document.getElementById('info');
const btnActivar    = document.getElementById('btn-activar');
const btnDetect     = document.getElementById('btn-detect');
const btnClear      = document.getElementById('btn-clear');
const visorRow      = document.getElementById('visor-row');
const camPlaceholder = document.getElementById('cam-placeholder');
const bigNumber     = document.getElementById('bigNumber');
const confFill      = document.getElementById('confFill');
const confText      = document.getElementById('confText');
const engineStatus  = document.getElementById('engine-status');

// =============================================================================
// ESTADO GLOBAL
// =============================================================================
let cameraActive = false;
let processingFrame = false;
const FPS_INTERVAL = 120; // ms entre frames enviados al servidor (~8 fps)
let lastFrameTime = 0;

// =============================================================================
// ROI — ZONA DE DETECCIÓN
// =============================================================================
const ROI_W = 180;
const ROI_H = 230;

function getROI() {
  const cw = canvasRaw.width, ch = canvasRaw.height;
  return {
    x: Math.floor((cw - ROI_W) / 2),
    y: Math.floor((ch - ROI_H) / 2),
    w: ROI_W,
    h: ROI_H
  };
}

function drawROI(ctx, roi) {
  const { x, y, w, h } = roi;
  const cornerLen = 22;

  ctx.save();
  ctx.strokeStyle = '#44ff88';
  ctx.lineWidth = 2.5;
  ctx.shadowColor = '#44ff88';
  ctx.shadowBlur = 10;
  ctx.lineCap = 'round';

  ctx.beginPath();
  ctx.moveTo(x, y + cornerLen); ctx.lineTo(x, y); ctx.lineTo(x + cornerLen, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + w - cornerLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLen);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x, y + h - cornerLen); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLen, y + h);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x + w - cornerLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLen);
  ctx.stroke();

  ctx.setLineDash([4, 6]);
  ctx.strokeStyle = 'rgba(68,255,136,0.3)';
  ctx.shadowBlur = 0;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  ctx.setLineDash([]);
  ctx.font = "bold 11px 'Share Tech Mono', monospace";
  ctx.fillStyle = 'rgba(68,255,136,0.7)';
  ctx.fillText('ZONA DE DETECCIÓN', x, y - 8);
  ctx.restore();
}

// =============================================================================
// VERIFICAR CONEXIÓN CON EL BACKEND
// =============================================================================
async function checkBackend() {
  try {
    const res = await fetch(API_URL + '/api/status');
    const data = await res.json();
    if (data.ready) {
      engineStatus.textContent = '✅ Python OpenCV conectado — Modelo KNN listo';
      engineStatus.style.color = '#44ff88';
      info.textContent = 'Backend Python OpenCV listo ✓ — Pulsa "Activar Cámara" o dibuja un número abajo.';
      return true;
    }
  } catch (e) {
    // Backend no disponible
  }
  engineStatus.textContent = '❌ Backend no disponible — Ejecuta: python app.py';
  engineStatus.style.color = '#ff5555';
  info.textContent = 'Error: No se puede conectar al servidor Python. Ejecuta "python app.py" en una terminal.';
  return false;
}

// =============================================================================
// ENVIAR IMAGEN AL BACKEND PARA CLASIFICACIÓN
// =============================================================================
async function sendToBackend(imageDataURL, mode) {
  try {
    const res = await fetch(API_URL + '/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageDataURL, mode: mode })
    });
    return await res.json();
  } catch (e) {
    console.warn('Error enviando al backend:', e);
    return null;
  }
}

// =============================================================================
// ACTUALIZAR PANTALLA DE RESULTADO
// =============================================================================
let lastDetectedDigit = null;
let stableCount = 0;
const STABLE_THRESHOLD = 2;

function updateResult(result) {
  if (!result || result.digit === null || result.digit === undefined) {
    bigNumber.textContent = "—";
    bigNumber.classList.add('detecting');
    confFill.style.width = '0%';
    confText.textContent = 'Sin detección';
    return;
  }

  if (result.digit === lastDetectedDigit) {
    stableCount++;
  } else {
    stableCount = 1;
    lastDetectedDigit = result.digit;
  }

  const showDigit = stableCount >= STABLE_THRESHOLD ? result.digit : bigNumber.textContent;

  bigNumber.textContent = showDigit;
  bigNumber.classList.remove('detecting');
  confFill.style.width = result.confidence + '%';
  confText.textContent = 'Confianza: ' + result.confidence + '%';

  if (result.confidence >= 70) {
    confFill.style.background = 'linear-gradient(90deg, #00d4ff, #44ff88)';
  } else if (result.confidence >= 45) {
    confFill.style.background = 'linear-gradient(90deg, #00d4ff, #ffd60a)';
  } else {
    confFill.style.background = 'linear-gradient(90deg, #ff5555, #ff6b35)';
  }
}

// =============================================================================
// BUCLE PRINCIPAL DE CÁMARA
// =============================================================================
function processFrame() {
  if (!cameraActive) return;

  const now = Date.now();
  const cw = canvasRaw.width, ch = canvasRaw.height;

  // Siempre dibujar en canvas original
  ctxRaw.drawImage(video, 0, 0, cw, ch);
  const roi = getROI();
  drawROI(ctxRaw, roi);

  // Throttle: solo enviar frames al backend cada FPS_INTERVAL ms
  if (now - lastFrameTime >= FPS_INTERVAL && !processingFrame) {
    processingFrame = true;
    lastFrameTime = now;

    // Capturar frame del canvas como JPEG
    const frameURL = canvasRaw.toDataURL('image/jpeg', 0.85);

    sendToBackend(frameURL, 'camera').then(result => {
      processingFrame = false;

      if (result && result.processed_image) {
        // Mostrar imagen procesada devuelta por Python
        const img = new Image();
        img.onload = () => {
          ctxProc.drawImage(img, 0, 0, cw, ch);
        };
        img.src = 'data:image/jpeg;base64,' + result.processed_image;
      }

      if (result && result.digit_preview) {
        // Mostrar preview del dígito extraído
        const img2 = new Image();
        img2.onload = () => {
          ctxDigit.drawImage(img2, 0, 0);
        };
        img2.src = 'data:image/png;base64,' + result.digit_preview;
      }

      updateResult(result);

      if (result && result.digit !== null && result.digit !== undefined) {
        const scores = result.scores || {};
        const top3 = Object.entries(scores)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([d, s]) => d + ':' + s + '%')
          .join('  ');
        info.textContent = '✓ Detectado: ' + result.digit + ' — Confianza: ' + result.confidence + '% | Top3: ' + top3 + ' | ' + (result.engine || 'Python OpenCV');
      } else {
        info.textContent = 'Buscando dígito... Apunta un número (0-9) al recuadro verde | Python OpenCV';
      }
    });
  }

  requestAnimationFrame(processFrame);
}

// =============================================================================
// ACTIVAR CÁMARA
// =============================================================================
btnActivar.addEventListener('click', () => {
  btnActivar.disabled = true;
  btnActivar.textContent = "Solicitando permiso...";

  navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
    .then(stream => {
      video.srcObject = stream;
      camPlaceholder.hidden = true;
      visorRow.hidden = false;
      btnActivar.hidden = true;
      cameraActive = true;
      bigNumber.textContent = "—";
      bigNumber.classList.add('detecting');
      info.textContent = "Cámara activada — Procesando con Python OpenCV...";
      processFrame();
    })
    .catch(err => {
      console.error("Error cámara:", err);
      btnActivar.disabled = false;
      btnActivar.innerHTML =
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg> Reintentar';
      info.textContent = "No se pudo acceder a la cámara. Verifica los permisos.";
    });
});

// =============================================================================
// CANVAS DE DIBUJO
// =============================================================================
let isDrawing = false;
let lastDrawX = 0, lastDrawY = 0;

ctxDraw.fillStyle = '#000';
ctxDraw.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
ctxDraw.lineCap = 'round';
ctxDraw.lineJoin = 'round';
ctxDraw.strokeStyle = '#ffffff';
ctxDraw.lineWidth = 20;

function getDrawPos(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  if (e.touches) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function startDraw(e) {
  e.preventDefault();
  isDrawing = true;
  const pos = getDrawPos(e);
  lastDrawX = pos.x;
  lastDrawY = pos.y;
  ctxDraw.beginPath();
  ctxDraw.arc(pos.x, pos.y, ctxDraw.lineWidth / 2, 0, Math.PI * 2);
  ctxDraw.fillStyle = '#ffffff';
  ctxDraw.fill();
}

function doDraw(e) {
  e.preventDefault();
  if (!isDrawing) return;
  const pos = getDrawPos(e);
  ctxDraw.beginPath();
  ctxDraw.moveTo(lastDrawX, lastDrawY);
  ctxDraw.lineTo(pos.x, pos.y);
  ctxDraw.stroke();
  lastDrawX = pos.x;
  lastDrawY = pos.y;
}

function endDraw(e) {
  e.preventDefault();
  isDrawing = false;
}

drawCanvas.addEventListener('mousedown', startDraw);
drawCanvas.addEventListener('mousemove', doDraw);
drawCanvas.addEventListener('mouseup', endDraw);
drawCanvas.addEventListener('mouseleave', endDraw);
drawCanvas.addEventListener('touchstart', startDraw, { passive: false });
drawCanvas.addEventListener('touchmove', doDraw, { passive: false });
drawCanvas.addEventListener('touchend', endDraw, { passive: false });

// =============================================================================
// BOTÓN DETECTAR DIBUJO
// =============================================================================
btnDetect.addEventListener('click', async () => {
  visorRow.hidden = false;
  btnDetect.disabled = true;
  btnDetect.textContent = 'Analizando...';
  info.textContent = 'Enviando dibujo a Python OpenCV...';

  const drawURL = drawCanvas.toDataURL('image/png');

  try {
    const result = await sendToBackend(drawURL, 'drawing');

    if (result && result.digit !== null && result.digit !== undefined) {
      bigNumber.textContent = result.digit;
      bigNumber.classList.remove('detecting');
      confFill.style.width = result.confidence + '%';
      confText.textContent = 'Confianza: ' + result.confidence + '%';

      if (result.confidence >= 70) {
        confFill.style.background = 'linear-gradient(90deg, #00d4ff, #44ff88)';
      } else if (result.confidence >= 45) {
        confFill.style.background = 'linear-gradient(90deg, #00d4ff, #ffd60a)';
      } else {
        confFill.style.background = 'linear-gradient(90deg, #ff5555, #ff6b35)';
      }

      if (result.digit_preview) {
        const img = new Image();
        img.onload = () => { ctxDigit.drawImage(img, 0, 0); };
        img.src = 'data:image/png;base64,' + result.digit_preview;
      }

      const scores = result.scores || {};
      const top3 = Object.entries(scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([d, s]) => d + ':' + s + '%')
        .join('  ');
      info.textContent = '✓ Detectado: ' + result.digit + ' — Confianza: ' + result.confidence + '% | Top3: ' + top3 + ' | Python OpenCV';
    } else {
      info.textContent = 'No se detectó ningún dígito. Intenta dibujar más grande y claro.';
    }
  } catch (e) {
    info.textContent = 'Error al conectar con el servidor. Ejecuta "python app.py".';
  }

  btnDetect.disabled = false;
  btnDetect.textContent = 'Detectar';
});

// =============================================================================
// BOTÓN LIMPIAR
// =============================================================================
btnClear.addEventListener('click', () => {
  ctxDraw.fillStyle = '#000';
  ctxDraw.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  bigNumber.textContent = '—';
  bigNumber.classList.add('detecting');
  confFill.style.width = '0%';
  confText.textContent = 'Confianza: 0%';
  ctxDigit.fillStyle = '#000';
  ctxDigit.fillRect(0, 0, canvasDigit.width, canvasDigit.height);
  info.textContent = 'Canvas limpio — dibuja un número (0-9)';
});

// =============================================================================
// INICIALIZACIÓN
// =============================================================================
ctxDigit.fillStyle = '#000';
ctxDigit.fillRect(0, 0, canvasDigit.width, canvasDigit.height);

// Verificar backend al cargar
checkBackend();
// Reintentar cada 3 segundos si no está disponible
setInterval(() => {
  if (engineStatus.textContent.includes('❌')) {
    checkBackend();
  }
}, 3000);