# =============================================================================
# RECONOCIMIENTO DE NÚMEROS (0-9) — PYTHON + OPENCV + FLASK
# =============================================================================
# Backend que usa OpenCV (cv2) para preprocesamiento y clasificación KNN
# con datos de entrenamiento sintéticos generados con cv2.putText.
# =============================================================================

import cv2
import numpy as np
import base64
import io
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

# =============================================================================
# MODELO KNN DE OPENCV
# =============================================================================
knn_model = cv2.ml.KNearest_create()
trained = False

# Tamaño estándar para el modelo
MODEL_W, MODEL_H = 28, 28

def generate_training_data():
    """Genera imágenes sintéticas de dígitos (0-9) usando cv2.putText."""
    images = []
    labels = []

    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    ]

    scales = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    thicknesses = [1, 2, 3]
    rotations = [-15, -10, -7, -4, -2, 0, 2, 4, 7, 10, 15]

    for digit in range(10):
        text = str(digit)
        for font in fonts:
            for scale in scales:
                for thick in thicknesses:
                    # Crear imagen con el dígito
                    img = np.zeros((60, 50), dtype=np.uint8)
                    text_size = cv2.getTextSize(text, font, scale, thick)[0]
                    tx = max(0, (50 - text_size[0]) // 2)
                    ty = min(58, (60 + text_size[1]) // 2)
                    cv2.putText(img, text, (tx, ty), font, scale, 255, thick, cv2.LINE_AA)

                    # Recortar al bounding box del dígito
                    coords = cv2.findNonZero(img)
                    if coords is None:
                        continue
                    x, y, w, h = cv2.boundingRect(coords)
                    if w < 3 or h < 5:
                        continue
                    digit_roi = img[y:y+h, x:x+w]

                    # Resize a 28x28 manteniendo aspect ratio
                    processed = resize_and_center(digit_roi, MODEL_W, MODEL_H)
                    images.append(processed.flatten().astype(np.float32))
                    labels.append(np.array([digit], dtype=np.float32))

                    # Variaciones con rotación
                    for angle in rotations:
                        if angle == 0:
                            continue
                        M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
                        rotated = cv2.warpAffine(processed, M, (28, 28))
                        images.append(rotated.flatten().astype(np.float32))
                        labels.append(np.array([digit], dtype=np.float32))

                    # Variaciones morfológicas
                    kernel_e = np.ones((2, 2), np.uint8)
                    eroded = cv2.erode(processed, kernel_e, iterations=1)
                    images.append(eroded.flatten().astype(np.float32))
                    labels.append(np.array([digit], dtype=np.float32))

                    kernel_d = np.ones((2, 2), np.uint8)
                    dilated = cv2.dilate(processed, kernel_d, iterations=1)
                    images.append(dilated.flatten().astype(np.float32))
                    labels.append(np.array([digit], dtype=np.float32))

    return np.array(images), np.array(labels)


def resize_and_center(img, target_w, target_h):
    """Redimensiona imagen manteniendo aspect ratio y centra en target_w x target_h."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_h, target_w), dtype=np.uint8)

    inner = 20
    scale = min(inner / w, inner / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    result = np.zeros((target_h, target_w), dtype=np.uint8)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    result[oy:oy+new_h, ox:ox+new_w] = resized
    return result


def train_model():
    """Entrena el modelo KNN con datos sintéticos."""
    global knn_model, trained
    print("[...] Generando datos de entrenamiento...")
    images, labels = generate_training_data()
    print(f"   {len(images)} muestras generadas ({len(images)//10} por dígito)")

    knn_model.train(images, cv2.ml.ROW_SAMPLE, labels)
    trained = True
    print("[OK] Modelo KNN entrenado correctamente")


# =============================================================================
# PREPROCESAMIENTO CON OPENCV
# =============================================================================

def preprocess_camera(img):
    """
    Pipeline de preprocesamiento para imagen de cámara.
    Retorna: (extracted_28x28, binarized_roi, contour_rect, roi_coords) o (None, ...)
    """
    h, w = img.shape[:2]

    # Definir ROI central (misma proporción que el frontend)
    roi_w, roi_h = min(180, w), min(230, h)
    rx = (w - roi_w) // 2
    ry = (h - roi_h) // 2

    # Extraer ROI
    roi = img[ry:ry+roi_h, rx:rx+roi_w]

    # 1. Escala de grises
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # 2. Gaussian blur para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Binarización adaptativa (tinta oscura sobre fondo claro)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 12
    )

    # 4. Limpieza morfológica
    kernel_open = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    kernel_close = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # 5. Encontrar contornos
    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None, None, (rx, ry, roi_w, roi_h), cleaned

    # Buscar el mejor contorno candidato
    best_contour = None
    best_rect = None
    max_area = 0
    total_pixels = roi_w * roi_h

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100 or area > total_pixels * 0.85:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0
        if aspect < 0.15 or aspect > 4.0:
            continue
        if cw < 15 or ch < 20:
            continue
        # Solo contornos externos (sin padre)
        if hierarchy is not None:
            parent_idx = hierarchy[0][i][3]
            if parent_idx != -1:
                continue
        if area > max_area:
            max_area = area
            best_contour = cnt
            best_rect = (x, y, cw, ch)

    if best_contour is None:
        return None, None, None, (rx, ry, roi_w, roi_h), cleaned

    # 6. Extraer dígito y redimensionar a 28x28
    x, y, cw, ch = best_rect
    digit_roi = cleaned[y:y+ch, x:x+cw]
    processed = resize_and_center(digit_roi, MODEL_W, MODEL_H)

    # 7. Dilatar ligeramente
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)

    return processed, cleaned, best_rect, (rx, ry, roi_w, roi_h), cleaned


def preprocess_drawing(img):
    """
    Pipeline para imagen de canvas de dibujo (fondo negro, tinta blanca).
    Retorna: (extracted_28x28, None) o (None, None)
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Binarizar (tinta blanca sobre fondo negro)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel_close = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

    # Encontrar contornos
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, cleaned

    # Tomar el más grande
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < 20:
        return None, cleaned

    x, y, w, h = cv2.boundingRect(best)
    if w < 8 or h < 12:
        return None, cleaned

    digit_roi = cleaned[y:y+h, x:x+w]
    processed = resize_and_center(digit_roi, MODEL_W, MODEL_H)

    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.dilate(processed, kernel, iterations=1)

    return processed, cleaned


# =============================================================================
# CLASIFICACIÓN
# =============================================================================

def classify_digit(img_28x28):
    """
    Clasifica una imagen 28x28 usando KNN.
    Retorna: { digit, confidence, all_scores }
    """
    if not trained:
        return None

    # Normalizar
    sample = img_28x28.flatten().astype(np.float32).reshape(1, -1)

    # KNN con k=7
    k = 7
    ret, results, neighbours, dist = knn_model.findNearest(sample, k)

    digit = int(ret)

    # Calcular confianza basada en vecinos y distancias
    neighbor_digits = [int(n) for n in neighbours[0]]
    distances = dist[0]

    # Votación ponderada por distancia inversa
    vote_scores = {}
    for i, nd in enumerate(neighbor_digits):
        d = max(distances[i], 1.0)
        weight = 1.0 / (d * d)
        vote_scores[nd] = vote_scores.get(nd, 0) + weight

    total_vote = sum(vote_scores.values())
    winner_vote = vote_scores.get(digit, 0)

    # Confianza base
    if total_vote > 0:
        confidence = int(min(100, (winner_vote / total_vote) * 100))
    else:
        confidence = 0

    # Boost si todos los vecinos son el mismo
    if all(n == digit for n in neighbor_digits):
        confidence = min(100, confidence + 15)

    # Boost si la distancia promedio es baja
    avg_dist = np.mean(distances)
    if avg_dist < 500:
        confidence = min(100, confidence + 10)

    # Scores por dígito
    all_scores = {}
    for d in range(10):
        val = vote_scores.get(d, 0) / total_vote * 100 if total_vote > 0 else 0
        all_scores[d] = round(float(val), 1)

    return {
        'digit': digit,
        'confidence': confidence,
        'neighbors': neighbor_digits,
        'distances': [round(float(d), 1) for d in distances],
        'scores': all_scores
    }


def count_holes(binary, rect):
    """Cuenta huecos internos en un contorno."""
    contours, hierarchy = cv2.findContours(
        binary.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0

    holes = 0
    rx, ry, rw, rh = rect

    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent >= 0:
            area = cv2.contourArea(cnt)
            br = cv2.boundingRect(cnt)
            if (area > 10 and
                br[0] >= rx - 5 and br[1] >= ry - 5 and
                br[0] + br[2] <= rx + rw + 5 and
                br[1] + br[3] <= ry + rh + 5):
                holes += 1
    return holes


# =============================================================================
# RUTAS FLASK
# =============================================================================

@app.route('/')
def index():
    return send_file('index.html')


@app.route('/CSS/<path:filename>')
def css_files(filename):
    return send_file(f'CSS/{filename}')


@app.route('/JS/<path:filename>')
def js_files(filename):
    return send_file(f'JS/{filename}')


@app.route('/api/status')
def status():
    return jsonify({'ready': trained, 'engine': 'Python OpenCV'})


@app.route('/api/detect', methods=['POST'])
def detect():
    """Endpoint principal de detección de dígitos."""
    try:
        data = request.json
        image_b64 = data.get('image', '')
        mode = data.get('mode', 'camera')  # 'camera' o 'drawing'

        # Decodificar imagen base64
        header = ''
        if ',' in image_b64:
            header, image_b64 = image_b64.split(',', 1)
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'No se pudo decodificar la imagen'}), 400

        processed = None
        result = None
        annotated_img = None

        if mode == 'camera':
            processed, binary, rect, roi_coords, cleaned = preprocess_camera(img)

            if processed is not None:
                result = classify_digit(processed)

                # Crear imagen anotada para el frontend
                rx, ry, rw, rh = roi_coords
                annotated = img.copy()

                # Dibujar ROI
                cv2.rectangle(annotated, (rx, ry), (rx+rw, ry+rh), (68, 255, 136), 1)

                # Dibujar bounding box del dígito
                if rect:
                    dx, dy, dw, dh = rect
                    cv2.rectangle(annotated,
                                  (rx+dx, ry+dy), (rx+dx+dw, ry+dy+dh),
                                  (255, 107, 53), 2)

                    # Etiqueta
                    if result:
                        label = f"{result['digit']} ({result['confidence']}%)"
                        cv2.rectangle(annotated,
                                      (rx+dx, ry+dy-22), (rx+dx+120, ry+dy-4),
                                      (0, 0, 0), -1)
                        cv2.putText(annotated, label, (rx+dx+4, ry+dy-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 107, 53), 1, cv2.LINE_AA)

                # Overlay binarizado
                if cleaned is not None:
                    overlay = annotated[ry:ry+rh, rx:rx+rw]
                    colored_binary = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
                    colored_binary[:, :, 1] = np.clip(
                        colored_binary[:, :, 1] * 0.6 + 100, 0, 255
                    ).astype(np.uint8)
                    blended = cv2.addWeighted(overlay, 0.5, colored_binary, 0.5, 0)
                    annotated[ry:ry+rh, rx:rx+rw] = blended

                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                annotated_img = base64.b64encode(buffer).decode('utf-8')

        elif mode == 'drawing':
            processed, cleaned = preprocess_drawing(img)

            if processed is not None:
                result = classify_digit(processed)

        if result:
            # Preparar imagen del dígito extraído para preview
            digit_preview = None
            if processed is not None:
                # Escalar a 140x140 para el canvas de preview
                preview = cv2.resize(processed, (140, 140), interpolation=cv2.INTER_NEAREST)
                colored = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
                # Dar tono naranja
                colored[:, :, 0] = 0
                colored[:, :, 1] = np.clip(processed_scaled(colored[:, :, 1]) * 0.85, 0, 255).astype(np.uint8)
                colored[:, :, 2] = preview
                _, buf = cv2.imencode('.png', colored)
                digit_preview = base64.b64encode(buf).decode('utf-8')

            return jsonify({
                'digit': result['digit'],
                'confidence': result['confidence'],
                'scores': result['scores'],
                'holes': -1,
                'processed_image': annotated_img,
                'digit_preview': digit_preview,
                'engine': 'Python OpenCV'
            })
        else:
            return jsonify({
                'digit': None,
                'confidence': 0,
                'message': 'Sin detección',
                'processed_image': annotated_img
            })

    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def processed_scaled(channel):
    """Helper para escalar canal de color."""
    return channel.astype(np.float32)


# =============================================================================
# INICIO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  RECONOCIMIENTO DE NÚMEROS (0-9) — Python + OpenCV + Flask")
    print("=" * 60)
    train_model()
    print()
    print(">> Servidor iniciando en http://localhost:5000")
    print("   Abre esa URL en tu navegador")
    print()
    app.run(debug=False, host='0.0.0.0', port=5000)