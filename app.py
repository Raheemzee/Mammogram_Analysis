import os
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template

# -----------------------------
# App Configuration
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Utility: CV-based Detection
# -----------------------------
def detect_suspicious_regions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Noise reduction
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41, 3
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 800 or area > 15000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        # Reject elongated ducts/vessels
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            continue

        # Create mask
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        region_pixels = gray[mask == 255]

        # Texture (variance)
        variance = np.var(region_pixels)
        if variance < 150:
            continue

        # Local contrast
        dilated = cv2.dilate(mask, None, iterations=5)
        neighborhood = gray[(dilated == 255) & (mask == 0)]
        if len(neighborhood) == 0:
            continue

        contrast = np.mean(region_pixels) - np.mean(neighborhood)
        if contrast < 12:
            continue

        # Suspicion score
        score = (contrast * 0.6) + (variance * 0.4)

        candidates.append({
            "cnt": cnt,
            "score": score
        })

    # Keep only TOP suspicious regions
    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:3]

    # Draw results
    for c in candidates:
        x, y, w, h = cv2.boundingRect(c["cnt"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 215, 255), 2)
        cv2.putText(
            img, "Suspicious",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 215, 255),
            1
        )

    return img

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "images" not in request.files:
        return jsonify([])

    files = request.files.getlist("images")
    results = []

    for file in files:
        if file.filename == "":
            continue

        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        upload_path = os.path.join(UPLOAD_DIR, filename)

        file.save(upload_path)

        processed = detect_suspicious_regions(upload_path)
        if processed is None:
            continue

        output_name = f"out_{filename}"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        cv2.imwrite(output_path, processed)

        results.append({
            "output": f"/static/outputs/{output_name}"
        })

    return jsonify(results)

# -----------------------------
# Static Files (Render-safe)
# -----------------------------
@app.route("/static/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
