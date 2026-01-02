from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

# ------------------ Paths ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# ------------------ CV-ONLY MAMMOGRAM DETECTION ------------------

def detect_suspicious_regions(image_path, filename):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Breast mask (remove background)
    _, breast_mask = cv2.threshold(enhanced, 10, 255, cv2.THRESH_BINARY)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=breast_mask)

    # Detect dense (bright) regions
    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -10
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600:
            continue

        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

        # Filter unlikely shapes
        if circularity < 0.15 or circularity > 0.85:
            continue

        suspicious_count += 1
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 215, 255), 2)
        cv2.putText(
            img,
            "Suspicious",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 215, 255),
            1
        )

    result_path = os.path.join(app.config["RESULT_FOLDER"], filename)
    cv2.imwrite(result_path, img)

    return suspicious_count, f"/static/results/{filename}"

# ------------------ ROUTES ------------------

@app.route("/", methods=["GET", "POST"])
def upload_files():
    results = []

    if request.method == "POST":
        uploaded_files = request.files.getlist("files")

        for file in uploaded_files:
            if not file:
                continue

            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(upload_path)

            count, processed_image_url = detect_suspicious_regions(
                upload_path, filename
            )

            results.append({
                "filename": filename,
                "regions": count,
                "processed_image": processed_image_url,
                "risk": (
                    "High" if count >= 5 else
                    "Moderate" if count >= 2 else
                    "Low"
                )
            })

    return render_template("index.html", results=results)

# ------------------ Run ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
