from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import os
import numpy as np
import pyodbc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # giới hạn file 10MB

# Giữ nguyên kết nối SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:asdjnu12uh12husa.database.windows.net;"
    "DATABASE=ai_2025;"
    "UID=sqladmin;"
    "PWD=YourPassword@123"
)
cursor = conn.cursor()

# Load YOLO model
YOLO_PATH = "best.pt"
if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"Model file not found: {YOLO_PATH}")
model = YOLO(YOLO_PATH)

@app.route("/")
def index():
    return "Face Recognition API is running"

def save_face_to_db(img, name):
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    cursor.execute("INSERT INTO Faces (Name, Image) VALUES (?, ?)", name, img_bytes.getvalue())
    conn.commit()

def get_all_faces_from_db():
    cursor.execute("SELECT Name, Image FROM Faces")
    return [(name, np.array(Image.open(BytesIO(data)).convert("RGB"))) for name, data in cursor.fetchall()]

def detect_face(image):
    results = model.predict(image, conf=0.3, save=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    x1, y1, x2, y2 = map(int, boxes[0].xyxy[0].tolist())
    return image.crop((x1, y1, x2, y2))

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return jsonify({"error": "Missing name or image"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        face = detect_face(image)
        if face:
            save_face_to_db(face, name)
            return jsonify({"message": "Face registered"}), 200
        return jsonify({"error": "No face detected"}), 400
    except Exception as e:
        return jsonify({"error": f"Image processing error: {str(e)}"}), 500

@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
        face = detect_face(image)
        detected_names = []

        if face:
            face_np = np.array(face)
            for name, db_img in get_all_faces_from_db():
                try:
                    result = DeepFace.verify(
                        face_np, db_img,
                        model_name="VGG-Face",  # ổn định hơn cho tốc độ và nhận diện
                        enforce_detection=False
                    )
                    if result.get("verified"):
                        detected_names.append(name)
                        break
                except:
                    continue

        return jsonify({
            "predictions": [{"class_name": name, "confidence": 1.0} for name in detected_names]
        }), 200
    except Exception as e:
        return jsonify({"error": f"Verification error: {str(e)}"}), 500

@app.route("/faces", methods=["GET"])
def list_faces():
    cursor.execute("SELECT Name, COUNT(*) FROM Faces GROUP BY Name")
    rows = cursor.fetchall()
    return jsonify({"faces": [{"name": name, "count": count} for name, count in rows]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
