from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
from io import BytesIO
import os
import uuid
import cv2
import numpy as np
import pyodbc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

# Kết nối SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:asdjnu12uh12husa.database.windows.net;"
    "DATABASE=ai_2025;"
    "UID=sqladmin;"
    "PWD=YourPassword@123"
)
cursor = conn.cursor()

# Load mô hình YOLO
model = YOLO("best.pt")

@app.route("/")
def index():
    return "Face Recognition API is running"

def save_face_to_db(img, name):
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    img_data = img_bytes.getvalue()
    cursor.execute("INSERT INTO Faces (Name, Image) VALUES (?, ?)", name, img_data)
    conn.commit()

def get_all_faces_from_db():
    cursor.execute("SELECT Name, Image FROM Faces")
    rows = cursor.fetchall()
    faces = []
    for name, image_data in rows:
        image = Image.open(BytesIO(image_data)).convert("RGB")
        faces.append((name, np.array(image)))
    return faces

def detect_face(image):
    results = model.predict(image, conf=0.3, save=False)
    boxes = results[0].boxes
    if not boxes:
        return None
    xyxy = boxes[0].xyxy[0].tolist()
    x1, y1, x2, y2 = map(int, xyxy)
    return image.crop((x1, y1, x2, y2))

@app.route("/register", methods=["POST"])
def register():
    name = request.form.get("name")
    file = request.files.get("image")
    if not name or not file:
        return jsonify({"error": "Missing name or image"}), 400

    image = Image.open(file.stream).convert("RGB")
    face = detect_face(image)
    if face:
        save_face_to_db(face, name)
        return jsonify({"message": "Face registered"}), 200
    return jsonify({"error": "No face detected"}), 400

@app.route("/verify_frame", methods=["POST"])
def verify_frame():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image provided"}), 400

    try:
        image = Image.open(file.stream).convert("RGB")
    except:
        return jsonify({"error": "Invalid image"}), 400

    face = detect_face(image)
    detected_names = []

    if face:
        face_np = np.array(face)
        faces_in_db = get_all_faces_from_db()
        for name, db_img in faces_in_db:
            try:
                result = DeepFace.verify(face_np, db_img, enforce_detection=False)
                if result["verified"]:
                    detected_names.append(name)
                    break
            except:
                continue

    predictions = [{
        "class_name": name,
        "confidence": 1.0
    } for name in detected_names]

    return jsonify({"predictions": predictions}), 200

@app.route("/verify", methods=["POST"])
def verify_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video provided"}), 400

    temp_path = "temp_input_video.mp4"
    file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return jsonify({"error": "Cannot read video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)
    frame_count = 0
    detected_names = set()
    faces_in_db = get_all_faces_from_db()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval != 0:
            frame_count += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        face = detect_face(image)

        if face:
            face_np = np.array(face)
            for name, db_img in faces_in_db:
                try:
                    result = DeepFace.verify(face_np, db_img, enforce_detection=False)
                    if result["verified"]:
                        detected_names.add(name)
                        break
                except:
                    continue

        frame_count += 1

    cap.release()
    os.remove(temp_path)

    predictions = [{
        "class_id": 0,
        "class_name": name,
        "confidence": 1.0
    } for name in detected_names]

    return jsonify({"predictions": predictions}), 200

@app.route("/faces", methods=["GET"])
def list_faces():
    cursor.execute("SELECT Name, COUNT(*) FROM Faces GROUP BY Name")
    rows = cursor.fetchall()
    face_list = [{"name": name, "count": count} for name, count in rows]
    return jsonify({"faces": face_list})

@app.route("/classes", methods=["GET"])
def get_classes():
    class_list = [{"id": k, "name": v} for k, v in model.names.items()]
    return jsonify({"classes": class_list})

if __name__ == "__main__":
    app.run(debug=True)
