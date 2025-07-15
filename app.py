from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from deepface import DeepFace
from PIL import Image
import os
import uuid
import cv2
import numpy as np
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
CORS(app)

model = YOLO("best.pt")
FACE_DB = "face_db"
os.makedirs(FACE_DB, exist_ok=True)

@app.route("/")
def index():
    return "Face Recognition API is running"

def save_face(img, name):
    unique_id = str(uuid.uuid4())[:8]  # tạo ID ngắn
    filename = f"{name}_{unique_id}.jpg"
    path = os.path.join(FACE_DB, filename)
    img.save(path)
    return path

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
        save_face(face, name)
        return jsonify({"message": "Face registered"}), 200
    return jsonify({"error": "No face detected"}), 400

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
    frame_interval = int(fps * 0.5)  # Lấy 1 frame mỗi 0.5 giây

    frame_count = 0
    detected_names = set()

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
            for filename in os.listdir(FACE_DB):
                db_path = os.path.join(FACE_DB, filename)
                try:
                    result = DeepFace.verify(face_np, db_path, enforce_detection=False)
                    if result["verified"]:
                        name = os.path.splitext(filename)[0].split("_")[0]
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

@app.route("/classes", methods=["GET"])
def get_classes():
    class_list = [{"id": k, "name": v} for k, v in model.names.items()]
    return jsonify({"classes": class_list})

@app.route("/faces", methods=["GET"])
def list_faces():
    counter = Counter()
    for filename in os.listdir(FACE_DB):
        if filename.endswith(".jpg"):
            name = os.path.splitext(filename)[0].split("_")[0]
            counter[name] += 1
    face_list = [{"name": name, "count": count} for name, count in counter.items()]
    return jsonify({"faces": face_list})
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
        for filename in os.listdir(FACE_DB):
            db_path = os.path.join(FACE_DB, filename)
            try:
                result = DeepFace.verify(face_np, db_path, enforce_detection=False)
                if result["verified"]:
                    name = os.path.splitext(filename)[0].split("_")[0]
                    detected_names.append(name)
                    break  # chỉ lấy người đầu tiên trùng
            except:
                continue

    predictions = [{
        "class_name": name,
        "confidence": 1.0
    } for name in detected_names]

    return jsonify({"predictions": predictions}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
