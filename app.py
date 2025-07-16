from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import numpy as np
import cv2
import uuid
import pyodbc
import io
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Kết nối SQL Server
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=tcp:asdjnu12uh12husa.database.windows.net;"
    "DATABASE=sic2025;"
    "UID=sqladmin;"
    "PWD=YourPassword@123"
)
cursor = conn.cursor()

# API: Đăng ký khuôn mặt
@app.route('/register', methods=['POST'])
def register_face():
    name = request.form.get('name')
    image_file = request.files.get('image')

    if not name or not image_file:
        return jsonify({'status': 'fail', 'message': 'Missing name or image'}), 400

    filename = f"{name}_{uuid.uuid4().hex}.jpg"
    image_bytes = image_file.read()

    # Lưu vào database
    cursor.execute(
        "INSERT INTO Faces (Name, Image, Filename, UploadTime) VALUES (?, ?, ?, ?)",
        name, image_bytes, filename, datetime.now()
    )
    conn.commit()

    return jsonify({'status': 'success', 'message': f'Image registered for {name}', 'filename': filename})

# API: Nhận diện khuôn mặt
@app.route('/recognize', methods=['POST'])
def recognize_face():
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({'status': 'fail', 'message': 'Missing image'}), 400

    input_bytes = image_file.read()
    npimg = np.frombuffer(input_bytes, np.uint8)
    input_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Lấy ảnh từ DB
    cursor.execute("SELECT Id, Name, Image FROM Faces")
    rows = cursor.fetchall()

    best_match = None
    lowest_distance = float('inf')

    for row in rows:
        db_name = row.Name
        db_image_bytes = row.Image

        npimg_db = np.frombuffer(db_image_bytes, np.uint8)
        db_img = cv2.imdecode(npimg_db, cv2.IMREAD_COLOR)

        try:
            result = DeepFace.verify(input_img, db_img, enforce_detection=False)
            if result["verified"]:
                distance = result["distance"]
                if distance < lowest_distance:
                    lowest_distance = distance
                    best_match = {
                        "class_name": db_name,
                        "confidence": round(1 - distance, 4)  # chuyển sang độ tin cậy
                    }
        except Exception:
            continue

    if best_match:
        return jsonify(best_match)
    else:
        return jsonify({'status': 'fail', 'message': 'No face matched'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
