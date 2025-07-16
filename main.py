from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import os
import shutil
import uuid

app = FastAPI()

# Cho phép truy cập CORS (nếu gọi từ web frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

# 1. Đăng ký khuôn mặt
@app.post("/register")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    filename = f"{name}.{ext}"
    file_path = os.path.join(FACES_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "message": f"Đã lưu khuôn mặt cho {name}", "filename": filename}

# 2. Xác định khuôn mặt
@app.post("/recognize")
async def recognize_face(file: UploadFile = File(...)):
    temp_path = f"temp_{uuid.uuid4()}.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = DeepFace.find(
            img_path=temp_path,
            db_path=FACES_DIR,
            enforce_detection=False,
            detector_backend="retinaface"
        )

        if result[0].empty:
            return {"match": False, "message": "Không tìm thấy khuôn mặt phù hợp"}

        best_match = result[0].iloc[0]
        identity_path = best_match["identity"]
        matched_name = os.path.basename(identity_path).split(".")[0]
        distance = float(best_match["distance"])

        return {"match": True, "name": matched_name, "distance": distance}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
