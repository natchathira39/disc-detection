from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

app = FastAPI()

# ─── Model config ─────────────────────────────────────────────────────────────
MODEL_PATH = "best_model_20260208_073549.h5"
FILE_ID    = "1q0LGb0baNFdatTKM2PhGMI89GNPC4I5B"   # ← only line you need to change

CLASS_NAMES = ["good", "patches", "rolled_pits", "scratches", "waist_folding"]
IMG_SIZE    = (224, 224)   # MobileNetV2 input size used during training

# ─── Download model from Google Drive on first startup ────────────────────────
if not os.path.exists(MODEL_PATH):
    print("Downloading disc defect model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "Disc Defect Detection API Running", "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    # Preprocess exactly as done during training
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    image    = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    # Inference
    probs      = model.predict(img_array)[0]          # shape (5,)
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_class,
        "confidence":      round(confidence, 4),
        "is_defective":    pred_class != "good",
        "probabilities": {
            name: round(float(p), 4)
            for name, p in zip(CLASS_NAMES, probs)
        }
    }
