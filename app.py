from fastapi import FastAPI, UploadFile
import os
import io
import gdown
import numpy as np
from PIL import Image

os.environ["TF_USE_LEGACY_KERAS"] = "1"  # force Keras 2 behavior

import tensorflow as tf
import tf_keras as keras

app = FastAPI()

MODEL_PATH = "disc_model.keras"
FILE_ID    = "1iUABWQZnCs9EBfCtWJe2I8G5pfGdFstm"

CLASS_NAMES = ["good", "patches", "rolled_pits", "scratches", "waist_folding"]
IMG_SIZE    = (224, 224)

if not os.path.exists(MODEL_PATH):
    print("Downloading disc defect model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

@app.get("/")
def home():
    return {"status": "Disc Defect Detection API Running", "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile):
    contents  = await file.read()
    image     = Image.open(io.BytesIO(contents)).convert("RGB")
    image     = image.resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    probs      = model.predict(img_array)[0]
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
