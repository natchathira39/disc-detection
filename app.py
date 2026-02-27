from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

app = FastAPI()

MODEL_PATH = "disc_defect_model.keras"
FILE_ID = "11eFwJ0gLQMYvOu0bee6nwi2sMCtDxWDz"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"status": "Disc Defect Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    result = "FAIL" if prediction[0][0] < 0.5 else "PASS"
    confidence = float(prediction[0][0])
    return {"result": result, "confidence": confidence}
