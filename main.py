from fastapi import FastAPI,File,UploadFile
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
app = FastAPI()
model = tf.keras.models.load_model("../models/potato_model")
classes = ["Early blight","healthy","late blight"]
@app.post("/ping")



async def ping(file: UploadFile=File(...)):
    img = readimg(await file.read())
    imgb = np.expand_dims(img,0)
    pre = model.predict(imgb)
    print(classes[np.argmax(pre[0])])
def readimg(data):
    im = np.array(Image.open(BytesIO(data)))
    return im

if __name__ == "__main__":
    uvicorn.run(app,host='localhost',port=8000)