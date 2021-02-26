import numpy as np
from fastapi import FastAPI, Body
from model.utils import get_model, preprocess, LABELS

app = FastAPI()
model = get_model()

@app.post("/get-breed/")
def get_breed(image:str = Body(..., embed=True)):
    image = preprocess(image)
    if isinstance(image, dict):
        return image
    output = model.predict([image])
    index = np.argmax(output)
    pct = max(output)
    return {"Label": LABELS[index], "Probability": pct}