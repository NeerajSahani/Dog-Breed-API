import base64
import numpy as np
from tensorflow.keras.models import load_model


LABELS = ['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']

def get_model():
    return load_model('./modelv2.h5')

def preprocess(image):
    encoded_img = base64.b64decode(image)
    img = np.fromstring(encoded_img, dtype=np.uint8)
    try:
        img.shape = (224, 224, 3)
    except:
        return {"Error": "Image Dimension must be (224, 224, 3)"}
    return img
