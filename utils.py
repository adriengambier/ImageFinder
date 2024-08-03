from typing import List
import struct
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def load_img(input):
    img = Image.open(input)
    img = img.resize((224, 224))
    
    img_arr = np.expand_dims(np.array(img), axis=0)
    img_preprocessed = preprocess_input(img_arr)

    return img_preprocessed