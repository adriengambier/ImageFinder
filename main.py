import io
import sqlite3

import numpy as np
import sqlite_vec
from fastapi import FastAPI, UploadFile
from tensorflow.keras.applications.resnet50 import ResNet50

import utils

app = FastAPI()

db = sqlite3.connect("./db/images.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)

model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


@app.post("/similar")
async def similar_img(file: UploadFile):
    """Take an image as input and return the 3 most similar images with the euclidian distances between the vectors"""

    contents = await file.read()
    image_file = io.BytesIO(contents)

    img_preprocessed = utils.load_img(image_file)
    img_embed = model.predict(img_preprocessed)

    rows = db.execute(
        """
        SELECT
            path,
            distance
        FROM images
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 3
        """,
        [utils.serialize_f32(img_embed[0])],
    ).fetchall()

    return {
        "match_1": {"index": rows[0][0], "similarity": round(rows[0][1], 3)},
        "match_2": {"index": rows[1][0], "similarity": round(rows[1][1], 3)},
        "match_3": {"index": rows[2][0], "similarity": round(rows[2][1], 3)},
    }
