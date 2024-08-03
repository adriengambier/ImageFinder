import argparse
import os

import numpy as np
from PIL import UnidentifiedImageError
import sqlite3
import sqlite_vec
from tensorflow.keras.applications.resnet50 import ResNet50

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Folder containing images")
    args = parser.parse_args()

    return args


def load_images(folder: str, db: sqlite3.Connection) -> tuple:
    """Load and preprocess images from a specified folder if an embedding is not already computed in the table `images`"""
    list_img = os.listdir(folder)

    list_preprocessed_img = []
    paths = []
    for path in list_img:
        res = db.execute("SELECT path FROM images WHERE path = ?", [path]).fetchone()
        if res is not None:
            continue

        try:
            img_preprocessed = utils.load_img(os.path.join("./images", path))
        except UnidentifiedImageError:
            continue

        paths.append(path)
        list_preprocessed_img.append(img_preprocessed)

    return (paths, list_preprocessed_img)


def embed_images(folder: str):
    """Embed images from a specified folder and save the paths and results to the table `images`"""
    db = sqlite3.connect("./db/images.sqlite")
    db.enable_load_extension(True)
    sqlite_vec.load(db)

    db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS images USING vec0(path TEXT PRIMARY KEY, embedding float[2048])"
    )

    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    paths, list_img = load_images(folder, db)

    if len(paths) == 0:
        return

    embeddings = model.predict(np.vstack(list_img))

    for path, embedding in zip(paths, embeddings):
        db.execute(
            "INSERT INTO images(path, embedding) VALUES (?, ?)",
            [path, utils.serialize_f32(embedding)],
        )

    print(f"{len(paths)} images inserted!")

    db.commit()


if __name__ == "__main__":
    args = parse_args()

    folder = args.folder

    embed_images(folder)
