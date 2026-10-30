import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from src.utils import SEED

COLOR_TYPES = ["Tshirts", "Shirts", "Tops", "Casual Shoes", "Handbags"]     
COLOR_COLOURS = ["Black", "White", "Blue", "Red", "Green", "Yellow"]       
COLOR_IMG_SIZE = 28                                                         
MAX_PER_CELL = 100      
MIN_PER_CELL = 10       


def _to_square_small(img, size=COLOR_IMG_SIZE):
    img = img.convert("RGB")
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))  
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))     
    return np.asarray(canvas.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def prepare_color_subset(path="data/color_subset.npz", seed=SEED):
    if os.path.exists(path):                                     
        data = np.load(path)
        return data["images"], data["types"], data["colours"]

    from datasets import load_dataset                           
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")   


    meta = ds.select_columns(["articleType", "baseColour"]).to_pandas()
    sel = meta[meta.articleType.isin(COLOR_TYPES) & meta.baseColour.isin(COLOR_COLOURS)]

    rng = np.random.default_rng(seed)
    chosen = []
    for _, group in sel.groupby(["articleType", "baseColour"]):
        if len(group) < MIN_PER_CELL:
            continue
        n = min(len(group), MAX_PER_CELL)
        chosen.extend(rng.choice(group.index.to_numpy(), size=n, replace=False))  
    chosen = np.sort(np.array(chosen))

  
    sub = ds.select(chosen.tolist())
    images = np.stack([_to_square_small(ex["image"]) for ex in sub])
    types = np.array(sub["articleType"])
    colours = np.array(sub["baseColour"])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, images=images, types=types, colours=colours)
    return images, types, colours


def get_color_splits(query_size=0.2, seed=SEED):
    images, types, colours = prepare_color_subset()
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0     
    strata = np.char.add(np.char.add(types, "_"), colours)              
    idx_gal, idx_q = train_test_split(np.arange(len(X)), test_size=query_size,
                                      stratify=strata, random_state=seed)
    return (X[idx_gal], types[idx_gal], colours[idx_gal],
            X[idx_q], types[idx_q], colours[idx_q])



def to_grayscale(X):
    imgs = X.reshape(-1, 28, 28, 3)
    gray = imgs @ np.array([0.299, 0.587, 0.114], dtype=np.float32)  
    return gray.reshape(len(X), -1)


def color_histogram(X, bins=4, white_thr=0.94, min_pixels=20):
    imgs = X.reshape(len(X), -1, 3)                                    
    feats = np.zeros((len(X), bins ** 3), dtype=np.float32)
    for n, px in enumerate(imgs):
        fg = px[~np.all(px > white_thr, axis=1)]                       
        if len(fg) < min_pixels:                                     
            fg = px
        q = np.minimum((fg * bins).astype(int), bins - 1)             
        codes = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]      
        h = np.bincount(codes, minlength=bins ** 3).astype(np.float32)
        feats[n] = h / h.sum()                                        
    return feats