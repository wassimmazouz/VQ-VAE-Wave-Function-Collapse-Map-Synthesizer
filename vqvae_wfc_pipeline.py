"""
vqvae + wfc pipeline
--------------------
NEW: Train / Generate from a single big image (e.g., 4096x4096) by subdividing into 256x256 tiles.

Examples
--------
# Train from a single 4096x4096 image:
python vqvae_wfc_pipeline.py --mode train --big_image ./maps/huge_map.png --tile 256 --epochs 100 --batch 8 --save vqvae_model.keras

# Generate from the same single big image:
python vqvae_wfc_pipeline.py --mode gen --load vqvae_model.keras --big_image ./maps/huge_map.png --tile 256 --out_codes 8 8 --out_png generated_map.png

# (Old behavior) Train from a folder of tiles:
python vqvae_wfc_pipeline.py --mode train --data_dir ./tiles --epochs 100 --batch 8 --save vqvae_model.keras
"""
import argparse
import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from tensorflow import keras

# local imports (same directory)
import wfc as wfc
import vqvae as vq


# -----------------------
# Data loading utilities
# -----------------------

def load_images_folder(folder: str, imsize: int = 256) -> np.ndarray:
    """Load all images in a folder and resize to (imsize, imsize). Returns array scaled to [-0.5, 0.5]."""
    paths = sorted(
        glob.glob(os.path.join(folder, "*.png"))
        + glob.glob(os.path.join(folder, "*.jpg"))
        + glob.glob(os.path.join(folder, "*.jpeg"))
    )
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((imsize, imsize), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32) / 255.0 - 0.5
        imgs.append(arr)
    if not imgs:
        raise ValueError(f"No images found in {folder}.")
    return np.stack(imgs, axis=0)


def load_tiles_from_big_image(path: str, tile: int = 256, strict: bool = True) -> np.ndarray:
    """
    Load a single big image and subdivide it into non-overlapping tile x tile crops.
    Returns array of shape (N_tiles, tile, tile, 3) scaled to [-0.5, 0.5].

    If strict=True, raises if width/height are not exact multiples of tile.
    If strict=False, the image is center-cropped to the largest multiple.
    """
    im = Image.open(path).convert("RGB")
    W, H = im.size
    if (W % tile != 0 or H % tile != 0):
        if strict:
            raise ValueError(
                f"Image size {W}x{H} not divisible by tile={tile}. "
                f"Use a divisible size (e.g., 4096x4096 for tile=256), or run with strict=False and we will center-crop."
            )
        # center-crop to largest multiple of tile
        new_W = (W // tile) * tile
        new_H = (H // tile) * tile
        left = (W - new_W) // 2
        top = (H - new_H) // 2
        im = im.crop((left, top, left + new_W, top + new_H))
        W, H = im.size

    tiles = []
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            crop = im.crop((x, y, x + tile, y + tile))
            arr = np.asarray(crop).astype(np.float32) / 255.0 - 0.5
            tiles.append(arr)
    if not tiles:
        raise ValueError("No tiles were extracted from the big image.")
    return np.stack(tiles, axis=0)  # (N, tile, tile, 3)


# -----------------------
# VQ-VAE train / encode
# -----------------------

def train_vqvae_from_array(x: np.ndarray, save_path: str,
                           latent_dim: int = 512, num_embeddings: int = 16,
                           epochs: int = 100, batch: int = 8):
    """
    Train VQ-VAE given an input array of shape (N, H, W, 3) in [-0.5, 0.5].
    """
    var = np.var((x + 0.5))  # variance of [0,1] scale
    trainer = vq.VQVAETrainer(var, imsize=x.shape[1], latent_dim=latent_dim, num_embeddings=num_embeddings)
    trainer.compile(optimizer=keras.optimizers.Adam())
    cb = keras.callbacks.EarlyStopping(monitor="loss", patience=10, min_delta=1e-3, restore_best_weights=True)
    trainer.fit(x, epochs=epochs, batch_size=batch, callbacks=[cb], verbose=1)
    trainer.vqvae.save(save_path)
    print(f"Saved VQ-VAE to {save_path}")


def train_vqvae(data_dir: str, save_path: str, imsize: int = 256, latent_dim: int = 512,
                num_embeddings: int = 16, epochs: int = 100, batch: int = 8):
    """
    Old behavior: train from a folder of already-sized tiles.
    """
    x = load_images_folder(data_dir, imsize)
    train_vqvae_from_array(x, save_path, latent_dim, num_embeddings, epochs, batch)


def train_vqvae_from_big_image(big_image: str, save_path: str, tile: int = 256,
                               latent_dim: int = 512, num_embeddings: int = 16,
                               epochs: int = 100, batch: int = 8, strict: bool = True):
    """
    New behavior: train from a single large image by subdividing into tiles.
    """
    x = load_tiles_from_big_image(big_image, tile=tile, strict=strict)
    train_vqvae_from_array(x, save_path, latent_dim, num_embeddings, epochs, batch)


def encode_dataset_codes(vqvae_model, arr: np.ndarray) -> Tuple[int, int, np.ndarray]:
    """
    Encode an in-memory dataset (N, H, W, 3) in [-0.5, 0.5] to VQ codes.
    Returns (Hc, Wc, codes) with codes shape (N, Hc, Wc).
    """
    Hc, Wc, codes = vq.encode_to_codes(vqvae_model, arr)
    return Hc, Wc, codes


# -----------------------
# WFC over codes & decode
# -----------------------

def synthesize_codes(codes_dataset: np.ndarray, out_hc: int, out_wc: int) -> np.ndarray:
    """
    Learn WFC patterns from many code-index grids, then synthesize a new grid.
    Concatenate datasets along height with separators to avoid mixing edges (simple trick).
    """
    B, Hc, Wc = codes_dataset.shape
    if B == 1:
        big = codes_dataset[0]
    else:
        sep_val = int(codes_dataset.max())
        sep = np.full((1, Wc), sep_val, dtype=codes_dataset.dtype)
        rows = []
        for b in range(B):
            rows.append(codes_dataset[b])
            rows.append(sep)
        big = np.concatenate(rows, axis=0)  # (B*Hc + (B-1), Wc)

    big_rgb = np.stack([big, big, big], axis=-1).astype(np.uint8)
    out_rgb = wfc.wave_function_collapse(big_rgb, pattern_size=1, out_h=out_hc, out_w=out_wc, flip=False, rotate=False)
    out_codes = out_rgb[..., 0].astype(np.int32)
    return out_codes


def generate_maps_from_array(model_path: str, arr: np.ndarray, out_codes_hw: Tuple[int, int], out_png: str):
    """
    Encode an array (N, H, W, 3) in [-0.5, 0.5] to codes, run WFC, decode and save PNG.
    """
    model = keras.models.load_model(model_path, compile=False)
    Hc, Wc, codes = encode_dataset_codes(model, arr)
    out_hc, out_wc = out_codes_hw
    new_codes = synthesize_codes(codes, out_hc, out_wc)[None, ...]  # (1, out_hc, out_wc)
    xhat = vq.decode_from_codes(model, new_codes)  # in [-0.5, 0.5]
    img = np.clip((xhat[0] + 0.5) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_png)
    print(f"Saved generated map to {out_png} (shape {img.shape})")


# -----------------------
# CLI
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "gen"], required=True)

    # Data sources
    parser.add_argument("--data_dir", type=str, help="Folder of tiles (old behavior).")
    parser.add_argument("--big_image", type=str, help="Path to one large image to be subdivided into tiles.")
    parser.add_argument("--tile", type=int, default=256, help="Tile size used when --big_image is provided.")
    parser.add_argument("--strict", action="store_true", help="Fail if big image is not divisible by tile (default False).")
    parser.set_defaults(strict=False)

    # VQ-VAE config
    parser.add_argument("--imsize", type=int, default=256, help="(Folder mode only) resize images to this square size.")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--num_embeddings", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--save", type=str, default="vqvae_model.keras")
    parser.add_argument("--load", type=str, help="Path to saved VQ-VAE (.keras) for gen mode.")

    # Generation
    parser.add_argument("--out_codes", type=int, nargs=2, metavar=("Hc", "Wc"), help="Output code-grid size for gen.")
    parser.add_argument("--out_png", type=str, default="generated_map.png")

    args = parser.parse_args()

    # ---------------- Train ----------------
    if args.mode == "train":
        if args.big_image:
            print(f"Training from big image: {args.big_image}, tile={args.tile}")
            train_vqvae_from_big_image(
                big_image=args.big_image,
                save_path=args.save,
                tile=args.tile,
                latent_dim=args.latent_dim,
                num_embeddings=args.num_embeddings,
                epochs=args.epochs,
                batch=args.batch,
                strict=args.strict,
            )
        elif args.data_dir:
            print(f"Training from folder: {args.data_dir}")
            train_vqvae(
                data_dir=args.data_dir,
                save_path=args.save,
                imsize=args.imsize,
                latent_dim=args.latent_dim,
                num_embeddings=args.num_embeddings,
                epochs=args.epochs,
                batch=args.batch,
            )
        else:
            raise ValueError("Provide either --big_image or --data_dir for training.")

    # ---------------- Generate ----------------
    else:  # gen
        if not args.load:
            raise ValueError("--load path is required for gen mode.")
        if not args.out_codes:
            raise ValueError("--out_codes Hc Wc is required for gen mode.")

        if args.big_image:
            print(f"Generating from big image: {args.big_image}, tile={args.tile}")
            arr = load_tiles_from_big_image(args.big_image, tile=args.tile, strict=args.strict)
            generate_maps_from_array(args.load, arr, tuple(args.out_codes), args.out_png)
        elif args.data_dir:
            print(f"Generating from folder: {args.data_dir}")
            arr = load_images_folder(args.data_dir, imsize=args.imsize)
            generate_maps_from_array(args.load, arr, tuple(args.out_codes), args.out_png)
        else:
            raise ValueError("Provide either --big_image or --data_dir for generation.")


if __name__ == "__main__":
    main()
