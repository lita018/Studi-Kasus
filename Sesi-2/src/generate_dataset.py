"""Membuat dataset sintetis: gambar asli vs gambar hasil splicing.

Tidak butuh dataset eksternal -> mudah direproduksi (prinsip artifact
description / artifact evaluation yang disebut pemateri).
"""
import argparse
import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_scene(rng, size):
    x = np.linspace(0, 1, size)
    base = np.outer(x, np.ones(size))
    ch = [base * rng.uniform(80, 255), base.T * rng.uniform(80, 255),
          (base + base.T) / 2 * rng.uniform(80, 255)]
    arr = np.stack(ch, axis=2) + rng.normal(0, 6, (size, size, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = (int(v) for v in rng.integers(0, size - 40, 2))
        d.ellipse([x0, y0, x0 + int(rng.integers(15, 60)), y0 + int(rng.integers(15, 60))],
                  fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=40, help="jumlah pasang asli/palsu")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(os.path.join(a.out, "authentic"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "tampered"), exist_ok=True)
    for i in range(a.n):
        scene = make_scene(rng, a.size)
        # gambar asli: JPEG kualitas tinggi (95)
        auth = os.path.join(a.out, "authentic", f"img_{i:04d}.jpg")
        scene.save(auth, "JPEG", quality=95)
        # tampered: tempel patch dari sumber lain yang sebelumnya dikompres kualitas rendah (40)
        base = Image.open(auth).convert("RGB")
        donor = make_scene(rng, a.size)
        buf = io.BytesIO()
        donor.save(buf, "JPEG", quality=40)
        buf.seek(0)
        donor = Image.open(buf).convert("RGB")
        pw, ph = (int(v) for v in rng.integers(a.size // 8, a.size // 3, 2))
        sx = int(rng.integers(0, a.size - pw))
        sy = int(rng.integers(0, a.size - ph))
        base.paste(donor.crop((sx, sy, sx + pw, sy + ph)), (sx, sy))
        base.save(os.path.join(a.out, "tampered", f"img_{i:04d}.jpg"), "JPEG", quality=95)
    print(f"Dataset siap di '{a.out}': {a.n} authentic + {a.n} tampered")


if __name__ == "__main__":
    main()
