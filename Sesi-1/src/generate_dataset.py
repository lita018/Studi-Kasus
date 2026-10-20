"""
Generator dataset sintetik: gambar asli vs gambar yang dimanipulasi.

Mirip "PT Maju Bersama" di materi Pak Andria: data sintetik dipakai supaya
seluruh pipeline bisa direproduksi tanpa mengunduh dataset besar.

Kelas  : 0 = asli, 1 = dimanipulasi (sekitar 15% -> data TIDAK seimbang)
Jenis manipulasi:
  - copy_move : menyalin satu area ke area lain pada gambar yang sama
  - splice    : menempel area dari gambar lain
  - text_edit : menimpa area dengan angka/teks baru (mis. mengubah nominal)

Semua gambar (asli maupun palsu) melewati riwayat kompresi yang sama
(simpan JPEG Q acak -> baca -> [manipulasi] -> simpan JPEG Q=90). Ini sengaja:
kalau hanya gambar palsu yang dikompresi dua kali, model cukup mendeteksi
"dikompresi dua kali" = data leakage, bukan mendeteksi manipulasi.
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

SIZE = 128
FINAL_Q = 90


def make_base_image(rng):
    """Gambar dasar: gradien halus + bentuk acak + angka + noise sensor."""
    h = w = SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    c0 = rng.uniform(60, 200, 3)
    gx, gy = rng.uniform(-0.6, 0.6, 2)
    base = c0[None, None, :] + gx * xx[..., None] + gy * yy[..., None]
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for _ in range(rng.integers(5, 9)):
        x0, y0 = rng.integers(0, w - 30, 2)
        x1, y1 = x0 + rng.integers(15, 45), y0 + rng.integers(15, 45)
        col = tuple(int(v) for v in rng.integers(20, 235, 3))
        if rng.random() < 0.5:
            draw.rectangle([x0, y0, x1, y1], fill=col)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=col)
    for _ in range(3):
        txt = "".join(str(d) for d in rng.integers(0, 10, rng.integers(3, 6)))
        pos = (int(rng.integers(2, w - 40)), int(rng.integers(2, h - 16)))
        draw.text(pos, txt, fill=(10, 10, 10))
    arr = np.asarray(img).astype(np.float32)
    sigma = rng.uniform(2.0, 5.0)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def camera_history(img, rng):
    """Simulasi riwayat kompresi: simpan JPEG kualitas acak lalu baca ulang."""
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(rng.integers(70, 96)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def copy_move(img, rng):
    s = int(rng.integers(24, 41))
    x0, y0 = rng.integers(0, SIZE - s, 2)
    for _ in range(50):
        x1, y1 = rng.integers(0, SIZE - s, 2)
        if abs(x1 - x0) > s + 4 or abs(y1 - y0) > s + 4:
            break
    patch = img.crop((x0, y0, x0 + s, y0 + s))
    out = img.copy()
    out.paste(patch, (int(x1), int(y1)))
    return out


def splice(img, donor, rng):
    s = int(rng.integers(24, 41))
    dx, dy = rng.integers(0, SIZE - s, 2)
    x1, y1 = rng.integers(0, SIZE - s, 2)
    patch = donor.crop((dx, dy, dx + s, dy + s))
    out = img.copy()
    out.paste(patch, (int(x1), int(y1)))
    return out


def text_edit(img, rng):
    w, h = int(rng.integers(30, 52)), int(rng.integers(14, 22))
    x0, y0 = rng.integers(0, SIZE - w), rng.integers(0, SIZE - h)
    region = img.crop((x0, y0, x0 + w, y0 + h))
    fill = tuple(int(v) for v in np.asarray(region).reshape(-1, 3).mean(0))
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=fill)
    txt = "".join(str(v) for v in rng.integers(0, 10, 4))
    d.text((int(x0 + 3), int(y0 + 3)), txt, fill=(10, 10, 10))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--tamper-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(args.n):
        base = camera_history(make_base_image(rng), rng)
        tampered = rng.random() < args.tamper_rate
        ttype = "none"
        out = base
        if tampered:
            ttype = str(rng.choice(["copy_move", "splice", "text_edit"]))
            if ttype == "copy_move":
                out = copy_move(base, rng)
            elif ttype == "splice":
                donor = camera_history(make_base_image(rng), rng)
                out = splice(base, donor, rng)
            else:
                out = text_edit(base, rng)
        path = os.path.join(img_dir, f"img_{i:04d}.jpg")
        out.save(path, "JPEG", quality=FINAL_Q)
        rows.append({"id": i, "path": path, "label": int(tampered),
                     "tamper_type": ttype})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "metadata.csv"), index=False)
    print(f"{len(df)} gambar -> {img_dir}")
    print(df["tamper_type"].value_counts().to_string())
    print(f"proporsi dimanipulasi: {df['label'].mean():.1%}")


if __name__ == "__main__":
    main()
