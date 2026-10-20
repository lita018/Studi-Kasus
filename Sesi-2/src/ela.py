"""Error Level Analysis (ELA) + skor kecurigaan per-blok.

Ide: gambar yang sebagian areanya ditempel (splicing) dari sumber lain
punya riwayat kompresi JPEG berbeda. Jika disimpan ulang pada kualitas
tertentu, area tempelan menghasilkan error level yang berbeda dari
area sekitarnya.
"""
import io
import numpy as np
from PIL import Image


def ela_map(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Kembalikan peta selisih (H x W) antara gambar asli dan versi re-JPEG."""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(img, dtype=np.int16) - np.asarray(resaved, dtype=np.int16))
    return diff.mean(axis=2)  # rata-rata 3 kanal


def block_scores(emap: np.ndarray, block: int = 16) -> np.ndarray:
    h, w = emap.shape
    h2, w2 = h - h % block, w - w % block
    e = emap[:h2, :w2].reshape(h2 // block, block, w2 // block, block)
    return e.mean(axis=(1, 3))


def suspicion_score(path: str, quality: int = 75, block: int = 16, low_ratio: float = 0.6) -> dict:
    """Skor kecurigaan per gambar.

    frac_low : porsi blok yang error level-nya jauh di bawah median
               (< low_ratio * median). Area tempelan yang sebelumnya sudah
               dikompres berat cenderung punya error level rendah.
    max_z    : deviasi robust terbesar (info tambahan).
    """
    img = Image.open(path)
    bs = block_scores(ela_map(img, quality), block)
    med = np.median(bs)
    mad = np.median(np.abs(bs - med)) + 1e-6
    z = (bs - med) / (1.4826 * mad)
    return {
        "file": path,
        "frac_low": float(np.mean(bs < low_ratio * med)),
        "max_z": float(np.max(np.abs(z))),
    }
