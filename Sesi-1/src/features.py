"""
Ekstraksi fitur pengolahan citra digital (PCD) untuk deteksi manipulasi.

Fitur dirancang dari bukti forensik klasik:
  1. ELA (Error Level Analysis): area yang riwayat kompresinya berbeda
     memberi error rekompresi yang berbeda.
  2. Residu noise: area hasil tempel/hapus biasanya noise-nya tidak konsisten.
  3. Copy-move: blok yang hampir identik di dua lokasi berjauhan.
"""
import io

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

BLK = 16


def _block_stat(arr, fn, blk=BLK):
    h, w = arr.shape
    h2, w2 = h - h % blk, w - w % blk
    a = arr[:h2, :w2].reshape(h2 // blk, blk, w2 // blk, blk)
    return fn(a, axis=(1, 3)).ravel()


def ela_map(img, quality=75):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re = Image.open(buf).convert("RGB")
    d = np.abs(np.asarray(img).astype(np.float32) - np.asarray(re).astype(np.float32))
    return d.mean(axis=2)


def noise_residual(gray):
    return gray - median_filter(gray, size=3)


def copy_move_score(gray, blk=BLK, stride=4, flat_std=8.0, min_dist=24, radius=6.0):
    """Jumlah pasangan blok bertekstur yang hampir identik tapi berjauhan."""
    h, w = gray.shape
    feats, pos = [], []
    for y in range(0, h - blk + 1, stride):
        for x in range(0, w - blk + 1, stride):
            b = gray[y:y + blk, x:x + blk]
            if b.std() < flat_std:
                continue
            pooled = b.reshape(4, blk // 4, 4, blk // 4).mean(axis=(1, 3)).ravel()
            feats.append(pooled)
            pos.append((y, x))
    if len(feats) < 2:
        return 0.0
    feats, pos = np.asarray(feats), np.asarray(pos)
    pairs = np.array(list(cKDTree(feats).query_pairs(r=radius)))
    if len(pairs) == 0:
        return 0.0
    dist = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    return float((dist >= min_dist).sum())


FEATURE_NAMES = [
    "ela_mean", "ela_std", "ela_p99",
    "ela_blk_std", "ela_blk_range", "ela_blk_max_over_median",
    "noise_mean", "noise_blk_std", "noise_blk_cv",
    "noise_min_over_median", "noise_max_over_median",
    "copy_move_pairs",
]


def extract(path):
    img = Image.open(path).convert("RGB")
    gray = np.asarray(img.convert("L")).astype(np.float32)

    ela = ela_map(img)
    eb = _block_stat(ela, np.mean)

    res = np.abs(noise_residual(gray))
    nb = _block_stat(res, np.mean)
    med_n = np.median(nb) + 1e-6

    return {
        "ela_mean": ela.mean(),
        "ela_std": ela.std(),
        "ela_p99": np.percentile(ela, 99),
        "ela_blk_std": eb.std(),
        "ela_blk_range": eb.max() - eb.min(),
        "ela_blk_max_over_median": eb.max() / (np.median(eb) + 1e-6),
        "noise_mean": res.mean(),
        "noise_blk_std": nb.std(),
        "noise_blk_cv": nb.std() / (nb.mean() + 1e-6),
        "noise_min_over_median": nb.min() / med_n,
        "noise_max_over_median": nb.max() / med_n,
        "copy_move_pairs": copy_move_score(gray),
    }
