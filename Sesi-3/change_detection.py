import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def ndvi(red, nir):
    """NDVI = (NIR - Red) / (NIR + Red); nilai tinggi = vegetasi lebat."""
    red, nir = red.astype("float32"), nir.astype("float32")
    return (nir - red) / (nir + red + 1e-6)


def synthetic_scene(size=200, seed=7):
    """Buat dua 'citra' sintetis: t1 hijau, t2 sebagian lahan jadi terbangun."""
    rng = np.random.default_rng(seed)
    red1 = rng.normal(0.10, 0.02, (size, size))
    nir1 = rng.normal(0.55, 0.04, (size, size))
    red2, nir2 = red1.copy(), nir1.copy()
    # tiga area berubah (vegetasi -> lahan terbuka/bangunan)
    for (r, c, h, w) in [(30, 40, 35, 45), (110, 120, 40, 30), (150, 30, 25, 25)]:
        red2[r:r + h, c:c + w] = rng.normal(0.30, 0.03, (h, w))
        nir2[r:r + h, c:c + w] = rng.normal(0.28, 0.03, (h, w))
    return (red1, nir1), (red2, nir2)


def load_tif(path, red_band, nir_band):
    import rasterio  # diimpor di sini agar mode demo tidak butuh rasterio

    with rasterio.open(path) as src:
        return src.read(red_band).astype("float32"), src.read(nir_band).astype("float32")


def detect_change(ndvi1, ndvi2, threshold=-0.25, min_area=50):
    """Image differencing: penurunan NDVI besar = kehilangan vegetasi."""
    diff = ndvi2 - ndvi1
    mask = diff < threshold
    # bersihkan noise (morphological opening) lalu hapus objek kecil
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3)))
    labels, n = ndi.label(mask)
    areas = ndi.sum(mask, labels, range(1, n + 1))
    keep = np.isin(labels, [i + 1 for i, a in enumerate(areas) if a >= min_area])
    return diff, keep


def summarize(keep, pixel_area_m2=100.0):
    """Ringkasan statistik + bounding box tiap area berubah (mirip atribut GIS)."""
    labels, n = ndi.label(keep)
    features = []
    for i, sl in enumerate(ndi.find_objects(labels), start=1):
        area_px = int((labels[sl] == i).sum())
        features.append({
            "id": i,
            "luas_m2": area_px * pixel_area_m2,
            "bbox_row": [sl[0].start, sl[0].stop],
            "bbox_col": [sl[1].start, sl[1].stop],
        })
    total = keep.sum() * pixel_area_m2
    return {"jumlah_area_berubah": n, "total_luas_m2": float(total), "features": features}


def plot(ndvi1, ndvi2, diff, keep, out="hasil_change_detection.png"):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for a, img, t, cm in zip(
        ax, [ndvi1, ndvi2, diff, keep],
        ["NDVI t1", "NDVI t2", "Selisih NDVI", "Area Berubah"],
        ["RdYlGn", "RdYlGn", "coolwarm", "gray"],
    ):
        a.imshow(img, cmap=cm)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Gambar disimpan: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t1")
    p.add_argument("--t2")
    p.add_argument("--red", type=int, default=3)
    p.add_argument("--nir", type=int, default=4)
    p.add_argument("--threshold", type=float, default=-0.25)
    a = p.parse_args()

    if a.t1 and a.t2:
        (r1, n1), (r2, n2) = load_tif(a.t1, a.red, a.nir), load_tif(a.t2, a.red, a.nir)
    else:
        print("Mode demo: memakai data sintetis")
        (r1, n1), (r2, n2) = synthetic_scene()

    v1, v2 = ndvi(r1, n1), ndvi(r2, n2)
    diff, keep = detect_change(v1, v2, threshold=a.threshold)
    report = summarize(keep)
    print(json.dumps(report, indent=2))
    with open("laporan_perubahan.json", "w") as f:
        json.dump(report, f, indent=2)
    plot(v1, v2, diff, keep)


if __name__ == "__main__":
    main()
