"""
Studi Kasus: Deteksi Citra Termanipulasi dengan Machine Learning
Sesi 4 - Introduction to Machine Learning (Informatics Practitioner Lecture Series 2026)

Alur mengikuti workflow materi:
  data -> preprocessing -> split train/test -> pilih model -> train -> evaluasi
Dua pendekatan dibandingkan:
  1. Supervised   : Random Forest (data berlabel asli/dimanipulasi)
  2. Unsupervised : Isolation Forest (anomaly detection, tanpa label saat training)
"""
import io
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

rng = np.random.default_rng(42)
SIZE = 128


# ---------- 1. DATA COLLECTION (sintetis, supaya bisa jalan tanpa dataset) ----------
def make_authentic():
    """Citra 'asli': gradien halus + noise ringan, lalu kompresi JPEG sekali."""
    x = np.linspace(0, 255, SIZE)
    base = np.add.outer(x, x[::-1]) / 2
    img = np.stack([base, base * 0.8, base * 0.6], axis=-1)
    img += rng.normal(0, 6, img.shape)
    im = Image.fromarray(np.clip(img, 0, 255).astype("uint8"))
    return jpeg_roundtrip(im, 90)


def make_tampered():
    """Citra 'dimanipulasi': tempel patch dari citra lain (splicing) lalu simpan JPEG."""
    im = make_authentic()
    donor = make_authentic().filter(ImageFilter.GaussianBlur(1.5))
    patch = donor.crop((10, 10, 50, 50))
    patch = patch.point(lambda p: min(255, int(p * 1.25)))   # beda pencahayaan
    x, y = rng.integers(20, SIZE - 60, 2)
    im.paste(patch, (int(x), int(y)))
    return jpeg_roundtrip(im, 90)


def jpeg_roundtrip(im, q):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------- 2. PREPROCESSING / FEATURE ENGINEERING (Error Level Analysis) ----------
def ela_features(im, q=75):
    """Simpan ulang JPEG kualitas q, hitung selisih -> ambil statistik sebagai fitur."""
    diff = np.asarray(ImageChops.difference(im, jpeg_roundtrip(im, q)), dtype=float).mean(axis=2)
    return [diff.mean(), diff.std(), diff.max(),
            np.percentile(diff, 95), (diff > diff.mean() + 2 * diff.std()).mean()]


N = 300
X = [ela_features(make_authentic()) for _ in range(N)] + \
    [ela_features(make_tampered()) for _ in range(N)]
y = np.array([0] * N + [1] * N)          # 0 = asli, 1 = dimanipulasi
X = np.array(X)

# Normalisasi (min-max) seperti dijelaskan pada materi preprocessing
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-9)

# ---------- 3. SPLIT DATA 80% train / 20% test ----------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          stratify=y, random_state=42)


def report(nama, y_true, y_pred):
    print(f"\n=== {nama} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_true, y_pred))


# ---------- 4A. SUPERVISED: Random Forest (klasifikasi) ----------
rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_tr, y_tr)
report("Supervised - Random Forest", y_te, rf.predict(X_te))

# ---------- 4B. UNSUPERVISED: Isolation Forest (anomaly detection) ----------
# Dilatih HANYA dengan citra asli (tanpa label) -> citra manipulasi dianggap anomali.
iso = IsolationForest(contamination=0.1, random_state=42).fit(X_tr[y_tr == 0])
pred_iso = (iso.predict(X_te) == -1).astype(int)   # -1 = anomali -> 1 (dimanipulasi)
report("Unsupervised - Isolation Forest", y_te, pred_iso)

# ---------- 5. Cek overfitting (poin materi: akurasi terlalu sempurna perlu dicurigai) ----------
print(f"\nAkurasi train RF: {rf.score(X_tr, y_tr):.3f} | test RF: {rf.score(X_te, y_te):.3f}")
print("Catatan: data sintetis -> hasil terlalu mulus; uji ulang dengan dataset nyata (mis. CASIA v2).")
