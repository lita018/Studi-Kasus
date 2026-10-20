"""Deteksi serial vs paralel (multiprocessing) + keluaran CSV."""
import argparse
import csv
import glob
import os
import time
from multiprocessing import Pool

from ela import suspicion_score


def list_images(root):
    items = []
    for label in ("authentic", "tampered"):
        for p in sorted(glob.glob(os.path.join(root, label, "*.jpg"))):
            items.append((p, label))
    return items


def _work(args):
    path, label = args
    r = suspicion_score(path)
    r["label"] = label
    return r


def run(items, workers):
    t0 = time.perf_counter()
    if workers <= 1:
        res = [_work(i) for i in items]
    else:
        with Pool(workers) as pool:
            res = pool.map(_work, items, chunksize=max(1, len(items) // (workers * 4)))
    return res, time.perf_counter() - t0


def evaluate(res, thr):
    tp = sum(1 for r in res if r["label"] == "tampered" and r["frac_low"] > thr)
    tn = sum(1 for r in res if r["label"] == "authentic" and r["frac_low"] <= thr)
    fp = sum(1 for r in res if r["label"] == "authentic" and r["frac_low"] > thr)
    fn = sum(1 for r in res if r["label"] == "tampered" and r["frac_low"] <= thr)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=(tp + tn) / max(1, len(res)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--thr", type=float, default=0.02,
                    help="ambang frac_low (porsi blok mencurigakan) untuk menandai gambar")
    ap.add_argument("--out", default="results/detections.csv")
    a = ap.parse_args()

    items = list_images(a.data)
    res, dt = run(items, a.workers)
    ev = evaluate(res, a.thr)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "frac_low", "max_z"])
        w.writeheader()
        for r in res:
            w.writerow({k: r[k] for k in w.fieldnames})
    print(f"{len(items)} gambar | worker={a.workers} | waktu={dt:.2f}s | thr={a.thr:.2f}")
    print(f"TP={ev['tp']} TN={ev['tn']} FP={ev['fp']} FN={ev['fn']} | akurasi={ev['acc']:.2%}")


if __name__ == "__main__":
    main()
