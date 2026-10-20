"""Benchmark skalabilitas: speedup dan efisiensi paralel."""
import argparse
import csv
import os

from detect import list_images, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--workers", default="1,2,4,8")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", default="results/benchmark.csv")
    a = ap.parse_args()

    items = list_images(a.data)
    ws = [int(x) for x in a.workers.split(",")]
    times = {w: min(run(items, w)[1] for _ in range(a.repeat)) for w in ws}
    t1 = times[ws[0]]
    rows = []
    for w in ws:
        s = t1 / times[w]
        rows.append({"workers": w, "time_s": round(times[w], 3),
                     "speedup": round(s, 2), "efficiency": round(s / w, 2)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{'workers':>8}{'time(s)':>10}{'speedup':>10}{'eff':>8}")
    for r in rows:
        print(f"{r['workers']:>8}{r['time_s']:>10}{r['speedup']:>10}{r['efficiency']:>8}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(ws, [r["speedup"] for r in rows], "o-", label="terukur")
        plt.plot(ws, ws, "--", label="ideal")
        plt.xlabel("jumlah worker"); plt.ylabel("speedup"); plt.legend(); plt.grid(alpha=.3)
        plt.title("Speedup deteksi manipulasi citra")
        plt.savefig(os.path.join(os.path.dirname(a.out), "speedup.png"), dpi=150, bbox_inches="tight")
    except Exception as e:
        print("grafik dilewati:", e)


if __name__ == "__main__":
    main()
