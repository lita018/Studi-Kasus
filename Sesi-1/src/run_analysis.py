"""
Pipeline CRISP-DM: deteksi gambar dimanipulasi (data sintetik).

Memetakan empat level analitik dari materi:
  Descriptive  -> statistik & distribusi fitur per kelas
  Diagnostic   -> jenis manipulasi mana yang sering lolos (recall per jenis)
  Predictive   -> model klasifikasi (baseline vs LogReg vs RandomForest)
  Prescriptive -> pilih ambang batas berdasarkan biaya lolos vs biaya review manual
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES, extract

SEED = 42
DATA, RES = "data", "results"
COST_MISS, COST_REVIEW = 10.0, 1.0  # asumsi: lolos 10x lebih mahal dari review


def load_features():
    cache = os.path.join(DATA, "features.csv")
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
    if os.path.exists(cache):
        return pd.read_csv(cache)
    feats = pd.DataFrame([extract(p) for p in meta["path"]])
    df = pd.concat([meta, feats], axis=1)
    df.to_csv(cache, index=False)
    return df


def evaluate(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


def expected_cost(y, proba, thr):
    pred = (proba >= thr).astype(int)
    fn = ((pred == 0) & (y == 1)).sum()
    reviewed = pred.sum()
    return float(fn * COST_MISS + reviewed * COST_REVIEW)


def main():
    os.makedirs(RES, exist_ok=True)
    df = load_features()
    X, y = df[FEATURE_NAMES], df["label"].values

    # ---------- 1. DESCRIPTIVE ----------
    print("=== DESCRIPTIVE ===")
    print(f"jumlah gambar: {len(df)} | dimanipulasi: {y.mean():.1%}")
    desc = df.groupby("label")[FEATURE_NAMES].mean().T
    desc.columns = ["asli", "dimanipulasi"]
    desc["rasio"] = desc["dimanipulasi"] / (desc["asli"] + 1e-9)
    desc.round(3).to_csv(os.path.join(RES, "descriptive_feature_means.csv"))
    print(desc.round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, f in zip(axes.ravel(), ["ela_blk_range", "ela_blk_max_over_median",
                                    "noise_blk_cv", "noise_min_over_median",
                                    "copy_move_pairs", "ela_p99"]):
        ax.boxplot([df.loc[df.label == 0, f], df.loc[df.label == 1, f]],
                   tick_labels=["asli", "palsu"])
        ax.set_title(f)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "01_descriptive_boxplots.png"), dpi=130)
    plt.close(fig)

    # ---------- split: stratified; data uji TIDAK dipakai untuk tuning ----------
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.3,
                                      stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]

    models = {
        "baseline_mayoritas": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(StandardScaler(),
                                LogisticRegression(class_weight="balanced",
                                                   max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=SEED, n_jobs=-1),
    }

    # ---------- 2. PREDICTIVE ----------
    print("\n=== PREDICTIVE (data uji, ambang 0.5) ===")
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows, proba_te, proba_oof = [], {}, {}
    for name, m in models.items():
        oof = cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        proba_oof[name], proba_te[name] = oof, p
        rows.append({"model": name, **evaluate(yte, p, 0.5)})
    res = pd.DataFrame(rows).set_index("model").round(3)
    res.to_csv(os.path.join(RES, "02_model_comparison.csv"))
    print(res.to_string())
    best = res.drop(index="baseline_mayoritas")["pr_auc"].idxmax()
    print(f"\nmodel terbaik (PR-AUC): {best}")

    # ---------- 3. PRESCRIPTIVE: ambang dipilih dari CV pada data latih ----------
    grid = np.linspace(0.05, 0.95, 91)
    costs_oof = [expected_cost(ytr, proba_oof[best], t) for t in grid]
    thr = float(grid[int(np.argmin(costs_oof))])
    print(f"\n=== PRESCRIPTIVE ===\nambang optimal (biaya terendah di CV latih): {thr:.2f}")
    tuned = evaluate(yte, proba_te[best], thr)
    print("hasil data uji pada ambang tsb:", {k: round(v, 3) for k, v in tuned.items()})
    cm = confusion_matrix(yte, (proba_te[best] >= thr).astype(int))
    print("confusion matrix [[TN FP],[FN TP]]:\n", cm)

    n_pos = int(yte.sum())
    policies = {
        "tanpa review (loloskan semua)": n_pos * COST_MISS,
        "review manual semua gambar": len(yte) * COST_REVIEW,
        f"model {best} @0.5": expected_cost(yte, proba_te[best], 0.5),
        f"model {best} @{thr:.2f}": expected_cost(yte, proba_te[best], thr),
    }
    pol = pd.Series(policies, name="biaya_total").round(1)
    pol.to_csv(os.path.join(RES, "03_policy_cost.csv"))
    print("\nbiaya total per kebijakan (asumsi lolos=10, review=1):")
    print(pol.to_string())

    # ---------- 4. DIAGNOSTIC: recall per jenis manipulasi ----------
    print("\n=== DIAGNOSTIC (recall per jenis manipulasi, data uji) ===")
    te = df.iloc[idx_te].copy()
    te["pred"] = (proba_te[best] >= thr).astype(int)
    diag = (te[te.label == 1].groupby("tamper_type")
            .agg(jumlah=("pred", "size"), terdeteksi=("pred", "sum")))
    diag["recall"] = (diag["terdeteksi"] / diag["jumlah"]).round(3)
    diag.to_csv(os.path.join(RES, "04_recall_per_tamper_type.csv"))
    print(diag.to_string())

    # ---------- figures ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name in models:
        if name == "baseline_mayoritas":
            continue
        pr, rc, _ = precision_recall_curve(yte, proba_te[name])
        ax[0].plot(rc, pr, label=name)
    ax[0].axhline(yte.mean(), ls="--", c="gray", label="acak")
    ax[0].set(xlabel="recall", ylabel="precision", title="Precision-Recall (data uji)")
    ax[0].legend()
    ax[1].plot(grid, costs_oof)
    ax[1].axvline(thr, c="r", ls="--")
    ax[1].set(xlabel="ambang", ylabel="biaya (CV latih)", title="Ambang optimal")
    ax[2].bar(diag.index, diag["recall"])
    ax[2].set(ylim=(0, 1), title="Recall per jenis manipulasi")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "05_pr_cost_diagnostic.png"), dpi=130)
    plt.close(fig)

    if hasattr(models[best], "feature_importances_"):
        imp = pd.Series(models[best].feature_importances_, index=FEATURE_NAMES)
        imp.sort_values().plot.barh(figsize=(6, 4), title="Feature importance")
        plt.tight_layout()
        plt.savefig(os.path.join(RES, "06_feature_importance.png"), dpi=130)
        plt.close()
        print("\nfitur terpenting:\n", imp.sort_values(ascending=False).head(5).round(3).to_string())

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump({"best_model": best, "threshold": thr,
                   "test_metrics_at_threshold": tuned,
                   "cost_assumption": {"miss": COST_MISS, "review": COST_REVIEW}},
                  f, indent=2)


if __name__ == "__main__":
    main()
