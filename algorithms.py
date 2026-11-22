"""Algoritma SAW, Naive Bayes, dan C.4.5 untuk SPK UMKM Ayam Geprek."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = [
    "kategori",
    "rentang_harga",
    "level_margin",
    "frekuensi_kebutuhan",
]

SAW_CRITERIA_DEFAULT = {
    "rata_jual_mingguan": {"label": "Rata-rata Jual Mingguan", "type": "Benefit", "weight": 0.35},
    "margin_persen": {"label": "Margin (%)", "type": "Benefit", "weight": 0.25},
    "stok_saat_ini": {"label": "Stok Saat Ini", "type": "Cost", "weight": 0.20},
    "lead_time_hari": {"label": "Lead Time (Hari)", "type": "Cost", "weight": 0.20},
}


def enrich_menu_row(row: dict[str, Any]) -> dict[str, Any]:
    """Hitung kolom turunan (rata-rata, margin, rentang_harga, level_margin, label)."""
    m1 = int(row["jual_minggu1"])
    m2 = int(row["jual_minggu2"])
    m3 = int(row["jual_minggu3"])
    m4 = int(row["jual_minggu4"])
    beli = int(row["harga_beli"])
    jual = int(row["harga_jual"])

    avg = round((m1 + m2 + m3 + m4) / 4.0, 1)
    margin = round(((jual - beli) / beli) * 100.0, 1) if beli > 0 else 0.0
    rentang_harga = "Murah" if jual < 10000 else ("Sedang" if jual <= 30000 else "Mahal")
    level_margin = "Rendah" if margin < 10.0 else ("Sedang" if margin <= 20.0 else "Tinggi")
    label = row.get("label") or ("Fast" if avg >= 10.0 else "Slow")

    return {
        **row,
        "rata_jual_mingguan": avg,
        "margin_persen": margin,
        "rentang_harga": rentang_harga,
        "level_margin": level_margin,
        "label": label,
    }


def calculate_saw(
    df: pd.DataFrame,
    weights: dict[str, float],
    criteria_types: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hitung matriks normalisasi R_ij dan nilai preferensi V_i metode SAW."""
    criteria = list(weights.keys())
    total_w = sum(weights.values())
    norm_weights = {k: (v / total_w if total_w > 0 else 0.0) for k, v in weights.items()}

    decision_df = df[["kode", "nama_barang", *criteria]].copy()
    norm_df = df[["kode", "nama_barang"]].copy()

    for col in criteria:
        col_vals = df[col].astype(float)
        if criteria_types[col] == "Benefit":
            max_val = col_vals.max()
            norm_df[col] = (col_vals / max_val) if max_val > 0 else 0.0
        else:
            min_val = col_vals.min()
            norm_df[col] = np.where(col_vals > 0, min_val / col_vals, 0.0)
        norm_df[col] = norm_df[col].round(4)

    scores = np.zeros(len(df), dtype=float)
    for col in criteria:
        scores += norm_df[col].astype(float).to_numpy() * norm_weights[col]

    result_df = df[["kode", "nama_barang", "kategori", "stok_saat_ini", "rata_jual_mingguan", "label"]].copy()
    result_df["nilai_preferensi"] = np.round(scores, 4)
    result_df = result_df.sort_values(by="nilai_preferensi", ascending=False).reset_index(drop=True)
    result_df.insert(0, "peringkat", range(1, len(result_df) + 1))

    return decision_df, norm_df, result_df


def train_naive_bayes(df: pd.DataFrame, features: list[str] | None = None) -> dict[str, Any]:
    """Latih model Categorical Naive Bayes dengan Laplace smoothing."""
    feats = features or CATEGORICAL_FEATURES
    classes = sorted(df["label"].unique().tolist())
    total = len(df)

    priors = {c: (df["label"] == c).sum() / total for c in classes}
    likelihoods: dict[str, dict[str, dict[str, float]]] = {}
    vocab_sizes: dict[str, int] = {}

    for feat in feats:
        values = sorted(df[feat].unique().tolist())
        vocab_sizes[feat] = len(values)
        likelihoods[feat] = {}
        for c in classes:
            sub = df[df["label"] == c]
            c_count = len(sub)
            likelihoods[feat][c] = {
                val: ((sub[feat] == val).sum() + 1) / (c_count + len(values))
                for val in values
            }

    return {
        "features": feats,
        "classes": classes,
        "priors": priors,
        "likelihoods": likelihoods,
        "vocab_sizes": vocab_sizes,
    }


def predict_naive_bayes(model: dict[str, Any], sample: dict[str, str]) -> tuple[str, dict[str, float]]:
    """Prediksi kelas Fast/Slow dan probabilitas posteriornya."""
    scores: dict[str, float] = {}
    for c in model["classes"]:
        prob = model["priors"][c]
        for feat in model["features"]:
            val = sample[feat]
            feat_table = model["likelihoods"][feat][c]
            if val in feat_table:
                prob *= feat_table[val]
            else:
                prob *= 1.0 / (model["vocab_sizes"][feat] + 1)
        scores[c] = prob

    total_score = sum(scores.values())
    posteriors = {c: (s / total_score if total_score > 0 else 0.0) for c, s in scores.items()}
    best_class = max(posteriors, key=posteriors.get)
    return best_class, posteriors


def _entropy(series: pd.Series) -> float:
    total = len(series)
    if total == 0:
        return 0.0
    counts = series.value_counts()
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)


def _gain_ratio(df: pd.DataFrame, feature: str, target: str = "label") -> float:
    base_entropy = _entropy(df[target])
    total = len(df)
    if total == 0 or base_entropy == 0:
        return 0.0

    cond_entropy = 0.0
    split_info = 0.0
    for _, subset in df.groupby(feature):
        w = len(subset) / total
        cond_entropy += w * _entropy(subset[target])
        if w > 0:
            split_info -= w * math.log2(w)

    info_gain = base_entropy - cond_entropy
    return (info_gain / split_info) if split_info > 0 else 0.0


def train_c45(df: pd.DataFrame, features: list[str] | None = None) -> dict[str, Any]:
    """Bangun pohon keputusan C.4.5 sederhana berbasis Gain Ratio."""
    feats = list(features or CATEGORICAL_FEATURES)
    majority = df["label"].mode()[0]

    def build(sub: pd.DataFrame, avail_feats: list[str]) -> Any:
        labels = sub["label"].unique()
        if len(labels) == 1:
            return {"type": "leaf", "label": labels[0], "samples": len(sub)}
        if not avail_feats:
            return {"type": "leaf", "label": sub["label"].mode()[0], "samples": len(sub)}

        ratios = {f: _gain_ratio(sub, f) for f in avail_feats}
        best_feat = max(ratios, key=ratios.get)
        if ratios[best_feat] <= 0:
            return {"type": "leaf", "label": sub["label"].mode()[0], "samples": len(sub)}

        next_feats = [f for f in avail_feats if f != best_feat]
        branches = {}
        for val, grp in sub.groupby(best_feat):
            branches[val] = build(grp, next_feats)

        return {
            "type": "node",
            "feature": best_feat,
            "gain_ratio": round(ratios[best_feat], 4),
            "default": sub["label"].mode()[0],
            "branches": branches,
        }

    return {"tree": build(df, feats), "majority": majority, "features": feats}


def predict_c45(model: dict[str, Any], sample: dict[str, str]) -> str:
    """Prediksi label Fast/Slow menggunakan pohon keputusan C.4.5."""
    node = model["tree"]
    while node["type"] == "node":
        val = sample.get(node["feature"])
        if val in node["branches"]:
            node = node["branches"][val]
        else:
            return node["default"]
    return node["label"]


def extract_c45_rules(node: dict[str, Any], prefix: str = "") -> list[str]:
    """Ekstrak aturan IF-THEN dari pohon keputusan C.4.5 untuk ditampilkan."""
    if node["type"] == "leaf":
        cond = prefix if prefix else "SEMUA DATA"
        return [f"IF {cond} THEN label = {node['label']} (n={node['samples']})"]

    rules: list[str] = []
    feat = node["feature"]
    for val, child in node["branches"].items():
        new_prefix = f"{prefix} AND {feat} = '{val}'" if prefix else f"{feat} = '{val}'"
        rules.extend(extract_c45_rules(child, new_prefix))
    return rules


def evaluate_classifier(df: pd.DataFrame, algo: str) -> tuple[float, pd.DataFrame]:
    """Hitung akurasi dan confusion matrix pada dataset."""
    y_true = df["label"].tolist()
    if algo == "Naive Bayes":
        model = train_naive_bayes(df)
        y_pred = [predict_naive_bayes(model, row)[0] for row in df.to_dict("records")]
    else:
        model = train_c45(df)
        y_pred = [predict_c45(model, row) for row in df.to_dict("records")]

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = (correct / len(y_true)) * 100.0 if y_true else 0.0
    cm = pd.crosstab(
        pd.Series(y_true, name="Aktual"),
        pd.Series(y_pred, name="Prediksi"),
        dropna=False,
    )
    return round(acc, 2), cm
