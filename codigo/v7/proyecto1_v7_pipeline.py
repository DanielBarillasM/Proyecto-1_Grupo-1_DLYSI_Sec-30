"""Proyecto 1 V7: selección tabular train-only e integración rubricada A/B/C.

El benchmark 85--100 % es histórico y nunca participa en decisiones. Los modelos
se eligen, calibran y umbralizan en bloques cronológicos disjuntos del 70--85 %.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import catboost
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(os.environ.get("PROYECTO1_ROOT", Path(__file__).resolve().parents[2])).resolve()
RAW = Path(os.environ.get("PROYECTO1_RAW", ROOT / "datos" / "raw")).resolve()
ART = ROOT / "artefactos" / "v7"
FIG = ROOT / "evidencia" / "figuras" / "v7"
PROCESSED = ROOT / "datos" / "processed" / "v7"
V6_ART = ROOT / "artefactos" / "v6"


@dataclass(frozen=True)
class ConfigV7:
    seed: int = 2026
    train_fraction: float = 0.70
    development_fraction: float = 0.85
    cost_fn_q: float = 4200.0
    cost_fp_q: float = 180.0
    recall_floor: float = 0.75
    correlation_threshold: float = 0.995
    correlation_sample: int = 30_000
    pca_fit_sample: int = 120_000
    pca_components: int = 128
    catboost_features: int = 150
    logistic_features: int = 100
    lightgbm_estimators: int = 900
    catboost_iterations: int = 650
    hypothesis_ap_gain: float = 0.01
    hypothesis_cost_reduction: float = 0.05
    alert_growth_tolerance: float = 0.10
    monthly_cards: int = 1_400_000
    monthly_transactions_scenarios: tuple[int, ...] = (5, 12, 20)


HYPOTHESIS_C = (
    "Creemos que una fusión de predicciones tabulares, secuenciales y de anomalía "
    "mejorará AP y costo porque sus errores pueden ser complementarios. C será útil "
    "solo si aumenta AP al menos 0.01, reduce el costo al menos 5 %, mantiene recall "
    ">= 0.75, no aumenta las alertas más de 10 % y mejora en al menos tres de cuatro "
    "ventanas temporales internas."
)

CAT_HINTS = {
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType",
    "DeviceInfo", *{f"M{i}" for i in range(1, 10)},
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
    "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
}

IDENTITY_DEFINITIONS = {
    "tarjeta_direccion": ["card1", "card2", "card3", "card5", "addr1"],
    "tarjeta_direccion_correo": ["card1", "card2", "addr1", "P_emaildomain"],
    "tarjeta_dispositivo_producto": ["card1", "DeviceInfo", "DeviceType", "ProductCD"],
    "tarjeta_dispositivo": ["card1", "DeviceInfo", "DeviceType"],
}


def ensure_dirs() -> None:
    for path in (ART, FIG, PROCESSED):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def resolve_raw(name: str) -> Path:
    direct = RAW / name
    if direct.exists() and direct.stat().st_size > 1_000_000:
        return direct
    cache = RAW / "kagglehub_cache"
    matches = list(cache.rglob(name)) if cache.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No se encontró {name} en {RAW}")


def reduce_memory(frame: pd.DataFrame) -> pd.DataFrame:
    for column in frame.select_dtypes(include=["float64"]).columns:
        frame[column] = frame[column].astype("float32")
    for column in frame.select_dtypes(include=["int64"]).columns:
        if column not in {"TransactionID", "TransactionDT"}:
            frame[column] = pd.to_numeric(frame[column], downcast="integer")
    return frame


def read_csv_compact(path: Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Lee todas las columnas sin el pico de RAM de una inferencia monolítica."""
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, low_memory=True, chunksize=chunksize):
        chunks.append(reduce_memory(chunk))
    result = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()
    return result


def load_full_data() -> pd.DataFrame:
    print("      leyendo 394 columnas transaccionales y 41 de identidad...", flush=True)
    tx = read_csv_compact(resolve_raw("train_transaction.csv"))
    identity = read_csv_compact(resolve_raw("train_identity.csv"))
    frame = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    del tx, identity
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    assert frame["TransactionDT"].is_monotonic_increasing
    return reduce_memory(frame)


def proxy_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    parts = [frame[c].fillna("MISSING").astype(str) if c in frame else pd.Series("MISSING", index=frame.index) for c in columns]
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def identity_diagnostics(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.Series]:
    diagnostics: dict[str, Any] = {}
    keys: dict[str, pd.Series] = {}
    for name, columns in IDENTITY_DEFINITIONS.items():
        key = proxy_key(frame, columns)
        keys[name] = key
        counts = key.value_counts(dropna=False)
        prior = key.groupby(key, sort=False).cumcount().to_numpy() + 1
        diagnostics[name] = {
            "columnas": columns,
            "entidades": int(counts.size),
            "mediana_transacciones_entidad": float(counts.median()),
            "p90_transacciones_entidad": float(counts.quantile(.90)),
            **{f"porcentaje_con_{k}": float((prior >= k).mean() * 100) for k in (3, 8, 16, 32)},
            "porcentaje_campos_faltantes": float(frame[columns].isna().mean().mean() * 100),
        }
    # La clave estable de V6 se conserva para comparabilidad; el diagnóstico evita
    # seleccionar una identidad por desempeño en el benchmark.
    return diagnostics, keys["tarjeta_direccion"]


def row_block_summary(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Media, desviación y faltantes por fila sin materializar un bloque float64."""
    n = len(frame)
    total = np.zeros(n, dtype=np.float64)
    total_sq = np.zeros(n, dtype=np.float64)
    count = np.zeros(n, dtype=np.int16)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(values)
        clean = np.where(finite, values, 0.0).astype(np.float64)
        total += clean
        total_sq += clean * clean
        count += finite.astype(np.int16)
    denominator = np.maximum(count, 1).astype(np.float64)
    mean = total / denominator
    variance = np.maximum(total_sq / denominator - mean * mean, 0.0)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32), (len(columns) - count).astype(np.float32)


def add_causal_features(frame: pd.DataFrame, entity_key: pd.Series) -> pd.DataFrame:
    seconds = frame["TransactionDT"].astype("float64")
    amount = frame["TransactionAmt"].fillna(0).astype("float64")
    day = seconds / 86400.0
    group = frame.groupby(entity_key, sort=False, observed=True)
    prior_count = group.cumcount().astype("float32")
    denom = prior_count.replace(0, np.nan)
    prior_sum = amount.groupby(entity_key, sort=False).cumsum() - amount
    prior_mean = prior_sum / denom
    previous_time = group["TransactionDT"].shift(1)
    frame["amount_log1p_v7"] = np.log1p(amount.clip(lower=0)).astype("float32")
    frame["amount_cents_v7"] = np.round((amount - np.floor(amount)) * 100).astype("float32")
    frame["hour_sin_v7"] = np.sin(2 * np.pi * ((seconds / 3600) % 24) / 24).astype("float32")
    frame["hour_cos_v7"] = np.cos(2 * np.pi * ((seconds / 3600) % 24) / 24).astype("float32")
    frame["weekday_sin_v7"] = np.sin(2 * np.pi * (day % 7) / 7).astype("float32")
    frame["weekday_cos_v7"] = np.cos(2 * np.pi * (day % 7) / 7).astype("float32")
    frame["entity_prior_count_v7"] = prior_count
    frame["entity_prior_amt_mean_v7"] = prior_mean.fillna(0).astype("float32")
    frame["amount_to_prior_mean_v7"] = amount.div(prior_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 100).astype("float32")
    frame["hours_since_prior_v7"] = ((seconds - previous_time) / 3600).fillna(0).clip(0, 24 * 365).astype("float32")
    feature_source = frame.drop(columns=["isFraud", "TransactionID", "TransactionDT"], errors="ignore")
    frame["missing_count_v7"] = feature_source.isna().sum(axis=1).astype("float32")
    for prefix, expected in (("V", range(1, 340)), ("C", range(1, 15)), ("D", range(1, 16)), ("id_", range(1, 39))):
        columns = [f"{prefix}{i}" if prefix != "id_" else f"id_{i:02d}" for i in expected]
        columns = [c for c in columns if c in frame]
        if columns:
            clean = prefix.replace("_", "")
            mean, std, missing = row_block_summary(frame, columns)
            frame[f"{clean}_missing_v7"] = missing
            frame[f"{clean}_mean_v7"] = mean
            frame[f"{clean}_std_v7"] = std
    # Frecuencias estrictamente previas: nunca cuentan la fila actual ni el futuro.
    for column in ("card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "DeviceInfo", "ProductCD"):
        values = frame[column].fillna("MISSING").astype(str)
        frame[f"prior_frequency_{column}_v7"] = values.groupby(values, sort=False).cumcount().astype("float32")
    return frame


def temporal_split(n: int, cfg: ConfigV7) -> dict[str, np.ndarray]:
    train_end = int(n * cfg.train_fraction)
    dev_end = int(n * cfg.development_fraction)
    result = {
        "train": np.arange(train_end, dtype=np.int64),
        "validation": np.arange(train_end, dev_end, dtype=np.int64),
        "benchmark_historico": np.arange(dev_end, n, dtype=np.int64),
    }
    assert result["train"][-1] < result["validation"][0] < result["benchmark_historico"][0]
    return result


def validation_bounds(n: int) -> dict[str, np.ndarray]:
    points = {
        "early": (0.00, 0.35), "meta_fit": (0.35, 0.50), "model_select": (0.50, 0.60),
        "calibration": (0.60, 0.70), "threshold": (0.70, 0.80), "evaluation": (0.80, 1.00),
    }
    return {name: np.arange(int(n * left), int(n * right), dtype=np.int64) for name, (left, right) in points.items()}


def feature_inventory(frame: pd.DataFrame, train_rows: np.ndarray) -> tuple[list[str], list[str], dict[str, Any]]:
    excluded = {"isFraud", "TransactionID", "TransactionDT"}
    candidates = [c for c in frame.columns if c not in excluded]
    missing = frame.loc[train_rows, candidates].isna().mean()
    kept_missing = missing[missing <= .995].index.tolist()
    nunique = frame.loc[train_rows[: min(120_000, len(train_rows))], kept_missing].nunique(dropna=False)
    kept = nunique[nunique > 1].index.tolist()
    categorical = [c for c in kept if c in CAT_HINTS or frame[c].dtype == "object"]
    numeric = [c for c in kept if c not in categorical]
    audit = {
        "columnas_crudas_union": int(len(frame.columns)), "candidatas_sin_id_tiempo_target": len(candidates),
        "eliminadas_faltantes_gt_99_5": sorted(set(candidates) - set(kept_missing)),
        "eliminadas_constantes_train": sorted(set(kept_missing) - set(kept)),
        "numericas": len(numeric), "categoricas": len(categorical), "retenidas": len(kept),
    }
    return numeric, categorical, audit


def encode_train_only(frame: pd.DataFrame, numeric: list[str], categorical: list[str], train_rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    n = len(frame)
    columns = [*numeric, *categorical]
    matrix = np.empty((n, len(columns)), dtype=np.float32)
    medians: dict[str, float] = {}
    frequency_maps: dict[str, dict[str, float]] = {}
    for j, column in enumerate(numeric):
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.iloc[train_rows].median())
        if not np.isfinite(median):
            median = 0.0
        medians[column] = median
        matrix[:, j] = values.fillna(median).clip(-1e12, 1e12).to_numpy(dtype=np.float32)
    offset = len(numeric)
    train_n = float(len(train_rows))
    for j, column in enumerate(categorical):
        values = frame[column].fillna("MISSING").astype(str)
        counts = values.iloc[train_rows].value_counts(dropna=False)
        mapping = (counts / train_n).to_dict()
        frequency_maps[column] = {str(k): float(v) for k, v in mapping.items()}
        matrix[:, offset + j] = values.map(mapping).fillna(0).to_numpy(dtype=np.float32)
    return matrix, {"columns": columns, "numeric": numeric, "categorical": categorical, "medians": medians, "frequency_maps": frequency_maps}


def association_scores(matrix: np.ndarray, y: np.ndarray, rows: np.ndarray, columns: list[str], sample: int = 120_000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    chosen = rows if len(rows) <= sample else np.sort(rng.choice(rows, sample, replace=False))
    target = y[chosen].astype(float)
    centered_y = target - target.mean()
    scores = []
    for j, column in enumerate(columns):
        values = matrix[chosen, j].astype(float)
        std = values.std()
        corr = 0.0 if std < 1e-12 else float(abs(np.mean((values - values.mean()) * centered_y) / (std * max(centered_y.std(), 1e-12))))
        scores.append({"variable": column, "asociacion_point_biserial_abs_train": corr})
    return pd.DataFrame(scores).sort_values("asociacion_point_biserial_abs_train", ascending=False).reset_index(drop=True)


def correlation_representatives(matrix: np.ndarray, rows: np.ndarray, columns: list[str], association: pd.DataFrame, cfg: ConfigV7) -> tuple[list[int], pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    sample_rows = rows if len(rows) <= cfg.correlation_sample else np.sort(rng.choice(rows, cfg.correlation_sample, replace=False))
    # Se limita a las 240 variables con mayor asociación train-only; las restantes
    # permanecen en el modelo completo y no se eliminan por una aproximación pobre.
    ranked = association.head(min(240, len(association)))["variable"].tolist()
    ranked_idx = [columns.index(c) for c in ranked]
    sample = pd.DataFrame(matrix[np.ix_(sample_rows, ranked_idx)], columns=ranked)
    corr = sample.corr(method="spearman").abs()
    dropped: set[str] = set()
    pairs: list[dict[str, Any]] = []
    score_map = association.set_index("variable")["asociacion_point_biserial_abs_train"].to_dict()
    for i, left in enumerate(ranked):
        if left in dropped:
            continue
        for right in ranked[i + 1:]:
            if right in dropped:
                continue
            rho = corr.at[left, right]
            if np.isfinite(rho) and rho >= cfg.correlation_threshold:
                keep, remove = (left, right) if score_map[left] >= score_map[right] else (right, left)
                dropped.add(remove)
                pairs.append({"variable_1": left, "variable_2": right, "rho_spearman_abs_train": float(rho), "conservada": keep, "eliminada": remove})
    kept_idx = [j for j, c in enumerate(columns) if c not in dropped]
    return kept_idx, pd.DataFrame(pairs)


def fit_lgbm(x: np.ndarray, y: np.ndarray, train_rows: np.ndarray, early_rows: np.ndarray, cfg: ConfigV7, seed_offset: int = 0, estimators: int | None = None) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=estimators or cfg.lightgbm_estimators, learning_rate=.025,
        num_leaves=47, max_depth=-1, min_child_samples=80, subsample=.85, colsample_bytree=.82,
        reg_alpha=.4, reg_lambda=1.2, random_state=cfg.seed + seed_offset,
        n_jobs=max(1, min(8, os.cpu_count() or 1)), verbosity=-1,
    )
    positive = max(1, int(y[train_rows].sum()))
    scale = math.sqrt(float((len(train_rows) - positive) / positive))
    weights = np.where(y[train_rows] == 1, scale, 1.0)
    model.fit(x[train_rows], y[train_rows], sample_weight=weights,
              eval_set=[(x[early_rows], y[early_rows])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(80, verbose=False)])
    return model


def predict(model: Any, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return model.predict_proba(x[rows])[:, 1]


def fit_incremental_pca(matrix: np.ndarray, train_rows: np.ndarray, v_indices: list[int], cfg: ConfigV7) -> tuple[IncrementalPCA, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    fit_rows = train_rows if len(train_rows) <= cfg.pca_fit_sample else np.sort(rng.choice(train_rows, cfg.pca_fit_sample, replace=False))
    block = matrix[np.ix_(fit_rows, v_indices)].astype(np.float64)
    mean = block.mean(axis=0)
    std = block.std(axis=0)
    std[std < 1e-6] = 1.0
    n_components = min(cfg.pca_components, len(v_indices), len(fit_rows) - 1)
    pca = IncrementalPCA(n_components=n_components, batch_size=max(512, n_components * 4))
    scaled = (block - mean) / std
    for start in range(0, len(scaled), 10_000):
        chunk = scaled[start:start + 10_000]
        if len(chunk) >= n_components:
            pca.partial_fit(chunk)
    transformed = np.empty((len(matrix), n_components), dtype=np.float32)
    for start in range(0, len(matrix), 20_000):
        chunk = (matrix[start:start + 20_000, v_indices].astype(np.float64) - mean) / std
        transformed[start:start + len(chunk)] = pca.transform(chunk).astype(np.float32)
    return pca, mean, std, transformed


def metric_set(y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV7) -> dict[str, Any]:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    k = max(1, int(round(.01 * len(y))))
    top = np.argsort(score)[-k:]
    return {
        "auc_pr": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0), "brier": brier_score_loss(y, score),
        "precision_at_1pct": float(y[top].mean()), "recall_at_1pct": float(y[top].sum() / max(1, y.sum())),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "cost_q": cfg.cost_fn_q * fn + cfg.cost_fp_q * fp,
        "cost_per_decision_q": (cfg.cost_fn_q * fn + cfg.cost_fp_q * fp) / len(y),
        "alertas_por_100k": pred.mean() * 100_000, "threshold": threshold,
    }


def choose_threshold(y: np.ndarray, score: np.ndarray, cfg: ConfigV7) -> tuple[float, pd.DataFrame]:
    _, _, pr_thresholds = precision_recall_curve(y, score)
    candidates = np.unique(np.r_[pr_thresholds[:: max(1, len(pr_thresholds) // 350)], np.quantile(score, np.linspace(.005, .995, 250))])
    rows = []
    for threshold in candidates:
        pred = score >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision_value = tp / max(1, tp + fp)
        recall_value = tp / max(1, tp + fn)
        rows.append({"threshold": float(threshold), "precision": precision_value, "recall": recall_value,
                     "f1": 2 * precision_value * recall_value / max(1e-12, precision_value + recall_value),
                     "cost_q": cfg.cost_fn_q * fn + cfg.cost_fp_q * fp, "alertas_por_100k": pred.mean() * 100_000})
    curve = pd.DataFrame(rows)
    feasible = curve[curve["recall"] >= cfg.recall_floor]
    best = (feasible if len(feasible) else curve).sort_values(["cost_q", "f1"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), curve


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values)).reshape(-1, 1)


def fit_calibrator(score: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1500, random_state=2026)
    model.fit(logit(score), y)
    return model


def apply_calibrator(model: LogisticRegression, score: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit(score))[:, 1]


def load_v6_scores(frame: pd.DataFrame, split: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"validation": {}, "benchmark_historico": {}}
    for split_name, filename in (("validation", "predicciones_validacion_v6.csv"), ("benchmark_historico", "predicciones_benchmark_v6.csv")):
        pred = pd.read_csv(V6_ART / filename)
        rows = split[split_name]
        assert np.array_equal(pred["TransactionID"].to_numpy(), frame.loc[rows, "TransactionID"].to_numpy())
        for name in ("A", "B", "D"):
            result[split_name][name] = pred[f"score_{name}"].to_numpy(float)
    return result


def fit_meta(features: np.ndarray, y: np.ndarray, rows: np.ndarray, seed: int) -> Pipeline:
    model = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(max_iter=2500, C=.5, random_state=seed))])
    model.fit(features[rows], y[rows])
    return model


def plot_pr(y: np.ndarray, scores: dict[str, np.ndarray], destination: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, lw=2, label=f"{name} · AP={average_precision_score(y, score):.3f}")
    ax.axhline(float(y.mean()), color="#6b7280", ls="--", label=f"Prevalencia={y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(destination, dpi=180); plt.close(fig)


def main_v7() -> None:
    started = time.perf_counter()
    cfg = ConfigV7()
    ensure_dirs(); set_seed(cfg.seed)
    print("[1/12] Cargando datos completos y ordenando temporalmente...", flush=True)
    frame = load_full_data()
    split = temporal_split(len(frame), cfg)
    y = frame["isFraud"].to_numpy(np.int8)
    val_rows, bench_rows = split["validation"], split["benchmark_historico"]
    y_val, y_bench = y[val_rows], y[bench_rows]
    bounds = validation_bounds(len(val_rows))
    early_global = val_rows[bounds["early"]]

    print("[2/12] Diagnosticando identidades y creando variables causales...", flush=True)
    identity_info, entity_key = identity_diagnostics(frame)
    frame = add_causal_features(frame, entity_key)
    del entity_key
    numeric_cols, categorical_cols, feature_audit = feature_inventory(frame, split["train"])

    print("[3/12] Ajustando codificación, imputación y asociación solo con train...", flush=True)
    matrix, preprocessing = encode_train_only(frame, numeric_cols, categorical_cols, split["train"])
    columns = preprocessing["columns"]
    association = association_scores(matrix, y, split["train"], columns)
    association.to_csv(PROCESSED / "asociacion_variables_train_v7.csv", index=False)
    corr_idx, corr_pairs = correlation_representatives(matrix, split["train"], columns, association, cfg)
    corr_pairs.to_csv(PROCESSED / "pares_correlacionados_train_v7.csv", index=False)
    joblib.dump(preprocessing, ART / "preprocesamiento_tabular_v7.joblib")
    inherited = load_v6_scores(frame, split)

    print("[4/12] Ejecutando ablation PCA del bloque V...", flush=True)
    v_idx = [j for j, c in enumerate(columns) if c.startswith("V") and c[1:].isdigit()]
    non_v_idx = [j for j in range(len(columns)) if j not in set(v_idx)]
    pca, pca_mean, pca_std, pca_values = fit_incremental_pca(matrix, split["train"], v_idx, cfg)
    pca_info = {
        "bloque": "V1-V339", "ajuste": "muestra determinista contenida solo en train",
        "muestra_ajuste": min(cfg.pca_fit_sample, len(split["train"])), "componentes_ajustados": pca.n_components_,
        "varianza_acumulada": np.cumsum(pca.explained_variance_ratio_),
        "componentes_para_90": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .90) + 1),
        "componentes_para_95": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .95) + 1),
        "componentes_para_99": int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), .99) + 1),
    }
    joblib.dump({"pca": pca, "mean": pca_mean, "std": pca_std, "v_columns": [columns[j] for j in v_idx]}, ART / "pca_bloque_v_v7.joblib")

    print("[5/12] Entrenando A0–A4 sin consultar el benchmark...", flush=True)
    score_cache = ART / "puntajes_crudos_candidatos_a_v7.npz"
    a_val: dict[str, np.ndarray] = {}
    a_bench: dict[str, np.ndarray] = {}
    model_metadata: dict[str, Any] = {}
    ranked_idx = [columns.index(c) for c in association["variable"].tolist()]

    # A0: control lineal sobre las variables train-only más asociadas.
    logistic_idx = ranked_idx[: min(cfg.logistic_features, len(ranked_idx))]
    logistic = Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, C=.2, solver="lbfgs", random_state=cfg.seed))])
    logistic.fit(matrix[np.ix_(split["train"], logistic_idx)], y[split["train"]])
    a_val["A0_logistica"] = logistic.predict_proba(matrix[np.ix_(val_rows, logistic_idx)])[:, 1]
    a_bench["A0_logistica"] = logistic.predict_proba(matrix[np.ix_(bench_rows, logistic_idx)])[:, 1]
    joblib.dump({"model": logistic, "features": [columns[j] for j in logistic_idx]}, ART / "modelo_A0_regresion_logistica_v7.joblib")

    # A1: todas las variables retenidas.
    a1 = fit_lgbm(matrix, y, split["train"], early_global, cfg, seed_offset=1)
    a_val["A1_lgbm_ampliado"] = predict(a1, matrix, val_rows)
    a_bench["A1_lgbm_ampliado"] = predict(a1, matrix, bench_rows)
    joblib.dump({"model": a1, "features": columns}, ART / "modelo_A1_lightgbm_ampliado_v7.joblib")

    # A2: representantes de redundancia extrema.
    a2 = fit_lgbm(matrix[:, corr_idx], y, split["train"], early_global, cfg, seed_offset=2)
    a_val["A2_lgbm_correlacion"] = predict(a2, matrix[:, corr_idx], val_rows)
    a_bench["A2_lgbm_correlacion"] = predict(a2, matrix[:, corr_idx], bench_rows)
    joblib.dump({"model": a2, "features": [columns[j] for j in corr_idx], "threshold": cfg.correlation_threshold}, ART / "modelo_A2_lightgbm_correlacion_v7.joblib")

    # A3: tres cortes anidados de un PCA train-only de V; el mejor se decide en model_select.
    pca_scores: dict[str, tuple[np.ndarray, np.ndarray, Any]] = {}
    non_v_ranked = [j for j in ranked_idx if j in set(non_v_idx)][:100]
    for components in (32, 64, 128):
        use = min(components, pca_values.shape[1])
        hybrid = np.column_stack([matrix[:, non_v_ranked], pca_values[:, :use]]).astype(np.float32)
        model = fit_lgbm(hybrid, y, split["train"], early_global, cfg, seed_offset=components, estimators=700)
        pca_scores[f"A3_pca_{components}"] = (predict(model, hybrid, val_rows), predict(model, hybrid, bench_rows), model)
        del hybrid
    pca_ap_select = {name: average_precision_score(y_val[bounds["model_select"]], scores[0][bounds["model_select"]]) for name, scores in pca_scores.items()}
    best_pca_name = max(pca_ap_select, key=pca_ap_select.get)
    a_val[best_pca_name], a_bench[best_pca_name], best_pca_model = pca_scores[best_pca_name]
    joblib.dump({"model": best_pca_model, "components": int(best_pca_name.rsplit("_", 1)[1]), "non_v_features": [columns[j] for j in non_v_ranked]}, ART / "modelo_A3_lightgbm_pca_v7.joblib")
    del pca_scores, pca_values
    gc.collect()

    # A4: CatBoost sobre selección train-only; las categóricas permanecen nativas.
    cat_selected = association.head(min(cfg.catboost_features, len(association)))["variable"].tolist()
    cat_features = [c for c in cat_selected if c in categorical_cols]
    cat_frame = frame[cat_selected].copy()
    for column in cat_features:
        cat_frame[column] = cat_frame[column].fillna("MISSING").astype(str)
    for column in [c for c in cat_selected if c not in cat_features]:
        cat_frame[column] = pd.to_numeric(cat_frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(preprocessing["medians"].get(column, 0.0))
    cat = CatBoostClassifier(iterations=cfg.catboost_iterations, depth=8, learning_rate=.04, loss_function="Logloss",
                             eval_metric="PRAUC", random_seed=cfg.seed, l2_leaf_reg=5, random_strength=.5,
                             auto_class_weights="SqrtBalanced", thread_count=max(1, min(8, os.cpu_count() or 1)),
                             verbose=100, allow_writing_files=False)
    cat.fit(cat_frame.iloc[split["train"]], y[split["train"]], cat_features=cat_features,
            eval_set=(cat_frame.iloc[early_global], y[early_global]), early_stopping_rounds=70, verbose=100)
    a_val["A4_catboost"] = cat.predict_proba(cat_frame.iloc[val_rows])[:, 1]
    a_bench["A4_catboost"] = cat.predict_proba(cat_frame.iloc[bench_rows])[:, 1]
    cat.save_model(str(ART / "modelo_A4_catboost_v7.cbm"))
    joblib.dump({"features": cat_selected, "categorical": cat_features}, ART / "preprocesamiento_catboost_v7.joblib")
    del cat_frame

    # A5: stacking tabular; entradas fuera de tiempo respecto del ajuste base.
    base_names = list(a_val)
    stack_inputs = [*base_names, "A_V6_control"]
    stack_val = np.column_stack([*[logit(a_val[name]).ravel() for name in base_names], logit(inherited["validation"]["A"]).ravel()])
    stack_bench = np.column_stack([*[logit(a_bench[name]).ravel() for name in base_names], logit(inherited["benchmark_historico"]["A"]).ravel()])
    a5 = fit_meta(stack_val, y_val, bounds["meta_fit"], cfg.seed)
    a_val["A5_ensamble_tabular"] = a5.predict_proba(stack_val)[:, 1]
    a_bench["A5_ensamble_tabular"] = a5.predict_proba(stack_bench)[:, 1]
    joblib.dump({"model": a5, "inputs": stack_inputs}, ART / "modelo_A5_ensamble_tabular_v7.joblib")
    np.savez_compressed(score_cache, **{f"val__{k}": v for k, v in a_val.items()}, **{f"bench__{k}": v for k, v in a_bench.items()})

    print("[6/12] Seleccionando A por AP y estabilidad, aún sin benchmark...", flush=True)
    select_ap = {name: float(average_precision_score(y_val[bounds["model_select"]], score[bounds["model_select"]])) for name, score in a_val.items()}
    select_windows = np.array_split(bounds["model_select"], 2)
    window_ap = {name: [float(average_precision_score(y_val[w], score[w])) for w in select_windows] for name, score in a_val.items()}
    best_a_name = max(select_ap, key=select_ap.get)
    a_selected_val, a_selected_bench = a_val[best_a_name], a_bench[best_a_name]

    print("[7/12] Integrando B y D congelados y ajustando controles C1–C3...", flush=True)
    b_val, b_bench = inherited["validation"]["B"], inherited["benchmark_historico"]["B"]
    d_val, d_bench = inherited["validation"]["D"], inherited["benchmark_historico"]["D"]
    old_a_val, old_a_bench = inherited["validation"]["A"], inherited["benchmark_historico"]["A"]
    length_file = ROOT / "datos" / "processed" / "v6" / "esquema_indices_secuencia_v6.npz"
    lengths = np.load(length_file)["lengths"]
    identity_present = frame[[c for c in ("DeviceType", "id_01") if c in frame]].notna().any(axis=1).to_numpy(float)
    quality_val = np.clip(np.log1p(lengths[val_rows]) / np.log1p(32), 0, 1) * (.6 + .4 * identity_present[val_rows])
    quality_bench = np.clip(np.log1p(lengths[bench_rows]) / np.log1p(32), 0, 1) * (.6 + .4 * identity_present[bench_rows])
    amount_val = np.log1p(frame.loc[val_rows, "TransactionAmt"].fillna(0).to_numpy(float))
    amount_bench = np.log1p(frame.loc[bench_rows, "TransactionAmt"].fillna(0).to_numpy(float))
    missing_val = frame.loc[val_rows, "missing_count_v7"].to_numpy(float)
    missing_bench = frame.loc[bench_rows, "missing_count_v7"].to_numpy(float)
    c_inputs = {
        "C1_A_B": (np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel()]), np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel()])),
        "C2_A_B_D": (np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel(), logit(d_val).ravel()]), np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel(), logit(d_bench).ravel()])),
        "C3_condicionada": (
            np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel(), logit(d_val).ravel(), (logit(b_val).ravel() - logit(a_selected_val).ravel()) * quality_val, quality_val, amount_val, missing_val]),
            np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel(), logit(d_bench).ravel(), (logit(b_bench).ravel() - logit(a_selected_bench).ravel()) * quality_bench, quality_bench, amount_bench, missing_bench]),
        ),
    }
    c_models: dict[str, Any] = {}
    c_val_options: dict[str, np.ndarray] = {}
    c_bench_options: dict[str, np.ndarray] = {}
    for i, (name, (z_val, z_bench)) in enumerate(c_inputs.items()):
        model = fit_meta(z_val, y_val, bounds["meta_fit"], cfg.seed + i)
        c_models[name] = model
        c_val_options[name] = model.predict_proba(z_val)[:, 1]
        c_bench_options[name] = model.predict_proba(z_bench)[:, 1]
    c_select_ap = {name: float(average_precision_score(y_val[bounds["model_select"]], score[bounds["model_select"]])) for name, score in c_val_options.items()}
    best_c_name = max(c_select_ap, key=c_select_ap.get)
    c_val, c_bench = c_val_options[best_c_name], c_bench_options[best_c_name]
    joblib.dump({"models": c_models, "selected": best_c_name, "hypothesis": HYPOTHESIS_C}, ART / "modelos_C_fusion_v7.joblib")

    print("[8/12] Calibrando y fijando umbrales en bloques independientes...", flush=True)
    raw_val = {"A": a_selected_val, "B": b_val, "C": c_val, "D": d_val, "A_V6_control": old_a_val}
    raw_bench = {"A": a_selected_bench, "B": b_bench, "C": c_bench, "D": d_bench, "A_V6_control": old_a_bench}
    calibrated_val: dict[str, np.ndarray] = {}; calibrated_bench: dict[str, np.ndarray] = {}
    calibrators: dict[str, Any] = {}; calibration_info: dict[str, Any] = {}
    for name in raw_val:
        calibrator = fit_calibrator(raw_val[name][bounds["calibration"]], y_val[bounds["calibration"]])
        calibrators[name] = calibrator
        calibrated_val[name] = apply_calibrator(calibrator, raw_val[name])
        calibrated_bench[name] = apply_calibrator(calibrator, raw_bench[name])
        calibration_info[name] = {
            "brier_raw": brier_score_loss(y_val[bounds["calibration"]], raw_val[name][bounds["calibration"]]),
            "brier_calibrado": brier_score_loss(y_val[bounds["calibration"]], calibrated_val[name][bounds["calibration"]]),
        }
    joblib.dump(calibrators, ART / "calibradores_v7.joblib")
    thresholds: dict[str, float] = {}; internal: dict[str, Any] = {}; benchmark: dict[str, Any] = {}; threshold_curves = []
    for name, score in calibrated_val.items():
        threshold, curve = choose_threshold(y_val[bounds["threshold"]], score[bounds["threshold"]], cfg)
        thresholds[name] = threshold; curve.insert(0, "modelo", name); threshold_curves.append(curve)
        internal[name] = metric_set(y_val[bounds["evaluation"]], score[bounds["evaluation"]], threshold, cfg)
        benchmark[name] = metric_set(y_bench, calibrated_bench[name], threshold, cfg)
    pd.concat(threshold_curves, ignore_index=True).to_csv(ART / "curvas_umbral_v7.csv", index=False)
    write_json(ART / "umbrales_v7.json", thresholds)

    print("[9/12] Aplicando gates predeclarados y estabilidad en cuatro ventanas...", flush=True)
    eval_windows = np.array_split(bounds["evaluation"], 4)
    c_deltas = []
    promotion_deltas = []
    for i, window in enumerate(eval_windows, 1):
        c_delta = average_precision_score(y_val[window], calibrated_val["C"][window]) - average_precision_score(y_val[window], calibrated_val["A"][window])
        a_delta = average_precision_score(y_val[window], calibrated_val["A"][window]) - average_precision_score(y_val[window], calibrated_val["A_V6_control"][window])
        c_deltas.append({"ventana": i, "delta_ap_C_vs_A": float(c_delta)})
        promotion_deltas.append({"ventana": i, "delta_ap_V7_vs_V6": float(a_delta)})
    c_ap_gain = internal["C"]["auc_pr"] - internal["A"]["auc_pr"]
    c_cost_reduction = (internal["A"]["cost_q"] - internal["C"]["cost_q"]) / max(1, internal["A"]["cost_q"])
    c_alert_growth = internal["C"]["alertas_por_100k"] / max(1e-9, internal["A"]["alertas_por_100k"]) - 1
    c_success = bool(c_ap_gain >= cfg.hypothesis_ap_gain and c_cost_reduction >= cfg.hypothesis_cost_reduction and internal["C"]["recall"] >= cfg.recall_floor and c_alert_growth <= cfg.alert_growth_tolerance and sum(r["delta_ap_C_vs_A"] > 0 for r in c_deltas) >= 3)
    promotion_ap_gain = internal["A"]["auc_pr"] - internal["A_V6_control"]["auc_pr"]
    promotion_cost_reduction = (internal["A_V6_control"]["cost_q"] - internal["A"]["cost_q"]) / max(1, internal["A_V6_control"]["cost_q"])
    promotion_alert_growth = internal["A"]["alertas_por_100k"] / max(1e-9, internal["A_V6_control"]["alertas_por_100k"]) - 1
    promotion_success = bool(promotion_ap_gain >= .01 and promotion_cost_reduction >= .05 and internal["A"]["recall"] >= cfg.recall_floor and promotion_alert_growth <= cfg.alert_growth_tolerance and sum(r["delta_ap_V7_vs_V6"] > 0 for r in promotion_deltas) >= 3 and min(r["delta_ap_V7_vs_V6"] for r in promotion_deltas) >= -.005)
    candidate = "C" if c_success else "A"

    print("[10/12] Incorporando falsificaciones congeladas y diagnóstico PCA/correlación...", flush=True)
    v6_results = json.loads((V6_ART / "resultados_v6.json").read_text(encoding="utf-8"))
    falsification = v6_results["falsificaciones"]
    falsification["procedencia_v7"] = "inferencia congelada V6 sobre el mismo B; no se reentrenó para los recortes"
    falsification["historia_32"] = falsification["original_internal"]
    falsification["criterio_material_delta_ap"] = .01

    # Walk-forward ligero del recipe A2: cada pliegue reajusta imputación/codificación.
    print("[11/12] Ejecutando tres ventanas walk-forward con preprocesamiento reajustado...", flush=True)
    walk_rows = []
    fold_specs = ((.45, .05), (.55, .05), (.65, .05))
    wf_columns = [columns[j] for j in corr_idx]
    wf_numeric = [c for c in wf_columns if c in numeric_cols]
    wf_categorical = [c for c in wf_columns if c in categorical_cols]
    for fold, (train_end_f, width) in enumerate(fold_specs, 1):
        fit_rows = np.arange(int(len(frame) * train_end_f), dtype=np.int64)
        eval_rows = np.arange(int(len(frame) * train_end_f), int(len(frame) * (train_end_f + width)), dtype=np.int64)
        fold_matrix, _ = encode_train_only(frame, wf_numeric, wf_categorical, fit_rows)
        # El último 10 % del bloque de ajuste funciona como early stopping; el modelo
        # efectivo usa únicamente el 90 % anterior para preservar orden.
        cut = int(len(fit_rows) * .90)
        model = fit_lgbm(fold_matrix, y, fit_rows[:cut], fit_rows[cut:], cfg, seed_offset=100 + fold, estimators=450)
        fold_score = predict(model, fold_matrix, eval_rows)
        walk_rows.append({"fold": fold, "train_inicio": int(fit_rows[0]), "train_fin_exclusivo": int(fit_rows[-1] + 1), "evaluacion_inicio": int(eval_rows[0]), "evaluacion_fin_exclusivo": int(eval_rows[-1] + 1), "n_evaluacion": len(eval_rows), "prevalencia": float(y[eval_rows].mean()), "auc_pr": float(average_precision_score(y[eval_rows], fold_score)), "roc_auc": float(roc_auc_score(y[eval_rows], fold_score))})
        del fold_matrix, model
        gc.collect()
    pd.DataFrame(walk_rows).to_csv(ART / "validacion_walk_forward_v7.csv", index=False)

    print("[12/12] Guardando evidencia, contrato, figuras y resultados...", flush=True)
    pred_val = pd.DataFrame({"indice": val_rows, "TransactionID": frame.loc[val_rows, "TransactionID"].to_numpy(), "y": y_val, **{f"score_{k}": v for k, v in calibrated_val.items()}})
    pred_bench = pd.DataFrame({"indice": bench_rows, "TransactionID": frame.loc[bench_rows, "TransactionID"].to_numpy(), "y": y_bench, **{f"score_{k}": v for k, v in calibrated_bench.items()}})
    pred_val.to_csv(ART / "predicciones_validacion_v7.csv", index=False)
    pred_bench.to_csv(ART / "predicciones_benchmark_v7.csv", index=False)
    selection = {"A": {"auc_pr_model_select": select_ap, "auc_pr_subventanas": window_ap, "seleccionado": best_a_name, "pca_auc_pr_model_select": pca_ap_select}, "C": {"auc_pr_model_select": c_select_ap, "seleccionado": best_c_name}}
    write_json(ART / "seleccion_modelos_v7.json", selection)
    write_json(PROCESSED / "auditoria_variables_v7.json", {"inventario": feature_audit, "pca": pca_info, "identidades": identity_info})

    economics: dict[str, Any] = {}
    for name in ("A", "B", "C", "D"):
        economics[name] = {}
        for tx_per_card in cfg.monthly_transactions_scenarios:
            decisions = cfg.monthly_cards * tx_per_card
            economics[name][str(tx_per_card)] = {"decisiones_mensuales": decisions, "costo_mensual_q": benchmark[name]["cost_per_decision_q"] * decisions, "diferencia_vs_A_q": (benchmark[name]["cost_per_decision_q"] - benchmark["A"]["cost_per_decision_q"]) * decisions}

    schema = {
        "version": "7.0", "candidate": candidate, "A_seleccionado": best_a_name, "C_seleccionado": best_c_name,
        "entrada": {"unidad": "una transacción", "variables": [columns[j] for j in (corr_idx if best_a_name == "A2_lgbm_correlacion" else range(len(columns)))], "ids_excluidos": ["TransactionID"], "tiempo_crudo_excluido": ["TransactionDT"]},
        "salida": {"risk_score": "puntaje continuo calibrado en [0,1]", "threshold": thresholds[candidate], "predicted_label": f"risk_score >= {thresholds[candidate]:.8f}"},
        "faltantes": "imputación/codificación aprendida solo con train; categorías nuevas reciben frecuencia 0",
        "restricciones": ["ordenar por TransactionDT antes de crear agregados", "ninguna variable histórica puede usar la fila actual ni eventos futuros"],
    }
    write_json(ART / "contrato_entrada_salida_v7.json", schema)

    result = {
        "version": "7.0", "estado_benchmark": "historico_reutilizado_no_ciego", "pregunta": "¿El orden aporta señal incremental, bajo qué condiciones y cuánto vale económicamente?",
        "protocolo_congelado": "configuracion/v7/PROTOCOLO_EXPERIMENTAL_V7.md", "configuracion": asdict(cfg),
        "entorno": {"python": sys.version, "lightgbm": lgb.__version__, "catboost": catboost.__version__, "plataforma": platform.platform(), "cpu": os.cpu_count()},
        "datos": {"origen": "IEEE-CIS Fraud Detection / Vesta Corporation (Kaggle)", "url": "https://www.kaggle.com/competitions/ieee-fraud-detection/overview", "filas": len(frame), "columnas_union": feature_audit["columnas_crudas_union"], "fraudes": int(y.sum()), "prevalencia": float(y.mean()), "particiones": {name: {"n": len(rows), "prevalencia": float(y[rows].mean()), "dt_min": int(frame.loc[rows, "TransactionDT"].min()), "dt_max": int(frame.loc[rows, "TransactionDT"].max())} for name, rows in split.items()}},
        "variables": {"auditoria": feature_audit, "correlacion": {"umbral": cfg.correlation_threshold, "pares_eliminados": len(corr_pairs), "variables_retenidas": len(corr_idx)}, "pca": pca_info},
        "identidades": identity_info, "seleccion": selection, "calibracion": calibration_info, "umbrales": thresholds,
        "evaluacion_interna": internal, "benchmark_historico": benchmark,
        "hipotesis_C": {"declaracion_previa": HYPOTHESIS_C, "control": "A", "delta_ap": c_ap_gain, "reduccion_costo": c_cost_reduction, "crecimiento_alertas": c_alert_growth, "ventanas": c_deltas, "success": c_success},
        "promocion_V7": {"control": "A_V6_control", "delta_ap": promotion_ap_gain, "reduccion_costo": promotion_cost_reduction, "crecimiento_alertas": promotion_alert_growth, "ventanas": promotion_deltas, "success": promotion_success, "confirmatoria": False},
        "falsificaciones": falsification, "walk_forward": walk_rows,
        "candidato": {"modelo": candidate, "detalle": best_c_name if candidate == "C" else best_a_name, "threshold": thresholds[candidate], "confirmatorio": False},
        "economia_mensual": economics,
        "decision": "Promover V7 solo si satisface todos los gates internos; exigir cohorte futura para confirmación externa.",
        "limitaciones": ["Benchmark histórico reutilizado y no ciego", "Identidad proxy anonimizada", "Costos y volumen mensual académicos", "B y D se congelaron desde V6 para aislar la mejora tabular", "Las falsificaciones V7 reutilizan inferencia congelada de B y no reentrenan cada longitud", "No existe cohorte futura etiquetada para confirmación"],
        "duracion_segundos": time.perf_counter() - started,
    }
    write_json(ART / "resultados_v7.json", result)

    plot_pr(y_val[bounds["evaluation"]], {name: calibrated_val[name][bounds["evaluation"]] for name in ("A", "B", "C", "D", "A_V6_control")}, FIG / "01_comparacion_interna_v7.png", "V7 · evaluación temporal interna")
    plot_pr(y_bench, {name: calibrated_bench[name] for name in ("A", "B", "C", "D")}, FIG / "02_benchmark_historico_v7.png", "V7 · benchmark histórico reutilizado")
    fig, ax = plt.subplots(figsize=(9, 5)); names = list(select_ap); ax.barh(names, [select_ap[n] for n in names], color="#184e77"); ax.set(xlabel="AP en model_select", title="Selección train-only de A"); fig.tight_layout(); fig.savefig(FIG / "03_seleccion_modelos_a_v7.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.bar(["Completo", "Correlación", "PCA"], [select_ap.get("A1_lgbm_ampliado", 0), select_ap.get("A2_lgbm_correlacion", 0), select_ap.get(best_pca_name, 0)], color=["#184e77", "#2a9d8f", "#e9c46a"]); ax.set(ylabel="AP", title="Ablation de reducción dimensional"); fig.tight_layout(); fig.savefig(FIG / "04_correlacion_pca_v7.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); table = pd.DataFrame(internal).T.loc[["A", "B", "C", "D"]]; ax.bar(table.index, table["cost_q"] / 1e6, color=["#184e77", "#e9c46a", "#2a9d8f", "#e76f51"]); ax.set(ylabel="Costo interno (millones Q)", title="Costo bajo umbral predefinido"); fig.tight_layout(); fig.savefig(FIG / "05_costos_v7.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6));
    for name in ("A", "B", "C", "D"):
        observed, predicted = calibration_curve(y_val[bounds["calibration"]], calibrated_val[name][bounds["calibration"]], n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#6b7280"); ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Calibración independiente V7"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "06_calibracion_v7.png", dpi=180); plt.close(fig)

    manifest = [{"archivo": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size} for path in sorted(ART.rglob("*")) if path.is_file()]
    write_json(ART / "manifiesto_v7.json", manifest)
    print(json.dumps(ready({"A": best_a_name, "C": best_c_name, "candidato": candidate, "interno": internal[candidate], "promueve_v7": promotion_success, "C_util": c_success}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main_v7()
