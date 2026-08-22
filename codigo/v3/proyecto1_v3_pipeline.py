from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codigo" / "v3"))
from dataset_v3_support import (  # noqa: E402
    add_causal_features,
    load_all,
    set_seed,
    temporal_boundaries,
)

ART = ROOT / "artefactos" / "v3"
PROCESSED = ROOT / "datos" / "processed" / "v3"
FIG = ROOT / "evidencia" / "figuras" / "v3"
V2_ART = ART / "referencia_v2"


@dataclass(frozen=True)
class ConfigV3:
    seed: int = 2026
    train_fraction: float = 0.70
    development_fraction: float = 0.85
    audit_train_fraction: float = 0.55
    max_numeric: int = 220
    max_categorical: int = 24
    corr_threshold: float = 0.995
    corr_sample: int = 45_000
    rare_min_count: int = 20
    max_category_levels: int = 1000
    lgb_iterations: int = 1400
    early_stopping_rounds: int = 75
    recency_half_life_days: float = 75.0
    logistic_train_rows: int = 250_000
    logistic_numeric: int = 90
    logistic_categorical: int = 12
    cost_fn_q: float = 4200.0
    cost_fp_q: float = 180.0
    recall_floor: float = 0.70
    bootstrap_repetitions: int = 400
    promotion_auc_pr_delta: float = 0.015
    promotion_cost_reduction: float = 0.03
    promotion_recall_tolerance: float = 0.01


def ensure_dirs() -> None:
    for path in (ART, PROCESSED, FIG):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )


def ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def metricas(y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV3) -> dict[str, Any]:
    pred = (score >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_pr": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "costo_q": float(fn * cfg.cost_fn_q + fp * cfg.cost_fp_q),
        "alertas_por_100k": float(pred.mean() * 100_000),
    }


def choose_threshold(
    y: np.ndarray, score: np.ndarray, cfg: ConfigV3, recall_floor: float | None = None
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.quantile(score, np.linspace(0.40, 0.9995, 700)))
    table = pd.DataFrame([metricas(y, score, float(t), cfg) for t in candidates])
    eligible = table if recall_floor is None else table.loc[table["recall"] >= recall_floor]
    if eligible.empty:
        eligible = table
    best = eligible.sort_values(["costo_q", "threshold"]).iloc[0]
    return float(best["threshold"]), table


def ece(y: np.ndarray, score: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return float(total)


def select_features(df: pd.DataFrame, audit_idx: np.ndarray, cfg: ConfigV3) -> tuple[list[str], list[str], dict[str, Any]]:
    audit = pd.read_csv(PROCESSED / "auditoria_fuente_variables.csv")
    excluded_decisions = {"excluida_alta_ausencia", "excluida_constante"}
    ranked = audit.loc[
        ~audit["decision"].isin(excluded_decisions)
        & audit["variable"].isin(df.columns)
        & ~audit["variable"].isin(["isFraud", "TransactionID", "TransactionDT"])
    ].sort_values("puntaje_relevancia", ascending=False)
    mandatory = [
        "TransactionAmt", "missing_count", "entity_prior_count",
        "entity_prior_amt_mean", "entity_prior_amt_std", "amount_to_prior_mean",
        "hours_since_prior", "entity_count_1h", "entity_count_6h",
        "entity_count_24h", "entity_count_72h", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
    ]
    pool = list(dict.fromkeys([x for x in mandatory if x in df] + ranked["variable"].tolist()))
    pool = pool[: min(300, len(pool))]
    rng = np.random.default_rng(cfg.seed)
    sample = np.sort(rng.choice(audit_idx, min(cfg.corr_sample, len(audit_idx)), replace=False))
    corr = df.loc[sample, pool].corr(method="spearman").abs()
    selected: list[str] = []
    removed: list[dict[str, Any]] = []
    for col in pool:
        conflict = next((kept for kept in selected if corr.loc[col, kept] >= cfg.corr_threshold), None)
        if conflict is None:
            selected.append(col)
        else:
            removed.append({
                "variable": col,
                "retenida": conflict,
                "rho_spearman_abs": float(corr.loc[col, conflict]),
            })
        if len(selected) >= cfg.max_numeric:
            break

    cat_audit = pd.read_csv(PROCESSED / "auditoria_fuente_categoricas.csv")
    categorical = cat_audit.loc[cat_audit["variable"].isin(df.columns)].head(cfg.max_categorical)["variable"].tolist()
    corr.loc[selected, selected].to_csv(PROCESSED / "matriz_correlacion_v3.csv")
    pd.DataFrame(removed).to_csv(PROCESSED / "variables_redundantes_v3.csv", index=False)
    result = {
        "numericas": selected,
        "categoricas": categorical,
        "n_numericas": len(selected),
        "n_categoricas": len(categorical),
        "umbral_redundancia": cfg.corr_threshold,
        "redundantes_eliminadas": removed,
        "exclusion_ids": ["TransactionID", "TransactionDT"],
        "criterio": "relevancia aprendida en 55% inicial + poda Spearman; baja correlación marginal no implica exclusión automática",
    }
    write_json(PROCESSED / "seleccion_variables_v3.json", result)
    return selected, categorical, result


class NativeEncoder:
    def __init__(self, numeric: list[str], categorical: list[str], cfg: ConfigV3):
        self.numeric = numeric
        self.categorical = categorical
        self.cfg = cfg
        self.levels: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, idx: np.ndarray) -> "NativeEncoder":
        for col in self.categorical:
            counts = df.loc[idx, col].fillna("MISSING").astype(str).value_counts()
            levels = counts.loc[counts >= self.cfg.rare_min_count].index[: self.cfg.max_category_levels].tolist()
            for special in ("MISSING", "OTHER"):
                if special not in levels:
                    levels.append(special)
            self.levels[col] = levels
        return self

    def transform(self, df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        out = df.loc[idx, self.numeric].replace([np.inf, -np.inf], np.nan).astype("float32").copy()
        for col in self.categorical:
            values = df.loc[idx, col].fillna("MISSING").astype(str)
            levels = self.levels[col]
            values = values.where(values.isin(levels), "OTHER")
            out[col] = pd.Categorical(values, categories=levels)
        return out

    def serializable(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "levels": self.levels}


def lgb_model(cfg: ConfigV3, seed_offset: int = 0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.lgb_iterations,
        learning_rate=0.025,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=65,
        subsample=0.90,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.10,
        reg_lambda=1.40,
        max_bin=255,
        min_data_per_group=80,
        cat_smooth=20,
        random_state=cfg.seed + seed_offset,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )


def folds(n: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    specs = [(0.55, 0.55, 0.65), (0.65, 0.65, 0.75), (0.75, 0.75, 0.85)]
    return [(np.arange(int(n * a)), np.arange(int(n * b), int(n * c)), f"F{i+1}") for i, (a, b, c) in enumerate(specs)]


def recency_weights(df: pd.DataFrame, idx: np.ndarray, cfg: ConfigV3) -> np.ndarray:
    seconds = df.loc[idx, "TransactionDT"].to_numpy(float)
    age_days = (seconds.max() - seconds) / 86400.0
    return np.clip(np.power(0.5, age_days / cfg.recency_half_life_days), 0.15, 1.0).astype("float32")


def evaluate_walk_forward(df: pd.DataFrame, numeric: list[str], categorical: list[str], cfg: ConfigV3) -> pd.DataFrame:
    y = df["isFraud"].to_numpy(np.int8)
    rows: list[dict[str, Any]] = []
    variants = ("LGB_native_uniform", "LGB_native_recency", "LGB_native_recent300k")
    for tr, va, fold_name in folds(len(df)):
        encoder = NativeEncoder(numeric, categorical, cfg).fit(df, tr)
        Xtr, Xva = encoder.transform(df, tr), encoder.transform(df, va)
        for j, name in enumerate(variants):
            chosen = tr
            weight = None
            if name == "LGB_native_recency":
                weight = recency_weights(df, tr, cfg)
            elif name == "LGB_native_recent300k":
                chosen = tr[-min(300_000, len(tr)) :]
            Xt = Xtr if len(chosen) == len(tr) else Xtr.iloc[-len(chosen) :]
            start = time.perf_counter()
            model = lgb_model(cfg, 10 * int(fold_name[-1]) + j).fit(
                Xt,
                y[chosen],
                sample_weight=weight,
                eval_X=Xva,
                eval_y=y[va],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
                categorical_feature=categorical,
            )
            score = model.predict_proba(Xva)[:, 1]
            rows.append({
                "fold": fold_name,
                "modelo": name,
                "auc_pr": float(average_precision_score(y[va], score)),
                "roc_auc": float(roc_auc_score(y[va], score)),
                "segundos": float(time.perf_counter() - start),
                "mejor_iteracion": int(model.best_iteration_ or cfg.lgb_iterations),
                "variables": int(Xt.shape[1]),
            })
            del model, score
            gc.collect()
        del Xtr, Xva
        gc.collect()
    table = pd.DataFrame(rows)
    table.to_csv(ART / "validacion_walk_forward_v3.csv", index=False)
    summary = table.groupby("modelo").agg(
        auc_pr_media=("auc_pr", "mean"), auc_pr_sd=("auc_pr", "std"),
        roc_auc_media=("roc_auc", "mean"), segundos_media=("segundos", "mean"),
        mejor_iteracion=("mejor_iteracion", "median"), variables=("variables", "median"),
    ).sort_values("auc_pr_media", ascending=False)
    summary.to_csv(ART / "resumen_walk_forward_v3.csv")
    return table


def logistic_baselines(
    df: pd.DataFrame, train: np.ndarray, val: np.ndarray,
    numeric: list[str], categorical: list[str], cfg: ConfigV3,
) -> dict[str, Any]:
    chosen = train[-min(cfg.logistic_train_rows, len(train)) :]
    num = numeric[: cfg.logistic_numeric]
    cat = categorical[: cfg.logistic_categorical]
    Xtr = df.loc[chosen, num + cat].copy()
    Xva = df.loc[val, num + cat].copy()
    ytr = df.loc[chosen, "isFraud"].to_numpy(np.int8)
    yva = df.loc[val, "isFraud"].to_numpy(np.int8)
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, dtype=np.float32))]), cat),
    ], sparse_threshold=0.20)
    Ztr = prep.fit_transform(Xtr)
    Zva = prep.transform(Xva)
    specs = {
        "Logistica_L2": dict(C=1.0, l1_ratio=0.0),
        "Logistica_L1": dict(C=0.25, l1_ratio=1.0),
        "Logistica_ElasticNet": dict(C=0.50, l1_ratio=0.5),
    }
    rows: list[dict[str, Any]] = []
    fitted: dict[str, LogisticRegression] = {}
    for i, (name, params) in enumerate(specs.items()):
        start = time.perf_counter()
        model = LogisticRegression(
            solver="saga", max_iter=100, tol=1e-3,
            random_state=cfg.seed + i, **params,
        ).fit(Ztr, ytr)
        score = model.predict_proba(Zva)[:, 1]
        rows.append({
            "modelo": name, "auc_pr_validacion": float(average_precision_score(yva, score)),
            "roc_auc_validacion": float(roc_auc_score(yva, score)),
            "segundos": float(time.perf_counter() - start), "iteraciones": int(np.max(model.n_iter_)),
            "variables_transformadas": int(Ztr.shape[1]),
            "convergio": bool(np.max(model.n_iter_) < model.max_iter),
        })
        fitted[name] = model

    pca_num = num[: min(100, len(num))]
    pca_imputer = SimpleImputer(strategy="median")
    pca_scaler = StandardScaler()
    Ptrain = pca_scaler.fit_transform(pca_imputer.fit_transform(df.loc[chosen, pca_num]))
    Pval = pca_scaler.transform(pca_imputer.transform(df.loc[val, pca_num]))
    components = min(64, Ptrain.shape[1], Ptrain.shape[0] - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=cfg.seed)
    Ttrain, Tval = pca.fit_transform(Ptrain), pca.transform(Pval)
    start = time.perf_counter()
    pca_model = LogisticRegression(max_iter=300, solver="lbfgs", random_state=cfg.seed).fit(Ttrain, ytr)
    pca_score = pca_model.predict_proba(Tval)[:, 1]
    rows.append({
        "modelo": "Logistica_PCA64", "auc_pr_validacion": float(average_precision_score(yva, pca_score)),
        "roc_auc_validacion": float(roc_auc_score(yva, pca_score)),
        "segundos": float(time.perf_counter() - start), "iteraciones": int(np.max(pca_model.n_iter_)),
        "variables_transformadas": int(components), "varianza_pca": float(pca.explained_variance_ratio_.sum()),
        "convergio": bool(np.max(pca_model.n_iter_) < pca_model.max_iter),
    })
    table = pd.DataFrame(rows).sort_values("auc_pr_validacion", ascending=False)
    table.to_csv(ART / "baselines_logisticos_v3.csv", index=False)
    winner = str(table.iloc[0]["modelo"])
    if winner in fitted:
        joblib.dump({"preprocessor": prep, "model": fitted[winner], "numeric": num, "categorical": cat}, ART / "baseline_logistico_ganador.joblib")
    joblib.dump({"imputer": pca_imputer, "scaler": pca_scaler, "pca": pca, "model": pca_model, "numeric": pca_num}, ART / "baseline_logistico_pca.joblib")
    del Ztr, Zva, Ptrain, Pval, Ttrain, Tval
    gc.collect()
    return {"resultados": table.to_dict("records"), "ganador": winner}


def fit_final(
    df: pd.DataFrame, split: dict[str, np.ndarray], numeric: list[str], categorical: list[str],
    winner: str, cfg: ConfigV3,
) -> dict[str, Any]:
    train, val, bench = split["train"], split["validation"], split["benchmark_historico"]
    y = df["isFraud"].to_numpy(np.int8)
    encoder = NativeEncoder(numeric, categorical, cfg).fit(df, train)
    Xtr, Xva, Xbe = encoder.transform(df, train), encoder.transform(df, val), encoder.transform(df, bench)
    chosen = train
    weight = None
    if winner == "LGB_native_recency":
        weight = recency_weights(df, train, cfg)
    elif winner == "LGB_native_recent300k":
        chosen = train[-min(300_000, len(train)) :]
    Xt = Xtr if len(chosen) == len(train) else Xtr.iloc[-len(chosen) :]
    model = lgb_model(cfg, 901).fit(
        Xt, y[chosen], sample_weight=weight,
        eval_X=Xva.iloc[: int(len(val) * 0.40)],
        eval_y=y[val][: int(len(val) * 0.40)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        categorical_feature=categorical,
    )
    val_raw = model.predict_proba(Xva)[:, 1]
    bench_raw = model.predict_proba(Xbe)[:, 1]
    n = len(val)
    calibration_idx = np.arange(int(n * 0.40), int(n * 0.70))
    threshold_idx = np.arange(int(n * 0.70), n)
    eps = 1e-6
    def logit(s: np.ndarray) -> np.ndarray:
        p = np.clip(s, eps, 1 - eps)
        return np.log(p / (1 - p)).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    calibrator.fit(logit(val_raw[calibration_idx]), y[val][calibration_idx])
    val_score = calibrator.predict_proba(logit(val_raw))[:, 1]
    bench_score = calibrator.predict_proba(logit(bench_raw))[:, 1]
    economic_threshold, curve = choose_threshold(y[val][threshold_idx], val_score[threshold_idx], cfg)
    eligible_balanced = curve.loc[curve["recall"] >= cfg.recall_floor]
    if eligible_balanced.empty:
        eligible_balanced = curve
    balanced_threshold = float(
        eligible_balanced.sort_values(["f1", "costo_q"], ascending=[False, True]).iloc[0]["threshold"]
    )
    curve.to_csv(ART / "curva_umbral_v3.csv", index=False)
    model.booster_.save_model(str(ART / "modelo_lightgbm_v3.txt"))
    joblib.dump(encoder.serializable(), ART / "codificador_nativo_v3.joblib")
    joblib.dump(calibrator, ART / "calibrador_sigmoide_v3.joblib")
    pd.DataFrame({"indice": val, "TransactionID": df.loc[val, "TransactionID"], "y": y[val], "score_raw": val_raw, "score_calibrado": val_score}).to_csv(ART / "predicciones_validacion_v3.csv", index=False)
    pd.DataFrame({"indice": bench, "TransactionID": df.loc[bench, "TransactionID"], "y": y[bench], "score_raw": bench_raw, "score_calibrado": bench_score}).to_csv(ART / "predicciones_benchmark_v3.csv", index=False)
    result = {
        "modelo": winner,
        "mejor_iteracion": int(model.best_iteration_ or cfg.lgb_iterations),
        "threshold": balanced_threshold,
        "threshold_recomendado_balanceado": balanced_threshold,
        "threshold_economico": economic_threshold,
        "particiones_validacion": {"early_stopping": [0, int(n * 0.40)], "calibracion": [int(n * 0.40), int(n * 0.70)], "umbral": [int(n * 0.70), n]},
        "calibracion": {
            "brier_raw_validacion": float(brier_score_loss(y[val], val_raw)),
            "brier_calibrado_validacion": float(brier_score_loss(y[val], val_score)),
            "ece_raw_validacion": ece(y[val], val_raw),
            "ece_calibrado_validacion": ece(y[val], val_score),
        },
        "validacion_completa": metricas(y[val], val_score, balanced_threshold, cfg),
        "validacion_holdout_umbral": metricas(y[val][threshold_idx], val_score[threshold_idx], balanced_threshold, cfg),
        "validacion_holdout_economico": metricas(y[val][threshold_idx], val_score[threshold_idx], economic_threshold, cfg),
        "benchmark_historico": metricas(y[bench], bench_score, balanced_threshold, cfg),
        "benchmark_economico": metricas(y[bench], bench_score, economic_threshold, cfg),
    }
    del Xtr, Xva, Xbe, Xt, model
    gc.collect()
    return {"resultado": result, "val_score": val_score, "bench_score": bench_score, "threshold_idx": threshold_idx}


def paired_block_delta(y: np.ndarray, new: np.ndarray, old: np.ndarray, cfg: ConfigV3, blocks: int = 24) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    pieces = np.array_split(np.arange(len(y)), blocks)
    estimates: list[float] = []
    for _ in range(cfg.bootstrap_repetitions):
        idx = np.concatenate([pieces[i] for i in rng.integers(0, blocks, size=blocks)])
        if 0 < y[idx].sum() < len(idx):
            estimates.append(float(average_precision_score(y[idx], new[idx]) - average_precision_score(y[idx], old[idx])))
    delta = float(average_precision_score(y, new) - average_precision_score(y, old))
    return {"delta_auc_pr": delta, "li95": float(np.quantile(estimates, 0.025)), "ls95": float(np.quantile(estimates, 0.975)), "replicas": len(estimates), "bloques": blocks}


def top_k(y: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-score)
    rows = []
    for rate in (0.001, 0.005, 0.01, 0.02, 0.05):
        k = max(1, int(len(y) * rate))
        picked = y[order[:k]]
        rows.append({"tasa_revision": rate, "k": k, "precision_at_k": float(picked.mean()), "recall_at_k": float(picked.sum() / max(1, y.sum()))})
    return rows


def segment_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV3) -> pd.DataFrame:
    frame = pd.DataFrame({
        "ProductCD": df.loc[idx, "ProductCD"].fillna("MISSING").astype(str).to_numpy(),
        "DeviceType": df.loc[idx, "DeviceType"].fillna("MISSING").astype(str).to_numpy(),
        "history": pd.cut(df.loc[idx, "entity_prior_count"], [-1, 0, 2, 7, 15, np.inf], labels=["0", "1-2", "3-7", "8-15", "16+"]).astype(str).to_numpy(),
        "amount": df.loc[idx, "TransactionAmt"].to_numpy(), "y": y, "score": score,
    })
    frame["segmento_monto"] = pd.qcut(frame["amount"], 4, duplicates="drop").astype(str)
    rows = []
    for dimension in ("ProductCD", "DeviceType", "history", "segmento_monto"):
        for value, group in frame.groupby(dimension, observed=True):
            if len(group) >= 100 and group["y"].sum() > 0:
                rows.append({"dimension": dimension, "segmento": value, "n": len(group), "prevalencia": float(group["y"].mean()), **metricas(group["y"].to_numpy(np.int8), group["score"].to_numpy(float), threshold, cfg)})
    return pd.DataFrame(rows)


def comparisons_and_promotion(
    walk: pd.DataFrame, final: dict[str, Any], split: dict[str, np.ndarray], cfg: ConfigV3
) -> dict[str, Any]:
    v2 = json.loads((V2_ART / "resultados_v2.json").read_text(encoding="utf-8"))
    v2_walk = float(next(x["auc_pr_media"] for x in v2["validacion_walk_forward"]["resumen"] if x["modelo"] == "LightGBM_corr_pruned"))
    summary = walk.groupby("modelo")["auc_pr"].mean().sort_values(ascending=False)
    winner = str(summary.index[0])
    v3_walk = float(summary.iloc[0])
    wins = 0
    v2_folds = pd.read_csv(V2_ART / "validacion_walk_forward.csv")
    for fold in ("F1", "F2", "F3"):
        v3_ap = float(walk.loc[(walk["fold"] == fold) & (walk["modelo"] == winner), "auc_pr"].iloc[0])
        v2_ap = float(v2_folds.loc[(v2_folds["fold"] == fold) & (v2_folds["modelo"] == "LightGBM_corr_pruned"), "auc_pr"].iloc[0])
        wins += int(v3_ap > v2_ap)

    old_val = pd.read_csv(V2_ART / "predicciones_validacion.csv")
    ti = final["threshold_idx"]
    old_threshold = float(v2["modelo_tabular_v2"]["threshold"])
    old_holdout = metricas(old_val["y"].to_numpy(np.int8)[ti], old_val["score_tabular"].to_numpy(float)[ti], old_threshold, cfg)
    new_holdout = final["resultado"]["validacion_holdout_umbral"]
    cost_reduction = 1.0 - new_holdout["costo_q"] / old_holdout["costo_q"]
    criteria = {
        "auc_pr_walk_delta_min_0_015": v3_walk - v2_walk >= cfg.promotion_auc_pr_delta,
        "costo_holdout_reduccion_min_3pct": cost_reduction >= cfg.promotion_cost_reduction,
        "recall_no_cae_mas_1pp": new_holdout["recall"] >= old_holdout["recall"] - cfg.promotion_recall_tolerance,
        "gana_al_menos_2_de_3_folds": wins >= 2,
    }
    return {
        "referencia": "LightGBM_corr_pruned_V2",
        "candidato": winner,
        "v2_auc_pr_walk": v2_walk,
        "v3_auc_pr_walk": v3_walk,
        "delta_auc_pr_walk": v3_walk - v2_walk,
        "folds_ganados": wins,
        "v2_validacion_holdout": old_holdout,
        "v3_validacion_holdout": new_holdout,
        "reduccion_costo_holdout": cost_reduction,
        "criterios": criteria,
        "promover_v3": bool(all(criteria.values())),
        "regla": "Promoción exige los cuatro criterios; el benchmark histórico no decide promoción.",
    }


def plots(walk: pd.DataFrame, logistic: dict[str, Any], y: np.ndarray, v3_score: np.ndarray, v2_score: np.ndarray, curve: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#2a9d8f"); ax.set(xlabel="AUC-PR media", title="V3 · variantes LightGBM walk-forward"); fig.tight_layout(); fig.savefig(FIG / "01_walk_forward_v3.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, score, color in (("V3", v3_score, "#2a9d8f"), ("V2", v2_score, "#184e77")):
        p, r, _ = precision_recall_curve(y, score); ax.plot(r, p, lw=2.2, color=color, label=f"{label} · AP={average_precision_score(y, score):.3f}")
    ax.axhline(y.mean(), color="#6b7280", ls="--", label="Prevalencia"); ax.set(xlabel="Recall", ylabel="Precisión", title="Benchmark histórico reutilizado"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "02_curvas_pr_v2_v3.png", dpi=180); plt.close(fig)
    ldf = pd.DataFrame(logistic["resultados"]).sort_values("auc_pr_validacion")
    fig, ax = plt.subplots(figsize=(9, 5)); ax.barh(ldf["modelo"], ldf["auc_pr_validacion"], color="#376f9e"); ax.set(xlabel="AUC-PR validación", title="Baselines de regresión logística"); fig.tight_layout(); fig.savefig(FIG / "03_logistica_v3.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5)); ordered = curve.sort_values("recall"); ax.plot(ordered["recall"], ordered["costo_q"], color="#e76f51", lw=2); ax.set(xlabel="Recall", ylabel="Costo Q", title="V3 · frontera umbral, recall y costo"); fig.tight_layout(); fig.savefig(FIG / "04_costo_recall_v3.png", dpi=180); plt.close(fig)


def main() -> None:
    cfg = ConfigV3()
    ensure_dirs(); set_seed(cfg.seed); started = time.time()
    print("[1/8] Cargando IEEE-CIS e integrando identidad...")
    df = load_all()
    split = temporal_boundaries(df, cfg)
    print("[2/8] Construyendo agregados causales...")
    df, identities = add_causal_features(df)
    print("[3/8] Ampliando selección y revalidando correlación...")
    numeric, categorical, selection = select_features(df, split["audit_train"], cfg)
    print(f"Variables V3: {len(numeric)} numéricas + {len(categorical)} categóricas")
    print("[4/8] Walk-forward: LightGBM nativo, recencia y ventana reciente...")
    walk = evaluate_walk_forward(df, numeric, categorical, cfg)
    walk_summary = walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    winner = str(walk_summary.index[0])
    print(walk_summary)
    print("[5/8] Baselines logísticos L2/L1/ElasticNet/PCA...")
    logistic = logistic_baselines(df, split["train"], split["validation"], numeric, categorical, cfg)
    print("[6/8] Ajuste final, calibración y umbral separado...")
    final = fit_final(df, split, numeric, categorical, winner, cfg)
    promotion = comparisons_and_promotion(walk, final, split, cfg)
    print("[7/8] Comparación pareada, segmentos y capacidad...")
    bench = split["benchmark_historico"]
    yb = df.loc[bench, "isFraud"].to_numpy(np.int8)
    v2_pred = pd.read_csv(V2_ART / "predicciones_benchmark_historico.csv")
    v2_score = v2_pred["score_tabular"].to_numpy(float)
    pair = paired_block_delta(yb, final["bench_score"], v2_score, cfg)
    segments = segment_metrics(df, bench, yb, final["bench_score"], final["resultado"]["threshold"], cfg)
    segments.to_csv(ART / "metricas_segmentos_v3.csv", index=False)
    topk = top_k(yb, final["bench_score"])
    pd.DataFrame(topk).to_csv(ART / "metricas_top_k_v3.csv", index=False)
    curve = pd.read_csv(ART / "curva_umbral_v3.csv")
    plots(walk, logistic, yb, final["bench_score"], v2_score, curve)
    result = {
        "version": "3.0",
        "estado_benchmark": "historico_reutilizado_no_ciego",
        "configuracion": asdict(cfg),
        "entorno": {"python": platform.python_version(), "plataforma": platform.platform(), "lightgbm": lgb.__version__, "cpu_logicos": os.cpu_count()},
        "datos": {"filas": len(df), "columnas_integradas": 434, "prevalencia": float(df["isFraud"].mean()), "particiones": {k: {"n": len(v), "fraude": float(df.loc[v, "isFraud"].mean()), "dt_min": float(df.loc[v, "TransactionDT"].min()), "dt_max": float(df.loc[v, "TransactionDT"].max())} for k, v in split.items()}},
        "identidad_secuencial": identities,
        "seleccion_variables": selection,
        "validacion_walk_forward": {"ganador": winner, "resumen": walk_summary.reset_index().to_dict("records"), "detalle": walk.to_dict("records")},
        "baselines_logisticos": logistic,
        "modelo_v3": final["resultado"],
        "comparacion_pareada_benchmark": pair,
        "metricas_top_k": topk,
        "promocion": promotion,
        "duracion_segundos": time.time() - started,
        "recomendacion": "Promover V3" if promotion["promover_v3"] else "Conservar V2 y tratar V3 como experimento",
        "limitaciones": ["Benchmark reutilizado y no ciego", "Identidad aproximada", "Costos académicos", "La GRU V1 no mostró señal de orden suficiente; no se fabricó un stacking GRU sin predicciones OOF", "Una cohorte nueva es necesaria para confirmación"],
    }
    write_json(ART / "resultados_v3.json", ready(result))
    print("[8/8] Fuente única:", ART / "resultados_v3.json")
    print(json.dumps(promotion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
