"""Proyecto 1 V4: optimización temporal, expertos y stacking leakage-safe.

La promoción de V4 se decide únicamente con walk-forward y el último bloque de
validación. El benchmark final se conserva como referencia histórica reutilizada.
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
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
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codigo" / "v3"))
sys.path.insert(0, str(ROOT / "codigo" / "v4"))
from dataset_v4_support import (  # noqa: E402
    V4_CATEGORICAL,
    add_v4_features,
    load_all,
    set_seed,
    temporal_boundaries,
)

ART = ROOT / "artefactos" / "v4"
PROCESSED = ROOT / "datos" / "processed" / "v4"
FIG = ROOT / "evidencia" / "figuras" / "v4"
V3_ART = ROOT / "artefactos" / "v3"


@dataclass(frozen=True)
class ConfigV4:
    seed: int = 2026
    train_fraction: float = 0.70
    development_fraction: float = 0.85
    audit_train_fraction: float = 0.55
    max_numeric: int = 360
    max_categorical: int = 38
    corr_threshold: float = 0.999
    corr_sample: int = 30_000
    rare_min_count: int = 15
    max_category_levels: int = 2500
    optuna_trials: int = 18
    tuning_estimators: int = 900
    final_estimators: int = 1800
    early_stopping_rounds: int = 70
    recency_half_life_days: float = 90.0
    hard_negative_multiplier: float = 2.25
    hard_negative_quantile: float = 0.90
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


def ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(ready(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def ece(y: np.ndarray, score: np.ndarray, bins: int = 12) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (score >= left) & (score < right if right < 1 else score <= right)
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return float(total)


def metrics(y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV4) -> dict[str, Any]:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_pr": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "costo_q": float(fn * cfg.cost_fn_q + fp * cfg.cost_fp_q),
        "alertas_por_100k": float(pred.mean() * 100_000),
    }


def choose_threshold(y: np.ndarray, score: np.ndarray, cfg: ConfigV4) -> tuple[float, float, pd.DataFrame]:
    precision, recall, thresholds = precision_recall_curve(y, score)
    if len(thresholds) > 1500:
        chosen = np.unique(np.quantile(thresholds, np.linspace(0, 1, 1500)))
    else:
        chosen = thresholds
    rows = []
    for threshold in chosen:
        row = metrics(y, score, float(threshold), cfg)
        rows.append({k: row[k] for k in ("threshold", "precision", "recall", "f1", "costo_q", "alertas_por_100k")})
    curve = pd.DataFrame(rows)
    economic = float(curve.sort_values(["costo_q", "f1"], ascending=[True, False]).iloc[0]["threshold"])
    eligible = curve.loc[curve["recall"] >= cfg.recall_floor]
    if eligible.empty:
        eligible = curve
    balanced = float(eligible.sort_values(["f1", "costo_q"], ascending=[False, True]).iloc[0]["threshold"])
    return balanced, economic, curve


def select_features(df: pd.DataFrame, audit_idx: np.ndarray, cfg: ConfigV4) -> tuple[list[str], list[str], dict[str, Any]]:
    excluded = {"isFraud", "TransactionID", "TransactionDT"}
    categorical = [c for c in V4_CATEGORICAL if c in df][: cfg.max_categorical]
    numeric_candidates = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in excluded and c not in categorical
    ]
    rng = np.random.default_rng(cfg.seed)
    audit_sample = np.sort(rng.choice(audit_idx, min(120_000, len(audit_idx)), replace=False))
    audit = df.loc[audit_sample, numeric_candidates]
    present = audit.notna().mean()
    variable = audit.nunique(dropna=True)
    eligible = [c for c in numeric_candidates if present[c] >= 0.005 and variable[c] > 1]

    v3_audit_path = ROOT / "datos" / "processed" / "v3" / "auditoria_fuente_variables.csv"
    ranked_raw: list[str] = []
    if v3_audit_path.exists():
        source = pd.read_csv(v3_audit_path).sort_values("puntaje_relevancia", ascending=False)
        ranked_raw = [c for c in source["variable"].tolist() if c in eligible]
    engineered = [
        c for c in eligible
        if c not in ranked_raw or any(token in c for token in (
            "prior_", "freq_", "share_", "changed_", "_missing", "_mean",
            "_std", "_min", "_max", "minus_day", "amount_", "day_",
            "week_", "month_", "hour", "weekday", "weekend",
        ))
    ]
    pool = list(dict.fromkeys(engineered + ranked_raw + sorted(eligible)))
    pool = pool[: min(440, len(pool))]
    sample = np.sort(rng.choice(audit_idx, min(cfg.corr_sample, len(audit_idx)), replace=False))
    corr = df.loc[sample, pool].corr(method="spearman").abs()
    selected: list[str] = []
    removed: list[dict[str, Any]] = []
    for column in pool:
        conflict = next((kept for kept in selected if corr.loc[column, kept] >= cfg.corr_threshold), None)
        if conflict is None:
            selected.append(column)
        else:
            removed.append({"variable": column, "retenida": conflict, "rho": float(corr.loc[column, conflict])})
        if len(selected) >= cfg.max_numeric:
            break
    corr.loc[selected, selected].to_csv(PROCESSED / "matriz_correlacion_v4.csv")
    pd.DataFrame(removed).to_csv(PROCESSED / "variables_redundantes_v4.csv", index=False)
    result = {
        "numericas": selected,
        "categoricas": categorical,
        "n_numericas": len(selected),
        "n_categoricas": len(categorical),
        "candidatas_numericas": len(numeric_candidates),
        "elegibles_numericas": len(eligible),
        "ingenieria_prioritaria": len(engineered),
        "umbral_redundancia": cfg.corr_threshold,
        "redundantes_eliminadas": removed,
        "ids_excluidos": ["TransactionID", "TransactionDT"],
        "nota_pca": "PCA no se aplica al boosting: en V3 redujo AUC-PR pese a conservar 99.53% de varianza.",
    }
    write_json(PROCESSED / "seleccion_variables_v4.json", result)
    del audit, corr
    gc.collect()
    return selected, categorical, result


class NativeEncoder:
    def __init__(self, numeric: list[str], categorical: list[str], cfg: ConfigV4):
        self.numeric = numeric
        self.categorical = categorical
        self.cfg = cfg
        self.levels: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame, idx: np.ndarray) -> "NativeEncoder":
        for column in self.categorical:
            counts = df.loc[idx, column].fillna("MISSING").astype(str).value_counts()
            levels = counts.loc[counts >= self.cfg.rare_min_count].index[: self.cfg.max_category_levels].tolist()
            for special in ("MISSING", "OTHER"):
                if special not in levels:
                    levels.append(special)
            self.levels[column] = levels
        return self

    def transform(self, df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        out = df.loc[idx, self.numeric].replace([np.inf, -np.inf], np.nan).astype("float32").copy()
        for column in self.categorical:
            values = df.loc[idx, column].fillna("MISSING").astype(str)
            levels = self.levels[column]
            values = values.where(values.isin(levels), "OTHER")
            out[column] = pd.Categorical(values, categories=levels)
        return out

    def numeric_codes(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.categorical:
            out[column] = out[column].cat.codes.astype("int32")
        return out

    def serializable(self) -> dict[str, Any]:
        return {"numeric": self.numeric, "categorical": self.categorical, "levels": self.levels}


def recency_weights(df: pd.DataFrame, idx: np.ndarray, cfg: ConfigV4) -> np.ndarray:
    seconds = df.loc[idx, "TransactionDT"].to_numpy(float)
    age_days = (seconds.max() - seconds) / 86400.0
    return np.clip(np.power(0.5, age_days / cfg.recency_half_life_days), 0.15, 1.0).astype("float32")


def base_lgb_params(cfg: ConfigV4) -> dict[str, Any]:
    return {
        "objective": "binary", "n_estimators": cfg.final_estimators,
        "learning_rate": 0.025, "num_leaves": 64, "max_depth": -1,
        "min_child_samples": 65, "subsample": 0.90, "subsample_freq": 1,
        "colsample_bytree": 0.82, "reg_alpha": 0.10, "reg_lambda": 1.40,
        "max_bin": 255, "min_data_per_group": 80, "cat_smooth": 20,
        "random_state": cfg.seed, "n_jobs": max(1, os.cpu_count() or 1), "verbosity": -1,
    }


def make_lgb(cfg: ConfigV4, params: dict[str, Any], seed_offset: int = 0, estimators: int | None = None) -> lgb.LGBMClassifier:
    full = base_lgb_params(cfg)
    full.update(params)
    full["random_state"] = cfg.seed + seed_offset
    if estimators is not None:
        full["n_estimators"] = estimators
    return lgb.LGBMClassifier(**full)


def folds(n: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    specs = [(0.55, 0.55, 0.65), (0.65, 0.65, 0.75), (0.75, 0.75, 0.85)]
    return [
        (np.arange(int(n * a)), np.arange(int(n * b), int(n * c)), f"F{i + 1}")
        for i, (a, b, c) in enumerate(specs)
    ]


def tune_lightgbm(df: pd.DataFrame, numeric: list[str], categorical: list[str], cfg: ConfigV4) -> tuple[dict[str, Any], pd.DataFrame]:
    n = len(df)
    tr = np.arange(0, int(n * 0.65))
    va = np.arange(int(n * 0.65), int(n * 0.75))
    y = df["isFraud"].to_numpy(np.int8)
    encoder = NativeEncoder(numeric, categorical, cfg).fit(df, tr)
    Xtr, Xva = encoder.transform(df, tr), encoder.transform(df, va)
    weights = recency_weights(df, tr, cfg)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_child_samples": trial.suggest_int("min_child_samples", 35, 180),
            "subsample": trial.suggest_float("subsample", 0.70, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.05, 8.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.20),
            "cat_smooth": trial.suggest_float("cat_smooth", 5.0, 60.0),
            "cat_l2": trial.suggest_float("cat_l2", 1.0, 25.0),
        }
        model = make_lgb(cfg, params, trial.number, cfg.tuning_estimators)
        model.fit(
            Xtr, y[tr], sample_weight=weights,
            eval_set=[(Xva, y[va])], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            categorical_feature=categorical,
        )
        score = model.predict_proba(Xva)[:, 1]
        ap = float(average_precision_score(y[va], score))
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or cfg.tuning_estimators))
        return ap

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=cfg.optuna_trials, gc_after_trial=True)
    trials = study.trials_dataframe()
    trials.to_csv(ART / "optuna_lightgbm_v4.csv", index=False)
    best = dict(study.best_params)
    write_json(ART / "mejores_parametros_lightgbm_v4.json", {
        "best_value": study.best_value,
        "best_params": best,
        "best_iteration": study.best_trial.user_attrs.get("best_iteration"),
        "ventana_tuning": {"train": [0, len(tr)], "validation": [int(n * 0.65), int(n * 0.75)]},
    })
    del Xtr, Xva
    gc.collect()
    return best, trials


def cat_model(cfg: ConfigV4, seed_offset: int = 0) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=120, depth=7, learning_rate=0.075, loss_function="Logloss",
        eval_metric="PRAUC:type=Classic", l2_leaf_reg=6.0, random_strength=0.6,
        random_seed=cfg.seed + seed_offset, thread_count=max(1, os.cpu_count() or 1),
        verbose=False, allow_writing_files=False,
    )


def xgb_model(cfg: ConfigV4, seed_offset: int = 0) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=220, learning_rate=0.060, max_depth=8, min_child_weight=4,
        subsample=0.86, colsample_bytree=0.82, reg_alpha=0.12, reg_lambda=2.2,
        max_bin=256, objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        early_stopping_rounds=cfg.early_stopping_rounds,
        random_state=cfg.seed + seed_offset, n_jobs=max(1, os.cpu_count() or 1),
    )


def walk_forward_models(df: pd.DataFrame, numeric: list[str], categorical: list[str], params: dict[str, Any], cfg: ConfigV4) -> pd.DataFrame:
    y = df["isFraud"].to_numpy(np.int8)
    rows: list[dict[str, Any]] = []
    for tr, va, fold in folds(len(df)):
        encoder = NativeEncoder(numeric, categorical, cfg).fit(df, tr)
        Xtr, Xva = encoder.transform(df, tr), encoder.transform(df, va)
        weight = recency_weights(df, tr, cfg)

        start = time.perf_counter()
        model_lgb = make_lgb(cfg, params, 100 + int(fold[-1]))
        model_lgb.fit(
            Xtr, y[tr], sample_weight=weight, eval_set=[(Xva, y[va])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            categorical_feature=categorical,
        )
        score = model_lgb.predict_proba(Xva)[:, 1]
        rows.append({"fold": fold, "modelo": "LightGBM_tuned", "auc_pr": average_precision_score(y[va], score), "roc_auc": roc_auc_score(y[va], score), "segundos": time.perf_counter() - start})
        del model_lgb, score

        codes_tr, codes_va = encoder.numeric_codes(Xtr), encoder.numeric_codes(Xva)
        start = time.perf_counter()
        model_cat = cat_model(cfg, 200 + int(fold[-1]))
        model_cat.fit(codes_tr, y[tr], cat_features=categorical, sample_weight=weight, eval_set=(codes_va, y[va]), early_stopping_rounds=cfg.early_stopping_rounds)
        score = model_cat.predict_proba(codes_va)[:, 1]
        rows.append({"fold": fold, "modelo": "CatBoost_ordered", "auc_pr": average_precision_score(y[va], score), "roc_auc": roc_auc_score(y[va], score), "segundos": time.perf_counter() - start})
        del model_cat, score

        start = time.perf_counter()
        model_xgb = xgb_model(cfg, 300 + int(fold[-1]))
        model_xgb.fit(codes_tr, y[tr], sample_weight=weight, eval_set=[(codes_va, y[va])], verbose=False)
        score = model_xgb.predict_proba(codes_va)[:, 1]
        rows.append({"fold": fold, "modelo": "XGBoost_hist", "auc_pr": average_precision_score(y[va], score), "roc_auc": roc_auc_score(y[va], score), "segundos": time.perf_counter() - start})
        del model_xgb, score, Xtr, Xva, codes_tr, codes_va
        gc.collect()
        print(f"  {fold} completado")
    result = pd.DataFrame(rows)
    result.to_csv(ART / "validacion_walk_forward_v4.csv", index=False)
    result.groupby("modelo").agg(auc_pr_media=("auc_pr", "mean"), auc_pr_sd=("auc_pr", "std"), roc_auc_media=("roc_auc", "mean"), segundos=("segundos", "sum")).sort_values("auc_pr_media", ascending=False).to_csv(ART / "resumen_walk_forward_v4.csv")
    return result


def _fit_lgb_final(model: lgb.LGBMClassifier, Xtr: pd.DataFrame, ytr: np.ndarray, weight: np.ndarray, Xes: pd.DataFrame, yes: np.ndarray, categorical: list[str], cfg: ConfigV4) -> lgb.LGBMClassifier:
    return model.fit(
        Xtr, ytr, sample_weight=weight, eval_set=[(Xes, yes)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
        categorical_feature=categorical,
    )


def final_models(df: pd.DataFrame, split: dict[str, np.ndarray], numeric: list[str], categorical: list[str], params: dict[str, Any], cfg: ConfigV4) -> dict[str, Any]:
    train, val, bench = split["train"], split["validation"], split["benchmark_historico"]
    y = df["isFraud"].to_numpy(np.int8)
    encoder = NativeEncoder(numeric, categorical, cfg).fit(df, train)
    Xtr, Xva, Xbe = encoder.transform(df, train), encoder.transform(df, val), encoder.transform(df, bench)
    n_val = len(val)
    bounds = {
        "early_stopping": [0, int(n_val * 0.25)],
        "meta_train": [int(n_val * 0.25), int(n_val * 0.50)],
        "calibracion": [int(n_val * 0.50), int(n_val * 0.70)],
        "umbral": [int(n_val * 0.70), int(n_val * 0.85)],
        "evaluacion": [int(n_val * 0.85), n_val],
    }
    es = np.arange(*bounds["early_stopping"])
    weight = recency_weights(df, train, cfg)
    scores_val: dict[str, np.ndarray] = {}
    scores_bench: dict[str, np.ndarray] = {}
    fitted: dict[str, Any] = {}

    print("    LightGBM global...")
    global_lgb = _fit_lgb_final(make_lgb(cfg, params, 501), Xtr, y[train], weight, Xva.iloc[es], y[val][es], categorical, cfg)
    scores_val["lgb"] = global_lgb.predict_proba(Xva)[:, 1]
    scores_bench["lgb"] = global_lgb.predict_proba(Xbe)[:, 1]
    global_lgb.booster_.save_model(str(ART / "modelo_lightgbm_global_v4.txt"))
    fitted["lgb_best_iteration"] = int(global_lgb.best_iteration_ or cfg.final_estimators)

    print("    LightGBM con hard negatives causales...")
    probe_end = int(len(df) * 0.55)
    probe_idx = np.arange(probe_end)
    hard_idx = np.arange(probe_end, len(train))
    probe = make_lgb(cfg, params, 502, 650)
    probe.fit(Xtr.iloc[:probe_end], y[probe_idx], sample_weight=weight[:probe_end], categorical_feature=categorical)
    hard_score = probe.predict_proba(Xtr.iloc[probe_end:])[:, 1]
    negative = y[hard_idx] == 0
    cutoff = float(np.quantile(hard_score[negative], cfg.hard_negative_quantile))
    hard_weight = weight.copy()
    selected_hard = negative & (hard_score >= cutoff)
    hard_tail_weight = hard_weight[probe_end:].copy()
    hard_tail_weight[selected_hard] *= cfg.hard_negative_multiplier
    hard_weight[probe_end:] = hard_tail_weight
    hard_lgb = _fit_lgb_final(make_lgb(cfg, params, 503), Xtr, y[train], hard_weight, Xva.iloc[es], y[val][es], categorical, cfg)
    scores_val["hard_lgb"] = hard_lgb.predict_proba(Xva)[:, 1]
    scores_bench["hard_lgb"] = hard_lgb.predict_proba(Xbe)[:, 1]
    hard_lgb.booster_.save_model(str(ART / "modelo_lightgbm_hard_negative_v4.txt"))
    fitted["hard_negatives"] = int(selected_hard.sum())
    fitted["hard_negative_cutoff"] = cutoff
    del probe, hard_score, hard_lgb
    gc.collect()

    print("    Expertos ProductCD=W y resto...")
    routed_val = np.zeros(len(val), dtype=float)
    routed_bench = np.zeros(len(bench), dtype=float)
    product_train = df.loc[train, "ProductCD"].astype(str).to_numpy()
    product_val = df.loc[val, "ProductCD"].astype(str).to_numpy()
    product_bench = df.loc[bench, "ProductCD"].astype(str).to_numpy()
    expert_info = {}
    for j, (name, value) in enumerate((("W", True), ("NO_W", False))):
        tr_mask = (product_train == "W") if value else (product_train != "W")
        va_mask = (product_val == "W") if value else (product_val != "W")
        be_mask = (product_bench == "W") if value else (product_bench != "W")
        es_mask = va_mask.copy()
        es_mask[np.arange(len(val)) >= bounds["early_stopping"][1]] = False
        model = _fit_lgb_final(
            make_lgb(cfg, {**params, "num_leaves": min(int(params.get("num_leaves", 64)), 79)}, 510 + j),
            Xtr.loc[tr_mask], y[train][tr_mask], weight[tr_mask], Xva.loc[es_mask], y[val][es_mask], categorical, cfg,
        )
        routed_val[va_mask] = model.predict_proba(Xva.loc[va_mask])[:, 1]
        routed_bench[be_mask] = model.predict_proba(Xbe.loc[be_mask])[:, 1]
        model.booster_.save_model(str(ART / f"modelo_experto_{name.lower()}_v4.txt"))
        expert_info[name] = {"train": int(tr_mask.sum()), "fraudes": int(y[train][tr_mask].sum()), "best_iteration": int(model.best_iteration_ or cfg.final_estimators)}
        del model
    scores_val["segment_lgb"] = routed_val
    scores_bench["segment_lgb"] = routed_bench
    fitted["expertos"] = expert_info

    print("    CatBoost y XGBoost...")
    codes_tr, codes_va, codes_be = encoder.numeric_codes(Xtr), encoder.numeric_codes(Xva), encoder.numeric_codes(Xbe)
    cat = cat_model(cfg, 520)
    cat.fit(codes_tr, y[train], cat_features=categorical, sample_weight=weight, eval_set=(codes_va.iloc[es], y[val][es]), early_stopping_rounds=cfg.early_stopping_rounds)
    scores_val["catboost"] = cat.predict_proba(codes_va)[:, 1]
    scores_bench["catboost"] = cat.predict_proba(codes_be)[:, 1]
    cat.save_model(str(ART / "modelo_catboost_v4.cbm"))
    fitted["catboost_best_iteration"] = int(cat.get_best_iteration())
    del cat
    gc.collect()

    xgb_final = xgb_model(cfg, 530)
    xgb_final.fit(codes_tr, y[train], sample_weight=weight, eval_set=[(codes_va.iloc[es], y[val][es])], verbose=False)
    scores_val["xgboost"] = xgb_final.predict_proba(codes_va)[:, 1]
    scores_bench["xgboost"] = xgb_final.predict_proba(codes_be)[:, 1]
    xgb_final.save_model(ART / "modelo_xgboost_v4.json")
    fitted["xgboost_best_iteration"] = int(xgb_final.best_iteration)
    del xgb_final, codes_tr, codes_va, codes_be
    gc.collect()

    # Stacking temporal: cada decisión usa un bloque posterior independiente.
    def meta_frame(indices: np.ndarray, base: dict[str, np.ndarray], absolute: np.ndarray) -> pd.DataFrame:
        result = pd.DataFrame({name: score[indices] for name, score in base.items()})
        result["log_amount"] = np.log1p(df.loc[absolute, "TransactionAmt"].fillna(0).to_numpy(float))
        result["missing_count"] = df.loc[absolute, "missing_count"].to_numpy(float)
        result["history"] = df.loc[absolute, "entity_prior_count"].to_numpy(float)
        result["product_w"] = (df.loc[absolute, "ProductCD"].astype(str).to_numpy() == "W").astype(float)
        result["device_missing"] = (df.loc[absolute, "DeviceType"].astype(str).to_numpy() == "MISSING").astype(float)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0)

    all_pos = np.arange(n_val)
    meta_positions = np.arange(*bounds["meta_train"])
    calibration_positions = np.arange(*bounds["calibracion"])
    threshold_positions = np.arange(*bounds["umbral"])
    evaluation_positions = np.arange(*bounds["evaluacion"])
    Zval = meta_frame(all_pos, scores_val, val)
    Zbench = meta_frame(np.arange(len(bench)), scores_bench, bench)
    meta = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=1500, class_weight=None, random_state=cfg.seed)),
    ])
    meta.fit(Zval.iloc[meta_positions], y[val][meta_positions])
    stack_raw_val = meta.predict_proba(Zval)[:, 1]
    stack_raw_bench = meta.predict_proba(Zbench)[:, 1]

    eps = 1e-6
    logit = lambda score: np.log(np.clip(score, eps, 1 - eps) / (1 - np.clip(score, eps, 1 - eps))).reshape(-1, 1)
    calibrator = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    calibrator.fit(logit(stack_raw_val[calibration_positions]), y[val][calibration_positions])
    stack_val = calibrator.predict_proba(logit(stack_raw_val))[:, 1]
    stack_bench = calibrator.predict_proba(logit(stack_raw_bench))[:, 1]
    balanced, economic, curve = choose_threshold(y[val][threshold_positions], stack_val[threshold_positions], cfg)
    curve.to_csv(ART / "curva_umbral_v4.csv", index=False)
    joblib.dump(encoder.serializable(), ART / "codificador_v4.joblib")
    joblib.dump(meta, ART / "stacking_meta_v4.joblib")
    joblib.dump(calibrator, ART / "calibrador_v4.joblib")

    pd.DataFrame({
        "indice": val, "TransactionID": df.loc[val, "TransactionID"].to_numpy(), "y": y[val],
        **{f"score_{name}": score for name, score in scores_val.items()},
        "score_stack_raw": stack_raw_val, "score_stack": stack_val,
    }).to_csv(ART / "predicciones_validacion_v4.csv", index=False)
    pd.DataFrame({
        "indice": bench, "TransactionID": df.loc[bench, "TransactionID"].to_numpy(), "y": y[bench],
        **{f"score_{name}": score for name, score in scores_bench.items()},
        "score_stack_raw": stack_raw_bench, "score_stack": stack_bench,
    }).to_csv(ART / "predicciones_benchmark_v4.csv", index=False)

    candidate_eval = []
    for name, score in {**scores_val, "stack": stack_val}.items():
        candidate_eval.append({
            "modelo": name,
            "auc_pr_evaluacion": float(average_precision_score(y[val][evaluation_positions], score[evaluation_positions])),
            "roc_auc_evaluacion": float(roc_auc_score(y[val][evaluation_positions], score[evaluation_positions])),
            "auc_pr_benchmark": float(average_precision_score(y[bench], ({**scores_bench, "stack": stack_bench}[name]))),
        })
    candidates = pd.DataFrame(candidate_eval).sort_values("auc_pr_evaluacion", ascending=False)
    candidates.to_csv(ART / "comparacion_candidatos_v4.csv", index=False)
    result = {
        "bloques_validacion": bounds,
        "ajuste": fitted,
        "columnas_meta": Zval.columns.tolist(),
        "coeficientes_meta": dict(zip(Zval.columns, meta.named_steps["model"].coef_[0].tolist())),
        "calibracion": {
            "brier_raw": brier_score_loss(y[val][calibration_positions], stack_raw_val[calibration_positions]),
            "brier_calibrado": brier_score_loss(y[val][calibration_positions], stack_val[calibration_positions]),
            "ece_raw": ece(y[val][calibration_positions], stack_raw_val[calibration_positions]),
            "ece_calibrado": ece(y[val][calibration_positions], stack_val[calibration_positions]),
        },
        "threshold_balanceado": balanced,
        "threshold_economico": economic,
        "evaluacion_balanceada": metrics(y[val][evaluation_positions], stack_val[evaluation_positions], balanced, cfg),
        "evaluacion_economica": metrics(y[val][evaluation_positions], stack_val[evaluation_positions], economic, cfg),
        "benchmark_balanceado": metrics(y[bench], stack_bench, balanced, cfg),
        "benchmark_economico": metrics(y[bench], stack_bench, economic, cfg),
        "candidatos": candidates.to_dict("records"),
    }
    del Xtr, Xva, Xbe, Zval, Zbench, global_lgb
    gc.collect()
    return {"resultado": result, "val_score": stack_val, "bench_score": stack_bench, "evaluation_positions": evaluation_positions}


def paired_block_delta(y: np.ndarray, new: np.ndarray, old: np.ndarray, cfg: ConfigV4, blocks: int = 20) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    pieces = np.array_split(np.arange(len(y)), blocks)
    estimates = []
    for _ in range(cfg.bootstrap_repetitions):
        idx = np.concatenate([pieces[i] for i in rng.integers(0, blocks, size=blocks)])
        if 0 < y[idx].sum() < len(idx):
            estimates.append(average_precision_score(y[idx], new[idx]) - average_precision_score(y[idx], old[idx]))
    delta = average_precision_score(y, new) - average_precision_score(y, old)
    return {"delta_auc_pr": delta, "li95": np.quantile(estimates, 0.025), "ls95": np.quantile(estimates, 0.975), "replicas": len(estimates), "bloques": blocks}


def top_k(y: np.ndarray, score: np.ndarray) -> list[dict[str, Any]]:
    order = np.argsort(-score)
    rows = []
    for rate in (0.001, 0.005, 0.01, 0.02, 0.05):
        k = max(1, int(len(y) * rate))
        picked = y[order[:k]]
        rows.append({"tasa_revision": rate, "k": k, "precision_at_k": picked.mean(), "recall_at_k": picked.sum() / max(1, y.sum())})
    return rows


def segment_metrics(df: pd.DataFrame, idx: np.ndarray, y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV4) -> pd.DataFrame:
    frame = pd.DataFrame({
        "ProductCD": df.loc[idx, "ProductCD"].astype(str).to_numpy(),
        "DeviceType": df.loc[idx, "DeviceType"].astype(str).to_numpy(),
        "history": pd.cut(df.loc[idx, "entity_prior_count"], [-1, 0, 2, 7, 15, np.inf], labels=["0", "1-2", "3-7", "8-15", "16+"]).astype(str).to_numpy(),
        "amount": df.loc[idx, "TransactionAmt"].to_numpy(), "y": y, "score": score,
    })
    frame["segmento_monto"] = pd.qcut(frame["amount"], 4, duplicates="drop").astype(str)
    rows = []
    for dimension in ("ProductCD", "DeviceType", "history", "segmento_monto"):
        for value, group in frame.groupby(dimension, observed=True):
            if len(group) >= 100 and group["y"].sum() > 0:
                rows.append({"dimension": dimension, "segmento": value, "n": len(group), "prevalencia": group["y"].mean(), **metrics(group["y"].to_numpy(np.int8), group["score"].to_numpy(float), threshold, cfg)})
    return pd.DataFrame(rows)


def adversarial_validation(df: pd.DataFrame, split: dict[str, np.ndarray], numeric: list[str], cfg: ConfigV4) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    train = split["train"]
    val = split["validation"]
    a = np.sort(rng.choice(train, min(60_000, len(train)), replace=False))
    b = np.sort(rng.choice(val, min(60_000, len(val)), replace=False))
    idx = np.concatenate([a, b])
    target = np.concatenate([np.zeros(len(a), dtype=np.int8), np.ones(len(b), dtype=np.int8)])
    X = df.loc[idx, numeric[: min(180, len(numeric))]].replace([np.inf, -np.inf], np.nan).astype("float32")
    cut = int(len(idx) * 0.80)
    order = rng.permutation(len(idx))
    tr, te = order[:cut], order[cut:]
    model = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.04, num_leaves=48, min_child_samples=80, verbosity=-1, n_jobs=max(1, os.cpu_count() or 1), random_state=cfg.seed)
    model.fit(X.iloc[tr], target[tr])
    score = model.predict_proba(X.iloc[te])[:, 1]
    importance = pd.DataFrame({"variable": X.columns, "ganancia": model.booster_.feature_importance(importance_type="gain")}).sort_values("ganancia", ascending=False)
    importance.head(30).to_csv(ART / "adversarial_importance_v4.csv", index=False)
    return {"roc_auc": roc_auc_score(target[te], score), "n": len(idx), "top_variables": importance.head(10).to_dict("records"), "interpretacion": "0.5 indica distribuciones indistinguibles; valores altos evidencian deriva."}


def compare_and_promote(df: pd.DataFrame, split: dict[str, np.ndarray], walk: pd.DataFrame, final: dict[str, Any], cfg: ConfigV4) -> dict[str, Any]:
    v3 = json.loads((V3_ART / "resultados_v3.json").read_text(encoding="utf-8"))
    v3_walk = float(next(x["mean"] for x in v3["validacion_walk_forward"]["resumen"] if x["modelo"] == "LGB_native_recency"))
    v4_walk = float(walk.loc[walk["modelo"] == "LightGBM_tuned"].groupby("modelo")["auc_pr"].mean().iloc[0])
    val_pred = pd.read_csv(V3_ART / "predicciones_validacion_v3.csv")
    bench_pred = pd.read_csv(V3_ART / "predicciones_benchmark_v3.csv")
    val = split["validation"]
    bench = split["benchmark_historico"]
    yv = df.loc[val, "isFraud"].to_numpy(np.int8)
    yb = df.loc[bench, "isFraud"].to_numpy(np.int8)
    ep = final["evaluation_positions"]
    threshold_v3 = float(v3["modelo_v3"]["threshold_recomendado_balanceado"])
    old_eval = metrics(yv[ep], val_pred["score_calibrado"].to_numpy(float)[ep], threshold_v3, cfg)
    new_eval = final["resultado"]["evaluacion_balanceada"]
    cost_reduction = 1 - new_eval["costo_q"] / old_eval["costo_q"]
    val_pair = paired_block_delta(yv[ep], final["val_score"][ep], val_pred["score_calibrado"].to_numpy(float)[ep], cfg, 12)
    bench_pair = paired_block_delta(yb, final["bench_score"], bench_pred["score_calibrado"].to_numpy(float), cfg, 24)
    criteria = {
        "lightgbm_walk_delta_min_0_015": v4_walk - v3_walk >= cfg.promotion_auc_pr_delta,
        "stack_eval_delta_min_0_015": val_pair["delta_auc_pr"] >= cfg.promotion_auc_pr_delta,
        "stack_eval_ci_no_negativo": val_pair["li95"] > 0,
        "costo_eval_reduccion_min_3pct": cost_reduction >= cfg.promotion_cost_reduction,
        "recall_eval_no_cae_mas_1pp": new_eval["recall"] >= old_eval["recall"] - cfg.promotion_recall_tolerance,
    }
    return {
        "v3_auc_pr_walk": v3_walk, "v4_auc_pr_walk": v4_walk, "delta_auc_pr_walk": v4_walk - v3_walk,
        "v3_evaluacion": old_eval, "v4_evaluacion": new_eval, "reduccion_costo_evaluacion": cost_reduction,
        "comparacion_pareada_evaluacion": val_pair,
        "comparacion_pareada_benchmark_historico": bench_pair,
        "criterios": criteria, "promover_v4": bool(all(criteria.values())),
        "regla": "Todos los criterios deben cumplirse; el benchmark histórico no decide promoción.",
    }


def plots(walk: pd.DataFrame, final: dict[str, Any], promotion: dict[str, Any], df: pd.DataFrame, split: dict[str, np.ndarray]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    summary = walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.barh(summary.index, summary["mean"], xerr=summary["std"], color="#2a9d8f"); ax.axvline(promotion["v3_auc_pr_walk"], color="#e76f51", ls="--", label="V3"); ax.set(xlabel="AUC-PR media", title="V4 · validación walk-forward"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "01_walk_forward_v4.png", dpi=180); plt.close(fig)

    candidates = pd.DataFrame(final["resultado"]["candidatos"]).sort_values("auc_pr_evaluacion")
    fig, ax = plt.subplots(figsize=(9, 6)); ax.barh(candidates["modelo"], candidates["auc_pr_evaluacion"], color="#376f9e"); ax.set(xlabel="AUC-PR evaluación temporal", title="V4 · candidatos y stacking"); fig.tight_layout(); fig.savefig(FIG / "02_candidatos_v4.png", dpi=180); plt.close(fig)

    bench = split["benchmark_historico"]
    yb = df.loc[bench, "isFraud"].to_numpy(np.int8)
    old = pd.read_csv(V3_ART / "predicciones_benchmark_v3.csv")["score_calibrado"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, score, color in (("V4", final["bench_score"], "#2a9d8f"), ("V3", old, "#184e77")):
        p, r, _ = precision_recall_curve(yb, score); ax.plot(r, p, lw=2.2, color=color, label=f"{label} · AP={average_precision_score(yb, score):.3f}")
    ax.axhline(yb.mean(), color="#6b7280", ls="--", label="Prevalencia"); ax.set(xlabel="Recall", ylabel="Precisión", title="Benchmark histórico reutilizado"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "03_curvas_pr_v3_v4.png", dpi=180); plt.close(fig)

    curve = pd.read_csv(ART / "curva_umbral_v4.csv").sort_values("recall")
    fig, ax = plt.subplots(figsize=(9, 5.5)); ax.plot(curve["recall"], curve["costo_q"], color="#e76f51", lw=2); ax.set(xlabel="Recall", ylabel="Costo Q", title="V4 · frontera económica"); fig.tight_layout(); fig.savefig(FIG / "04_costo_recall_v4.png", dpi=180); plt.close(fig)

    pred = pd.read_csv(ART / "predicciones_benchmark_v4.csv")
    prob_true, prob_pred = calibration_curve(pred["y"], pred["score_stack"], n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6.5, 6)); ax.plot(prob_pred, prob_true, marker="o", color="#2a9d8f", label="V4"); ax.plot([0, 1], [0, 1], ls="--", color="#6b7280"); ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Calibración V4"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "05_calibracion_v4.png", dpi=180); plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    cfg = ConfigV4()
    ensure_dirs(); set_seed(cfg.seed); started = time.time()
    print("[1/9] Cargando IEEE-CIS...")
    df = load_all()
    split = temporal_boundaries(df, cfg)
    print("[2/9] Construyendo características causales V4...")
    df, identities = add_v4_features(df)
    print(f"Columnas integradas después de ingeniería: {len(df.columns)}")
    print("[3/9] Seleccionando variables y auditando redundancia...")
    numeric, categorical, selection = select_features(df, split["audit_train"], cfg)
    print(f"V4: {len(numeric)} numéricas + {len(categorical)} categóricas")
    print("[4/9] Optuna para LightGBM...")
    cached_best = ART / "mejores_parametros_lightgbm_v4.json"
    cached_trials = ART / "optuna_lightgbm_v4.csv"
    if cached_best.exists() and cached_trials.exists():
        cached = json.loads(cached_best.read_text(encoding="utf-8"))
        best_params = cached["best_params"]
        trials = pd.read_csv(cached_trials)
        print("Reutilizando búsqueda Optuna V4 ya completada.")
    else:
        best_params, trials = tune_lightgbm(df, numeric, categorical, cfg)
    print(f"Mejor AUC-PR tuning: {trials['value'].max():.6f}")
    print("[5/9] Walk-forward LightGBM/CatBoost/XGBoost...")
    walk = walk_forward_models(df, numeric, categorical, best_params, cfg)
    print(walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).sort_values("mean", ascending=False))
    print("[6/9] Modelos finales, expertos, hard negatives y stacking...")
    final = final_models(df, split, numeric, categorical, best_params, cfg)
    print("[7/9] Deriva, segmentos, top-K y promoción...")
    drift = adversarial_validation(df, split, numeric, cfg)
    promotion = compare_and_promote(df, split, walk, final, cfg)
    bench = split["benchmark_historico"]
    yb = df.loc[bench, "isFraud"].to_numpy(np.int8)
    topk = top_k(yb, final["bench_score"])
    pd.DataFrame(topk).to_csv(ART / "metricas_top_k_v4.csv", index=False)
    segments = segment_metrics(df, bench, yb, final["bench_score"], final["resultado"]["threshold_balanceado"], cfg)
    segments.to_csv(ART / "metricas_segmentos_v4.csv", index=False)
    print("[8/9] Figuras...")
    plots(walk, final, promotion, df, split)
    result = {
        "version": "4.0", "estado_benchmark": "historico_reutilizado_no_ciego",
        "configuracion": asdict(cfg),
        "entorno": {"python": platform.python_version(), "plataforma": platform.platform(), "lightgbm": lgb.__version__, "catboost": __import__("catboost").__version__, "xgboost": xgb.__version__, "optuna": optuna.__version__, "cpu_logicos": os.cpu_count()},
        "datos": {"filas": len(df), "columnas_integradas": len(df.columns), "prevalencia": df["isFraud"].mean(), "particiones": {k: {"n": len(v), "fraude": df.loc[v, "isFraud"].mean(), "dt_min": df.loc[v, "TransactionDT"].min(), "dt_max": df.loc[v, "TransactionDT"].max()} for k, v in split.items()}},
        "identidad_y_features": identities, "seleccion_variables": selection,
        "optuna": {"trials": len(trials), "mejores_parametros": best_params},
        "validacion_walk_forward": {"detalle": walk.to_dict("records"), "resumen": walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).reset_index().to_dict("records")},
        "modelo_v4": final["resultado"], "deriva_adversarial": drift,
        "metricas_top_k": topk, "promocion": promotion,
        "duracion_segundos": time.time() - started,
        "recomendacion": "Promover V4" if promotion["promover_v4"] else "Conservar V3; V4 permanece experimental",
        "limitaciones": ["Benchmark reutilizado y no ciego", "Identidad aproximada", "Costos académicos", "Sin cohorte externa nueva", "No se promete 0.90 simultáneo en precisión y recall"],
    }
    write_json(ART / "resultados_v4.json", result)
    print("[9/9] Resultado:", result["recomendacion"])
    print(json.dumps(ready(promotion), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
