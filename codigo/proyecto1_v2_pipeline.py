"""Pipeline V2 auditable para el Proyecto 1 de monitoreo transaccional.

La V2 no reemplaza la evidencia V1. Amplia variables, aplica selección únicamente
con el pasado, compara correlación/PCA, CatBoost/LightGBM, calibración, costo y
validación temporal. El último 15 % se etiqueta siempre como benchmark histórico
reutilizado, nunca como prueba ciega.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
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


@dataclass(frozen=True)
class ConfigV2:
    seed: int = 2026
    development_fraction: float = 0.85
    train_fraction: float = 0.70
    benchmark_fraction: float = 0.15
    audit_train_fraction: float = 0.55
    max_selected_numeric: int = 110
    max_selected_categorical: int = 18
    corr_prune_threshold: float = 0.985
    missing_limit: float = 0.985
    audit_sample: int = 80_000
    mi_sample: int = 60_000
    correlation_sample: int = 50_000
    cost_fn_q: float = 4200.0
    cost_fp_q: float = 180.0
    pca_variance: float = 0.95
    pca_max_components: int = 64
    lgb_iterations: int = 500
    cat_iterations: int = 240
    early_stopping_rounds: int = 35
    bootstrap_repetitions: int = 300


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "raw"
PROCESSED = ROOT / "datos" / "processed" / "v2"
ART = ROOT / "artefactos" / "v2"
FIG = ROOT / "evidencia" / "figuras" / "v2"
CFG_DIR = ROOT / "configuracion" / "v2"

ID_COLUMNS = {
    "TransactionID",
    "TransactionDT",
    "isFraud",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
}
BASE_CATEGORICAL = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_31",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]
RAW_PROXY_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
MANDATORY_NUMERIC = [
    "TransactionAmt",
    "log_amount",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "missing_count",
    "entity_prior_count",
    "entity_prior_amt_mean",
    "entity_prior_amt_std",
    "amount_to_prior_mean",
    "hours_since_prior",
    "entity_count_1h",
    "entity_count_6h",
    "entity_count_24h",
    "entity_count_72h",
]


def ensure_dirs() -> None:
    for p in (PROCESSED, ART, FIG, CFG_DIR):
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ready(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [ready(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(ready(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_raw(name: str) -> Path:
    direct = RAW / name
    cached = RAW / "kagglehub_cache" / "competitions" / "ieee-fraud-detection" / name
    for candidate in (direct, cached):
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            return candidate
    raise FileNotFoundError(name)


def sha256(path: Path, block: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def load_all() -> pd.DataFrame:
    tx_path = resolve_raw("train_transaction.csv")
    id_path = resolve_raw("train_identity.csv")
    print("[1/9] Cargando las 394 variables transaccionales y 41 de identidad...")
    tx = pd.read_csv(tx_path, low_memory=False)
    identity = pd.read_csv(id_path, low_memory=False)
    df = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    del tx, identity
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(
        drop=True
    )
    return df


def stringify(frame: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in frame:
            frame[c] = frame[c].fillna("MISSING").astype(str)
    return frame


def proxy_key(df: pd.DataFrame, columns: list[str], name: str) -> pd.Series:
    parts = []
    for c in columns:
        if c in df:
            parts.append(df[c].fillna(-999999).astype(str))
        else:
            parts.append(pd.Series("MISSING", index=df.index))
    return pd.concat(parts, axis=1).agg("|".join, axis=1).rename(name)


def add_causal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(
        "[2/9] Construyendo variables temporales y agregados estrictamente causales..."
    )
    out = df
    out["entity_key"] = proxy_key(
        out, ["card1", "card2", "card3", "card5", "addr1"], "entity_key"
    )
    amount = out["TransactionAmt"].fillna(0).astype("float32")
    seconds = out["TransactionDT"].astype("float64")
    out["log_amount"] = np.log1p(amount.clip(lower=0)).astype("float32")
    hour = (seconds / 3600.0) % 24
    weekday = (seconds / 86400.0) % 7
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype("float32")
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype("float32")
    raw_feature_cols = [
        c
        for c in out.columns
        if c not in ("isFraud", "TransactionID", "TransactionDT", "entity_key")
    ]
    out["missing_count"] = out[raw_feature_cols].isna().sum(axis=1).astype("float32")

    group = out.groupby("entity_key", sort=False, observed=True)
    out["entity_prior_count"] = group.cumcount().astype("float32")
    prior_sum = group["TransactionAmt"].cumsum() - amount
    denom = out["entity_prior_count"].replace(0, np.nan)
    prior_mean = prior_sum / denom
    prior_sq_sum = amount.pow(2).groupby(
        out["entity_key"], sort=False
    ).cumsum() - amount.pow(2)
    prior_var = (prior_sq_sum / denom - prior_mean.pow(2)).clip(lower=0)
    out["entity_prior_amt_mean"] = prior_mean.fillna(0).astype("float32")
    out["entity_prior_amt_std"] = np.sqrt(prior_var).fillna(0).astype("float32")
    out["amount_to_prior_mean"] = (
        (amount / prior_mean.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1)
        .clip(0, 100)
        .astype("float32")
    )
    prior_time = group["TransactionDT"].shift(1)
    out["hours_since_prior"] = (
        ((seconds - prior_time) / 3600).fillna(0).clip(0, 24 * 365).astype("float32")
    )

    windows = {"1h": 3600, "6h": 21600, "24h": 86400, "72h": 259200}
    histories: dict[str, deque[float]] = defaultdict(deque)
    counts = {label: np.zeros(len(out), dtype=np.float32) for label in windows}
    for i, (key, now) in enumerate(
        zip(out["entity_key"].to_numpy(), seconds.to_numpy())
    ):
        history = histories[key]
        while history and history[0] < now - windows["72h"]:
            history.popleft()
        snapshot = list(history)
        for label, width in windows.items():
            cutoff = now - width
            counts[label][i] = sum(t >= cutoff for t in snapshot)
        history.append(now)
    for label, values in counts.items():
        out[f"entity_count_{label}"] = values

    coverage = identity_coverage(out)
    stringify(out, BASE_CATEGORICAL)
    return out, coverage


def identity_coverage(df: pd.DataFrame) -> dict[str, Any]:
    definitions = {
        "tarjeta_direccion": ["card1", "card2", "card3", "card5", "addr1"],
        "tarjeta_direccion_correo": ["card1", "card2", "addr1", "P_emaildomain"],
        "tarjeta_dispositivo_producto": [
            "card1",
            "DeviceInfo",
            "DeviceType",
            "ProductCD",
        ],
        "tarjeta_dispositivo": ["card1", "DeviceInfo", "DeviceType"],
    }
    result: dict[str, Any] = {}
    for name, cols in definitions.items():
        key = proxy_key(df, cols, name)
        sizes = key.value_counts(dropna=False)
        mapped = key.map(sizes)
        result[name] = {
            "columnas": cols,
            "entidades": int(sizes.size),
            "mediana_transacciones": float(sizes.median()),
            "p90_transacciones": float(sizes.quantile(0.90)),
            "porcentaje_con_3": float((mapped >= 3).mean() * 100),
            "porcentaje_con_8": float((mapped >= 8).mean() * 100),
            "porcentaje_con_16": float((mapped >= 16).mean() * 100),
            "porcentaje_con_32": float((mapped >= 32).mean() * 100),
        }
    return result


def temporal_boundaries(df: pd.DataFrame, cfg: ConfigV2) -> dict[str, np.ndarray]:
    n = len(df)

    def cut(fraction: float) -> int:
        return int(math.floor(n * fraction))

    return {
        "audit_train": np.arange(0, cut(cfg.audit_train_fraction)),
        "train": np.arange(0, cut(cfg.train_fraction)),
        "validation": np.arange(cut(cfg.train_fraction), cut(cfg.development_fraction)),
        "benchmark_historico": np.arange(cut(cfg.development_fraction), n),
    }


def feature_audit(
    df: pd.DataFrame, audit_idx: np.ndarray, cfg: ConfigV2
) -> tuple[list[str], list[str], dict[str, Any]]:
    print(
        "[3/9] Auditando ausencia, cardinalidad, correlación, información mutua y redundancia..."
    )
    y = df.loc[audit_idx, "isFraud"].to_numpy(dtype=np.int8)
    candidates = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in ID_COLUMNS
    ]
    rng = np.random.default_rng(cfg.seed)
    sample_idx = np.sort(
        rng.choice(audit_idx, size=min(cfg.audit_sample, len(audit_idx)), replace=False)
    )
    audit_rows: list[dict[str, Any]] = []
    for c in candidates:
        s = df.loc[audit_idx, c]
        missing = float(s.isna().mean())
        unique = int(s.nunique(dropna=True))
        variance = float(s.var(skipna=True)) if unique > 1 else 0.0
        sampled = df.loc[sample_idx, c]
        pearson = (
            float(sampled.corr(df.loc[sample_idx, "isFraud"], method="pearson"))
            if unique > 1
            else 0.0
        )
        spearman = (
            float(sampled.corr(df.loc[sample_idx, "isFraud"], method="spearman"))
            if unique > 1
            else 0.0
        )
        reason = "candidata"
        if missing > cfg.missing_limit:
            reason = "excluida_alta_ausencia"
        elif unique <= 1 or not np.isfinite(variance) or variance == 0:
            reason = "excluida_constante"
        audit_rows.append(
            {
                "variable": c,
                "tipo": str(s.dtype),
                "ausencia": missing,
                "unicos": unique,
                "varianza": variance,
                "pearson": 0.0 if not np.isfinite(pearson) else pearson,
                "spearman": 0.0 if not np.isfinite(spearman) else spearman,
                "decision": reason,
            }
        )
    audit = pd.DataFrame(audit_rows)
    eligible = audit.loc[audit["decision"].eq("candidata")].copy()
    eligible["relevancia_corr"] = eligible[["pearson", "spearman"]].abs().max(axis=1)
    mi_candidates = eligible.nlargest(min(160, len(eligible)), "relevancia_corr")[
        "variable"
    ].tolist()
    mi_idx = np.sort(
        rng.choice(audit_idx, size=min(cfg.mi_sample, len(audit_idx)), replace=False)
    )
    X_mi = df.loc[mi_idx, mi_candidates].replace([np.inf, -np.inf], np.nan)
    X_mi = X_mi.fillna(X_mi.median()).astype("float32")
    mi = mutual_info_classif(
        X_mi, df.loc[mi_idx, "isFraud"], random_state=cfg.seed, n_neighbors=3
    )
    mi_map = dict(zip(mi_candidates, mi))
    audit["informacion_mutua"] = audit["variable"].map(mi_map).fillna(0.0)
    audit["puntaje_relevancia"] = (
        audit[["pearson", "spearman"]].abs().max(axis=1)
        + 0.25 * audit["informacion_mutua"]
    )

    ranked = audit.loc[audit["decision"].eq("candidata")].sort_values(
        "puntaje_relevancia", ascending=False
    )
    pool = list(
        dict.fromkeys(
            [c for c in MANDATORY_NUMERIC if c in ranked["variable"].values]
            + ranked["variable"].head(150).tolist()
        )
    )
    corr_idx = np.sort(
        rng.choice(
            audit_idx, size=min(cfg.correlation_sample, len(audit_idx)), replace=False
        )
    )
    corr = df.loc[corr_idx, pool].corr(method="spearman").abs()
    selected: list[str] = []
    removed: list[dict[str, Any]] = []
    for c in pool:
        conflict = next(
            (
                kept
                for kept in selected
                if corr.loc[c, kept] >= cfg.corr_prune_threshold
            ),
            None,
        )
        if conflict is None:
            selected.append(c)
        else:
            removed.append(
                {
                    "variable": c,
                    "retenida": conflict,
                    "rho_spearman_abs": float(corr.loc[c, conflict]),
                    "motivo": "redundancia",
                }
            )
            audit.loc[audit["variable"].eq(c), "decision"] = (
                f"excluida_redundante_con_{conflict}"
            )
        if len(selected) >= cfg.max_selected_numeric:
            break

    cat_rows = []
    for c in BASE_CATEGORICAL:
        if c not in df:
            continue
        train_s = df.loc[audit_idx, c].fillna("MISSING").astype(str)
        table = (
            pd.DataFrame({"x": train_s, "y": y})
            .groupby("x", observed=True)["y"]
            .agg(["mean", "size"])
        )
        global_rate = max(float(y.mean()), 1e-8)
        weighted_lift = float(
            (((table["mean"] - global_rate).abs() * table["size"]).sum() / len(y))
            / global_rate
        )
        cat_rows.append(
            {
                "variable": c,
                "cardinalidad": int(train_s.nunique()),
                "ausencia_original": float((train_s == "MISSING").mean()),
                "asociacion_lift_ponderado": weighted_lift,
            }
        )
    cat_audit = pd.DataFrame(cat_rows).sort_values(
        "asociacion_lift_ponderado", ascending=False
    )
    selected_cat = cat_audit.head(cfg.max_selected_categorical)["variable"].tolist()
    audit.to_csv(PROCESSED / "auditoria_variables.csv", index=False)
    pd.DataFrame(removed).to_csv(PROCESSED / "variables_redundantes.csv", index=False)
    cat_audit.to_csv(PROCESSED / "auditoria_categoricas.csv", index=False)
    corr.loc[selected, selected].to_csv(
        PROCESSED / "matriz_correlacion_seleccionadas.csv"
    )
    result = {
        "numericas_disponibles": len(candidates),
        "numericas_seleccionadas": selected,
        "categoricas_seleccionadas": selected_cat,
        "redundantes_eliminadas": removed,
        "umbral_redundancia": cfg.corr_prune_threshold,
        "nota_ids": "TransactionID y TransactionDT se usan solo para unión/orden; proxies de tarjeta/dirección no entran como magnitudes continuas.",
    }
    write_json(PROCESSED / "seleccion_variables.json", result)
    return selected, selected_cat, result


class FrameEncoder:
    def __init__(self, numeric: list[str], categorical: list[str]):
        self.numeric = numeric
        self.categorical = categorical
        self.medians: dict[str, float] = {}
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, idx: np.ndarray) -> "FrameEncoder":
        self.medians = df.loc[idx, self.numeric].median().fillna(0).to_dict()
        for c in self.categorical:
            counts = df.loc[idx, c].fillna("MISSING").astype(str).value_counts()
            self.maps[c] = {value: i + 1 for i, value in enumerate(counts.index[:500])}
        return self

    def transform(self, df: pd.DataFrame, idx: np.ndarray) -> pd.DataFrame:
        out = (
            df.loc[idx, self.numeric]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
            .astype("float32")
            .copy()
        )
        for c in self.categorical:
            out[c] = (
                df.loc[idx, c]
                .fillna("MISSING")
                .astype(str)
                .map(self.maps[c])
                .fillna(0)
                .astype("int32")
            )
        return out


def folds(n: int) -> list[tuple[np.ndarray, np.ndarray, str]]:
    specs = [(0.55, 0.55, 0.65), (0.65, 0.65, 0.75), (0.75, 0.75, 0.85)]
    return [
        (np.arange(0, int(n * tr)), np.arange(int(n * va), int(n * vb)), f"F{i + 1}")
        for i, (tr, va, vb) in enumerate(specs)
    ]


def lgb_model(cfg: ConfigV2, seed_offset: int = 0) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=cfg.lgb_iterations,
        learning_rate=0.035,
        num_leaves=48,
        max_depth=-1,
        min_child_samples=90,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=1.2,
        class_weight=None,
        random_state=cfg.seed + seed_offset,
        n_jobs=max(1, os.cpu_count() or 1),
        verbosity=-1,
    )


def cat_model(cfg: ConfigV2, seed_offset: int = 0) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=cfg.cat_iterations,
        depth=7,
        learning_rate=0.065,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=cfg.seed + seed_offset,
        l2_leaf_reg=5,
        random_strength=0.4,
        border_count=64,
        thread_count=max(1, os.cpu_count() or 1),
        verbose=False,
        allow_writing_files=False,
    )


def metrics(
    y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV2
) -> dict[str, float]:
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
    y: np.ndarray, score: np.ndarray, cfg: ConfigV2
) -> tuple[float, pd.DataFrame]:
    candidates = np.unique(np.quantile(score, np.linspace(0.70, 0.9995, 500)))
    rows = [metrics(y, score, float(t), cfg) for t in candidates]
    table = pd.DataFrame(rows).sort_values(["costo_q", "threshold"])
    return float(table.iloc[0]["threshold"]), table


def expected_calibration_error(
    y: np.ndarray, score: np.ndarray, bins: int = 15
) -> float:
    edges = np.linspace(0, 1, bins + 1)
    result = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.any():
            result += mask.mean() * abs(
                float(y[mask].mean()) - float(score[mask].mean())
            )
    return float(result)


def evaluate_temporal_models(
    df: pd.DataFrame, numeric: list[str], categorical: list[str], cfg: ConfigV2
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print("[4/9] Comparando LightGBM, CatBoost y PCA en tres ventanas walk-forward...")
    y_all = df["isFraud"].to_numpy(dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for tr, va, fold_name in folds(len(df)):
        encoder = FrameEncoder(numeric, categorical).fit(df, tr)
        Xtr, Xva = encoder.transform(df, tr), encoder.transform(df, va)
        ytr, yva = y_all[tr], y_all[va]
        callbacks = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
        start = time.perf_counter()
        model_lgb = lgb_model(cfg, int(fold_name[-1])).fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=callbacks
        )
        score = model_lgb.predict_proba(Xva)[:, 1]
        rows.append(
            {
                "fold": fold_name,
                "modelo": "LightGBM_corr_pruned",
                "auc_pr": average_precision_score(yva, score),
                "roc_auc": roc_auc_score(yva, score),
                "segundos": time.perf_counter() - start,
                "variables": Xtr.shape[1],
            }
        )

        cat_cols = [c for c in categorical if c in df]
        Xtr_cat = df.loc[tr, numeric + cat_cols].copy()
        Xva_cat = df.loc[va, numeric + cat_cols].copy()
        for c in numeric:
            median = float(Xtr_cat[c].median()) if Xtr_cat[c].notna().any() else 0.0
            Xtr_cat[c] = Xtr_cat[c].replace([np.inf, -np.inf], np.nan).fillna(median)
            Xva_cat[c] = Xva_cat[c].replace([np.inf, -np.inf], np.nan).fillna(median)
        stringify(Xtr_cat, cat_cols)
        stringify(Xva_cat, cat_cols)
        start = time.perf_counter()
        model_cat = cat_model(cfg, int(fold_name[-1])).fit(
            Xtr_cat,
            ytr,
            cat_features=cat_cols,
            eval_set=(Xva_cat, yva),
            early_stopping_rounds=cfg.early_stopping_rounds,
        )
        score_cat = model_cat.predict_proba(Xva_cat)[:, 1]
        rows.append(
            {
                "fold": fold_name,
                "modelo": "CatBoost_nativo",
                "auc_pr": average_precision_score(yva, score_cat),
                "roc_auc": roc_auc_score(yva, score_cat),
                "segundos": time.perf_counter() - start,
                "variables": Xtr_cat.shape[1],
            }
        )

        eligible_pca = [
            c
            for c in numeric
            if c.startswith("V")
            or c.startswith("C")
            or c.startswith("D")
            or c.startswith("id_")
        ]
        eligible_pca = eligible_pca[:140]
        passthrough = [c for c in numeric if c not in eligible_pca]
        if len(eligible_pca) >= 5:
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=cfg.pca_variance,
                            svd_solver="full",
                            random_state=cfg.seed,
                        ),
                    ),
                ]
            )
            Ztr = pipe.fit_transform(df.loc[tr, eligible_pca])
            Zva = pipe.transform(df.loc[va, eligible_pca])
            if Ztr.shape[1] > cfg.pca_max_components:
                Ztr, Zva = (
                    Ztr[:, : cfg.pca_max_components],
                    Zva[:, : cfg.pca_max_components],
                )
            pass_encoder = FrameEncoder(passthrough, categorical).fit(df, tr)
            Ptr, Pva = pass_encoder.transform(df, tr), pass_encoder.transform(df, va)
            Xp_tr = np.column_stack([Ptr.to_numpy(), Ztr]).astype("float32")
            Xp_va = np.column_stack([Pva.to_numpy(), Zva]).astype("float32")
            start = time.perf_counter()
            model_pca = lgb_model(cfg, 30 + int(fold_name[-1])).fit(
                Xp_tr, ytr, eval_set=[(Xp_va, yva)], callbacks=callbacks
            )
            score_pca = model_pca.predict_proba(Xp_va)[:, 1]
            rows.append(
                {
                    "fold": fold_name,
                    "modelo": "LightGBM_PCA95",
                    "auc_pr": average_precision_score(yva, score_pca),
                    "roc_auc": roc_auc_score(yva, score_pca),
                    "segundos": time.perf_counter() - start,
                    "variables": Xp_tr.shape[1],
                    "componentes_pca": Ztr.shape[1],
                }
            )
        del Xtr, Xva, Xtr_cat, Xva_cat, model_lgb, model_cat
        gc.collect()
    table = pd.DataFrame(rows)
    summary = (
        table.groupby("modelo")
        .agg(
            auc_pr_media=("auc_pr", "mean"),
            auc_pr_sd=("auc_pr", "std"),
            roc_auc_media=("roc_auc", "mean"),
            segundos_media=("segundos", "mean"),
            variables=("variables", "median"),
        )
        .sort_values("auc_pr_media", ascending=False)
    )
    table.to_csv(ART / "validacion_walk_forward.csv", index=False)
    summary.to_csv(ART / "resumen_walk_forward.csv")
    return table, {
        "resumen": summary.reset_index().to_dict("records"),
        "ganador": str(summary.index[0]),
    }


def fit_final_models(
    df: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    winner: str,
    split: dict[str, np.ndarray],
    cfg: ConfigV2,
) -> dict[str, Any]:
    print(
        "[5/9] Entrenando candidatos finales y ablación de tamaño sobre el 70 % histórico..."
    )
    train, val, bench = (
        split["train"],
        split["validation"],
        split["benchmark_historico"],
    )
    y = df["isFraud"].to_numpy(dtype=np.int8)
    encoder = FrameEncoder(numeric, categorical).fit(df, train)
    Xtr, Xva, Xbe = (
        encoder.transform(df, train),
        encoder.transform(df, val),
        encoder.transform(df, bench),
    )
    callbacks = [lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)]
    sample_rows = []
    for size in [180_000, 300_000, len(train)]:
        chosen = train[-min(size, len(train)) :]
        local_encoder = FrameEncoder(numeric, categorical).fit(df, chosen)
        Xt, Xv = local_encoder.transform(df, chosen), local_encoder.transform(df, val)
        m = lgb_model(cfg, size % 97).fit(
            Xt, y[chosen], eval_set=[(Xv, y[val])], callbacks=callbacks
        )
        s = m.predict_proba(Xv)[:, 1]
        sample_rows.append(
            {
                "n_entrenamiento": len(chosen),
                "auc_pr_validacion": average_precision_score(y[val], s),
                "roc_auc_validacion": roc_auc_score(y[val], s),
                "mejor_iteracion": int(m.best_iteration_ or cfg.lgb_iterations),
            }
        )
        del Xt, Xv, m
    pd.DataFrame(sample_rows).to_csv(
        ART / "ablacion_tamano_entrenamiento.csv", index=False
    )

    model = lgb_model(cfg, 777).fit(
        Xtr, y[train], eval_set=[(Xva, y[val])], callbacks=callbacks
    )
    val_score = model.predict_proba(Xva)[:, 1]
    bench_score = model.predict_proba(Xbe)[:, 1]
    threshold, threshold_table = choose_threshold(y[val], val_score, cfg)
    threshold_table.to_csv(ART / "curva_umbral_lightgbm.csv", index=False)
    joblib.dump(model, ART / "modelo_tabular_lightgbm.joblib")
    joblib.dump(encoder, ART / "codificador_tabular.joblib")
    pd.DataFrame(
        {
            "indice": val,
            "TransactionID": df.loc[val, "TransactionID"],
            "y": y[val],
            "score_tabular": val_score,
        }
    ).to_csv(ART / "predicciones_validacion.csv", index=False)
    pd.DataFrame(
        {
            "indice": bench,
            "TransactionID": df.loc[bench, "TransactionID"],
            "y": y[bench],
            "score_tabular": bench_score,
        }
    ).to_csv(ART / "predicciones_benchmark_historico.csv", index=False)
    return {
        "model": model,
        "encoder": encoder,
        "val_score": val_score,
        "benchmark_score": bench_score,
        "threshold": threshold,
        "validacion": metrics(y[val], val_score, threshold, cfg),
        "benchmark_historico": metrics(y[bench], bench_score, threshold, cfg),
        "ablacion_tamano": sample_rows,
    }


def calibrate_and_stack(
    df: pd.DataFrame,
    split: dict[str, np.ndarray],
    tabular: dict[str, Any],
    cfg: ConfigV2,
) -> dict[str, Any]:
    print(
        "[6/9] Calibrando puntajes y construyendo un ensamble temporal conservador..."
    )
    val, bench = split["validation"], split["benchmark_historico"]
    y = df["isFraud"].to_numpy(dtype=np.int8)
    cut = int(len(val) * 0.65)
    meta_train, threshold_val = np.arange(cut), np.arange(cut, len(val))
    # El segundo experto es una señal causal agregada independiente; evita fabricar
    # predicciones GRU in-sample. La documentación conserva B-GRU V1 como benchmark.
    prior = np.log1p(df.loc[val, "entity_prior_count"].to_numpy())
    ratio = np.log1p(df.loc[val, "amount_to_prior_mean"].to_numpy())
    prior_b = np.log1p(df.loc[bench, "entity_prior_count"].to_numpy())
    ratio_b = np.log1p(df.loc[bench, "amount_to_prior_mean"].to_numpy())
    X_meta = np.column_stack(
        [tabular["val_score"], prior, ratio, df.loc[val, "log_amount"].to_numpy()]
    )
    X_bench = np.column_stack(
        [
            tabular["benchmark_score"],
            prior_b,
            ratio_b,
            df.loc[bench, "log_amount"].to_numpy(),
        ]
    )
    stack = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=cfg.seed
                ),
            ),
        ]
    )
    stack.fit(X_meta[meta_train], y[val][meta_train])
    score_val = stack.predict_proba(X_meta)[:, 1]
    score_bench = stack.predict_proba(X_bench)[:, 1]
    calibrator = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    eps = 1e-6
    logit_val = np.log(
        np.clip(score_val, eps, 1 - eps) / np.clip(1 - score_val, eps, 1 - eps)
    ).reshape(-1, 1)
    calibrator.fit(logit_val[threshold_val], y[val][threshold_val])
    calibrated_val = calibrator.predict_proba(logit_val)[:, 1]
    logit_b = np.log(
        np.clip(score_bench, eps, 1 - eps) / np.clip(1 - score_bench, eps, 1 - eps)
    ).reshape(-1, 1)
    calibrated_bench = calibrator.predict_proba(logit_b)[:, 1]
    threshold, table = choose_threshold(
        y[val][threshold_val], calibrated_val[threshold_val], cfg
    )
    table.to_csv(ART / "curva_umbral_ensamble_calibrado.csv", index=False)
    joblib.dump(stack, ART / "modelo_stacking.joblib")
    joblib.dump(calibrator, ART / "calibrador_sigmoide.joblib")
    return {
        "metodo": "stacking_logistico_con_agregados_causales_y_calibracion_sigmoide",
        "advertencia": "No se denomina ensamble A+B-GRU porque no usa predicciones OOF de una GRU V2; B-GRU V1 se conserva por separado.",
        "threshold": threshold,
        "validacion_threshold_holdout": metrics(
            y[val][threshold_val], calibrated_val[threshold_val], threshold, cfg
        ),
        "benchmark_historico": metrics(y[bench], calibrated_bench, threshold, cfg),
        "calibracion": {
            "ece_tabular_validacion": expected_calibration_error(
                y[val], tabular["val_score"]
            ),
            "ece_ensamble_validacion": expected_calibration_error(
                y[val], calibrated_val
            ),
            "brier_tabular_validacion": brier_score_loss(y[val], tabular["val_score"]),
            "brier_ensamble_validacion": brier_score_loss(y[val], calibrated_val),
        },
        "val_score": calibrated_val,
        "benchmark_score": calibrated_bench,
    }


def top_k_metrics(
    y: np.ndarray, score: np.ndarray, rates=(0.001, 0.005, 0.01, 0.02)
) -> list[dict[str, float]]:
    order = np.argsort(-score)
    rows = []
    for rate in rates:
        k = max(1, int(len(y) * rate))
        selected = y[order[:k]]
        rows.append(
            {
                "tasa_revision": rate,
                "k": k,
                "precision_at_k": float(selected.mean()),
                "recall_at_k": float(selected.sum() / max(1, y.sum())),
            }
        )
    return rows


def block_bootstrap_ci(
    y: np.ndarray, score: np.ndarray, cfg: ConfigV2, blocks: int = 24
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)
    pieces = np.array_split(np.arange(len(y)), blocks)
    values = []
    for _ in range(cfg.bootstrap_repetitions):
        chosen = rng.integers(0, blocks, size=blocks)
        idx = np.concatenate([pieces[i] for i in chosen])
        if y[idx].sum() and y[idx].sum() < len(idx):
            values.append(average_precision_score(y[idx], score[idx]))
    return {
        "estimacion": float(average_precision_score(y, score)),
        "li95": float(np.quantile(values, 0.025)),
        "ls95": float(np.quantile(values, 0.975)),
        "metodo": f"bootstrap por {blocks} bloques temporales, {len(values)} réplicas",
    }


def segment_metrics(
    df: pd.DataFrame,
    idx: np.ndarray,
    y: np.ndarray,
    score: np.ndarray,
    threshold: float,
    cfg: ConfigV2,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "ProductCD": df.loc[idx, "ProductCD"].astype(str).to_numpy(),
            "DeviceType": df.loc[idx, "DeviceType"].astype(str).to_numpy(),
            "amount": df.loc[idx, "TransactionAmt"].to_numpy(),
            "y": y,
            "score": score,
        }
    )
    frame["segmento_monto"] = pd.qcut(frame["amount"], q=4, duplicates="drop").astype(
        str
    )
    rows = []
    for variable in ["ProductCD", "DeviceType", "segmento_monto"]:
        for value, g in frame.groupby(variable, observed=True):
            if len(g) < 100 or g["y"].sum() == 0:
                continue
            m = metrics(g["y"].to_numpy(), g["score"].to_numpy(), threshold, cfg)
            rows.append(
                {
                    "dimension": variable,
                    "segmento": value,
                    "n": len(g),
                    "prevalencia": g["y"].mean(),
                    **m,
                }
            )
    return pd.DataFrame(rows)


def plots(
    df: pd.DataFrame,
    split: dict[str, np.ndarray],
    tabular: dict[str, Any],
    stack: dict[str, Any],
    audit: dict[str, Any],
    walk: pd.DataFrame,
    cfg: ConfigV2,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    y = df["isFraud"].to_numpy(dtype=np.int8)
    bench = split["benchmark_historico"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    summary = walk.groupby("modelo")["auc_pr"].agg(["mean", "std"]).sort_values("mean")
    ax.barh(
        summary.index,
        summary["mean"],
        xerr=summary["std"].fillna(0),
        color=["#2a9d8f", "#376f9e", "#e9c46a"][: len(summary)],
    )
    ax.set_xlabel("AUC-PR media walk-forward")
    ax.set_title("V2 · comparación temporal de representaciones y modelos")
    fig.tight_layout()
    fig.savefig(FIG / "01_validacion_walk_forward.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for label, score, color in [
        ("LightGBM V2", tabular["benchmark_score"], "#184e77"),
        ("Ensamble calibrado", stack["benchmark_score"], "#2a9d8f"),
    ]:
        p, r, _ = precision_recall_curve(y[bench], score)
        ax.plot(
            r,
            p,
            label=f"{label} · AP={average_precision_score(y[bench], score):.3f}",
            color=color,
            lw=2.2,
        )
    ax.axhline(y[bench].mean(), color="#6b7280", ls="--", label="Prevalencia")
    ax.set(
        xlabel="Recall",
        ylabel="Precisión",
        title="Benchmark histórico reutilizado · curvas precisión–recall",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "02_curvas_pr_v2.png", dpi=180)
    plt.close(fig)

    audit_df = (
        pd.read_csv(PROCESSED / "auditoria_variables.csv")
        .nlargest(18, "puntaje_relevancia")
        .sort_values("puntaje_relevancia")
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(audit_df["variable"], audit_df["puntaje_relevancia"], color="#376f9e")
    ax.set(
        xlabel="Puntaje combinado (correlación + MI)",
        title="Señal univariada estimada solo con el pasado inicial",
    )
    fig.tight_layout()
    fig.savefig(FIG / "03_relevancia_variables.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    prob_true, prob_pred = calibration_curve(
        y[bench], stack["benchmark_score"], n_bins=12, strategy="quantile"
    )
    ax.plot(
        prob_pred, prob_true, marker="o", color="#2a9d8f", label="Ensamble calibrado"
    )
    ax.plot([0, 1], [0, 1], ls="--", color="#6b7280", label="Calibración ideal")
    ax.set(
        xlabel="Probabilidad media predicha",
        ylabel="Fracción observada de fraude",
        title="Calibración en benchmark histórico",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_calibracion_v2.png", dpi=180)
    plt.close(fig)


def main() -> None:
    cfg = ConfigV2()
    ensure_dirs()
    set_seed(cfg.seed)
    started = time.time()
    df = load_all()
    split = temporal_boundaries(df, cfg)
    df, coverage = add_causal_features(df)
    numeric, categorical, audit = feature_audit(df, split["audit_train"], cfg)
    walk, comparison = evaluate_temporal_models(df, numeric, categorical, cfg)
    final = fit_final_models(
        df, numeric, categorical, comparison["ganador"], split, cfg
    )
    stack = calibrate_and_stack(df, split, final, cfg)
    print("[7/9] Calculando intervalos, capacidad operativa, segmentos y deriva...")
    y = df["isFraud"].to_numpy(dtype=np.int8)
    bench = split["benchmark_historico"]
    ci = block_bootstrap_ci(y[bench], stack["benchmark_score"], cfg)
    topk = top_k_metrics(y[bench], stack["benchmark_score"])
    segments = segment_metrics(
        df, bench, y[bench], stack["benchmark_score"], stack["threshold"], cfg
    )
    segments.to_csv(ART / "metricas_segmentos_benchmark.csv", index=False)
    drift_rows = []
    for name, idx in split.items():
        drift_rows.append(
            {
                "periodo": name,
                "n": len(idx),
                "fraude": float(y[idx].mean()),
                "monto_media": float(df.loc[idx, "TransactionAmt"].mean()),
                "ausencia_media": float(df.loc[idx, "missing_count"].mean()),
            }
        )
    pd.DataFrame(drift_rows).to_csv(ART / "resumen_deriva_temporal.csv", index=False)
    pd.DataFrame(topk).to_csv(ART / "metricas_top_k.csv", index=False)
    plots(df, split, final, stack, audit, walk, cfg)
    print("[8/9] Consolidando la fuente única de verdad resultados_v2.json...")
    result = {
        "version": "2.0",
        "estado_benchmark": "histórico_reutilizado_no_ciego",
        "configuracion": asdict(cfg),
        "entorno": {
            "python": platform.python_version(),
            "plataforma": platform.platform(),
            "cpu_logicos": os.cpu_count(),
            "lightgbm": lgb.__version__,
        },
        "datos": {
            "filas": len(df),
            "columnas_originales_integradas": 434,
            "prevalencia": float(y.mean()),
            "dias": float(
                (df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400
            ),
            "sha256_train_transaction": sha256(resolve_raw("train_transaction.csv")),
            "sha256_train_identity": sha256(resolve_raw("train_identity.csv")),
            "particiones": {
                k: {
                    "n": len(v),
                    "fraude": float(y[v].mean()),
                    "dt_min": float(df.loc[v, "TransactionDT"].min()),
                    "dt_max": float(df.loc[v, "TransactionDT"].max()),
                }
                for k, v in split.items()
            },
        },
        "identidad_secuencial": coverage,
        "seleccion_variables": audit,
        "validacion_walk_forward": comparison,
        "modelo_tabular_v2": {
            k: v
            for k, v in final.items()
            if k not in ("model", "encoder", "val_score", "benchmark_score")
        },
        "ensamble_v2": {
            k: v for k, v in stack.items() if k not in ("val_score", "benchmark_score")
        },
        "intervalo_auc_pr_benchmark": ci,
        "metricas_top_k": topk,
        "limitaciones": [
            "El benchmark final ya fue observado en V1 y se reporta como histórico reutilizado.",
            "Las claves de identidad son proxies y pueden mezclar o fragmentar personas.",
            "Correlación e información mutua son filtros descriptivos, no pruebas de causalidad.",
            "PCA se trata como ablación; solo se adopta si mejora de forma estable la validación temporal.",
            "El costo FN/FP es un supuesto académico y requiere validación operativa antes de producción.",
        ],
        "duracion_segundos": time.time() - started,
    }
    write_json(ART / "resultados_v2.json", result)
    write_json(
        ART / "esquema_entrada_v2.json",
        {
            "numericas": numeric,
            "categoricas": categorical,
            "excluidas_como_ids": sorted(ID_COLUMNS),
            "orden_temporal": "TransactionDT",
        },
    )
    print("[9/9] V2 tabular terminada. Fuente única:", ART / "resultados_v2.json")


if __name__ == "__main__":
    main()
