"""Carga y creación causal de variables compartidas por el experimento V3."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datos" / "raw"

BASE_CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "DeviceType", "DeviceInfo", "id_12", "id_15", "id_16", "id_28",
    "id_29", "id_31", "id_34", "id_35", "id_36", "id_37", "id_38",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_raw(name: str) -> Path:
    candidates = (
        RAW / name,
        RAW / "kagglehub_cache" / "competitions" / "ieee-fraud-detection" / name,
    )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            return candidate
    raise FileNotFoundError(
        f"No se encontró {name}. Ejecute codigo/compartido/download_data.py desde la raíz."
    )


def load_all() -> pd.DataFrame:
    tx_path = resolve_raw("train_transaction.csv")
    identity_path = resolve_raw("train_identity.csv")
    print("Cargando 394 variables transaccionales y 41 de identidad...")
    tx = pd.read_csv(tx_path, low_memory=False)
    identity = pd.read_csv(identity_path, low_memory=False)
    frame = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    del tx, identity
    return frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def stringify(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = frame[column].fillna("MISSING").astype(str)
    return frame


def proxy_key(frame: pd.DataFrame, columns: list[str], name: str) -> pd.Series:
    parts = []
    for column in columns:
        if column in frame:
            parts.append(frame[column].fillna(-999999).astype(str))
        else:
            parts.append(pd.Series("MISSING", index=frame.index))
    return pd.concat(parts, axis=1).agg("|".join, axis=1).rename(name)


def identity_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    definitions = {
        "tarjeta_direccion": ["card1", "card2", "card3", "card5", "addr1"],
        "tarjeta_direccion_correo": ["card1", "card2", "addr1", "P_emaildomain"],
        "tarjeta_dispositivo_producto": ["card1", "DeviceInfo", "DeviceType", "ProductCD"],
        "tarjeta_dispositivo": ["card1", "DeviceInfo", "DeviceType"],
    }
    result: dict[str, Any] = {}
    for name, columns in definitions.items():
        key = proxy_key(frame, columns, name)
        sizes = key.value_counts(dropna=False)
        mapped = key.map(sizes)
        result[name] = {
            "columnas": columns,
            "entidades": int(sizes.size),
            "mediana_transacciones": float(sizes.median()),
            "p90_transacciones": float(sizes.quantile(0.90)),
            "porcentaje_con_3": float((mapped >= 3).mean() * 100),
            "porcentaje_con_8": float((mapped >= 8).mean() * 100),
            "porcentaje_con_16": float((mapped >= 16).mean() * 100),
            "porcentaje_con_32": float((mapped >= 32).mean() * 100),
        }
    return result


def add_causal_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Añade variables que, para cada fila, usan únicamente eventos anteriores."""
    out = frame
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
    raw_columns = [
        column for column in out.columns
        if column not in ("isFraud", "TransactionID", "TransactionDT", "entity_key")
    ]
    out["missing_count"] = out[raw_columns].isna().sum(axis=1).astype("float32")

    group = out.groupby("entity_key", sort=False, observed=True)
    out["entity_prior_count"] = group.cumcount().astype("float32")
    denominator = out["entity_prior_count"].replace(0, np.nan)
    prior_sum = group["TransactionAmt"].cumsum() - amount
    prior_mean = prior_sum / denominator
    prior_sq_sum = amount.pow(2).groupby(out["entity_key"], sort=False).cumsum() - amount.pow(2)
    prior_variance = (prior_sq_sum / denominator - prior_mean.pow(2)).clip(lower=0)
    out["entity_prior_amt_mean"] = prior_mean.fillna(0).astype("float32")
    out["entity_prior_amt_std"] = np.sqrt(prior_variance).fillna(0).astype("float32")
    out["amount_to_prior_mean"] = (
        amount.div(prior_mean.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 100).astype("float32")
    )
    prior_time = group["TransactionDT"].shift(1)
    out["hours_since_prior"] = (
        ((seconds - prior_time) / 3600).fillna(0).clip(0, 24 * 365).astype("float32")
    )

    windows = {"1h": 3600, "6h": 21600, "24h": 86400, "72h": 259200}
    histories: dict[str, deque[float]] = defaultdict(deque)
    counts = {label: np.zeros(len(out), dtype=np.float32) for label in windows}
    for row, (key, current) in enumerate(zip(out["entity_key"].to_numpy(), seconds.to_numpy())):
        history = histories[key]
        while history and history[0] < current - windows["72h"]:
            history.popleft()
        snapshot = list(history)
        for label, width in windows.items():
            cutoff = current - width
            counts[label][row] = sum(moment >= cutoff for moment in snapshot)
        history.append(current)
    for label, values in counts.items():
        out[f"entity_count_{label}"] = values

    coverage = identity_coverage(out)
    stringify(out, BASE_CATEGORICAL)
    return out, coverage


def temporal_boundaries(frame: pd.DataFrame, config: Any) -> dict[str, np.ndarray]:
    n = len(frame)
    cut = lambda fraction: int(np.floor(n * fraction))
    return {
        "audit_train": np.arange(0, cut(config.audit_train_fraction)),
        "train": np.arange(0, cut(config.train_fraction)),
        "validation": np.arange(cut(config.train_fraction), cut(config.development_fraction)),
        "benchmark_historico": np.arange(cut(config.development_fraction), n),
    }
