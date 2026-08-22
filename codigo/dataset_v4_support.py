"""Carga y características causales ampliadas para Proyecto 1, versión V4.

Todas las variables históricas se calculan después de ordenar por TransactionDT y
antes de observar la etiqueta de la fila actual. Ninguna usa isFraud como entrada.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dataset_v3_support import (
    BASE_CATEGORICAL,
    add_causal_features,
    load_all,
    proxy_key,
    set_seed,
    stringify,
    temporal_boundaries,
)

ROOT = Path(__file__).resolve().parents[1]


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def _historical_amount_features(
    frame: pd.DataFrame, key: pd.Series, prefix: str
) -> None:
    """Agrega conteo, monto histórico y recencia para una clave proxy."""
    amount = frame["TransactionAmt"].fillna(0).astype("float64")
    seconds = frame["TransactionDT"].astype("float64")
    group = frame.groupby(key, sort=False, observed=True)
    count = group.cumcount().astype("float32")
    denominator = count.replace(0, np.nan)
    prior_sum = amount.groupby(key, sort=False).cumsum() - amount
    prior_mean = prior_sum / denominator
    prior_sq = amount.pow(2).groupby(key, sort=False).cumsum() - amount.pow(2)
    prior_var = (prior_sq / denominator - prior_mean.pow(2)).clip(lower=0)
    previous_time = group["TransactionDT"].shift(1)

    frame[f"{prefix}_prior_count"] = count
    frame[f"{prefix}_prior_amt_mean"] = prior_mean.fillna(0).astype("float32")
    frame[f"{prefix}_prior_amt_std"] = np.sqrt(prior_var).fillna(0).astype("float32")
    frame[f"{prefix}_amount_ratio"] = (
        amount.div(prior_mean.replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1)
        .clip(0, 100)
        .astype("float32")
    )
    frame[f"{prefix}_hours_since_prior"] = (
        ((seconds - previous_time) / 3600)
        .fillna(0)
        .clip(0, 24 * 365)
        .astype("float32")
    )


def _text_family(series: pd.Series, kind: str) -> pd.Series:
    text = series.fillna("MISSING").astype(str).str.lower()
    if kind == "device":
        return text.str.split(r"[/ _-]", n=1, regex=True).str[0].replace("", "missing")
    if kind == "browser":
        return text.str.replace(r"[0-9._]+", "", regex=True).str.strip().replace("", "missing")
    if kind == "email":
        return text.str.split(".").str[-1].replace("", "missing")
    return text


def add_v4_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extiende V3 con interacción, velocidad, ausencia y deriva temporal."""
    initial_columns = len(frame.columns)
    original_id_columns = [f"id_{i:02d}" for i in range(1, 39) if f"id_{i:02d}" in frame]
    original_m_columns = [f"M{i}" for i in range(1, 10) if f"M{i}" in frame]
    original_id_missing = frame[original_id_columns].isna().sum(axis=1).astype("float32")
    original_m_missing = frame[original_m_columns].isna().sum(axis=1).astype("float32")
    out, coverage = add_causal_features(frame)
    seconds = out["TransactionDT"].astype("float64")
    day = seconds / 86400.0
    amount = out["TransactionAmt"].fillna(0).astype("float64")

    # Calendario y forma del monto.
    out["day_index"] = day.astype("float32")
    out["week_index"] = np.floor(day / 7).astype("float32")
    out["month_proxy"] = np.floor(day / 30).astype("float32")
    out["hour"] = np.floor((seconds / 3600) % 24).astype("float32")
    out["weekday"] = np.floor(day % 7).astype("float32")
    out["is_weekend"] = (out["weekday"] >= 5).astype("float32")
    out["amount_cents"] = np.round((amount - np.floor(amount)) * 100).astype("float32")
    out["amount_is_integer"] = np.isclose(amount, np.round(amount)).astype("float32")
    out["amount_log1p"] = np.log1p(amount.clip(lower=0)).astype("float32")
    out["amount_first_digit"] = (
        np.floor(amount / np.power(10.0, np.floor(np.log10(amount.clip(lower=1e-6)))))
        .clip(0, 9)
        .astype("float32")
    )

    # Resúmenes por bloques anonimizados: preservan señal aunque haya ausencia.
    for prefix, expected in (("V", range(1, 340)), ("C", range(1, 15)), ("D", range(1, 16))):
        columns = [f"{prefix}{i}" for i in expected if f"{prefix}{i}" in out]
        if not columns:
            continue
        block = out[columns]
        out[f"{prefix}_missing"] = block.isna().sum(axis=1).astype("float32")
        out[f"{prefix}_mean"] = block.mean(axis=1).astype("float32")
        out[f"{prefix}_std"] = block.std(axis=1).fillna(0).astype("float32")
        out[f"{prefix}_min"] = block.min(axis=1).astype("float32")
        out[f"{prefix}_max"] = block.max(axis=1).astype("float32")

    out["id_missing"] = original_id_missing
    out["M_missing"] = original_m_missing

    # Normalización temporal de D* y transformaciones estables de C*.
    for i in range(1, 16):
        column = f"D{i}"
        if column in out:
            out[f"{column}_minus_day"] = (out[column].astype("float64") - day).astype("float32")
    for i in range(1, 15):
        column = f"C{i}"
        if column in out:
            out[f"{column}_log1p"] = np.log1p(out[column].clip(lower=0)).astype("float32")

    # Claves proxy. Se usan para agregados, nunca como un ID numérico continuo.
    definitions = {
        "card_addr": ["card1", "card2", "addr1"],
        "card_email": ["card1", "card2", "P_emaildomain"],
        "card_device": ["card1", "DeviceInfo", "DeviceType"],
        "card_product": ["card1", "card2", "ProductCD"],
    }
    keys: dict[str, pd.Series] = {}
    for name, columns in definitions.items():
        key = proxy_key(out, columns, f"key_{name}")
        keys[name] = key
        _historical_amount_features(out, key, name)
        out[f"cat_{name}"] = key

    # Frecuencias estrictamente previas de campos de entidad y contexto.
    frequency_columns = [
        "card1", "card2", "card3", "card5", "addr1", "addr2",
        "P_emaildomain", "R_emaildomain", "DeviceInfo", "DeviceType",
        "ProductCD", "id_31",
    ]
    row_number = np.arange(len(out), dtype=np.float64)
    for column in frequency_columns:
        if column not in out:
            continue
        values = out[column].fillna("MISSING").astype(str)
        prior = values.groupby(values, sort=False).cumcount().astype("float32")
        out[f"freq_prior_{_safe_name(column)}"] = prior
        out[f"share_prior_{_safe_name(column)}"] = (
            prior.to_numpy(dtype=np.float64) / np.maximum(row_number, 1.0)
        ).astype("float32")

    # Cambios de contexto respecto al evento anterior de card1.
    card_key = out["card1"].fillna(-999999).astype(str)
    for column in ("addr1", "P_emaildomain", "DeviceInfo", "DeviceType", "ProductCD"):
        if column not in out:
            continue
        current = out[column].fillna("MISSING").astype(str)
        previous = current.groupby(card_key, sort=False).shift(1)
        out[f"card_changed_{_safe_name(column)}"] = (
            previous.notna() & current.ne(previous)
        ).astype("float32")

    # Familias categóricas legibles y categorías explícitas de tarjeta/dirección.
    out["device_family"] = _text_family(out["DeviceInfo"], "device")
    out["browser_family"] = _text_family(out["id_31"], "browser")
    out["p_email_tld"] = _text_family(out["P_emaildomain"], "email")
    out["r_email_tld"] = _text_family(out["R_emaildomain"], "email")
    for column in ("card1", "card2", "card3", "card5", "addr1", "addr2"):
        if column in out:
            out[f"cat_{column}"] = out[column].fillna("MISSING").astype(str)

    categorical = list(dict.fromkeys(
        BASE_CATEGORICAL
        + [f"cat_{name}" for name in definitions]
        + ["device_family", "browser_family", "p_email_tld", "r_email_tld"]
        + [f"cat_{column}" for column in ("card1", "card2", "card3", "card5", "addr1", "addr2")]
    ))
    stringify(out, categorical)
    coverage["claves_v4"] = {
        name: {
            "columnas": columns,
            "entidades": int(keys[name].nunique(dropna=False)),
            "mediana_transacciones": float(keys[name].value_counts().median()),
        }
        for name, columns in definitions.items()
    }
    coverage["variables_ingenieria_v4"] = int(len(out.columns) - initial_columns)
    del keys
    gc.collect()
    return out, coverage


V4_CATEGORICAL = list(dict.fromkeys(
    BASE_CATEGORICAL
    + ["cat_card_addr", "cat_card_email", "cat_card_device", "cat_card_product"]
    + ["device_family", "browser_family", "p_email_tld", "r_email_tld"]
    + ["cat_card1", "cat_card2", "cat_card3", "cat_card5", "cat_addr1", "cat_addr2"]
))
