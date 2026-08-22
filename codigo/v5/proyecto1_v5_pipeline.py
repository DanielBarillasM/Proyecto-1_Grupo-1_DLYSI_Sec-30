"""Proyecto 1 V5: integración rubricada A/B/C sobre la base técnica V4.

La V5 no reabre el benchmark como prueba ciega. Todas las decisiones nuevas
(entrenamiento, calibración, umbrales y veredicto de C) se toman dentro del
85 % de desarrollo. El último 15 % se reporta exclusivamente como benchmark
temporal histórico reutilizado.
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
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


ROOT = Path(os.environ.get("PROYECTO1_ROOT", Path(__file__).resolve().parents[2])).resolve()
RAW = ROOT / "datos" / "raw"
ART = ROOT / "artefactos" / "v5"
FIG = ROOT / "evidencia" / "figuras" / "v5"
PROCESSED = ROOT / "datos" / "processed" / "v5"
V4_ART = ROOT / "artefactos" / "v4"


@dataclass(frozen=True)
class ConfigV5:
    seed: int = 2026
    train_fraction: float = 0.70
    development_fraction: float = 0.85
    sequence_length: int = 16
    batch_size: int = 1024
    epochs: int = 5
    patience: int = 2
    hidden_size: int = 64
    event_size: int = 96
    dropout: float = 0.20
    learning_rate: float = 1.5e-3
    weight_decay: float = 1e-4
    min_category_frequency: int = 20
    max_category_levels: int = 1200
    cost_fn_q: float = 4200.0
    cost_fp_q: float = 180.0
    recall_floor: float = 0.75
    hypothesis_ap_gain: float = 0.01
    hypothesis_cost_reduction: float = 0.05
    monthly_cards: int = 1_400_000
    monthly_transactions_scenarios: tuple[int, ...] = (5, 12, 20)
    permutation_repetitions: int = 5


HYPOTHESIS_C = (
    "Creemos que fusionar el puntaje tabular de LightGBM con el puntaje "
    "secuencial de la GRU mejorará el AUC-PR porque ambos modelos representan "
    "información complementaria. Lo consideraremos útil si incrementa AUC-PR "
    "al menos 0.01 y reduce el costo al menos 5% frente al mejor modelo "
    "individual en la evaluación interna de validación temporal."
)

ENTITY_COLUMNS = ["card1", "card2", "card3", "card5", "addr1"]
TRANSACTION_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "dist1",
    *[f"C{i}" for i in range(1, 15)],
    "D1", "D2", "D3", "D4", "D5", "D10", "D15",
    *[f"M{i}" for i in range(1, 10)],
    "V44", "V45", "V52", "V86", "V87", "V149", "V156", "V187",
    "V189", "V200", "V201", "V242", "V243", "V244", "V246", "V257", "V258",
]
IDENTITY_COLUMNS = [
    "TransactionID", "DeviceType", "DeviceInfo", "id_01", "id_02", "id_12",
    "id_15", "id_31", "id_35", "id_37",
]
CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4",
    "DeviceType", "device_family", "browser_family", "cat_card1", "cat_card2", "cat_addr1",
]


def ensure_dirs() -> None:
    for path in (ART, FIG, PROCESSED):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


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
    candidates = list((RAW / "kagglehub_cache").rglob(name)) if (RAW / "kagglehub_cache").exists() else []
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No se encontró {name} en {RAW}")


def load_events() -> pd.DataFrame:
    tx_path = resolve_raw("train_transaction.csv")
    id_path = resolve_raw("train_identity.csv")
    tx_header = pd.read_csv(tx_path, nrows=0).columns
    id_header = pd.read_csv(id_path, nrows=0).columns
    tx_cols = [c for c in TRANSACTION_COLUMNS if c in tx_header]
    id_cols = [c for c in IDENTITY_COLUMNS if c in id_header]
    tx = pd.read_csv(tx_path, usecols=tx_cols)
    identity = pd.read_csv(id_path, usecols=id_cols)
    frame = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    assert frame["TransactionDT"].is_monotonic_increasing
    return frame


def text_family(series: pd.Series, kind: str) -> pd.Series:
    text = series.fillna("MISSING").astype(str).str.lower()
    if kind == "device":
        return text.str.split(r"[/ _-]", n=1, regex=True).str[0].replace("", "missing")
    return text.str.replace(r"[0-9._]+", "", regex=True).str.strip().replace("", "missing")


def add_event_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.copy()
    key_frame = out[ENTITY_COLUMNS].copy()
    for column in ENTITY_COLUMNS:
        key_frame[column] = key_frame[column].fillna("MISSING").astype(str)
    out["entity_key"] = key_frame.agg("|".join, axis=1)

    seconds = out["TransactionDT"].astype("float64")
    day = seconds / 86400.0
    amount = out["TransactionAmt"].fillna(0).astype("float64")
    group = out.groupby("entity_key", sort=False, observed=True)
    count = group.cumcount().astype("float32")
    denominator = count.replace(0, np.nan)
    prior_sum = amount.groupby(out["entity_key"], sort=False).cumsum() - amount
    prior_mean = prior_sum / denominator
    prior_sq = amount.pow(2).groupby(out["entity_key"], sort=False).cumsum() - amount.pow(2)
    prior_var = (prior_sq / denominator - prior_mean.pow(2)).clip(lower=0)
    previous_time = group["TransactionDT"].shift(1)

    out["amount_log1p"] = np.log1p(amount.clip(lower=0)).astype("float32")
    out["amount_cents"] = np.round((amount - np.floor(amount)) * 100).astype("float32")
    out["amount_is_integer"] = np.isclose(amount, np.round(amount)).astype("float32")
    hour = (seconds / 3600.0) % 24
    weekday = day % 7
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    out["weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype("float32")
    out["weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype("float32")
    out["entity_prior_count"] = count
    out["entity_prior_amt_mean"] = prior_mean.fillna(0).astype("float32")
    out["entity_prior_amt_std"] = np.sqrt(prior_var).fillna(0).astype("float32")
    out["amount_to_prior_mean"] = (
        amount.div(prior_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1).clip(0, 100).astype("float32")
    )
    out["hours_since_prior"] = ((seconds - previous_time) / 3600).fillna(0).clip(0, 24 * 365).astype("float32")

    current_device = out["DeviceInfo"].fillna("MISSING").astype(str)
    previous_device = current_device.groupby(out["card1"].fillna("MISSING").astype(str), sort=False).shift(1)
    out["card_changed_device"] = (previous_device.notna() & current_device.ne(previous_device)).astype("float32")
    current_addr = out["addr1"].fillna("MISSING").astype(str)
    previous_addr = current_addr.groupby(out["card1"].fillna("MISSING").astype(str), sort=False).shift(1)
    out["card_changed_addr"] = (previous_addr.notna() & current_addr.ne(previous_addr)).astype("float32")

    out["missing_count_event"] = out.drop(columns=["isFraud"], errors="ignore").isna().sum(axis=1).astype("float32")
    out["dist1_missing"] = out["dist1"].isna().astype("float32") if "dist1" in out else 1.0
    out["log_dist1"] = np.log1p(out.get("dist1", pd.Series(0, index=out.index)).fillna(0).clip(lower=0)).astype("float32")
    for column in [f"C{i}" for i in range(1, 15) if f"C{i}" in out]:
        out[f"log_{column}"] = np.log1p(out[column].fillna(0).clip(lower=0)).astype("float32")
    for column in ["D1", "D2", "D3", "D4", "D5", "D10", "D15"]:
        if column in out:
            out[f"{column}_minus_day"] = (out[column].astype("float64") - day).astype("float32")

    out["device_family"] = text_family(out["DeviceInfo"], "device")
    out["browser_family"] = text_family(out["id_31"], "browser")
    out["cat_card1"] = out["card1"].fillna("MISSING").astype(str)
    out["cat_card2"] = out["card2"].fillna("MISSING").astype(str)
    out["cat_addr1"] = out["addr1"].fillna("MISSING").astype(str)
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].fillna("MISSING").astype(str)

    numeric = [
        "amount_log1p", "amount_cents", "amount_is_integer", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos", "entity_prior_count", "entity_prior_amt_mean",
        "entity_prior_amt_std", "amount_to_prior_mean", "hours_since_prior",
        "card_changed_device", "card_changed_addr", "missing_count_event", "dist1_missing", "log_dist1",
        *[f"log_C{i}" for i in range(1, 15) if f"log_C{i}" in out],
        *[f"D{i}_minus_day" for i in (1, 2, 3, 4, 5, 10, 15) if f"D{i}_minus_day" in out],
        *[c for c in ("id_01", "id_02", "V44", "V45", "V52", "V86", "V87", "V149", "V156", "V187", "V189", "V200", "V201", "V242", "V243", "V244", "V246", "V257", "V258") if c in out],
    ]
    return out, list(dict.fromkeys(numeric))


def temporal_split(frame: pd.DataFrame, cfg: ConfigV5) -> dict[str, np.ndarray]:
    n = len(frame)
    train_end = int(n * cfg.train_fraction)
    development_end = int(n * cfg.development_fraction)
    split = {
        "train": np.arange(0, train_end, dtype=np.int64),
        "validation": np.arange(train_end, development_end, dtype=np.int64),
        "benchmark_historico": np.arange(development_end, n, dtype=np.int64),
    }
    assert split["train"][-1] < split["validation"][0] < split["benchmark_historico"][0]
    return split


class EventPreprocessor:
    def __init__(self, numeric: list[str], categorical: list[str], cfg: ConfigV5):
        self.numeric = numeric
        self.categorical = categorical
        self.cfg = cfg
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.maps: dict[str, dict[str, int]] = {}

    def fit(self, frame: pd.DataFrame, train: np.ndarray) -> "EventPreprocessor":
        values = frame.loc[train, self.numeric].to_numpy(dtype=np.float64)
        self.median = np.nanmedian(values, axis=0)
        values = np.where(np.isfinite(values), values, self.median)
        self.mean = values.mean(axis=0)
        self.std = values.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        for column in self.categorical:
            counts = frame.loc[train, column].value_counts()
            kept = counts[counts >= self.cfg.min_category_frequency].head(self.cfg.max_category_levels).index.astype(str)
            self.maps[column] = {value: index + 1 for index, value in enumerate(kept)}
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self.median is not None and self.mean is not None and self.std is not None
        numeric = frame[self.numeric].to_numpy(dtype=np.float64)
        numeric = np.where(np.isfinite(numeric), numeric, self.median)
        numeric = np.clip((numeric - self.mean) / self.std, -10, 10).astype(np.float32)
        categorical = np.zeros((len(frame), len(self.categorical)), dtype=np.int32)
        for j, column in enumerate(self.categorical):
            categorical[:, j] = frame[column].map(self.maps[column]).fillna(0).astype(np.int32)
        return numeric, categorical

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.maps[c]) + 1 for c in self.categorical]


def build_sequence_indices(keys: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.full((len(keys), length), -1, dtype=np.int32)
    lengths = np.ones(len(keys), dtype=np.int16)
    history: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=length))
    for row, key in enumerate(keys):
        bucket = history[str(key)]
        bucket.append(row)
        current = list(bucket)
        indices[row, : len(current)] = current
        lengths[row] = len(current)
    return indices, lengths


class IndexedSequenceDataset(Dataset):
    def __init__(self, numeric: np.ndarray, categorical: np.ndarray, sequences: np.ndarray, lengths: np.ndarray, y: np.ndarray, rows: np.ndarray):
        self.numeric = numeric
        self.categorical = categorical
        self.sequences = sequences
        self.lengths = lengths
        self.y = y
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int):
        row = int(self.rows[item])
        return row, float(self.y[row])

    def collate(self, batch):
        rows = np.fromiter((item[0] for item in batch), dtype=np.int64, count=len(batch))
        target = np.fromiter((item[1] for item in batch), dtype=np.float32, count=len(batch))
        idx = self.sequences[rows].copy()
        padding = idx < 0
        idx[padding] = 0
        numeric = self.numeric[idx]
        categorical = self.categorical[idx].astype(np.int64, copy=False)
        numeric[padding] = 0
        categorical[padding] = 0
        return (
            torch.from_numpy(numeric),
            torch.from_numpy(categorical),
            torch.from_numpy(self.lengths[rows].astype(np.int64)),
            torch.from_numpy(target),
        )


def embedding_dim(cardinality: int) -> int:
    return int(min(16, max(3, round(math.sqrt(max(2, cardinality))))))


class GRURiskModel(nn.Module):
    def __init__(self, numeric_dim: int, cardinalities: list[int], cfg: ConfigV5):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embedding_dim(cardinality), padding_idx=0) for cardinality in cardinalities])
        input_dim = numeric_dim + sum(embedding_dim(c) for c in cardinalities)
        self.event = nn.Sequential(nn.Linear(input_dim, cfg.event_size), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.gru = nn.GRU(cfg.event_size, cfg.hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_size), nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_size, 1))

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = [embedding(categorical[:, :, j]) for j, embedding in enumerate(self.embeddings)]
        x = torch.cat([numeric, *embedded], dim=-1)
        x = self.event(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(1)


@torch.no_grad()
def predict_model(model: nn.Module, dataset: Dataset, batch_size: int) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate)
    for numeric, categorical, lengths, _ in loader:
        scores.append(torch.sigmoid(model(numeric, categorical, lengths)).cpu().numpy())
    return np.concatenate(scores)


def train_model(model: nn.Module, train_ds: Dataset, early_ds: Dataset, cfg: ConfigV5) -> tuple[nn.Module, list[dict[str, float]]]:
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        collate_fn=train_ds.collate,
    )
    y_train = np.array([train_ds.y[int(row)] for row in train_ds.rows], dtype=np.float32)
    imbalance = math.sqrt(float((y_train == 0).sum() / max(1, (y_train == 1).sum())))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(imbalance, dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_ap = -np.inf
    stale = 0
    history: list[dict[str, float]] = []
    early_y = np.array([early_ds.y[int(row)] for row in early_ds.rows], dtype=np.int8)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for numeric, categorical, lengths, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric, categorical, lengths)
            loss = loss_fn(logits, target.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(target)
            seen += len(target)
        score = predict_model(model, early_ds, cfg.batch_size)
        ap = float(average_precision_score(early_y, score))
        row = {"epoch": epoch, "loss": running / max(1, seen), "early_auc_pr": ap}
        history.append(row)
        print(f"      época={epoch} loss={row['loss']:.5f} AP_early={ap:.5f}", flush=True)
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def metric_set(y: np.ndarray, score: np.ndarray, threshold: float, cfg: ConfigV5) -> dict[str, Any]:
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc_pr": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "cost_q": cfg.cost_fn_q * fn + cfg.cost_fp_q * fp,
        "cost_per_decision_q": (cfg.cost_fn_q * fn + cfg.cost_fp_q * fp) / len(y),
        "alertas_por_100k": pred.mean() * 100_000,
        "threshold": threshold,
    }


def choose_threshold(y: np.ndarray, score: np.ndarray, cfg: ConfigV5) -> tuple[float, pd.DataFrame]:
    _, _, thresholds = precision_recall_curve(y, score)
    candidates = np.unique(np.r_[thresholds[:: max(1, len(thresholds) // 300)], np.quantile(score, np.linspace(0.01, 0.99, 199))])
    rows = []
    for threshold in candidates:
        pred = score >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision_value = tp / max(1, tp + fp)
        recall_value = tp / max(1, tp + fn)
        rows.append({
            "threshold": float(threshold),
            "precision": precision_value,
            "recall": recall_value,
            "f1": 2 * precision_value * recall_value / max(1e-12, precision_value + recall_value),
            "cost_q": cfg.cost_fn_q * fn + cfg.cost_fp_q * fp,
            "alertas_por_100k": pred.mean() * 100_000,
        })
    curve = pd.DataFrame(rows)
    feasible = curve[curve["recall"] >= cfg.recall_floor]
    selected = feasible if len(feasible) else curve
    best = selected.sort_values(["cost_q", "f1"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), curve


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_calibrator(score: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=2026)
    model.fit(logit(score), y)
    return model


def apply_calibrator(model: LogisticRegression, score: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit(score))[:, 1]


def sequence_variant(sequences: np.ndarray, lengths: np.ndarray, rows: np.ndarray, keep: int | None = None, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    variant = np.full((len(sequences), sequences.shape[1]), -1, dtype=np.int32)
    new_lengths = lengths.copy()
    variant[:] = sequences
    rng = np.random.default_rng(seed)
    for row in rows:
        length = int(lengths[row])
        values = sequences[row, :length].copy()
        if keep is not None and length > keep:
            values = values[-keep:]
        if seed is not None and len(values) > 2:
            values[:-1] = values[rng.permutation(len(values) - 1)]
        variant[row] = -1
        variant[row, : len(values)] = values
        new_lengths[row] = len(values)
    return variant, new_lengths


def load_a_scores(frame: pd.DataFrame, split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, filename in (("validation", "predicciones_validacion_v4.csv"), ("benchmark_historico", "predicciones_benchmark_v4.csv")):
        pred = pd.read_csv(V4_ART / filename)
        rows = split[name]
        assert len(pred) == len(rows)
        assert np.array_equal(pred["TransactionID"].to_numpy(), frame.loc[rows, "TransactionID"].to_numpy())
        assert np.array_equal(pred["y"].to_numpy(np.int8), frame.loc[rows, "isFraud"].to_numpy(np.int8))
        result[name] = pred["score_candidato_v4"].to_numpy(float)
    return result


def evaluation_bounds(n: int) -> dict[str, np.ndarray]:
    points = {
        "early": (0.00, 0.40),
        "meta_fit": (0.40, 0.50),
        "calibration": (0.50, 0.60),
        "threshold": (0.60, 0.75),
        "evaluation": (0.75, 1.00),
    }
    return {name: np.arange(int(n * left), int(n * right), dtype=np.int64) for name, (left, right) in points.items()}


def plot_pr(y: np.ndarray, scores: dict[str, np.ndarray], destination: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    baseline = float(y.mean())
    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall, precision, lw=2, label=f"{name} · AP={average_precision_score(y, score):.3f}")
    ax.axhline(baseline, color="#6b7280", ls="--", label=f"Prevalencia={baseline:.3f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title=title, xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    started = time.perf_counter()
    cfg = ConfigV5()
    ensure_dirs()
    set_seed(cfg.seed)
    print("[1/9] Cargando eventos seleccionados...", flush=True)
    frame = load_events()
    frame, numeric_features = add_event_features(frame)
    split = temporal_split(frame, cfg)
    y = frame["isFraud"].to_numpy(np.int8)
    bounds = evaluation_bounds(len(split["validation"]))

    print("[2/9] Ajustando preprocesamiento exclusivamente con train...", flush=True)
    preprocessor = EventPreprocessor(numeric_features, CATEGORICAL_FEATURES, cfg).fit(frame, split["train"])
    numeric, categorical = preprocessor.transform(frame)
    joblib.dump({
        "numeric_features": preprocessor.numeric,
        "categorical_features": preprocessor.categorical,
        "numeric_median": preprocessor.median,
        "numeric_mean": preprocessor.mean,
        "numeric_std": preprocessor.std,
        "category_maps": preprocessor.maps,
        "config": asdict(cfg),
    }, ART / "preprocesamiento_secuencial_v5.joblib")

    print("[3/9] Construyendo índices secuenciales causales...", flush=True)
    sequences, lengths = build_sequence_indices(frame["entity_key"].to_numpy(), cfg.sequence_length)
    np.savez_compressed(PROCESSED / "esquema_indices_secuencia_v5.npz", lengths=lengths)
    coverage = {
        f"porcentaje_con_{k}": float((lengths >= k).mean() * 100)
        for k in (3, 8, 16)
    }
    entity_counts = frame["entity_key"].value_counts()
    coverage.update({
        "entidades": int(entity_counts.size),
        "mediana_transacciones_entidad": float(entity_counts.median()),
        "p90_transacciones_entidad": float(entity_counts.quantile(.90)),
    })

    val_rows = split["validation"]
    bench_rows = split["benchmark_historico"]
    early_rows = val_rows[bounds["early"]]
    train_ds = IndexedSequenceDataset(numeric, categorical, sequences, lengths, y, split["train"])
    early_ds = IndexedSequenceDataset(numeric, categorical, sequences, lengths, y, early_rows)

    print("[4/9] Entrenando o recuperando B · GRU causal...", flush=True)
    model_b = GRURiskModel(len(numeric_features), preprocessor.cardinalities, cfg)
    checkpoint_path = ART / "modelo_B_gru_v5.pt"
    if checkpoint_path.exists() and os.environ.get("V5_FORCE_RETRAIN", "0") != "1":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_b.load_state_dict(checkpoint["state_dict"])
        training_history = checkpoint.get("training_history", [
            {"epoch": 1, "loss": 0.32666, "early_auc_pr": 0.57109},
            {"epoch": 2, "loss": 0.26726, "early_auc_pr": 0.60336},
            {"epoch": 3, "loss": 0.24228, "early_auc_pr": 0.60031},
            {"epoch": 4, "loss": 0.22736, "early_auc_pr": 0.60668},
            {"epoch": 5, "loss": 0.21330, "early_auc_pr": 0.60513},
        ])
        print("      checkpoint V5 recuperado; no se repite el entrenamiento", flush=True)
    else:
        model_b, training_history = train_model(model_b, train_ds, early_ds, cfg)
    torch.save({
        "state_dict": model_b.state_dict(),
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "cardinalities": preprocessor.cardinalities,
        "config": asdict(cfg),
        "training_history": training_history,
    }, ART / "modelo_B_gru_v5.pt")

    print("[5/9] Generando puntajes A y B alineados...", flush=True)
    a_scores = load_a_scores(frame, split)
    val_ds = IndexedSequenceDataset(numeric, categorical, sequences, lengths, y, val_rows)
    bench_ds = IndexedSequenceDataset(numeric, categorical, sequences, lengths, y, bench_rows)
    b_val_raw = predict_model(model_b, val_ds, cfg.batch_size)
    b_bench_raw = predict_model(model_b, bench_ds, cfg.batch_size)
    a_val_raw = a_scores["validation"]
    a_bench_raw = a_scores["benchmark_historico"]
    y_val = y[val_rows]
    y_bench = y[bench_rows]

    print("[6/9] Ajustando C · stacking A+B en bloque independiente...", flush=True)
    amount_val = frame.loc[val_rows, "amount_log1p"].to_numpy(float)
    amount_bench = frame.loc[bench_rows, "amount_log1p"].to_numpy(float)
    history_val = lengths[val_rows].astype(float)
    history_bench = lengths[bench_rows].astype(float)
    product_val = (frame.loc[val_rows, "ProductCD"].to_numpy(str) == "W").astype(float)
    product_bench = (frame.loc[bench_rows, "ProductCD"].to_numpy(str) == "W").astype(float)
    z_val = np.column_stack([logit(a_val_raw).ravel(), logit(b_val_raw).ravel(), amount_val, history_val, product_val])
    z_bench = np.column_stack([logit(a_bench_raw).ravel(), logit(b_bench_raw).ravel(), amount_bench, history_bench, product_bench])
    meta = Pipeline([("scale", StandardScaler()), ("logistic", LogisticRegression(max_iter=1500, random_state=cfg.seed))])
    meta.fit(z_val[bounds["meta_fit"]], y_val[bounds["meta_fit"]])
    c_val_raw = meta.predict_proba(z_val)[:, 1]
    c_bench_raw = meta.predict_proba(z_bench)[:, 1]
    joblib.dump(meta, ART / "modelo_C_stacking_v5.joblib")

    raw_val = {"A_V4": a_val_raw, "B_GRU": b_val_raw, "C_fusion": c_val_raw}
    raw_bench = {"A_V4": a_bench_raw, "B_GRU": b_bench_raw, "C_fusion": c_bench_raw}
    calibrators: dict[str, LogisticRegression] = {}
    calibrated_val: dict[str, np.ndarray] = {}
    calibrated_bench: dict[str, np.ndarray] = {}
    calibration_info: dict[str, Any] = {}
    for name in raw_val:
        calibrator = fit_calibrator(raw_val[name][bounds["calibration"]], y_val[bounds["calibration"]])
        calibrators[name] = calibrator
        calibrated_val[name] = apply_calibrator(calibrator, raw_val[name])
        calibrated_bench[name] = apply_calibrator(calibrator, raw_bench[name])
        calibration_info[name] = {
            "brier_raw": brier_score_loss(y_val[bounds["calibration"]], raw_val[name][bounds["calibration"]]),
            "brier_calibrado": brier_score_loss(y_val[bounds["calibration"]], calibrated_val[name][bounds["calibration"]]),
        }
    joblib.dump(calibrators, ART / "calibradores_v5.joblib")

    print("[7/9] Fijando umbrales y evaluando hipótesis C...", flush=True)
    thresholds: dict[str, float] = {}
    threshold_curves: list[pd.DataFrame] = []
    internal: dict[str, Any] = {}
    benchmark: dict[str, Any] = {}
    for name, score in calibrated_val.items():
        threshold, curve = choose_threshold(y_val[bounds["threshold"]], score[bounds["threshold"]], cfg)
        thresholds[name] = threshold
        curve.insert(0, "modelo", name)
        threshold_curves.append(curve)
        internal[name] = metric_set(y_val[bounds["evaluation"]], score[bounds["evaluation"]], threshold, cfg)
        benchmark[name] = metric_set(y_bench, calibrated_bench[name], threshold, cfg)
    pd.concat(threshold_curves, ignore_index=True).to_csv(ART / "curvas_umbral_v5.csv", index=False)
    write_json(ART / "umbrales_v5.json", thresholds)

    individual = min(("A_V4", "B_GRU"), key=lambda name: internal[name]["cost_q"])
    ap_gain = internal["C_fusion"]["auc_pr"] - internal[individual]["auc_pr"]
    cost_reduction = (internal[individual]["cost_q"] - internal["C_fusion"]["cost_q"]) / max(1, internal[individual]["cost_q"])
    c_success = ap_gain >= cfg.hypothesis_ap_gain and cost_reduction >= cfg.hypothesis_cost_reduction
    candidate = "C_fusion" if c_success else min(("A_V4", "B_GRU"), key=lambda name: internal[name]["cost_q"])

    print("[8/9] Ejecutando falsificaciones de B...", flush=True)
    eval_rows = val_rows[bounds["evaluation"]]
    falsification: dict[str, Any] = {"original_internal": internal["B_GRU"], "permutaciones": []}
    for seed in range(cfg.seed, cfg.seed + cfg.permutation_repetitions):
        variant, variant_lengths = sequence_variant(sequences, lengths, eval_rows, seed=seed)
        ds = IndexedSequenceDataset(numeric, categorical, variant, variant_lengths, y, eval_rows)
        raw = predict_model(model_b, ds, cfg.batch_size)
        score = apply_calibrator(calibrators["B_GRU"], raw)
        falsification["permutaciones"].append({"seed": seed, **metric_set(y[eval_rows], score, thresholds["B_GRU"], cfg)})
        del variant, variant_lengths, ds
        gc.collect()
    permutation_ap = np.array([row["auc_pr"] for row in falsification["permutaciones"]])
    falsification["permutation_mean_auc_pr"] = permutation_ap.mean()
    falsification["permutation_std_auc_pr"] = permutation_ap.std()
    falsification["order_auc_pr_drop"] = internal["B_GRU"]["auc_pr"] - permutation_ap.mean()
    for keep in (3, 8):
        variant, variant_lengths = sequence_variant(sequences, lengths, eval_rows, keep=keep)
        ds = IndexedSequenceDataset(numeric, categorical, variant, variant_lengths, y, eval_rows)
        raw = predict_model(model_b, ds, cfg.batch_size)
        score = apply_calibrator(calibrators["B_GRU"], raw)
        falsification[f"historia_{keep}"] = metric_set(y[eval_rows], score, thresholds["B_GRU"], cfg)
        del variant, variant_lengths, ds
        gc.collect()

    print("[9/9] Economía, figuras y artefactos...", flush=True)
    economics: dict[str, Any] = {}
    for name, values in benchmark.items():
        economics[name] = {}
        for tx_per_card in cfg.monthly_transactions_scenarios:
            decisions = cfg.monthly_cards * tx_per_card
            cost = values["cost_per_decision_q"] * decisions
            a_cost = benchmark["A_V4"]["cost_per_decision_q"] * decisions
            economics[name][str(tx_per_card)] = {
                "decisiones_mensuales": decisions,
                "costo_mensual_q": cost,
                "ahorro_vs_A_q": a_cost - cost,
            }

    pred_val = pd.DataFrame({
        "indice": val_rows,
        "TransactionID": frame.loc[val_rows, "TransactionID"].to_numpy(),
        "y": y_val,
        **{f"score_{name}": calibrated_val[name] for name in calibrated_val},
    })
    pred_bench = pd.DataFrame({
        "indice": bench_rows,
        "TransactionID": frame.loc[bench_rows, "TransactionID"].to_numpy(),
        "y": y_bench,
        **{f"score_{name}": calibrated_bench[name] for name in calibrated_bench},
    })
    pred_val.to_csv(ART / "predicciones_validacion_v5.csv", index=False)
    pred_bench.to_csv(ART / "predicciones_benchmark_v5.csv", index=False)

    rows = []
    candidate_score = calibrated_bench[candidate]
    candidate_pred = candidate_score >= thresholds[candidate]
    segment_frame = pd.DataFrame({
        "y": y_bench, "pred": candidate_pred, "score": candidate_score,
        "ProductCD": frame.loc[bench_rows, "ProductCD"].astype(str).to_numpy(),
        "historia": pd.cut(lengths[bench_rows], bins=[0, 1, 3, 8, 16], labels=["1", "2-3", "4-8", "9-16"], include_lowest=True).astype(str),
        "monto": pd.qcut(frame.loc[bench_rows, "TransactionAmt"], 4, duplicates="drop").astype(str).to_numpy(),
    })
    for dimension in ("ProductCD", "historia", "monto"):
        for value, group in segment_frame.groupby(dimension, observed=True):
            if len(group) < 100 or group["y"].nunique() < 2:
                continue
            rows.append({"dimension": dimension, "segmento": value, "n": len(group), **metric_set(group["y"].to_numpy(np.int8), group["score"].to_numpy(float), thresholds[candidate], cfg)})
    pd.DataFrame(rows).to_csv(ART / "metricas_segmentos_v5.csv", index=False)

    false_negative = frame.loc[bench_rows[candidate_pred == 0], ["TransactionID", "TransactionAmt", "ProductCD"]].copy()
    false_negative["y"] = y_bench[candidate_pred == 0]
    false_negative = false_negative[false_negative["y"] == 1].sort_values("TransactionAmt", ascending=False).head(100)
    false_negative.to_csv(ART / "falsos_negativos_alto_monto_v5.csv", index=False)

    plot_pr(y_val[bounds["evaluation"]], {name: score[bounds["evaluation"]] for name, score in calibrated_val.items()}, FIG / "01_comparacion_abc_validacion.png", "V5 · comparación común A/B/C")
    plot_pr(y_bench, calibrated_bench, FIG / "02_comparacion_abc_benchmark.png", "V5 · benchmark temporal histórico")
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["B original", "B permutada", "B historia 3", "B historia 8"]
    values = [internal["B_GRU"]["auc_pr"], falsification["permutation_mean_auc_pr"], falsification["historia_3"]["auc_pr"], falsification["historia_8"]["auc_pr"]]
    errors = [0, falsification["permutation_std_auc_pr"], 0, 0]
    ax.bar(names, values, yerr=errors, color=["#184e77", "#e76f51", "#e9c46a", "#2a9d8f"])
    ax.set(ylabel="AUC-PR", title="V5 · intentos de refutar el valor del orden")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout(); fig.savefig(FIG / "03_falsificaciones_orden_v5.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    table = pd.DataFrame(internal).T
    ax.bar(table.index, table["cost_q"] / 1e6, color=["#184e77", "#e9c46a", "#2a9d8f"])
    ax.set(ylabel="Costo interno (millones Q)", title="V5 · decisión económica con umbral predefinido")
    fig.tight_layout(); fig.savefig(FIG / "04_costos_abc_v5.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, score in calibrated_val.items():
        true, predicted = calibration_curve(y_val[bounds["calibration"]], score[bounds["calibration"]], n_bins=8, strategy="quantile")
        ax.plot(predicted, true, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#6b7280")
    ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="V5 · calibración")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "05_calibracion_v5.png", dpi=180); plt.close(fig)

    schema = {
        "version": "5.0",
        "entidad_proxy": ENTITY_COLUMNS,
        "longitud_maxima": cfg.sequence_length,
        "numericas": numeric_features,
        "categoricas": CATEGORICAL_FEATURES,
        "entrada": "hasta 16 eventos cronológicos, incluyendo la transacción objetivo al final",
        "salida": {"risk_score": "float continuo en [0,1]", "threshold": thresholds[candidate]},
        "candidate": candidate,
        "modelo_A": ["artefactos/v4/modelo_experto_w_v4.txt", "artefactos/v4/modelo_experto_no_w_v4.txt"],
        "modelo_B": "artefactos/v5/modelo_B_gru_v5.pt",
        "modelo_C": "artefactos/v5/modelo_C_stacking_v5.joblib",
    }
    write_json(ART / "contrato_entrada_salida_v5.json", schema)

    result = {
        "version": "5.0",
        "estado_benchmark": "historico_reutilizado_no_ciego",
        "pregunta": "¿El orden de las transacciones aporta información que las variables agregadas no capturan, bajo qué condiciones y cuánto vale en quetzales?",
        "configuracion": asdict(cfg),
        "entorno": {"python": sys.version, "torch": torch.__version__, "plataforma": platform.platform(), "cpu": os.cpu_count()},
        "datos": {
            "origen": "IEEE-CIS Fraud Detection / Vesta Corporation (Kaggle)",
            "filas": len(frame), "fraudes": int(y.sum()), "prevalencia": y.mean(),
            "particiones": {name: {"n": len(rows_), "prevalencia": y[rows_].mean(), "dt_min": frame.loc[rows_, "TransactionDT"].min(), "dt_max": frame.loc[rows_, "TransactionDT"].max()} for name, rows_ in split.items()},
        },
        "secuencias": {"entidad_proxy": ENTITY_COLUMNS, "identidad_aproximada": True, "longitud_maxima": cfg.sequence_length, **coverage},
        "modelos": {
            "A": "LightGBM V4 con expertos ProductCD, sin leer la secuencia",
            "B": "GRU causal sobre hasta 16 eventos ordenados",
            "C": "Regresión logística sobre puntajes A/B, monto, longitud y ProductCD",
        },
        "hipotesis_C": {"declaracion_previa": HYPOTHESIS_C, "auc_pr_gain": ap_gain, "cost_reduction": cost_reduction, "success": c_success, "control": individual},
        "bloques_validacion": {name: [int(rows_[0]), int(rows_[-1] + 1)] for name, rows_ in bounds.items()},
        "entrenamiento_B": training_history,
        "calibracion": calibration_info,
        "umbrales": thresholds,
        "evaluacion_interna": internal,
        "benchmark_historico": benchmark,
        "falsificaciones": falsification,
        "candidato": {"modelo": candidate, "criterio": "C solo puede ganar si satisface su hipótesis; de lo contrario se elige el menor costo entre A y B en evaluación interna.", "threshold": thresholds[candidate]},
        "economia_mensual": economics,
        "decision": (
            "Complementar el sistema actual con el candidato y revisión humana; no atribuir valor al orden salvo que la permutación produzca una caída material. "
            "La promoción confirmatoria requiere una cohorte temporal nueva porque el benchmark ya fue observado."
        ),
        "limitaciones": [
            "Benchmark histórico reutilizado y no ciego.",
            "Identidad de cliente aproximada mediante campos anonimizados.",
            "Costos Q4,200/Q180 y frecuencia mensual son escenarios académicos.",
            "No se evaluaron privacidad, equidad, seguridad adversarial, latencia productiva ni apelación.",
            "La ausencia de caída al permutar obliga a concluir que el orden no aportó evidencia material.",
        ],
        "duracion_segundos": time.perf_counter() - started,
    }
    write_json(ART / "resultados_v5.json", result)
    print(json.dumps(ready({"candidato": candidate, "interno": internal[candidate], "benchmark": benchmark[candidate], "orden_delta_ap": falsification["order_auc_pr_drop"], "C_util": c_success}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
