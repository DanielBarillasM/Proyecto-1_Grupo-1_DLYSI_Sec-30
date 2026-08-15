from __future__ import annotations

import json
import math
import random
import shutil
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    seq_len: int = 8
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    max_train_targets: int = 180_000
    batch_size: int = 512
    epochs: int = 6
    patience: int = 2
    hidden_size: int = 32
    learning_rate: float = 2e-3
    min_category_frequency: int = 20
    max_categories: int = 120
    cost_fn_q: float = 4200.0
    cost_fp_q: float = 180.0
    cards_bank: int = 1_400_000
    transactions_per_card_month: int = 12
    hypothesis_ap_gain: float = 0.01
    hypothesis_cost_reduction: float = 0.05


TRANSACTION_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
    "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", "dist1", "C1", "C2", "C13", "D1", "M4",
]
IDENTITY_COLUMNS = ["TransactionID", "DeviceType", "id_31"]
CARD_KEY_COLUMNS = ["card1", "card2", "card3", "card5", "addr1"]
NUMERIC_FEATURES = [
    "log_amount", "log_delta_hours", "hour_sin", "hour_cos", "log_dist1",
    "dist1_missing", "log_C1", "log_C2", "log_C13", "log_D1",
]
CATEGORICAL_FEATURES = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "id_31"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def paths() -> dict[str, Path]:
    root = project_root()
    return {
        "root": root,
        "raw": root / "datos" / "raw",
        "processed": root / "datos" / "processed",
        "artifacts": root / "artefactos",
        "figures": root / "evidencia" / "figuras",
    }


def ensure_directories() -> None:
    for p in paths().values():
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_raw_file(name: str) -> Path:
    p = paths()["raw"] / name
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    cached = paths()["raw"] / "kagglehub_cache" / "competitions" / "ieee-fraud-detection" / name
    if cached.exists():
        return cached
    raise FileNotFoundError(f"No se encontró {name}. Revise .github/README.md.")


def load_raw_data() -> pd.DataFrame:
    tx = pd.read_csv(resolve_raw_file("train_transaction.csv"), usecols=TRANSACTION_COLUMNS)
    identity = pd.read_csv(resolve_raw_file("train_identity.csv"), usecols=IDENTITY_COLUMNS)
    df = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    return df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key_frame = out[CARD_KEY_COLUMNS].copy()
    for c in CARD_KEY_COLUMNS:
        key_frame[c] = key_frame[c].fillna(-999999).astype("int64").astype(str)
    out["card_key"] = key_frame.agg("|".join, axis=1)

    out["log_amount"] = np.log1p(out["TransactionAmt"].clip(lower=0))
    delta = out.groupby("card_key", sort=False)["TransactionDT"].diff().fillna(0) / 3600.0
    out["log_delta_hours"] = np.log1p(delta.clip(0, 24 * 30))
    hour = (out["TransactionDT"] / 3600.0) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dist1_missing"] = out["dist1"].isna().astype(float)
    out["log_dist1"] = np.log1p(out["dist1"].fillna(0).clip(lower=0))
    for c in ["C1", "C2", "C13", "D1"]:
        out[f"log_{c}"] = np.log1p(out[c].fillna(0).clip(lower=0))
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].fillna("MISSING").astype(str)
    return out


def temporal_split(df: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    q_train = float(df["TransactionDT"].quantile(cfg.train_fraction))
    q_val = float(df["TransactionDT"].quantile(cfg.train_fraction + cfg.val_fraction))
    train_idx = np.flatnonzero(df["TransactionDT"].to_numpy() <= q_train)
    val_idx = np.flatnonzero((df["TransactionDT"].to_numpy() > q_train) & (df["TransactionDT"].to_numpy() <= q_val))
    test_idx = np.flatnonzero(df["TransactionDT"].to_numpy() > q_val)
    assert train_idx.max() < val_idx.min() < test_idx.min(), "La partición no es cronológica"
    return train_idx, val_idx, test_idx, {"train_cutoff": q_train, "val_cutoff": q_val}


def choose_training_targets(df: pd.DataFrame, train_idx: np.ndarray, cfg: Config) -> tuple[np.ndarray, dict[str, float]]:
    y = df["isFraud"].to_numpy(dtype=np.int8)
    positives = train_idx[y[train_idx] == 1]
    negatives = train_idx[y[train_idx] == 0]
    rng = np.random.default_rng(cfg.seed)
    max_neg = max(0, cfg.max_train_targets - len(positives))
    selected_neg = rng.choice(negatives, size=min(max_neg, len(negatives)), replace=False)
    selected = np.sort(np.concatenate([positives, selected_neg]))
    weights = {
        "positive": 1.0,
        "negative": float(len(negatives) / max(1, len(selected_neg))),
        "train_population": int(len(train_idx)),
        "train_selected": int(len(selected)),
    }
    return selected, weights


class Preprocessor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.numeric_medians: np.ndarray | None = None
        self.numeric_means: np.ndarray | None = None
        self.numeric_stds: np.ndarray | None = None
        self.category_maps: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, train_idx: np.ndarray) -> "Preprocessor":
        numeric = df.loc[train_idx, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
        self.numeric_medians = np.nanmedian(numeric, axis=0)
        numeric = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        self.numeric_means = numeric.mean(axis=0)
        self.numeric_stds = numeric.std(axis=0)
        self.numeric_stds[self.numeric_stds < 1e-7] = 1.0
        for c in CATEGORICAL_FEATURES:
            counts = df.loc[train_idx, c].value_counts()
            kept = counts[counts >= self.cfg.min_category_frequency].head(self.cfg.max_categories).index.astype(str)
            self.category_maps[c] = {value: i + 1 for i, value in enumerate(kept)}
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self.numeric_medians is not None and self.numeric_means is not None and self.numeric_stds is not None
        numeric = df[NUMERIC_FEATURES].to_numpy(dtype=np.float64)
        numeric = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        numeric = ((numeric - self.numeric_means) / self.numeric_stds).astype(np.float32)
        categorical = np.zeros((len(df), len(CATEGORICAL_FEATURES)), dtype=np.int32)
        for j, c in enumerate(CATEGORICAL_FEATURES):
            categorical[:, j] = df[c].map(self.category_maps[c]).fillna(0).astype(np.int32)
        return numeric, categorical

    @property
    def cardinalities(self) -> list[int]:
        return [len(self.category_maps[c]) + 1 for c in CATEGORICAL_FEATURES]


def build_sequences(
    df: pd.DataFrame,
    numeric: np.ndarray,
    categorical: np.ndarray,
    target_indices: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    target_indices = np.sort(np.asarray(target_indices, dtype=np.int64))
    slot_by_row = np.full(len(df), -1, dtype=np.int64)
    slot_by_row[target_indices] = np.arange(len(target_indices), dtype=np.int64)
    seq_num = np.zeros((len(target_indices), cfg.seq_len, numeric.shape[1]), dtype=np.float32)
    seq_cat = np.zeros((len(target_indices), cfg.seq_len, categorical.shape[1]), dtype=np.int32)
    lengths = np.zeros(len(target_indices), dtype=np.int64)
    history: dict[str, deque[tuple[np.ndarray, np.ndarray]]] = defaultdict(lambda: deque(maxlen=cfg.seq_len))

    keys = df["card_key"].to_numpy()
    for row in range(len(df)):
        h = history[keys[row]]
        h.append((numeric[row], categorical[row]))
        slot = slot_by_row[row]
        if slot >= 0:
            values = list(h)
            length = len(values)
            lengths[slot] = length
            seq_num[slot, :length] = np.stack([v[0] for v in values])
            seq_cat[slot, :length] = np.stack([v[1] for v in values])

    y = df.loc[target_indices, "isFraud"].to_numpy(dtype=np.int64)
    meta = df.loc[target_indices, ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "card6", "card_key"]].reset_index(drop=True)
    assert np.all(lengths >= 1) and np.all(lengths <= cfg.seq_len)
    assert np.array_equal(y, df.loc[target_indices, "isFraud"].to_numpy(dtype=np.int64))
    return seq_num, seq_cat, lengths, y, meta


def aggregate_features(seq_num: np.ndarray, seq_cat: np.ndarray, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, _, n_num = seq_num.shape
    n_cat = seq_cat.shape[2]
    continuous = np.zeros((n, n_num * 5 + 1 + n_cat), dtype=np.float32)
    current_cat = np.zeros((n, n_cat), dtype=np.int32)
    for i in range(n):
        length = int(lengths[i])
        values = seq_num[i, :length]
        cats = seq_cat[i, :length]
        continuous[i] = np.concatenate([
            values[-1], values.mean(axis=0), values.std(axis=0), values.min(axis=0), values.max(axis=0),
            np.array([length], dtype=np.float32),
            np.array([len(np.unique(cats[:, j])) for j in range(n_cat)], dtype=np.float32),
        ])
        current_cat[i] = cats[-1]
    all_features = np.concatenate([continuous, current_cat.astype(np.float32)], axis=1)
    categorical_mask = np.zeros(all_features.shape[1], dtype=bool)
    categorical_mask[-n_cat:] = True
    return continuous, all_features, categorical_mask


class SequenceDataset(Dataset):
    def __init__(self, seq_num, seq_cat, lengths, y, aggregates=None):
        self.seq_num = torch.from_numpy(seq_num).float()
        self.seq_cat = torch.from_numpy(seq_cat).long()
        self.lengths = torch.from_numpy(lengths).long()
        self.y = torch.from_numpy(y).float()
        self.aggregates = None if aggregates is None else torch.from_numpy(aggregates).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.aggregates is None:
            return self.seq_num[index], self.seq_cat[index], self.lengths[index], self.y[index]
        return self.seq_num[index], self.seq_cat[index], self.lengths[index], self.aggregates[index], self.y[index]


def embedding_dim(cardinality: int) -> int:
    return int(min(12, max(2, round(math.sqrt(cardinality)))))


class GRURiskModel(nn.Module):
    def __init__(self, n_numeric: int, cardinalities: list[int], hidden_size: int = 32, aggregate_dim: int = 0):
        super().__init__()
        self.aggregate_dim = aggregate_dim
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim(cardinality), padding_idx=0)
            for cardinality in cardinalities
        ])
        embedded_dim = sum(e.embedding_dim for e in self.embeddings)
        self.gru = nn.GRU(n_numeric + embedded_dim, hidden_size, batch_first=True)
        head_input = hidden_size + aggregate_dim
        self.head = nn.Sequential(nn.Linear(head_input, 24), nn.ReLU(), nn.Dropout(0.10), nn.Linear(24, 1))

    def forward(self, seq_num, seq_cat, lengths, aggregates=None):
        embedded = [emb(seq_cat[:, :, j]) for j, emb in enumerate(self.embeddings)]
        x = torch.cat([seq_num, *embedded], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        representation = hidden[-1]
        if self.aggregate_dim:
            representation = torch.cat([representation, aggregates], dim=1)
        return self.head(representation).squeeze(1)


def predict_torch(model: nn.Module, dataset: SequenceDataset, batch_size: int) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                seq_num, seq_cat, lengths, _ = batch
                logits = model(seq_num, seq_cat, lengths)
            else:
                seq_num, seq_cat, lengths, aggregates, _ = batch
                logits = model(seq_num, seq_cat, lengths, aggregates)
            scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def train_torch_model(
    model: nn.Module,
    train_dataset: SequenceDataset,
    val_dataset: SequenceDataset,
    y_train: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> tuple[nn.Module, list[dict[str, float]]]:
    set_seed(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_ap = -np.inf
    stale = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad()
            if len(batch) == 4:
                seq_num, seq_cat, lengths, yb = batch
                logits = model(seq_num, seq_cat, lengths)
            else:
                seq_num, seq_cat, lengths, aggregates, yb = batch
                logits = model(seq_num, seq_cat, lengths, aggregates)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item()) * len(yb)
            seen += len(yb)
        val_scores = predict_torch(model, val_dataset, cfg.batch_size)
        val_ap = float(average_precision_score(y_val, val_scores))
        history.append({"epoch": epoch, "loss": running / max(1, seen), "val_auc_pr": val_ap})
        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model, history


def choose_threshold(y: np.ndarray, scores: np.ndarray, cfg: Config) -> dict[str, float]:
    quantiles = np.linspace(0.0, 1.0, 501)
    candidates = np.unique(np.quantile(scores, quantiles))
    best = None
    curve = []
    for threshold in candidates:
        pred = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        cost = cfg.cost_fn_q * fn + cfg.cost_fp_q * fp
        curve.append((float(threshold), float(cost)))
        row = {"threshold": float(threshold), "cost": float(cost), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        if best is None or row["cost"] < best["cost"]:
            best = row
    assert best is not None
    best["curve"] = curve
    return best


def metrics_at_threshold(y: np.ndarray, scores: np.ndarray, threshold: float, cfg: Config) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cost = cfg.cost_fn_q * fn + cfg.cost_fp_q * fp
    return {
        "auc_pr": float(average_precision_score(y, scores)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost_q": float(cost),
        "cost_per_decision_q": float(cost / len(y)),
    }


def permute_history(seq_num: np.ndarray, seq_cat: np.ndarray, lengths: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    out_num = seq_num.copy()
    out_cat = seq_cat.copy()
    for i, length in enumerate(lengths):
        history_len = int(length) - 1
        if history_len > 1:
            order = rng.permutation(history_len)
            out_num[i, :history_len] = seq_num[i, order]
            out_cat[i, :history_len] = seq_cat[i, order]
    return out_num, out_cat


def truncate_history(seq_num: np.ndarray, seq_cat: np.ndarray, lengths: np.ndarray, keep: int = 3):
    out_num = np.zeros_like(seq_num)
    out_cat = np.zeros_like(seq_cat)
    out_lengths = np.minimum(lengths, keep).astype(np.int64)
    for i, (length, new_length) in enumerate(zip(lengths, out_lengths)):
        start = int(length - new_length)
        out_num[i, :new_length] = seq_num[i, start:int(length)]
        out_cat[i, :new_length] = seq_cat[i, start:int(length)]
    return out_num, out_cat, out_lengths


def plot_temporal_data(df: pd.DataFrame, split_info: dict[str, float], destination: Path) -> None:
    days = (df["TransactionDT"] - df["TransactionDT"].min()) / 86400
    bins = pd.cut(days, bins=18)
    summary = df.assign(time_bin=bins).groupby("time_bin", observed=True).agg(
        transactions=("isFraud", "size"), fraud_rate=("isFraud", "mean")
    )
    x = np.arange(len(summary))
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(x, summary["transactions"], color="#8ecae6", alpha=.75, label="Transacciones")
    ax1.set_ylabel("Transacciones")
    ax1.set_xlabel("Tiempo cronológico (18 intervalos)")
    ax2 = ax1.twinx()
    ax2.plot(x, summary["fraud_rate"] * 100, color="#d1495b", marker="o", label="Tasa de fraude")
    ax2.set_ylabel("Fraude (%)")
    ax1.set_title("Volumen y prevalencia a través del tiempo")
    fig.tight_layout()
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], destination: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    colors = {"A": "#457b9d", "B": "#2a9d8f", "C": "#e76f51"}
    for name, values in scores.items():
        precision, recall, _ = precision_recall_curve(y, values)
        ap = average_precision_score(y, values)
        ax.plot(recall, precision, lw=2.2, color=colors[name], label=f"{name} · AUC-PR={ap:.3f}")
    ax.axhline(y.mean(), color="#6c757d", ls="--", lw=1.2, label=f"Prevalencia={y.mean():.3f}")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Comparación común en prueba cronológica")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_falsification(original_ap: float, permutation_aps: list[float], truncated_ap: float, destination: Path) -> None:
    values = [original_ap, float(np.mean(permutation_aps)), truncated_ap]
    errors = [0, float(np.std(permutation_aps)), 0]
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    bars = ax.bar(["Secuencia original", "Historia permutada", "Últimos 3 eventos"], values,
                  yerr=errors, color=["#2a9d8f", "#d1495b", "#f4a261"], capsize=5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .01, f"{value:.3f}", ha="center", fontweight="bold")
    ax.set_ylabel("AUC-PR"); ax.set_title("Pruebas de falsificación del valor del orden")
    ax.set_ylim(0, max(values) * 1.18); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_cost_curve(threshold_info: dict[str, float], destination: Path) -> None:
    curve = np.asarray(threshold_info["curve"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot(curve[:, 0], curve[:, 1] / 1e6, color="#184e77", lw=2)
    ax.axvline(threshold_info["threshold"], color="#d1495b", ls="--", label=f"τ={threshold_info['threshold']:.3f}")
    ax.scatter([threshold_info["threshold"]], [threshold_info["cost"] / 1e6], color="#d1495b", zorder=3)
    ax.set(xlabel="Umbral", ylabel="Costo de validación (millones de Q)", title="Selección económica del umbral")
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(destination, dpi=180, bbox_inches="tight"); plt.close(fig)


def error_breakdown(meta: pd.DataFrame, y: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    frame = meta.copy()
    frame["y"] = y
    frame["pred"] = (scores >= threshold).astype(int)
    frame["error"] = np.select(
        [(frame.y == 1) & (frame.pred == 0), (frame.y == 0) & (frame.pred == 1)],
        ["FN", "FP"], default="Correcto"
    )
    frame["amount_band"] = pd.qcut(frame["TransactionAmt"], q=4, duplicates="drop").astype(str)
    return frame.groupby(["error", "ProductCD", "amount_band"], observed=True).size().rename("n").reset_index().sort_values("n", ascending=False)


def run_experiment(cfg: Config | None = None, force: bool = False) -> dict[str, Any]:
    cfg = cfg or Config()
    ensure_directories()
    set_seed(cfg.seed)
    p = paths()
    result_path = p["artifacts"] / "resultados.json"
    if result_path.exists() and not force:
        return json.loads(result_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    df = add_features(load_raw_data())
    train_population, val_idx, test_idx, split_cutoffs = temporal_split(df, cfg)
    train_idx, sampling_weights = choose_training_targets(df, train_population, cfg)
    all_targets = np.concatenate([train_idx, val_idx, test_idx])

    preprocessor = Preprocessor(cfg).fit(df, train_population)
    numeric, categorical = preprocessor.transform(df)
    seq_num, seq_cat, lengths, y_all, meta_all = build_sequences(df, numeric, categorical, all_targets, cfg)

    n_train, n_val = len(train_idx), len(val_idx)
    slices = {
        "train": slice(0, n_train),
        "val": slice(n_train, n_train + n_val),
        "test": slice(n_train + n_val, len(all_targets)),
    }
    arrays = {}
    for name, sl in slices.items():
        arrays[name] = {
            "seq_num": seq_num[sl], "seq_cat": seq_cat[sl], "lengths": lengths[sl],
            "y": y_all[sl], "meta": meta_all.iloc[sl].reset_index(drop=True),
        }
        cont, all_agg, cat_mask = aggregate_features(arrays[name]["seq_num"], arrays[name]["seq_cat"], arrays[name]["lengths"])
        arrays[name]["agg_cont"] = cont
        arrays[name]["agg_all"] = all_agg

    # Pieza A: línea base competitiva y ajena al orden.
    y_train = arrays["train"]["y"]
    sample_weight = np.where(y_train == 0, sampling_weights["negative"], sampling_weights["positive"])
    model_a = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=220, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=0.15, categorical_features=cat_mask, random_state=cfg.seed,
    )
    model_a.fit(arrays["train"]["agg_all"], y_train, sample_weight=sample_weight)
    scores_a_val = model_a.predict_proba(arrays["val"]["agg_all"])[:, 1]

    # Pieza B: GRU secuencial.
    train_ds_b = SequenceDataset(arrays["train"]["seq_num"], arrays["train"]["seq_cat"], arrays["train"]["lengths"], y_train)
    val_ds_b = SequenceDataset(arrays["val"]["seq_num"], arrays["val"]["seq_cat"], arrays["val"]["lengths"], arrays["val"]["y"])
    model_b = GRURiskModel(len(NUMERIC_FEATURES), preprocessor.cardinalities, cfg.hidden_size)
    model_b, history_b = train_torch_model(model_b, train_ds_b, val_ds_b, y_train, arrays["val"]["y"], cfg)
    scores_b_val = predict_torch(model_b, val_ds_b, cfg.batch_size)

    # Pieza C: apuesta híbrida predeclarada, GRU + agregados.
    aggregate_scaler = StandardScaler().fit(arrays["train"]["agg_cont"])
    for name in arrays:
        arrays[name]["agg_scaled"] = aggregate_scaler.transform(arrays[name]["agg_cont"]).astype(np.float32)
    train_ds_c = SequenceDataset(arrays["train"]["seq_num"], arrays["train"]["seq_cat"], arrays["train"]["lengths"], y_train, arrays["train"]["agg_scaled"])
    val_ds_c = SequenceDataset(arrays["val"]["seq_num"], arrays["val"]["seq_cat"], arrays["val"]["lengths"], arrays["val"]["y"], arrays["val"]["agg_scaled"])
    model_c = GRURiskModel(len(NUMERIC_FEATURES), preprocessor.cardinalities, cfg.hidden_size, arrays["train"]["agg_scaled"].shape[1])
    model_c, history_c = train_torch_model(model_c, train_ds_c, val_ds_c, y_train, arrays["val"]["y"], cfg)
    scores_c_val = predict_torch(model_c, val_ds_c, cfg.batch_size)

    val_scores = {"A": scores_a_val, "B": scores_b_val, "C": scores_c_val}
    thresholds = {name: choose_threshold(arrays["val"]["y"], score, cfg) for name, score in val_scores.items()}
    validation = {name: metrics_at_threshold(arrays["val"]["y"], score, thresholds[name]["threshold"], cfg) for name, score in val_scores.items()}

    ap_gain = validation["C"]["auc_pr"] - validation["B"]["auc_pr"]
    cost_reduction = (validation["B"]["cost_q"] - validation["C"]["cost_q"]) / max(1, validation["B"]["cost_q"])
    hypothesis_success = ap_gain >= cfg.hypothesis_ap_gain and cost_reduction >= cfg.hypothesis_cost_reduction
    candidate = "C" if hypothesis_success else min(validation, key=lambda name: validation[name]["cost_q"])

    # El conjunto final se abre una vez, después de congelar modelos, umbrales y candidato.
    test_ds_b = SequenceDataset(arrays["test"]["seq_num"], arrays["test"]["seq_cat"], arrays["test"]["lengths"], arrays["test"]["y"])
    test_ds_c = SequenceDataset(arrays["test"]["seq_num"], arrays["test"]["seq_cat"], arrays["test"]["lengths"], arrays["test"]["y"], arrays["test"]["agg_scaled"])
    test_scores = {
        "A": model_a.predict_proba(arrays["test"]["agg_all"])[:, 1],
        "B": predict_torch(model_b, test_ds_b, cfg.batch_size),
        "C": predict_torch(model_c, test_ds_c, cfg.batch_size),
    }
    test_metrics = {
        name: metrics_at_threshold(arrays["test"]["y"], score, thresholds[name]["threshold"], cfg)
        for name, score in test_scores.items()
    }

    # Prueba obligatoria: permutar solo la historia y conservar la transacción objetivo al final.
    permutation_aps = []
    permutation_metrics = []
    for seed in [11, 23, 37, 41, 59]:
        perm_num, perm_cat = permute_history(arrays["test"]["seq_num"], arrays["test"]["seq_cat"], arrays["test"]["lengths"], seed)
        perm_ds = SequenceDataset(perm_num, perm_cat, arrays["test"]["lengths"], arrays["test"]["y"])
        perm_scores = predict_torch(model_b, perm_ds, cfg.batch_size)
        permutation_aps.append(float(average_precision_score(arrays["test"]["y"], perm_scores)))
        permutation_metrics.append(metrics_at_threshold(arrays["test"]["y"], perm_scores, thresholds["B"]["threshold"], cfg))

    # Segunda falsificación elegida: conservar únicamente los tres eventos más recientes.
    trunc_num, trunc_cat, trunc_lengths = truncate_history(arrays["test"]["seq_num"], arrays["test"]["seq_cat"], arrays["test"]["lengths"], 3)
    trunc_ds = SequenceDataset(trunc_num, trunc_cat, trunc_lengths, arrays["test"]["y"])
    trunc_scores = predict_torch(model_b, trunc_ds, cfg.batch_size)
    truncated_metrics = metrics_at_threshold(arrays["test"]["y"], trunc_scores, thresholds["B"]["threshold"], cfg)

    monthly_decisions = cfg.cards_bank * cfg.transactions_per_card_month
    economics = {}
    for name, metrics in test_metrics.items():
        monthly_cost = metrics["cost_per_decision_q"] * monthly_decisions
        economics[name] = {
            "monthly_decisions": monthly_decisions,
            "monthly_cost_q": monthly_cost,
            "cost_per_100k_q": metrics["cost_per_decision_q"] * 100_000,
            "monthly_savings_vs_A_q": test_metrics["A"]["cost_per_decision_q"] * monthly_decisions - monthly_cost,
        }

    breakdown = error_breakdown(arrays["test"]["meta"], arrays["test"]["y"], test_scores[candidate], thresholds[candidate]["threshold"])
    breakdown.to_csv(p["artifacts"] / "patrones_error.csv", index=False)

    # Artefactos reproducibles.
    joblib.dump(model_a, p["artifacts"] / "modelo_A_histgradientboosting.joblib")
    joblib.dump(preprocessor, p["artifacts"] / "preprocesamiento.joblib")
    joblib.dump(aggregate_scaler, p["artifacts"] / "escalador_agregados.joblib")
    torch.save({"state_dict": model_b.state_dict(), "cardinalities": preprocessor.cardinalities, "config": asdict(cfg)}, p["artifacts"] / "modelo_B_gru.pt")
    torch.save({"state_dict": model_c.state_dict(), "cardinalities": preprocessor.cardinalities, "aggregate_dim": arrays["train"]["agg_scaled"].shape[1], "config": asdict(cfg)}, p["artifacts"] / "modelo_C_hibrido.pt")
    candidate_source = {"A": "modelo_A_histgradientboosting.joblib", "B": "modelo_B_gru.pt", "C": "modelo_C_hibrido.pt"}[candidate]
    shutil.copy2(p["artifacts"] / candidate_source, p["artifacts"] / f"modelo_candidato_{candidate}{Path(candidate_source).suffix}")

    threshold_export = {name: {k: v for k, v in info.items() if k != "curve"} for name, info in thresholds.items()}
    write_json(p["artifacts"] / "umbrales.json", threshold_export)
    write_json(p["artifacts"] / "configuracion.json", asdict(cfg))
    write_json(p["artifacts"] / "esquema_entrada.json", {
        "entity_proxy": CARD_KEY_COLUMNS, "sequence_length": cfg.seq_len,
        "numeric_event_features": NUMERIC_FEATURES, "categorical_event_features": CATEGORICAL_FEATURES,
        "output": {"risk_score": "float in [0,1]", "threshold": threshold_export[candidate]["threshold"]},
    })

    plot_temporal_data(df, split_cutoffs, p["figures"] / "01_integridad_temporal.png")
    plot_pr_curves(arrays["test"]["y"], test_scores, p["figures"] / "02_curvas_precision_recall.png")
    plot_falsification(test_metrics["B"]["auc_pr"], permutation_aps, truncated_metrics["auc_pr"], p["figures"] / "03_falsificaciones_orden.png")
    plot_cost_curve(thresholds[candidate], p["figures"] / "04_curva_costo_umbral.png")

    results = {
        "project": "Monitoreo transaccional: detectar lo que el orden revela",
        "dataset": {
            "source": "Kaggle IEEE-CIS Fraud Detection / Vesta Corporation",
            "rows": len(df), "frauds": int(df["isFraud"].sum()), "fraud_rate": float(df["isFraud"].mean()),
            "time_span_days": float((df["TransactionDT"].max() - df["TransactionDT"].min()) / 86400),
            "entity_proxy": CARD_KEY_COLUMNS, "identity_is_approximate": True,
        },
        "splits": {
            "train_population": {"n": len(train_population), "fraud_rate": float(df.loc[train_population, "isFraud"].mean())},
            "train_modeling": {"n": len(train_idx), "fraud_rate": float(df.loc[train_idx, "isFraud"].mean())},
            "validation": {"n": len(val_idx), "fraud_rate": float(df.loc[val_idx, "isFraud"].mean())},
            "test": {"n": len(test_idx), "fraud_rate": float(df.loc[test_idx, "isFraud"].mean())},
            "cutoffs": split_cutoffs,
        },
        "hypothesis": {
            "statement": "Creemos que fusionar la representación secuencial de la GRU con variables agregadas mejorará el AUC-PR porque la evidencia temporal y global es complementaria. Lo consideraremos útil si aumenta AUC-PR al menos 0.01 y reduce el costo al menos 5% en validación.",
            "ap_gain": ap_gain, "cost_reduction": cost_reduction, "success": hypothesis_success,
        },
        "validation": validation,
        "thresholds": threshold_export,
        "candidate": candidate,
        "test": test_metrics,
        "falsification": {
            "original_B": test_metrics["B"],
            "permutation_auc_pr": permutation_aps,
            "permutation_mean_auc_pr": float(np.mean(permutation_aps)),
            "permutation_std_auc_pr": float(np.std(permutation_aps)),
            "permutation_metrics": permutation_metrics,
            "truncated_to_3": truncated_metrics,
            "order_auc_pr_drop": float(test_metrics["B"]["auc_pr"] - np.mean(permutation_aps)),
        },
        "economics": economics,
        "training": {"B": history_b, "C": history_c, "seconds_total": time.perf_counter() - started},
        "reproducibility": {"config": asdict(cfg), "sampling": sampling_weights},
    }
    write_json(result_path, results)
    return results


if __name__ == "__main__":
    result = run_experiment(force=True)
    print(json.dumps(json_ready({
        "candidate": result["candidate"],
        "validation": result["validation"],
        "test": result["test"],
        "falsification": result["falsification"],
        "economics": result["economics"],
    }), ensure_ascii=False, indent=2))
