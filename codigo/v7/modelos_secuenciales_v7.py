"""Arquitecturas secuenciales autocontenidas de Proyecto 1 V7.

Este módulo conserva en V7 la implementación verificable de B (GRU/TCN causal)
y D (encoder--decoder). Los checkpoints usados por el experimento comparativo se
mantienen congelados para no confundir una mejora tabular con un reentrenamiento
del control; las clases y rutinas de entrenamiento quedan aquí para reproducirlos.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset


@dataclass(frozen=True)
class SequenceConfigV7:
    seed: int = 2026
    sequence_length: int = 32
    batch_size: int = 4096
    epochs: int = 6
    patience: int = 2
    hidden_size: int = 96
    event_size: int = 128
    dropout: float = 0.18
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4


def set_sequence_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IndexedSequenceDataset(Dataset):
    """Materializa en cada batch únicamente las historias solicitadas."""

    def __init__(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        sequences: np.ndarray,
        lengths: np.ndarray,
        target: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        self.numeric = numeric
        self.categorical = categorical
        self.sequences = sequences
        self.lengths = lengths
        self.target = target
        self.rows = np.asarray(rows, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> tuple[int, float]:
        row = int(self.rows[item])
        return row, float(self.target[row])

    def collate(self, batch):
        rows = np.fromiter((item[0] for item in batch), dtype=np.int64, count=len(batch))
        target = np.fromiter((item[1] for item in batch), dtype=np.float32, count=len(batch))
        indices = self.sequences[rows].copy()
        padding = indices < 0
        indices[padding] = 0
        numeric = self.numeric[indices]
        categorical = self.categorical[indices].astype(np.int64, copy=False)
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
    """Modelo B1: codifica cada evento y resume la historia con una GRU."""

    def __init__(self, numeric_dim: int, cardinalities: list[int], cfg: SequenceConfigV7):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim(cardinality), padding_idx=0)
            for cardinality in cardinalities
        )
        input_dim = numeric_dim + sum(embedding_dim(value) for value in cardinalities)
        self.event = nn.Sequential(
            nn.Linear(input_dim, cfg.event_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.gru = nn.GRU(cfg.event_size, cfg.hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, :, column])
            for column, embedding in enumerate(self.embeddings)
        ]
        events = self.event(torch.cat([numeric, *embedded], dim=-1))
        packed = pack_padded_sequence(
            events,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.head(hidden[-1]).squeeze(1)


class CausalResidualBlock(nn.Module):
    """Convolución causal: el recorte impide observar eventos posteriores."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=self.padding,
        )
        self.norm = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        transformed = self.conv(values)
        if self.padding:
            transformed = transformed[:, :, :-self.padding]
        transformed = self.dropout(torch.relu(self.norm(transformed)))
        return torch.relu(residual + transformed)


class TCNRiskModel(nn.Module):
    """Modelo B2: red convolucional temporal causal con dilataciones."""

    def __init__(self, numeric_dim: int, cardinalities: list[int], cfg: SequenceConfigV7):
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, embedding_dim(cardinality), padding_idx=0)
            for cardinality in cardinalities
        )
        input_dim = numeric_dim + sum(embedding_dim(value) for value in cardinalities)
        self.event = nn.Sequential(
            nn.Linear(input_dim, cfg.event_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        self.blocks = nn.Sequential(
            CausalResidualBlock(cfg.event_size, dilation=1, dropout=cfg.dropout),
            CausalResidualBlock(cfg.event_size, dilation=2, dropout=cfg.dropout),
            CausalResidualBlock(cfg.event_size, dilation=4, dropout=cfg.dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.event_size),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.event_size, 1),
        )

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, :, column])
            for column, embedding in enumerate(self.embeddings)
        ]
        events = self.event(torch.cat([numeric, *embedded], dim=-1))
        encoded = self.blocks(events.transpose(1, 2)).transpose(1, 2)
        batch = torch.arange(encoded.size(0), device=encoded.device)
        last = encoded[batch, lengths.to(encoded.device) - 1]
        return self.head(last).squeeze(1)


class TransactionAutoencoder(nn.Module):
    """Modelo D: encoder--decoder tabular ajustado solo con clase legítima."""

    def __init__(self, input_dim: int):
        super().__init__()
        hidden = min(256, max(64, input_dim * 2))
        bottleneck = min(48, max(16, input_dim // 3))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def train_sequence_model(
    model: nn.Module,
    train_data: IndexedSequenceDataset,
    early_data: IndexedSequenceDataset,
    cfg: SequenceConfigV7,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Entrena B con BCE ponderada, AdamW, clipping y early stopping por AP."""

    set_sequence_seed(cfg.seed)
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=train_data.collate,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    early_loader = DataLoader(
        early_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=early_data.collate,
    )
    positives = float(train_data.target[train_data.rows].sum())
    negatives = float(len(train_data.rows) - positives)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / max(1.0, positives)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    best_state = None
    best_ap = -np.inf
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for numeric, categorical, lengths, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(numeric, categorical, lengths)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        score = predict_sequence_model(model, early_loader)
        early_target = early_data.target[early_data.rows]
        early_ap = float(average_precision_score(early_target, score))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "early_ap": early_ap})
        if early_ap > best_ap + 1e-5:
            best_ap = early_ap
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.inference_mode()
def predict_sequence_model(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    predictions = []
    for numeric, categorical, lengths, _ in loader:
        predictions.append(torch.sigmoid(model(numeric, categorical, lengths)).cpu().numpy())
    return np.concatenate(predictions)


def train_autoencoder(
    model: TransactionAutoencoder,
    train_legitimate: np.ndarray,
    early_legitimate: np.ndarray,
    seed: int = 2026,
    epochs: int = 8,
    batch_size: int = 4096,
) -> tuple[TransactionAutoencoder, list[dict[str, float]]]:
    """Entrena D únicamente con isFraud=0 y selecciona por reconstrucción legítima."""

    set_sequence_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_legitimate.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    early = torch.from_numpy(early_legitimate.astype(np.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    best_state = None
    best_loss = np.inf
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for (values,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), values)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            early_loss = float(criterion(model(early), early))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "early_loss": early_loss})
        if early_loss < best_loss:
            best_loss = early_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.inference_mode()
def reconstruction_error(
    model: TransactionAutoencoder,
    values: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(values.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
    )
    errors = []
    for (batch,) in loader:
        reconstructed = model(batch)
        errors.append(torch.mean((batch - reconstructed) ** 2, dim=1).cpu().numpy())
    return np.concatenate(errors)


def sequence_variant(
    sequences: np.ndarray,
    lengths: np.ndarray,
    rows: np.ndarray,
    keep: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Permuta antecedentes o recorta historia sin mover el objetivo final."""

    variant = sequences.copy()
    new_lengths = lengths.copy()
    generator = np.random.default_rng(seed)
    for row in rows:
        length = int(lengths[row])
        values = sequences[row, :length].copy()
        if keep is not None and length > keep:
            values = values[-keep:]
        if seed is not None and len(values) > 2:
            values[:-1] = values[generator.permutation(len(values) - 1)]
        variant[row] = -1
        variant[row, : len(values)] = values
        new_lengths[row] = len(values)
    return variant, new_lengths
