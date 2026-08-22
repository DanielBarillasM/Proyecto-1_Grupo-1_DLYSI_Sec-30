from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artefactos/v2/resultados_v2.json"


def block_ci(
    y: np.ndarray,
    score: np.ndarray,
    seed: int,
    repetitions: int = 300,
    blocks: int = 24,
) -> dict:
    rng = np.random.default_rng(seed)
    pieces = np.array_split(np.arange(len(y)), blocks)
    estimates = []
    for _ in range(repetitions):
        idx = np.concatenate([pieces[i] for i in rng.integers(0, blocks, size=blocks)])
        if 0 < y[idx].sum() < len(idx):
            estimates.append(average_precision_score(y[idx], score[idx]))
    return {
        "estimacion": float(average_precision_score(y, score)),
        "li95": float(np.quantile(estimates, 0.025)),
        "ls95": float(np.quantile(estimates, 0.975)),
        "metodo": f"bootstrap por {blocks} bloques temporales, {len(estimates)} réplicas",
    }


def top_k(y: np.ndarray, score: np.ndarray) -> list[dict]:
    order = np.argsort(-score)
    rows = []
    for rate in (0.001, 0.005, 0.01, 0.02):
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


def main() -> None:
    result = json.loads(RESULTS.read_text(encoding="utf-8"))
    predictions = pd.read_csv(
        ROOT / "artefactos/v2/predicciones_benchmark_historico.csv"
    )
    y = predictions["y"].to_numpy(dtype=np.int8)
    score = predictions["score_tabular"].to_numpy(dtype=float)

    result["ensamble_v2"]["intervalo_auc_pr_benchmark"] = result[
        "intervalo_auc_pr_benchmark"
    ]
    result["ensamble_v2"]["metricas_top_k"] = result["metricas_top_k"]
    result["intervalo_auc_pr_benchmark"] = block_ci(
        y, score, result["configuracion"]["seed"]
    )
    result["metricas_top_k"] = top_k(y, score)
    result["recomendacion_final"] = {
        "modelo": "LightGBM_corr_pruned_V2",
        "motivo": "Mayor AUC-PR walk-forward, mejor AUC-PR y menor costo que A-V1 en benchmark histórico; el ensamble y PCA no mejoraron.",
        "condicion": "Benchmark reutilizado: requiere cohorte temporal nueva para confirmación.",
    }
    RESULTS.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(json.dumps(result["recomendacion_final"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
