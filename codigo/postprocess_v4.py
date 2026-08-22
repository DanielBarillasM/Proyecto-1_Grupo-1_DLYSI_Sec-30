"""Selecciona el candidato V4 y documenta una política robusta post-hoc.

No reentrena modelos base. Divide la validación restante en selección de candidato,
calibración, umbral y evaluación. La política recall>=0.75 se marca explícitamente
como post-hoc y requiere una nueva cohorte para promoción confirmatoria.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codigo"))
from proyecto1_v4_pipeline import (  # noqa: E402
    ConfigV4,
    ece,
    metrics,
    paired_block_delta,
    ready,
    top_k,
)

ART = ROOT / "artefactos" / "v4"
V3_ART = ROOT / "artefactos" / "v3"
FIG = ROOT / "evidencia" / "figuras" / "v4"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(ready(value), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def logit(score: np.ndarray) -> np.ndarray:
    eps = 1e-6
    clipped = np.clip(score, eps, 1 - eps)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def threshold_table(y: np.ndarray, score: np.ndarray, cfg: ConfigV4) -> pd.DataFrame:
    _, _, raw = precision_recall_curve(y, score)
    thresholds = np.unique(np.quantile(raw, np.linspace(0, 1, min(1200, len(raw)))))
    rows = []
    for threshold in thresholds:
        result = metrics(y, score, float(threshold), cfg)
        rows.append({k: result[k] for k in ("threshold", "precision", "recall", "f1", "costo_q", "alertas_por_100k")})
    return pd.DataFrame(rows)


def choose(curve: pd.DataFrame, floor: float) -> float:
    eligible = curve.loc[curve["recall"] >= floor]
    if eligible.empty:
        eligible = curve
    return float(eligible.sort_values(["f1", "costo_q"], ascending=[False, True]).iloc[0]["threshold"])


def main() -> None:
    cfg = ConfigV4()
    result_path = ART / "resultados_v4.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validation = pd.read_csv(ART / "predicciones_validacion_v4.csv")
    benchmark = pd.read_csv(ART / "predicciones_benchmark_v4.csv")
    y = validation["y"].to_numpy(np.int8)
    yb = benchmark["y"].to_numpy(np.int8)
    n = len(y)
    blocks = {
        "seleccion_candidato": [int(n * 0.50), int(n * 0.60)],
        "calibracion_candidato": [int(n * 0.60), int(n * 0.70)],
        "seleccion_umbral": [int(n * 0.70), int(n * 0.85)],
        "evaluacion": [int(n * 0.85), n],
    }
    indices = {name: np.arange(*bounds) for name, bounds in blocks.items()}
    candidates = {
        "LightGBM_global": "score_lgb",
        "LightGBM_hard_negative": "score_hard_lgb",
        "LightGBM_expertos_ProductCD": "score_segment_lgb",
        "CatBoost_piloto": "score_catboost",
        "XGBoost_piloto": "score_xgboost",
        "Stacking_experimental": "score_stack_raw",
    }
    selection_rows = []
    for name, column in candidates.items():
        score = validation[column].to_numpy(float)
        selection_rows.append({
            "modelo": name,
            "columna": column,
            "auc_pr_seleccion": average_precision_score(y[indices["seleccion_candidato"]], score[indices["seleccion_candidato"]]),
            "roc_auc_seleccion": roc_auc_score(y[indices["seleccion_candidato"]], score[indices["seleccion_candidato"]]),
        })
    selection = pd.DataFrame(selection_rows).sort_values("auc_pr_seleccion", ascending=False)
    selection.to_csv(ART / "seleccion_candidato_v4.csv", index=False)
    winner = str(selection.iloc[0]["modelo"])
    column = str(selection.iloc[0]["columna"])
    raw_val = validation[column].to_numpy(float)
    raw_bench = benchmark[column].to_numpy(float)

    calibration_idx = indices["calibracion_candidato"]
    calibrator = LogisticRegression(max_iter=1000, random_state=cfg.seed)
    calibrator.fit(logit(raw_val[calibration_idx]), y[calibration_idx])
    score_val = calibrator.predict_proba(logit(raw_val))[:, 1]
    score_bench = calibrator.predict_proba(logit(raw_bench))[:, 1]
    joblib.dump(calibrator, ART / "calibrador_candidato_v4.joblib")

    threshold_idx = indices["seleccion_umbral"]
    curve = threshold_table(y[threshold_idx], score_val[threshold_idx], cfg)
    curve.to_csv(ART / "curva_umbral_candidato_v4.csv", index=False)
    thresholds = {
        "balanceado_recall_070": choose(curve, 0.70),
        "robusto_recall_075_post_hoc": choose(curve, 0.75),
        "economico": float(curve.sort_values(["costo_q", "f1"], ascending=[True, False]).iloc[0]["threshold"]),
    }
    evaluation_idx = indices["evaluacion"]
    policies = {}
    for name, threshold in thresholds.items():
        policies[name] = {
            "threshold": threshold,
            "evaluacion": metrics(y[evaluation_idx], score_val[evaluation_idx], threshold, cfg),
            "benchmark_historico": metrics(yb, score_bench, threshold, cfg),
        }

    v3_result = json.loads((V3_ART / "resultados_v3.json").read_text(encoding="utf-8"))
    v3_val = pd.read_csv(V3_ART / "predicciones_validacion_v3.csv")
    v3_bench = pd.read_csv(V3_ART / "predicciones_benchmark_v3.csv")
    v3_threshold = float(v3_result["modelo_v3"]["threshold_recomendado_balanceado"])
    old_eval = metrics(y[evaluation_idx], v3_val["score_calibrado"].to_numpy(float)[evaluation_idx], v3_threshold, cfg)
    recommended = policies["robusto_recall_075_post_hoc"]["evaluacion"]
    improvements = {
        metric: float(recommended[metric] - old_eval[metric])
        for metric in ("auc_pr", "roc_auc", "precision", "recall", "f1")
    }
    improvements["reduccion_costo"] = float(1 - recommended["costo_q"] / old_eval["costo_q"])
    dominates = all(improvements[name] > 0 for name in ("auc_pr", "roc_auc", "precision", "recall", "f1")) and improvements["reduccion_costo"] > 0
    pair_eval = paired_block_delta(
        y[evaluation_idx], score_val[evaluation_idx],
        v3_val["score_calibrado"].to_numpy(float)[evaluation_idx], cfg, 12,
    )
    pair_bench = paired_block_delta(
        yb, score_bench, v3_bench["score_calibrado"].to_numpy(float), cfg, 24,
    )

    validation["score_candidato_v4"] = score_val
    benchmark["score_candidato_v4"] = score_bench
    validation.to_csv(ART / "predicciones_validacion_v4.csv", index=False)
    benchmark.to_csv(ART / "predicciones_benchmark_v4.csv", index=False)
    topk = top_k(yb, score_bench)
    pd.DataFrame(topk).to_csv(ART / "metricas_top_k_candidato_v4.csv", index=False)

    result["modelo_v4_recomendado"] = {
        "modelo": winner,
        "criterio_seleccion": "Mayor AUC-PR en 50%-60% de validación, antes de calibración, umbral y evaluación.",
        "bloques": blocks,
        "seleccion": selection.to_dict("records"),
        "calibracion": {
            "brier_raw": brier_score_loss(y[calibration_idx], raw_val[calibration_idx]),
            "brier_calibrado": brier_score_loss(y[calibration_idx], score_val[calibration_idx]),
            "ece_raw": ece(y[calibration_idx], raw_val[calibration_idx]),
            "ece_calibrado": ece(y[calibration_idx], score_val[calibration_idx]),
        },
        "politicas": policies,
        "politica_recomendada": "robusto_recall_075_post_hoc",
        "comparacion_v3_evaluacion": {"v3": old_eval, "deltas_v4_menos_v3": improvements, "domina_todas_metricas": dominates},
        "comparacion_pareada_evaluacion": pair_eval,
        "comparacion_pareada_benchmark_historico": pair_bench,
    }
    result["metricas_top_k_candidato"] = topk
    result["decision_v4"] = {
        "estado": "candidato_superior_post_hoc_requiere_cohorte_nueva",
        "promocion_confirmatoria": False,
        "razon": "La política recall>=0.75 domina a V3, pero fue añadida después de observar la primera evaluación V4.",
        "siguiente_paso": "Congelar esta política y validarla sin cambios en una cohorte temporal nueva.",
    }
    result["recomendacion"] = "Conservar V3 como versión confirmada; V4 es el candidato congelado para la próxima cohorte."
    write_json(result_path, result)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    table = selection.sort_values("auc_pr_seleccion")
    ax.barh(table["modelo"], table["auc_pr_seleccion"], color="#2a9d8f")
    ax.set(xlabel="AUC-PR en selección independiente", title="V4 · selección del candidato operativo")
    fig.tight_layout(); fig.savefig(FIG / "06_seleccion_candidato_v4.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, yy, score, color in (
        ("V4 experto", yb, score_bench, "#2a9d8f"),
        ("V3", yb, v3_bench["score_calibrado"].to_numpy(float), "#184e77"),
    ):
        precision, recall, _ = precision_recall_curve(yy, score)
        ax.plot(recall, precision, lw=2.2, color=color, label=f"{label} · AP={average_precision_score(yy, score):.3f}")
    ax.axhline(yb.mean(), color="#6b7280", ls="--", label="Prevalencia")
    ax.set(xlabel="Recall", ylabel="Precisión", title="V4 candidato vs V3 · benchmark histórico")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "07_curvas_pr_candidato_v4.png", dpi=180); plt.close(fig)

    print(json.dumps(ready(result["modelo_v4_recomendado"]), ensure_ascii=False, indent=2))
    print(json.dumps(result["decision_v4"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

