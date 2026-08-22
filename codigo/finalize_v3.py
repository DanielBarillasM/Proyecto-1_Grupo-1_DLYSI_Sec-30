from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codigo"))
from dataset_v3_support import add_causal_features, load_all, temporal_boundaries
from proyecto1_v3_pipeline import ConfigV3, metricas, segment_metrics

ART = ROOT / "artefactos" / "v3"
RESULT = ART / "resultados_v3.json"


def main() -> None:
    cfg = ConfigV3()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    curve = pd.read_csv(ART / "curva_umbral_v3.csv")
    eligible = curve.loc[curve["recall"] >= cfg.recall_floor]
    if eligible.empty:
        eligible = curve
    balanced = float(
        eligible.sort_values(["f1", "costo_q"], ascending=[False, True]).iloc[0]["threshold"]
    )
    economic = float(curve.sort_values(["costo_q", "threshold"]).iloc[0]["threshold"])

    validation = pd.read_csv(ART / "predicciones_validacion_v3.csv")
    benchmark = pd.read_csv(ART / "predicciones_benchmark_v3.csv")
    threshold_slice = slice(int(len(validation) * 0.70), len(validation))
    yv = validation["y"].to_numpy(np.int8)
    sv = validation["score_calibrado"].to_numpy(float)
    yb = benchmark["y"].to_numpy(np.int8)
    sb = benchmark["score_calibrado"].to_numpy(float)
    model = result["modelo_v3"]
    model["threshold"] = balanced
    model["threshold_recomendado_balanceado"] = balanced
    model["threshold_economico"] = economic
    model["validacion_completa"] = metricas(yv, sv, balanced, cfg)
    model["validacion_holdout_umbral"] = metricas(yv[threshold_slice], sv[threshold_slice], balanced, cfg)
    model["validacion_holdout_economico"] = metricas(yv[threshold_slice], sv[threshold_slice], economic, cfg)
    model["benchmark_historico"] = metricas(yb, sb, balanced, cfg)
    model["benchmark_economico"] = metricas(yb, sb, economic, cfg)

    v2 = result["promocion"]["v2_validacion_holdout"]
    v3 = model["validacion_holdout_umbral"]
    reduction = 1.0 - v3["costo_q"] / v2["costo_q"]
    result["promocion"]["v3_validacion_holdout"] = v3
    result["promocion"]["reduccion_costo_holdout"] = reduction
    result["promocion"]["criterios"]["costo_holdout_reduccion_min_3pct"] = reduction >= cfg.promotion_cost_reduction
    result["promocion"]["criterios"]["recall_no_cae_mas_1pp"] = v3["recall"] >= v2["recall"] - cfg.promotion_recall_tolerance
    result["promocion"]["promover_v3"] = all(result["promocion"]["criterios"].values())
    result["recomendacion"] = "Promover V3 con umbral balanceado"

    result["referencias_historicas"]["V3"] = {
        **model["benchmark_historico"],
        "modelo": "LGB_native_recency_V3",
        "estado": "benchmark_historico_reutilizado",
    }
    old = result["referencias_historicas"]["V2"]
    new = result["referencias_historicas"]["V3"]
    result["comparacion_benchmark_v3_vs_v2"] = {
        "delta_auc_pr": new["auc_pr"] - old["auc_pr"],
        "delta_precision": new["precision"] - old["precision"],
        "delta_recall": new["recall"] - old["recall"],
        "delta_f1": new["f1"] - old["f1"],
        "delta_costo_q": new["costo_q"] - old["costo_q"],
        "reduccion_costo_relativa": 1.0 - new["costo_q"] / old["costo_q"],
        "advertencia": "Comparación descriptiva: el benchmark ya fue observado y no decide promoción.",
    }

    print("Recalculando segmentos para el umbral recomendado...")
    frame = load_all()
    split = temporal_boundaries(frame, cfg)
    frame, _ = add_causal_features(frame)
    bench_idx = split["benchmark_historico"]
    segments = segment_metrics(frame, bench_idx, yb, sb, balanced, cfg)
    segments.to_csv(ART / "metricas_segmentos_v3.csv", index=False)

    RESULT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({
        "threshold_balanceado": balanced,
        "threshold_economico": economic,
        "benchmark_balanceado": model["benchmark_historico"],
        "benchmark_economico": model["benchmark_economico"],
        "promover_v3": result["promocion"]["promover_v3"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
