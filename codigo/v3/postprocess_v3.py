from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "artefactos" / "v3" / "resultados_v3.json"
REF = ROOT / "artefactos" / "v3" / "referencia_v2" / "resultados_v2.json"


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    reference = json.loads(REF.read_text(encoding="utf-8"))
    for row in result["baselines_logisticos"]["resultados"]:
        row["convergio"] = bool(row["iteraciones"] < (300 if row["modelo"] == "Logistica_PCA64" else 100))

    v1 = {
        "modelo": "HistGradientBoosting_A_V1",
        "auc_pr": 0.42850380972380175,
        "precision": 0.13392319730347899,
        "recall": 0.7216996432046707,
        "f1": 0.22592272934964716,
        "fp": 14389,
        "fn": 858,
        "tp": 2225,
        "tn": 71109,
        "costo_q": 6193620.0,
        "estado": "benchmark_historico_observado",
    }
    v2 = dict(reference["modelo_tabular_v2"]["benchmark_historico"])
    v2.update({"modelo": "LightGBM_corr_pruned_V2", "estado": "benchmark_historico_reutilizado"})
    v3 = dict(result["modelo_v3"]["benchmark_historico"])
    v3.update({"modelo": "LGB_native_recency_V3", "estado": "benchmark_historico_reutilizado"})
    result["referencias_historicas"] = {"V1": v1, "V2": v2, "V3": v3}
    result["comparacion_benchmark_v3_vs_v2"] = {
        "delta_auc_pr": v3["auc_pr"] - v2["auc_pr"],
        "delta_precision": v3["precision"] - v2["precision"],
        "delta_recall": v3["recall"] - v2["recall"],
        "delta_f1": v3["f1"] - v2["f1"],
        "delta_costo_q": v3["costo_q"] - v2["costo_q"],
        "reduccion_costo_relativa": 1.0 - v3["costo_q"] / v2["costo_q"],
        "advertencia": "Comparación descriptiva: el benchmark ya fue observado y no decide promoción.",
    }
    result["estado_versiones"] = {
        "v3_promovida": bool(result["promocion"]["promover_v3"]),
        "v1_v2_retirables_del_arbol_activo": bool(result["promocion"]["promover_v3"]),
        "preservacion": "La historia permanece recuperable mediante Git; V3 conserva referencias numéricas mínimas para auditoría.",
    }
    RESULT.write_text(
        json.dumps(clean(result), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result["comparacion_benchmark_v3_vs_v2"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
