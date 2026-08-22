from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    v1 = json.loads(
        (ROOT / "artefactos/v1/resultados.json").read_text(encoding="utf-8")
    )
    v2 = json.loads(
        (ROOT / "artefactos/v2/resultados_v2.json").read_text(encoding="utf-8")
    )
    rows = []
    for name, metrics in v1["test"].items():
        rows.append(
            {
                "version": "V1",
                "modelo": name,
                "estado_test": "benchmark histórico observado",
                **metrics,
            }
        )
    rows.append(
        {
            "version": "V2",
            "modelo": "LightGBM_depured",
            "estado_test": "benchmark histórico reutilizado",
            **v2["modelo_tabular_v2"]["benchmark_historico"],
        }
    )
    rows.append(
        {
            "version": "V2",
            "modelo": "Ensamble_calibrado",
            "estado_test": "benchmark histórico reutilizado",
            **v2["ensamble_v2"]["benchmark_historico"],
        }
    )
    table = pd.DataFrame(rows)
    table["delta_auc_pr_vs_A_v1"] = table["auc_pr"] - v1["test"]["A"]["auc_pr"]
    table["delta_costo_q_vs_A_v1"] = table["costo_q"] - v1["test"]["A"]["cost_q"]
    table["reduccion_costo_relativa_vs_A_v1"] = (
        -table["delta_costo_q_vs_A_v1"] / v1["test"]["A"]["cost_q"]
    )
    table.to_csv(
        ROOT / "artefactos/v2/comparacion_v1_v2.csv", index=False, lineterminator="\n"
    )
    best = table.loc[
        (table["version"] == "V2") & table["modelo"].eq("LightGBM_depured")
    ].iloc[0]
    summary = {
        "referencia": "A_V1",
        "candidato_recomendado": "LightGBM_depured_V2",
        "delta_auc_pr": float(best["delta_auc_pr_vs_A_v1"]),
        "delta_recall": float(best["recall"] - v1["test"]["A"]["recall"]),
        "delta_costo_q": float(best["delta_costo_q_vs_A_v1"]),
        "reduccion_costo_relativa": float(best["reduccion_costo_relativa_vs_A_v1"]),
        "interpretacion": "V2 mejora ranking y costo bajo el supuesto académico, pero reduce recall; requiere seleccionar el umbral según la prioridad operativa.",
    }
    (ROOT / "artefactos/v2/resumen_comparacion_v1_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
