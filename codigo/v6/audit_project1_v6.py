"""Auditoría automática del cumplimiento técnico y documental de Proyecto 1 V6."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import pymupdf
import torch


ROOT = Path(__file__).resolve().parents[2]
ERRORS: list[str] = []
WARNINGS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def warn(condition: bool, message: str) -> None:
    if not condition:
        WARNINGS.append(message)


def main() -> None:
    result_path = ROOT / "artefactos" / "v6" / "resultados_v6.json"
    require(result_path.exists(), "Falta resultados_v6.json")
    if not result_path.exists():
        finish()
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(result.get("estado_benchmark") == "historico_reutilizado_no_ciego", "El benchmark debe declararse histórico reutilizado")
    require(set(result.get("evaluacion_interna", {})) == {"A", "B", "C", "D"}, "Falta alguna pieza A/B/C o el control D")
    require(bool(result.get("hipotesis_C", {}).get("declaracion_previa")), "Falta hipótesis previa de C")
    require("success" in result.get("hipotesis_C", {}), "Falta veredicto de C")
    fals = result.get("falsificaciones", {})
    require(len(fals.get("permutaciones", [])) >= 5, "Faltan cinco permutaciones controladas")
    require(all(f"historia_{k}" in fals for k in (3, 8, 16)), "Faltan recortes de historia 3/8/16")
    require(result.get("datos", {}).get("filas") == 590540, "Cantidad de filas inesperada")
    require(result.get("secuencias", {}).get("longitud_maxima") == 32, "Longitud secuencial V6 no documentada")

    for block in ("evaluacion_interna", "benchmark_historico"):
        for model, metrics in result[block].items():
            for name in ("auc_pr", "roc_auc", "precision", "recall", "f1", "cost_q", "threshold"):
                require(name in metrics and np.isfinite(metrics[name]), f"Métrica inválida: {block}/{model}/{name}")
            require(0 <= metrics["auc_pr"] <= 1, f"AUC-PR fuera de rango: {block}/{model}")

    val_path = ROOT / "artefactos" / "v6" / "predicciones_validacion_v6.csv"
    bench_path = ROOT / "artefactos" / "v6" / "predicciones_benchmark_v6.csv"
    require(val_path.exists() and bench_path.exists(), "Faltan predicciones V6")
    if val_path.exists() and bench_path.exists():
        val = pd.read_csv(val_path)
        bench = pd.read_csv(bench_path)
        expected = {"score_A", "score_B", "score_C", "score_D", "score_control_logistico"}
        require(expected.issubset(val.columns) and expected.issubset(bench.columns), "Faltan puntajes continuos A/B/C")
        require(len(val) == 88581 and len(bench) == 88581, "Poblaciones de validación/benchmark inesperadas")
        require(val["TransactionID"].is_unique and bench["TransactionID"].is_unique, "TransactionID no es único en predicciones")
        require(all(val[c].between(0, 1).all() and bench[c].between(0, 1).all() for c in expected), "Puntaje fuera de [0,1]")

    artifacts = [
        "modelo_b_gru_v6.pt", "modelo_b_tcn_v6.pt", "modelo_D_autoencoder_v6.pt",
        "modelo_C_fusion_condicionada_v6.joblib", "modelo_A_refuerzo_causal_v6.joblib", "control_regresion_logistica_v6.joblib",
        "preprocesamiento_secuencial_v6.joblib",
        "calibradores_v6.joblib", "umbrales_v6.json", "contrato_entrada_salida_v6.json",
        "metricas_segmentos_v6.csv", "falsos_negativos_alto_monto_v6.csv",
    ]
    for name in artifacts:
        require((ROOT / "artefactos" / "v6" / name).exists(), f"Falta artefacto {name}")
    try:
        checkpoint = torch.load(ROOT / "artefactos" / "v6" / "modelo_b_gru_v6.pt", map_location="cpu", weights_only=False)
        require("state_dict" in checkpoint and len(checkpoint.get("numeric_features", [])) > 0, "Checkpoint B incompleto")
        joblib.load(ROOT / "artefactos" / "v6" / "modelo_C_fusion_condicionada_v6.joblib")
        joblib.load(ROOT / "artefactos" / "v6" / "preprocesamiento_secuencial_v6.joblib")
        joblib.load(ROOT / "artefactos" / "v6" / "calibradores_v6.joblib")
    except Exception as exc:
        ERRORS.append(f"No se pudieron cargar artefactos: {exc}")

    notebook_path = ROOT / "entregables" / "cuaderno" / "v6" / "proyecto1_calderon_barillas.ipynb"
    require(notebook_path.exists(), "Falta notebook con nombre rubricado")
    if notebook_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        require(code_cells and all(c.execution_count is not None for c in code_cells), "Notebook no está completamente ejecutado")
        errors = [o for c in code_cells for o in c.get("outputs", []) if o.output_type == "error"]
        require(not errors, "Notebook contiene salidas de error")
        source = "\n".join(c.source for c in notebook.cells)
        for term in ("Núcleo comparable A/B", "Encoder", "Hipótesis previa", "Dos intentos", "Decisión económica", "Matriz de evidencias", "Declaración de uso de inteligencia artificial"):
            require(term.lower() in source.lower(), f"Notebook no contiene: {term}")

    report_pdf = ROOT / "entregables" / "informe" / "v6" / "informe.pdf"
    slides_pdf = ROOT / "entregables" / "presentacion" / "v6" / "presentacion.pdf"
    require(report_pdf.exists(), "Falta informe.pdf")
    require(slides_pdf.exists(), "Falta presentacion.pdf")
    if report_pdf.exists():
        report = pymupdf.open(report_pdf)
        require(len(report) <= 7, f"Informe excede 7 páginas: {len(report)}")
        require(len(report) >= 5, f"Informe demasiado corto: {len(report)}")
        report_text = "\n".join(page.get_text() for page in report)
        for term in ("Integridad de datos", "Núcleo A/B", "Valor del orden", "Umbral", "Recomendación", "Matriz de evidencias"):
            require(term.lower() in report_text.lower(), f"Informe no contiene: {term}")
    if slides_pdf.exists():
        slides = pymupdf.open(slides_pdf)
        require(len(slides) == 8, f"Presentación debe tener 8 diapositivas, tiene {len(slides)}")

    readme_path = ROOT / "README.md"
    require(readme_path.exists(), "README.md debe existir en la raíz")
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        for term in ("Reproducción", "Tres decisiones técnicas importantes", "Candidato al Proyecto Final", "Declaración de uso de inteligencia artificial", "Limitaciones"):
            require(term in readme, f"README no contiene: {term}")

    for figure in ("01_comparacion_abc_validacion.png", "02_comparacion_abc_benchmark.png", "03_falsificaciones_orden_v6.png", "04_costos_abc_v6.png", "05_calibracion_v6.png"):
        require((ROOT / "evidencia" / "figuras" / "v6" / figure).exists(), f"Falta figura {figure}")

    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
        require(not any("datos/raw" in p.replace("\\", "/") and p.lower().endswith(".csv") for p in tracked), "Hay CSV crudos versionados")
        require(not any(p.lower().endswith("kaggle.json") for p in tracked), "Hay kaggle.json versionado")
        for path in tracked:
            if not re.search(r"\.(md|py|json|txt|tex|html)$", path, re.I):
                continue
            full = ROOT / path
            if not full.exists() or full.stat().st_size > 2_000_000:
                continue
            content = full.read_text(encoding="utf-8", errors="ignore")
            require(re.search(r"KGAT_[A-Za-z0-9]{20,}", content) is None, f"Posible token Kaggle en {path}")
    except Exception as exc:
        WARNINGS.append(f"No se completó auditoría Git: {exc}")

    finish()


def finish() -> None:
    print("AUDITORÍA PROYECTO 1 · V6")
    if ERRORS:
        print(f"Resultado: RECHAZADO ({len(ERRORS)} errores, {len(WARNINGS)} avisos)")
        for item in ERRORS:
            print("ERROR:", item)
    else:
        print(f"Resultado: APROBADO ({len(WARNINGS)} avisos)")
    for item in WARNINGS:
        print("AVISO:", item)
    raise SystemExit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
