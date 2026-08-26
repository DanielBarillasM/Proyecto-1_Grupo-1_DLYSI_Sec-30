"""Auditoría técnica, documental y rubricada de Proyecto 1 V7."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
import pymupdf
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
ERRORS: list[str] = []
WARNINGS: list[str] = []


def require(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def main() -> None:
    art = ROOT / "artefactos" / "v7"
    result_path = art / "resultados_v7.json"
    require(result_path.exists(), "Falta resultados_v7.json")
    if not result_path.exists():
        finish(); return
    r = json.loads(result_path.read_text(encoding="utf-8"))
    require(r.get("version") == "7.0", "Versión de resultados distinta de 7.0")
    require(r.get("estado_benchmark") == "historico_reutilizado_no_ciego", "Benchmark no declarado histórico reutilizado")
    require((ROOT / r.get("protocolo_congelado", "missing")).exists(), "Falta protocolo experimental congelado")
    require({"A", "B", "C", "D"}.issubset(r.get("evaluacion_interna", {})), "Falta núcleo A/B/C o control D")
    require(bool(r.get("hipotesis_C", {}).get("declaracion_previa")), "Falta hipótesis previa C")
    require("success" in r.get("hipotesis_C", {}), "Falta veredicto C")
    require("success" in r.get("promocion_V7", {}), "Falta gate V7")
    require(len(r.get("promocion_V7", {}).get("ventanas", [])) == 4, "Gate no contiene cuatro ventanas")
    require(len(r.get("walk_forward", [])) == 3, "Faltan tres refits walk-forward")
    require(r.get("datos", {}).get("filas") == 590540, "Cantidad de filas inesperada")
    require(r.get("datos", {}).get("columnas_union", 0) >= 430, "No se documenta la unión ampliada")
    require(r.get("variables", {}).get("correlacion", {}).get("pares_eliminados", 0) > 0, "No hay ablation de correlación")
    require(r.get("variables", {}).get("pca", {}).get("componentes_ajustados") == 128, "PCA train-only incompleto")

    fals = r.get("falsificaciones", {})
    require(len(fals.get("permutaciones", [])) >= 5, "Faltan cinco permutaciones")
    require(all(f"historia_{k}" in fals for k in (3, 8, 16, 32)), "Faltan recortes 3/8/16/32")
    for block in ("evaluacion_interna", "benchmark_historico"):
        for model in ("A", "B", "C", "D"):
            metrics = r[block][model]
            for name in ("auc_pr", "roc_auc", "precision", "recall", "f1", "brier", "cost_q", "threshold", "alertas_por_100k", "precision_at_1pct", "recall_at_1pct"):
                require(name in metrics and np.isfinite(metrics[name]), f"Métrica inválida: {block}/{model}/{name}")
            require(0 <= metrics["auc_pr"] <= 1 and 0 <= metrics["roc_auc"] <= 1, f"AUC fuera de rango: {block}/{model}")

    val_path, bench_path = art / "predicciones_validacion_v7.csv", art / "predicciones_benchmark_v7.csv"
    require(val_path.exists() and bench_path.exists(), "Faltan predicciones V7")
    if val_path.exists() and bench_path.exists():
        val, bench = pd.read_csv(val_path), pd.read_csv(bench_path)
        expected = {"score_A", "score_B", "score_C", "score_D", "score_A_V6_control"}
        require(expected.issubset(val.columns) and expected.issubset(bench.columns), "Faltan scores continuos comunes")
        require(len(val) == len(bench) == 88581, "Poblaciones temporalmente incompatibles")
        require(val["TransactionID"].is_unique and bench["TransactionID"].is_unique, "TransactionID repetido en scores")
        require(all(val[c].between(0, 1).all() and bench[c].between(0, 1).all() for c in expected), "Score fuera de [0,1]")
        evaluation = np.arange(int(len(val) * .80), len(val))
        for model in ("A", "B", "C", "D"):
            measured = average_precision_score(val.loc[evaluation, "y"], val.loc[evaluation, f"score_{model}"])
            require(abs(measured - r["evaluacion_interna"][model]["auc_pr"]) < 1e-10, f"AP no reproducible para {model}")

    required_artifacts = [
        "modelo_A0_regresion_logistica_v7.joblib", "modelo_A1_lightgbm_ampliado_v7.joblib",
        "modelo_A2_lightgbm_correlacion_v7.joblib", "modelo_A3_lightgbm_pca_v7.joblib",
        "modelo_A4_catboost_v7.cbm", "modelo_A5_ensamble_tabular_v7.joblib",
        "modelos_C_fusion_v7.joblib", "preprocesamiento_tabular_v7.joblib",
        "preprocesamiento_catboost_v7.joblib", "pca_bloque_v_v7.joblib",
        "calibradores_v7.joblib", "umbrales_v7.json", "contrato_entrada_salida_v7.json",
        "seleccion_modelos_v7.json", "validacion_walk_forward_v7.csv",
    ]
    for name in required_artifacts:
        require((art / name).exists() and (art / name).stat().st_size > 0, f"Falta artefacto {name}")
    try:
        for name in ("modelo_A0_regresion_logistica_v7.joblib", "modelo_A5_ensamble_tabular_v7.joblib", "modelos_C_fusion_v7.joblib", "preprocesamiento_tabular_v7.joblib", "pca_bloque_v_v7.joblib", "calibradores_v7.joblib"):
            joblib.load(art / name)
    except Exception as exc:
        ERRORS.append(f"No se cargaron artefactos joblib: {exc}")

    for name in ("asociacion_variables_train_v7.csv", "pares_correlacionados_train_v7.csv", "auditoria_variables_v7.json"):
        require((ROOT / "datos" / "processed" / "v7" / name).exists(), f"Falta evidencia train-only {name}")
    for name in ("01_comparacion_interna_v7.png", "02_benchmark_historico_v7.png", "03_seleccion_modelos_a_v7.png", "04_correlacion_pca_v7.png", "05_costos_v7.png", "06_calibracion_v7.png"):
        require((ROOT / "evidencia" / "figuras" / "v7" / name).exists(), f"Falta figura {name}")

    notebook_dir = ROOT / "entregables" / "cuaderno" / "v7"
    for name in ("proyecto1_calderon_barillas.ipynb", "EDA_IEEE_CIS_Diagnostico_Datos_V7.ipynb"):
        path = notebook_dir / name
        require(path.exists(), f"Falta notebook {name}")
        if path.exists():
            notebook = nbformat.read(path, as_version=4)
            code = [c for c in notebook.cells if c.cell_type == "code"]
            require(code and all(c.execution_count is not None for c in code), f"Notebook no ejecutado: {name}")
            errors = [o for c in code for o in c.get("outputs", []) if o.output_type == "error"]
            require(not errors, f"Notebook con errores: {name}")
            source = "\n".join(c.source for c in notebook.cells)
            require("<style>" in source and "class=\"hero\"" in source, f"Notebook sin HTML/CSS estético: {name}")

    report_pdf = ROOT / "entregables" / "informe" / "v7" / "informe.pdf"
    slides_pdf = ROOT / "entregables" / "presentacion" / "v7" / "presentacion.pdf"
    slides_html = slides_pdf.with_suffix(".html")
    require(report_pdf.exists() and slides_pdf.exists() and slides_html.exists(), "Falta informe o presentación")
    if report_pdf.exists():
        with pymupdf.open(report_pdf) as doc:
            require(doc.page_count == 7, f"Informe debe tener 7 páginas; tiene {doc.page_count}")
            text = "\n".join(page.get_text() for page in doc)
        for term in ("Datos, EDA y protocolo", "Modelos A", "Valor del orden", "Apuesta C", "Economía", "Matriz de evidencias"):
            require(term.lower() in text.lower(), f"Informe no contiene {term}")
    if slides_pdf.exists():
        with pymupdf.open(slides_pdf) as doc:
            require(doc.page_count == 8, f"Presentación debe tener 8 diapositivas; tiene {doc.page_count}")
    if slides_html.exists():
        html = slides_html.read_text(encoding="utf-8")
        require(html.count('<section class="slide') == 8, "HTML no contiene ocho diapositivas")
        require(html.count('class="notes"') == 8, "Faltan notas en alguna diapositiva")

    docs = [
        ROOT / "README.md", ROOT / "configuracion" / "v7" / "INSTRUCCIONES_V7.md",
        ROOT / "configuracion" / "v7" / "PROTOCOLO_EXPERIMENTAL_V7.md",
        ROOT / "entregables" / "presentacion" / "v7" / "GUION_EXPOSICION_V7.md",
        ROOT / "entregables" / "ficha" / "v7" / "Ficha_Repositorio_Proyecto_1_V7.docx",
        ROOT / "entregables" / "ficha" / "v7" / "Ficha_Repositorio_Proyecto_1_V7.pdf",
    ]
    for path in docs:
        require(path.exists() and path.stat().st_size > 0, f"Falta documento {path.relative_to(ROOT)}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for term in ("La versión que debe revisarse es V7", "Cómo interpretar las métricas", "Reproducción rápida", "Estructura y versiones", "Candidato al Proyecto Final", "Tres decisiones técnicas importantes", "Declaración de uso de inteligencia artificial", "Referencias APA 7"):
        require(term in readme, f"README no contiene: {term}")
    for match in re.finditer(r"\[[^]]+\]\((?!https?://)([^)#]+)(?:#[^)]+)?\)", readme):
        target = ROOT / match.group(1)
        require(target.exists(), f"Ruta local rota en README: {match.group(1)}")

    # Escaneo de secretos en archivos textuales V7 y raíz.
    candidates = [ROOT / "README.md", *list((ROOT / "codigo" / "v7").glob("*.py")), *list((ROOT / "configuracion" / "v7").glob("*"))]
    for path in candidates:
        content = path.read_text(encoding="utf-8", errors="ignore")
        require(re.search(r"KGAT_[A-Za-z0-9]{20,}", content) is None, f"Posible token Kaggle en {path.relative_to(ROOT)}")

    finish()


def finish() -> None:
    print("AUDITORÍA PROYECTO 1 · V7")
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
