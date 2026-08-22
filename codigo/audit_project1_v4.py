from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import fitz
import nbformat

ROOT = Path(__file__).resolve().parents[1]
ERRORS: list[str] = []
WARNINGS: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        ERRORS.append(message)


def main() -> int:
    result_path = ROOT / "artefactos/v4/resultados_v4.json"
    check(result_path.exists(), "Falta resultados_v4.json")
    if not result_path.exists():
        return finish()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    rec = result.get("modelo_v4_recomendado", {})
    policy = rec.get("politicas", {}).get("robusto_recall_075_post_hoc", {})
    evaluation = policy.get("evaluacion", {})
    v3 = rec.get("comparacion_v3_evaluacion", {}).get("v3", {})

    check(result.get("version") == "4.0", "Versión JSON distinta de 4.0")
    check(result.get("estado_benchmark") == "historico_reutilizado_no_ciego", "Benchmark mal etiquetado")
    check(result.get("decision_v4", {}).get("promocion_confirmatoria") is False, "V4 no debe declararse confirmatoria")
    check(rec.get("modelo") == "LightGBM_expertos_ProductCD", "Candidato V4 inesperado")
    check(rec.get("politica_recomendada") == "robusto_recall_075_post_hoc", "Política recomendada inesperada")
    for metric in ("auc_pr", "roc_auc", "precision", "recall", "f1"):
        check(float(evaluation.get(metric, -1)) > float(v3.get(metric, 2)), f"V4 no mejora {metric} frente a V3")
    check(float(evaluation.get("costo_q", 1e99)) < float(v3.get("costo_q", -1)), "V4 no reduce costo")
    check(float(evaluation.get("roc_auc", 0)) >= 0.90, "V4 no alcanza ROC-AUC 0.90")
    check(float(result["promocion"]["delta_auc_pr_walk"]) >= 0.015, "Mejora walk-forward insuficiente")
    check(float(rec["comparacion_pareada_evaluacion"]["li95"]) > 0, "IC pareado de evaluación cruza cero")

    required = [
        ".github/README.md",
        "codigo/dataset_v4_support.py", "codigo/proyecto1_v4_pipeline.py",
        "codigo/postprocess_v4.py", "codigo/build_v4_deliverables.py", "codigo/audit_project1_v4.py",
        "configuracion/v4/requirements-v4.txt", "configuracion/v4/requirements-docs-v4.txt", "configuracion/v4/INSTRUCCIONES_V4.md",
        "artefactos/v4/modelo_lightgbm_global_v4.txt", "artefactos/v4/modelo_experto_w_v4.txt",
        "artefactos/v4/modelo_experto_no_w_v4.txt", "artefactos/v4/modelo_xgboost_v4.json",
        "artefactos/v4/modelo_catboost_v4.cbm", "artefactos/v4/predicciones_validacion_v4.csv",
        "artefactos/v4/predicciones_benchmark_v4.csv", "artefactos/v4/seleccion_candidato_v4.csv",
        "entregables/cuaderno/Proyecto_1_Monitoreo_Transaccional_V4.ipynb",
        "entregables/informe/informe_proyecto1_v4.tex", "entregables/informe/informe_proyecto1_v4.pdf",
        "entregables/presentacion/presentacion_proyecto1_v4.html", "entregables/presentacion/presentacion_proyecto1_v4.pdf",
        "entregables/ficha/Ficha_Repositorio_Proyecto_1_V4.docx", "entregables/ficha/Ficha_Repositorio_Proyecto_1_V4.pdf",
    ]
    for relative in required:
        path = ROOT / relative
        check(path.exists() and path.stat().st_size > 0, f"Falta o está vacío: {relative}")

    readme = (ROOT / ".github/README.md").read_text(encoding="utf-8")
    check(len(readme.split()) >= 1500, "README V4 tiene menos de 1,500 palabras")
    for phrase in ("candidato congelado", "benchmark histórico reutilizado", "Referencias APA 7", "post-hoc", "ROC-AUC"):
        check(phrase.lower() in readme.lower(), f"README no contiene: {phrase}")

    notebook_path = ROOT / "entregables/cuaderno/Proyecto_1_Monitoreo_Transaccional_V4.ipynb"
    if notebook_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
        check(len(notebook.cells) >= 30, "Notebook V4 demasiado corto")
        markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
        check("<style>" in markdown, "Notebook sin CSS embebido")
        check("Referencias APA 7" in markdown, "Notebook sin referencias APA 7")
        errors = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.get("outputs", []) if out.get("output_type") == "error"]
        check(not errors, f"Notebook contiene {len(errors)} errores de ejecución")

    report_pdf = ROOT / "entregables/informe/informe_proyecto1_v4.pdf"
    if report_pdf.exists():
        pages = len(fitz.open(report_pdf))
        check(5 <= pages <= 7, f"Informe V4 tiene {pages} páginas; se esperan 5–7")
    presentation_pdf = ROOT / "entregables/presentacion/presentacion_proyecto1_v4.pdf"
    if presentation_pdf.exists():
        check(len(fitz.open(presentation_pdf)) == 8, "Presentación PDF no tiene 8 páginas")
    presentation_html = ROOT / "entregables/presentacion/presentacion_proyecto1_v4.html"
    if presentation_html.exists():
        html = presentation_html.read_text(encoding="utf-8")
        check(html.count('class="slide"') == 8, "Presentación HTML no tiene 8 diapositivas")
        check("<aside>" in html and "document.onkeydown" in html, "Presentación sin notas o navegación")
    ficha_pdf = ROOT / "entregables/ficha/Ficha_Repositorio_Proyecto_1_V4.pdf"
    if ficha_pdf.exists():
        check(len(fitz.open(ficha_pdf)) == 1, "Ficha PDF debe tener una página")

    tracked = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    check(not any(path.startswith("datos/raw/") and path.endswith(".csv") for path in tracked), "Hay CSV crudos rastreados por Git")
    secret_pattern = re.compile(r"KGAT_[A-Za-z0-9]+|\"key\"\s*:\s*\"[a-f0-9]{20,}\"", re.I)
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and path.suffix.lower() in {".py", ".md", ".txt", ".json", ".tex", ".html"}:
            try:
                if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                    ERRORS.append(f"Posible secreto en {path.relative_to(ROOT)}")
            except OSError:
                pass
    return finish()


def finish() -> int:
    print("AUDITORÍA PROYECTO 1 · V4")
    for message in WARNINGS:
        print("AVISO:", message)
    for message in ERRORS:
        print("ERROR:", message)
    if ERRORS:
        print(f"Resultado: RECHAZADO ({len(ERRORS)} errores, {len(WARNINGS)} avisos)")
        return 1
    print(f"Resultado: APROBADO ({len(WARNINGS)} avisos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
