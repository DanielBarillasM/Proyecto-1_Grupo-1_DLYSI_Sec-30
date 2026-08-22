from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import fitz
import joblib
import lightgbm as lgb
import nbformat
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
errors: list[str] = []
warnings: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    result_path = ROOT / "artefactos/v3/resultados_v3.json"
    check(result_path.exists(), "Falta resultados_v3.json")
    if not result_path.exists():
        return finish()
    result = json.loads(result_path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    check(result["promocion"]["promover_v3"] is True, "V3 no cumple la regla de promoción")
    check(all(result["promocion"]["criterios"].values()), "Hay criterios de promoción falsos")
    v2, v3 = result["referencias_historicas"]["V2"], result["referencias_historicas"]["V3"]
    for metric in ("auc_pr", "precision", "recall", "f1"):
        check(v3[metric] > v2[metric], f"V3 no mejora {metric} frente a V2")
    check(v3["costo_q"] < v2["costo_q"], "V3 no reduce costo frente a V2")
    check(result["comparacion_pareada_benchmark"]["li95"] > 0, "El intervalo pareado AP incluye cero")
    check(result["modelo_v3"]["threshold_recomendado_balanceado"] == result["modelo_v3"]["threshold"], "El umbral principal no es el balanceado")

    expected = [
        ".github/README.md",
        "codigo/v3/dataset_v3_support.py", "codigo/v3/proyecto1_v3_pipeline.py",
        "codigo/v3/postprocess_v3.py", "codigo/v3/finalize_v3.py",
        "codigo/v3/build_v3_deliverables.py", "codigo/v3/audit_project1_v3.py",
        "configuracion/v3/requirements-v3.txt", "configuracion/v3/requirements-docs-v3.txt",
        "configuracion/v3/INSTRUCCIONES_V3.md",
        "artefactos/v3/modelo_lightgbm_v3.txt", "artefactos/v3/calibrador_sigmoide_v3.joblib",
        "artefactos/v3/codificador_nativo_v3.joblib", "artefactos/v3/metricas_segmentos_v3.csv",
        "datos/processed/v3/seleccion_variables_v3.json",
        "entregables/cuaderno/v3/Proyecto_1_Monitoreo_Transaccional_V3.ipynb",
        "entregables/informe/v3/informe_proyecto1_v3.tex", "entregables/informe/v3/informe_proyecto1_v3.pdf",
        "entregables/presentacion/v3/presentacion_proyecto1_v3.html", "entregables/presentacion/v3/presentacion_proyecto1_v3.pdf",
        "entregables/ficha/v3/Ficha_Repositorio_Proyecto_1_V3.docx", "entregables/ficha/v3/Ficha_Repositorio_Proyecto_1_V3.pdf",
    ]
    for relative in expected:
        check((ROOT / relative).exists(), f"Falta {relative}")
    check(
        not [p.name for p in ROOT.iterdir() if p.is_file() and p.name != "README.md"],
        "Hay archivos sueltos en la raíz",
    )

    forbidden = [
        "artefactos/v1", "artefactos/v2", "codigo/v1", "configuracion/v2",
        "datos/processed/v2", "evidencia/figuras/v1", "evidencia/figuras/v2",
    ]
    for relative in forbidden:
        check(not (ROOT / relative).exists(), f"Permanece versión anterior activa: {relative}")

    readme = (ROOT / ".github/README.md").read_text(encoding="utf-8")
    check(len(readme.split()) >= 1300, "README V3 tiene menos de 1,300 palabras")
    for phrase in ("V3 cumplió", "benchmark histórico reutilizado", "Referencias APA 7", "umbral balanceado"):
        check(phrase in readme, f"README no contiene: {phrase}")

    notebook_path = ROOT / "entregables/cuaderno/v3/Proyecto_1_Monitoreo_Transaccional_V3.ipynb"
    if notebook_path.exists():
        notebook = nbformat.read(notebook_path, as_version=4)
        check(len(notebook.cells) >= 20, "Notebook V3 demasiado corto")
        check(any("<style>" in cell.source for cell in notebook.cells if cell.cell_type == "markdown"), "Notebook sin HTML/CSS")
        for cell in notebook.cells:
            if cell.cell_type == "code":
                check(cell.get("execution_count") is not None, "Notebook contiene código sin ejecutar")
            for output in cell.get("outputs", []):
                check(output.get("output_type") != "error", "Notebook contiene salida de error")

    report_pdf = ROOT / "entregables/informe/v3/informe_proyecto1_v3.pdf"
    if report_pdf.exists():
        pages = len(fitz.open(report_pdf))
        check(5 <= pages <= 7, f"Informe V3 tiene {pages} páginas; se esperan 5–7")
    slides_html = ROOT / "entregables/presentacion/v3/presentacion_proyecto1_v3.html"
    if slides_html.exists():
        slides = slides_html.read_text(encoding="utf-8")
        check(len(re.findall(r'<section class="slide"', slides)) == 8, "La presentación HTML no tiene ocho diapositivas")
        check(len(re.findall(r"<aside>", slides)) == 8, "Faltan notas del presentador")
    slides_pdf = ROOT / "entregables/presentacion/v3/presentacion_proyecto1_v3.pdf"
    if slides_pdf.exists():
        check(len(fitz.open(slides_pdf)) == 8, "La presentación PDF no tiene ocho páginas")
    ficha_pdf = ROOT / "entregables/ficha/v3/Ficha_Repositorio_Proyecto_1_V3.pdf"
    if ficha_pdf.exists():
        check(len(fitz.open(ficha_pdf)) == 1, "La ficha PDF no tiene una página")

    booster = lgb.Booster(model_file=str(ROOT / "artefactos/v3/modelo_lightgbm_v3.txt"))
    check(booster.num_trees() > 0, "El modelo LightGBM no carga")
    calibrator = joblib.load(ROOT / "artefactos/v3/calibrador_sigmoide_v3.joblib")
    encoder = joblib.load(ROOT / "artefactos/v3/codificador_nativo_v3.joblib")
    check(hasattr(calibrator, "predict_proba"), "El calibrador no carga")
    check(len(encoder.get("numeric", [])) == 220 and len(encoder.get("categorical", [])) == 24, "El codificador no coincide con 220+24")
    pred = pd.read_csv(ROOT / "artefactos/v3/predicciones_benchmark_v3.csv")
    ap = average_precision_score(pred["y"], pred["score_calibrado"])
    check(math.isclose(ap, v3["auc_pr"], rel_tol=0, abs_tol=1e-12), "AUC-PR no reproduce desde predicciones")

    tracked = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False).stdout.splitlines()
    check(not any(path.startswith("datos/raw/") and path.lower().endswith(".csv") for path in tracked), "Hay datos crudos versionados")
    secret_patterns = [re.compile(r"KGAT_[A-Za-z0-9]{20,}"), re.compile(r'"key"\s*:\s*"[a-fA-F0-9]{20,}"')]
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and "raw" not in path.parts and path.suffix.lower() in {".py", ".md", ".txt", ".json", ".tex", ".html", ".yml", ".yaml"} and path.stat().st_size < 5_000_000:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                check(not pattern.search(content), f"Posible secreto en {path.relative_to(ROOT)}")

    for requirement in ("requirements-v3.txt", "requirements-docs-v3.txt"):
        lines = [line.strip() for line in (ROOT / "configuracion/v3" / requirement).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        check(all("==" in line for line in lines), f"Dependencias no fijadas en {requirement}")
    return finish()


def finish() -> int:
    print("AUDITORÍA PROYECTO 1 · V3")
    for warning in warnings:
        print("[AVISO]", warning)
    for error in errors:
        print("[ERROR]", error)
    if errors:
        print(f"Resultado: FALLÓ ({len(errors)} errores, {len(warnings)} avisos)")
        return 1
    print(f"Resultado: APROBADO ({len(warnings)} avisos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
