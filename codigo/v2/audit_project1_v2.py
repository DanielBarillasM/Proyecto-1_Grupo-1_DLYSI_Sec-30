from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import fitz
import nbformat

ROOT = Path(__file__).resolve().parents[2]
errors: list[str] = []
warnings: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    result_path = ROOT / "artefactos/v2/resultados_v2.json"
    check(result_path.exists(), "Falta resultados_v2.json")
    if not result_path.exists():
        return finish()
    r = json.loads(result_path.read_text(encoding="utf-8"))
    check(
        r.get("estado_benchmark") == "histórico_reutilizado_no_ciego",
        "El benchmark debe etiquetarse como reutilizado",
    )
    check(
        (ROOT / "artefactos/v1/PRESERVACION_V1.json").exists(), "Falta manifiesto de V1"
    )
    check(
        len(r["seleccion_variables"]["numericas_seleccionadas"]) > 20,
        "Selección numérica V2 demasiado pequeña",
    )
    check(
        len(r["validacion_walk_forward"]["resumen"]) >= 2, "Faltan modelos walk-forward"
    )

    expected = [
        ".github/README.md",
        "codigo/v2/proyecto1_v2_pipeline.py",
        "codigo/v2/compare_versions.py",
        "codigo/v2/postprocess_v2.py",
        "codigo/v2/deliverables_v2.py",
        "codigo/v2/crear_ficha_repositorio_v2.py",
        "configuracion/v2/requirements-v2.txt",
        "configuracion/v2/requirements-docs-v2.txt",
        "configuracion/v2/INSTRUCCIONES_V2.md",
        "datos/processed/v2/auditoria_variables.csv",
        "artefactos/v2/resumen_walk_forward.csv",
        "artefactos/v2/comparacion_v1_v2.csv",
        "artefactos/v2/resumen_comparacion_v1_v2.json",
        "entregables/cuaderno/v2/Proyecto_1_Monitoreo_Transaccional_V2.ipynb",
        "entregables/informe/v2/informe_proyecto1_v2.tex",
        "entregables/informe/v2/informe_proyecto1_v2.pdf",
        "entregables/presentacion/v2/presentacion_proyecto1_v2.html",
        "entregables/presentacion/v2/presentacion_proyecto1_v2.pdf",
        "entregables/ficha/v2/Ficha_Repositorio_Proyecto_1_V2.docx",
        "entregables/ficha/v2/Ficha_Repositorio_Proyecto_1_V2.pdf",
    ]
    for rel in expected:
        check((ROOT / rel).exists(), f"Falta {rel}")

    root_files = [p.name for p in ROOT.iterdir() if p.is_file() and p.name != "README.md"]
    check(not root_files, f"Hay archivos sueltos en raíz: {root_files}")

    readme = (
        (ROOT / ".github/README.md").read_text(encoding="utf-8")
        if (ROOT / ".github/README.md").exists()
        else ""
    )
    check(len(readme.split()) >= 1300, "README V2 tiene menos de 1,300 palabras")
    check(
        "benchmark histórico reutilizado" in readme,
        "README no advierte benchmark reutilizado",
    )
    check("Referencias APA 7" in readme, "README no contiene referencias APA 7")
    tab_ap = r["modelo_tabular_v2"]["benchmark_historico"]["auc_pr"]
    check(f"{tab_ap:.3f}" in readme, "AUC-PR tabular no coincide en README")

    notebook_path = (
        ROOT / "entregables/cuaderno/v2/Proyecto_1_Monitoreo_Transaccional_V2.ipynb"
    )
    if notebook_path.exists():
        nb = nbformat.read(notebook_path, as_version=4)
        check(len(nb.cells) >= 16, "Notebook V2 tiene pocas celdas")
        check(
            any("<style>" in c.source for c in nb.cells if c.cell_type == "markdown"),
            "Notebook sin HTML/CSS",
        )
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                check(
                    output.get("output_type") != "error",
                    "Notebook contiene una salida de error",
                )

    slides_path = ROOT / "entregables/presentacion/v2/presentacion_proyecto1_v2.html"
    if slides_path.exists():
        slides = slides_path.read_text(encoding="utf-8")
        check(
            len(re.findall(r'<section class="slide"', slides)) == 8,
            "La presentación no tiene exactamente 8 diapositivas",
        )
        check(len(re.findall(r"<aside>", slides)) == 8, "Faltan notas del presentador")

    pdf = ROOT / "entregables/informe/v2/informe_proyecto1_v2.pdf"
    if pdf.exists():
        pages = len(fitz.open(pdf))
        check(5 <= pages <= 7, f"Informe V2 tiene {pages} páginas; se esperan 5–7")
    else:
        warnings.append(
            "No se compiló informe_proyecto1_v2.pdf; falta pdflatex o hubo error LaTeX"
        )

    slides_pdf = ROOT / "entregables/presentacion/v2/presentacion_proyecto1_v2.pdf"
    if slides_pdf.exists():
        check(
            len(fitz.open(slides_pdf)) == 8,
            "El PDF de presentación no tiene ocho páginas",
        )
    ficha_pdf = ROOT / "entregables/ficha/v2/Ficha_Repositorio_Proyecto_1_V2.pdf"
    if ficha_pdf.exists():
        check(len(fitz.open(ficha_pdf)) == 1, "La ficha PDF no tiene una página")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, text=True, capture_output=True, check=False
    ).stdout.splitlines()
    check(
        not any(
            x.startswith("datos/raw/") and x.lower().endswith(".csv") for x in tracked
        ),
        "Hay CSV crudos versionados",
    )
    secret_patterns = [
        re.compile(r"KGAT_[A-Za-z0-9]{20,}"),
        re.compile(r'"key"\s*:\s*"[a-fA-F0-9]{20,}"'),
        re.compile(r"kaggle\.json", re.I),
    ]
    text_ext = {".py", ".md", ".txt", ".json", ".tex", ".html", ".yml", ".yaml"}
    candidates = [
        p
        for p in ROOT.rglob("*")
        if p.is_file() and ".git" not in p.parts and "raw" not in p.parts
    ]
    for p in candidates:
        rel = str(p.relative_to(ROOT))
        if p.suffix.lower() in text_ext and p.stat().st_size < 5_000_000:
            content = p.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                check(
                    not pattern.search(content),
                    f"Posible secreto en {rel}: {pattern.pattern}",
                )

    for req in ["requirements-v2.txt", "requirements-docs-v2.txt"]:
        p = ROOT / "configuracion/v2" / req
        if p.exists():
            lines = [
                x.strip()
                for x in p.read_text(encoding="utf-8").splitlines()
                if x.strip() and not x.startswith("#")
            ]
            check(all("==" in x for x in lines), f"Dependencias no fijadas en {req}")
    return finish()


def finish() -> int:
    print("AUDITORÍA PROYECTO 1 · V2")
    for w in warnings:
        print("[AVISO]", w)
    for e in errors:
        print("[ERROR]", e)
    if errors:
        print(f"Resultado: FALLÓ ({len(errors)} errores, {len(warnings)} avisos)")
        return 1
    print(f"Resultado: APROBADO ({len(warnings)} avisos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
