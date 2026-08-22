from pathlib import Path
import json
import re
import sys
import zipfile

import nbformat
import pymupdf

root = Path(__file__).resolve().parents[2]
required = [
    "entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb",
    "entregables/informe/v1/informe.tex", "entregables/informe/v1/informe.pdf",
    "entregables/presentacion/v1/presentacion.html", "entregables/presentacion/v1/presentacion.pdf",
    "entregables/ficha/v1/Ficha_Repositorio_Proyecto1.docx",
    ".github/README.md", "configuracion/v1/requirements-v1.txt",
    "configuracion/v1/INSTRUCCIONES_V1.md",
    "codigo/v1/proyecto1_pipeline.py", "codigo/compartido/download_data.py",
    "codigo/v1/build_deliverables.py", "codigo/v1/crear_ficha_repositorio.py",
    "artefactos/v1/resultados.json",
    "artefactos/v1/manifiesto_datos.json", "artefactos/v1/preprocesamiento.joblib",
    "artefactos/v1/umbrales.json", "artefactos/v1/esquema_entrada.json",
]
missing = [name for name in required if not (root / name).exists()]
assert not missing, f"Faltan: {missing}"

results = json.loads((root / "artefactos/v1/resultados.json").read_text(encoding="utf-8"))
nb = nbformat.read(root / "entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
assert code and all(c.get("execution_count") is not None for c in code)
assert [c.execution_count for c in code] == list(range(1, len(code) + 1))
assert not errors

md = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
for phrase in [
    "hipótesis previa", "integridad de los datos", "núcleo común", "apuesta c",
    "valor del orden", "decisión económica", "matriz de evidencias",
    "Declaración de uso de inteligencia artificial", "Referencias",
]:
    assert phrase.lower() in md.lower(), phrase
assert "Bahdanau" not in md or "Cho" in md
assert md.count("https://") >= 7
assert md.lower().count("<div") == md.lower().count("</div>")

report_pages = len(pymupdf.open(root / "entregables/informe/v1/informe.pdf"))
presentation_pages = len(pymupdf.open(root / "entregables/presentacion/v1/presentacion.pdf"))
assert report_pages <= 7, report_pages
assert presentation_pages == 8, presentation_pages

html = (root / "entregables/presentacion/v1/presentacion.html").read_text(encoding="utf-8")
assert len(re.findall(r"<section(?:\s|>)", html)) == 8
assert "@media print" in html and "data:image/png;base64," in html

artifacts = list((root / "artefactos/v1").glob("*"))
assert any(p.name.startswith("modelo_candidato_") for p in artifacts)
assert all(m in results["test"] for m in ["A", "B", "C"])
assert len(results["falsification"]["permutation_auc_pr"]) == 5

loose_root_files = [p.name for p in root.iterdir() if p.is_file() and p.name != "README.md"]
assert not loose_root_files, f"Archivos sueltos en raíz: {loose_root_files}"

docx_path = root / "entregables/ficha/v1/Ficha_Repositorio_Proyecto1.docx"
assert zipfile.is_zipfile(docx_path)
with zipfile.ZipFile(docx_path) as package:
    assert "word/document.xml" in package.namelist()
    document_xml = package.read("word/document.xml").decode("utf-8")
    for phrase in ["Monitoreo transaccional", "Wilson Alejandro", "Pablo Daniel", "github.com"]:
        assert phrase in document_xml, phrase

summary = {
    "required_files": len(required),
    "notebook_cells": len(nb.cells),
    "code_cells_executed": len(code),
    "notebook_errors": len(errors),
    "report_pages": report_pages,
    "presentation_pages": presentation_pages,
    "presentation_slides_html": len(re.findall(r"<section(?:\s|>)", html)),
    "candidate": results["candidate"],
    "artifact_files": len(artifacts),
    "order_auc_pr_drop": results["falsification"]["order_auc_pr_drop"],
    "loose_root_files": len(loose_root_files),
    "repository_sheet_docx": "OK",
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
