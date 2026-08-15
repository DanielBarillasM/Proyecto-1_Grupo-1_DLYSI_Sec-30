from pathlib import Path
import json
import re
import sys

import nbformat
import pymupdf

root = Path(__file__).resolve().parent
required = [
    "proyecto1_calderon_barillas.ipynb", "informe.tex", "informe.pdf",
    "presentacion.html", "presentacion.pdf", "README.md", "requirements.txt",
    "src/proyecto1_pipeline.py", "artefactos/resultados.json",
    "artefactos/manifiesto_datos.json", "artefactos/preprocesamiento.joblib",
    "artefactos/umbrales.json", "artefactos/esquema_entrada.json",
]
missing = [name for name in required if not (root / name).exists()]
assert not missing, f"Faltan: {missing}"

results = json.loads((root / "artefactos/resultados.json").read_text(encoding="utf-8"))
nb = nbformat.read(root / "proyecto1_calderon_barillas.ipynb", as_version=4)
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

report_pages = len(pymupdf.open(root / "informe.pdf"))
presentation_pages = len(pymupdf.open(root / "presentacion.pdf"))
assert report_pages <= 7, report_pages
assert presentation_pages == 8, presentation_pages

html = (root / "presentacion.html").read_text(encoding="utf-8")
assert len(re.findall(r"<section(?:\s|>)", html)) == 8
assert "@media print" in html and "data:image/png;base64," in html

artifacts = list((root / "artefactos").glob("*"))
assert any(p.name.startswith("modelo_candidato_") for p in artifacts)
assert all(m in results["test"] for m in ["A", "B", "C"])
assert len(results["falsification"]["permutation_auc_pr"]) == 5

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
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
