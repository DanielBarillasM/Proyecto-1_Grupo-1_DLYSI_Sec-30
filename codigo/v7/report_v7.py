"""Compila y valida el informe canónico de Proyecto 1 V7."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def validate_percent_signs(tex_path: Path) -> None:
    invalid = []
    for number, line in enumerate(tex_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        if re.search(r"\d(?:[\d.,]*\d)?\s*(?<!\\)%", line):
            invalid.append(number)
    if invalid:
        raise ValueError("Porcentajes LaTeX sin escapar en líneas: " + ", ".join(map(str, invalid)))


def compile_report(root: Path) -> Path:
    report_dir = root / "entregables" / "informe" / "v7"
    tex_path = report_dir / "informe.tex"
    pdf_path = report_dir / "informe.pdf"
    validate_percent_signs(tex_path)
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex no está disponible en PATH")
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        result = subprocess.run(command, cwd=report_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError("Falló la compilación:\n" + "\n".join((result.stdout + result.stderr).splitlines()[-40:]))
    import pymupdf
    with pymupdf.open(pdf_path) as document:
        pages = document.page_count
    if pages != 7:
        raise RuntimeError(f"El informe debe tener exactamente 7 páginas; produjo {pages}")
    print(f"Informe V7 compilado: {pdf_path} · {pages} páginas")
    return pdf_path


if __name__ == "__main__":
    compile_report(Path(__file__).resolve().parents[2])
