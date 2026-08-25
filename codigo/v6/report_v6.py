"""Compila y valida el informe ejecutivo canónico de Proyecto 1 V6."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


def _validate_percent_signs(tex_path: Path) -> None:
    """Evita que un porcentaje sin escapar silencie el resto de una línea LaTeX."""

    invalid = []
    for number, line in enumerate(tex_path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("%"):
            continue
        # Los signos al final de comandos son continuaciones LaTeX válidas.
        # El fallo peligroso para este informe es un porcentaje numérico como
        # ``78%`` en lugar de ``78\%``, porque comenta el resto de la línea.
        if re.search(r"\d(?:[\d.,]*\d)?\s*%", line):
            invalid.append(number)
    if invalid:
        joined = ", ".join(map(str, invalid))
        raise ValueError(f"Porcentajes LaTeX sin escapar en líneas: {joined}")


def compile_report(root: Path) -> Path:
    """Compila dos veces el TeX versionado y exige exactamente siete páginas."""

    report_dir = root / "entregables" / "informe" / "v6"
    tex_path = report_dir / "informe.tex"
    pdf_path = report_dir / "informe.pdf"
    if not tex_path.exists():
        raise FileNotFoundError(f"No existe el informe canónico: {tex_path}")

    _validate_percent_signs(tex_path)
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex no está disponible en PATH; no se generó el PDF.")

    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name]
    for _ in range(2):
        result = subprocess.run(
            command,
            cwd=report_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            tail = "\n".join((result.stdout + result.stderr).splitlines()[-35:])
            raise RuntimeError(f"Falló la compilación de informe.tex:\n{tail}")

    if not pdf_path.exists():
        raise RuntimeError(f"pdflatex finalizó sin producir {pdf_path}")

    try:
        import pymupdf

        with pymupdf.open(pdf_path) as document:
            pages = document.page_count
        if pages != 7:
            raise RuntimeError(f"El informe debe tener 7 páginas exactas; produjo {pages}.")
    except ImportError as exc:
        raise RuntimeError("PyMuPDF es necesario para validar el límite de 7 páginas.") from exc

    return pdf_path


if __name__ == "__main__":
    compile_report(Path(__file__).resolve().parents[2])
