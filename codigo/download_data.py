"""Descarga y extrae de forma reproducible los datos oficiales IEEE-CIS.

El script usa las credenciales configuradas localmente por Kaggle. Nunca imprime,
lee de forma explícita ni incorpora tokens al repositorio.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import kagglehub


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "raw"
HANDLE = "ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def download_one(filename: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    destination = RAW / filename
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"Ya existe: {destination}")
        return destination

    cached = Path(kagglehub.competition_download(HANDLE, filename))
    if zipfile.is_zipfile(cached):
        with zipfile.ZipFile(cached) as archive:
            members = [m for m in archive.namelist() if Path(m).name == filename]
            if len(members) != 1:
                raise RuntimeError(f"No se encontró una entrada única para {filename}")
            with archive.open(members[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
    else:
        shutil.copy2(cached, destination)

    print(f"Descargado: {destination} ({destination.stat().st_size:,} bytes)")
    return destination


if __name__ == "__main__":
    for required_file in FILES:
        download_one(required_file)
