# Ejecución reproducible de la V6

La V6 compara un baseline tabular sin orden (A), un modelo secuencial causal (B) y una fusión condicionada (C) sobre las mismas transacciones. El último 15 % se conserva como **benchmark temporal histórico reutilizado**; las decisiones nuevas se realizan dentro del 15 % de validación, dividido cronológicamente.

## 1. Datos

Descargue la competencia `ieee-fraud-detection` de Kaggle y deje estos archivos sin descomprimir en `datos/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

Los datos crudos, credenciales y tokens no deben versionarse. El pipeline también admite una ubicación externa mediante `PROYECTO1_RAW`.

## 2. Entorno

Desde la raíz del repositorio:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v6/requirements-v6.txt
python -m pip install -r configuracion/v6/requirements-docs-v6.txt
```

## 3. Experimento completo

```powershell
$env:PROYECTO1_RAW=(Resolve-Path datos/raw)
python -u codigo/v6/proyecto1_v6_pipeline.py
```

El proceso entrena con las 413,378 transacciones más antiguas, compara GRU y TCN, calibra, fija umbrales, ejecuta cinco permutaciones y tres recortes de historia, y escribe resultados en `artefactos/v6/` y `evidencia/figuras/v6/`.

## 4. Entregables y auditoría

```powershell
python codigo/v6/build_v6_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 entregables/cuaderno/v6/proyecto1_calderon_barillas.ipynb
python codigo/v6/report_v6.py
python codigo/v6/audit_project1_v6.py
```

La generación documental lee `artefactos/v6/resultados_v6.json`; por ello las cifras del notebook, README, informe, presentación y ficha provienen de una sola fuente de verdad.

## 5. Criterio de promoción

C solo puede sustituir al mejor modelo individual si, en evaluación interna posterior e independiente, aumenta AP al menos `0.01`, reduce el costo al menos `5 %` y mantiene recall de `0.75`. Una afirmación confirmatoria requiere una cohorte temporal nueva porque el benchmark final ya fue observado en V1–V5.
