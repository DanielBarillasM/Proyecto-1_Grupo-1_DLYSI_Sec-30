# Ejecución reproducible de V3

Ejecute todos los comandos desde la raíz del repositorio. V3 fue validada con Python 3.13.1 en Windows y CPU.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v3/requirements-v3.txt
python -m pip install -r configuracion/v3/requirements-docs-v3.txt
```

Descargue IEEE-CIS únicamente si `datos/raw/train_transaction.csv` y `train_identity.csv` no existen:

```powershell
python codigo/compartido/download_data.py
```

Entrene y evalúe V3:

```powershell
python -u codigo/v3/proyecto1_v3_pipeline.py
python codigo/v3/postprocess_v3.py
python codigo/v3/finalize_v3.py
```

La ejecución completa tarda aproximadamente 28 minutos en el equipo comprobado. `finalize_v3.py` vuelve a cargar los datos para garantizar que los segmentos correspondan al umbral recomendado.

Construya y audite los entregables:

```powershell
python codigo/v3/build_v3_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/v3/Proyecto_1_Monitoreo_Transaccional_V3.ipynb
python codigo/v3/audit_project1_v3.py
```

La fuente única de verdad es `artefactos/v3/resultados_v3.json`. El umbral recomendado balancea F1 y un recall mínimo de 70%; el umbral económico alternativo minimiza `Q4,200·FN + Q180·FP`.

El último 15% es un benchmark histórico reutilizado. No debe describirse como test ciego ni usarse para decidir hiperparámetros o promoción.
