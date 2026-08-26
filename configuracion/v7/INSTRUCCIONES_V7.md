# Ejecución reproducible de Proyecto 1 · V7

La V7 amplía el baseline A con todas las familias de variables IEEE-CIS, correlación train-only, PCA del bloque V, CatBoost, regresión logística y stacking; integra B secuencial, C de fusión y D encoder–decoder bajo el mismo protocolo temporal. El benchmark final es histórico y reutilizado.

## 1. Datos

La fuente es [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview), publicada en Kaggle con datos anonimizados de Vesta Corporation. Después de aceptar las reglas, deje estos archivos en `datos/raw/`:

- `train_transaction.csv`
- `train_identity.csv`

Los CSV, `kaggle.json`, tokens y credenciales están excluidos del repositorio. También se admite una ruta externa mediante `PROYECTO1_RAW`.

## 2. Entorno

Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v7/requirements-v7.txt
python -m pip install -r configuracion/v7/requirements-docs-v7.txt
```

La exportación del informe requiere una distribución LaTeX con `pdflatex`; la presentación PDF requiere Google Chrome.

## 3. Protocolo congelado

Antes de entrenar, lea `configuracion/v7/PROTOCOLO_EXPERIMENTAL_V7.md`. Resume particiones, modelos, hipótesis C, falsificaciones, costos y gates. No se deben modificar reglas después de observar evaluación o benchmark.

## 4. Corrida completa

```powershell
$env:PROYECTO1_RAW=(Resolve-Path datos/raw)
python -u codigo/v7/proyecto1_v7_pipeline.py
```

La corrida usa las 590,540 filas, ajusta A0–A5, integra B/D congelados, compara C1–C3, calibra, selecciona umbrales y ejecuta walk-forward. En CPU puede tomar entre 25 y 45 minutos; CatBoost es la fase más larga.

Si los modelos base ya están entrenados y solo debe reconstruirse el stacking corregido A5 y las métricas comunes:

```powershell
python -u codigo/v7/finalize_v7_cached.py
```

## 5. Notebooks y documentación

```powershell
python codigo/v7/build_notebooks_v7.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 entregables/cuaderno/v7/proyecto1_calderon_barillas.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 entregables/cuaderno/v7/EDA_IEEE_CIS_Diagnostico_Datos_V7.ipynb
python codigo/v7/report_v7.py
python codigo/v7/presentation_v7.py
python codigo/v7/build_documentation_v7.py
python codigo/v7/audit_project1_v7.py
```

La documentación lee `artefactos/v7/resultados_v7.json` como fuente única de verdad. El informe debe producir siete páginas y la presentación ocho diapositivas.

## 6. Lectura del resultado

A5 mejora AP y costo promedio respecto de V6, pero no supera el gate de estabilidad porque una de cuatro ventanas cae más de 0.005 AP. Por tanto, se conserva como candidato exploratorio y no como reemplazo confirmatorio. B no demuestra valor material del orden; C y D no se promueven.

## 7. Salidas principales

```text
artefactos/v7/                 modelos, scores, calibradores, umbrales y contrato
datos/processed/v7/            asociación, correlación y auditoría de variables
evidencia/figuras/v7/          figuras reproducibles
entregables/cuaderno/v7/       notebook oficial y EDA ejecutados
entregables/informe/v7/        informe.tex e informe.pdf (7 páginas)
entregables/presentacion/v7/   presentación HTML/PDF y guion (8 diapositivas)
entregables/ficha/v7/          ficha DOCX/PDF del repositorio
```
