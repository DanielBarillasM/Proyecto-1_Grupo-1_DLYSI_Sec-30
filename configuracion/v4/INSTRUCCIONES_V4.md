# Ejecución reproducible de V4

Ejecute los comandos desde la raíz del repositorio. V4 fue validada con Python 3.13.1, Windows y ocho CPU lógicas.

## 1. Entorno

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v4/requirements-v4.txt
python -m pip install -r configuracion/v4/requirements-docs-v4.txt
```

## 2. Datos

Descargue IEEE-CIS solo si faltan `datos/raw/train_transaction.csv` o `datos/raw/train_identity.csv`:

```powershell
python codigo/download_data.py
```

Los CSV crudos permanecen fuera de Git. No incluya credenciales ni tokens de Kaggle en el repositorio.

## 3. Entrenamiento y evaluación

```powershell
python -u codigo/proyecto1_v4_pipeline.py
python codigo/postprocess_v4.py
```

El pipeline realiza ingeniería causal, selección en el 55 % inicial, 18 pruebas Optuna, tres folds walk-forward, LightGBM global y hard-negative, expertos `ProductCD`, CatBoost/XGBoost piloto, stacking, calibración, umbrales, bootstrap y segmentos. Si los archivos `optuna_lightgbm_v4.csv` y `mejores_parametros_lightgbm_v4.json` existen, reutiliza esa búsqueda para evitar repetirla accidentalmente.

CatBoost y XGBoost usan presupuestos piloto de 120 y 220 iteraciones. Una configuración CatBoost de 320 iteraciones superó 20 minutos por fold en el equipo de prueba y se descartó por costo computacional antes de producir la tabla definitiva.

`postprocess_v4.py` selecciona el candidato usando 50–60 % de validación, calibra con 60–70 %, selecciona umbral con 70–85 % y evalúa en 85–100 %. La política robusta `recall >= 0.75` se etiqueta post-hoc y debe validarse sin cambios en una cohorte nueva.

## 4. Entregables

```powershell
python codigo/build_v4_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/Proyecto_1_Monitoreo_Transaccional_V4.ipynb
python codigo/audit_project1_v4.py
```

Se generan:

- `.github/README.md`.
- Notebook V4 ejecutable.
- Informe LaTeX y PDF.
- Presentación HTML y PDF de ocho diapositivas.
- Ficha DOCX y PDF.
- Figuras, modelos y predicciones bajo carpetas `v4`.

## 5. Fuente y decisión

La fuente única de verdad es `artefactos/v4/resultados_v4.json`.

V4 logra ROC-AUC mayor a 0.90 y mejora todas las métricas frente a V3 bajo la política robusta. Sin embargo, esa política se añadió después de observar la primera evaluación V4. Por tanto:

1. V3 permanece como versión confirmada.
2. V4 queda congelada como candidato superior post-hoc.
3. La siguiente prueba debe usar una cohorte temporal nueva.
4. No se deben cambiar variables, hiperparámetros, calibración ni umbral después de verla.

El último 15 % actual es un benchmark histórico reutilizado y nunca debe describirse como test ciego.
