# Ejecución reproducible de la versión 2

La V2 preserva los resultados originales en `artefactos/v1/` y escribe toda su evidencia nueva en rutas terminadas en `v2`. El último 15 % cronológico es un **benchmark histórico reutilizado** porque ya fue observado durante la V1; no debe describirse como una prueba ciega.

## 1. Crear el entorno

Desde la raíz del repositorio, en PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v2/requirements-v2.txt
python -m pip install -r configuracion/v2/requirements-docs-v2.txt
python -m pip install -r configuracion/v2/requirements-dev-v2.txt
```

Si la política de PowerShell impide activar el entorno, puede invocarse directamente `.\.venv\Scripts\python.exe` en los comandos siguientes.

## 2. Obtener IEEE-CIS Fraud Detection

Nunca guarde el token de Kaggle en Git. Después de configurar Kaggle en el perfil local, ejecute:

```powershell
python codigo/download_data.py
```

Los archivos esperados son `datos/raw/train_transaction.csv` y `datos/raw/train_identity.csv`. La carpeta `datos/raw/` está ignorada por Git.

## 3. Ejecutar la V2

```powershell
python -u codigo/proyecto1_v2_pipeline.py
python codigo/postprocess_v2.py
python codigo/compare_versions.py
```

El proceso carga las 435 columnas disponibles entre ambas tablas (434 después de resolver la llave duplicada), construye variables causales, audita correlación y ruido, elimina redundancia, evalúa PCA, compara modelos mediante tres ventanas walk-forward y congela los resultados en `artefactos/v2/resultados_v2.json`.

## 4. Reconstruir documentación

```powershell
python codigo/build_deliverables_v2.py
python codigo/crear_ficha_repositorio_v2.py
python codigo/audit_project1_v2.py
```

Para compilar el informe manualmente cuando exista una distribución TeX:

```powershell
pdflatex -interaction=nonstopmode -output-directory entregables/informe entregables/informe/informe_proyecto1_v2.tex
pdflatex -interaction=nonstopmode -output-directory entregables/informe entregables/informe/informe_proyecto1_v2.tex
```

## 5. Fuente única y verificación

Las cifras de README, cuaderno, informe, presentación y ficha deben provenir de `artefactos/v2/resultados_v2.json`. El auditor comprueba estructura, ocho diapositivas, límite de siete páginas, referencias, ausencia de secretos, consistencia numérica y conservación de V1.

No se debe volver a seleccionar variables, hiperparámetros ni umbrales mirando el benchmark histórico. Una futura afirmación confirmatoria requiere una cohorte temporal nueva y etiquetada.
