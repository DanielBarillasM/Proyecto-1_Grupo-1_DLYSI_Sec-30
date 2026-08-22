# Ejecución reproducible · Proyecto 1 V5

V5 integra el núcleo rubricado A/B/C sin eliminar V1–V4. A reutiliza los expertos LightGBM V4; B es una GRU causal entrenada sobre secuencias de hasta 16 eventos; C fusiona puntajes A/B en un bloque temporal independiente.

## 1. Preparar el entorno

Desde la raíz del repositorio:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v5/requirements-v5.txt
python -m pip install -r configuracion/v5/requirements-docs-v5.txt
```

Las versiones están fijadas a las comprobadas en Windows 10, Python 3.13.1 y CPU.

## 2. Obtener IEEE-CIS

Primero acepte las reglas de la competencia `ieee-fraud-detection` en Kaggle. Configure su token fuera del repositorio y ejecute:

```powershell
python codigo/compartido/download_data.py
```

Se esperan:

```text
datos/raw/train_transaction.csv
datos/raw/train_identity.csv
```

Los CSV, `kaggle.json` y cualquier token están ignorados por Git y nunca deben versionarse.

## 3. Ejecutar V5

```powershell
python -u codigo/v5/proyecto1_v5_pipeline.py
```

El pipeline:

1. Carga únicamente las columnas necesarias para el modelo secuencial.
2. Ordena por `TransactionDT` y crea una identidad proxy.
3. Ajusta escalado y vocabularios solo con entrenamiento.
4. Construye secuencias causales de 16 eventos.
5. Recupera los puntajes de A-V4 y entrena B-GRU.
6. Ajusta C en un bloque independiente.
7. Calibra, fija umbrales y evalúa en bloques posteriores.
8. Ejecuta cinco permutaciones y recortes a 3/8 eventos.
9. Exporta pesos, predicciones, costos, contrato y figuras.

En CPU, la primera ejecución tarda aproximadamente 12–20 minutos. Si existe `artefactos/v5/modelo_B_gru_v5.pt`, se recupera el checkpoint. Para reentrenar intencionalmente:

```powershell
$env:V5_FORCE_RETRAIN='1'
python -u codigo/v5/proyecto1_v5_pipeline.py
Remove-Item Env:V5_FORCE_RETRAIN
```

## 4. Regenerar documentación

```powershell
python codigo/v5/build_v5_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/v5/proyecto1_calderon_barillas.ipynb
```

Para recompilar `informe.pdf` se requiere una distribución LaTeX con `pdflatex`. Para convertir la presentación se utiliza Microsoft Edge en modo headless.

## 5. Auditar

```powershell
python codigo/v5/audit_project1_v5.py
```

La auditoría verifica A/B/C, falsificaciones, hipótesis previa, economía, artefactos, notebook ejecutado, siete páginas máximas, ocho diapositivas, README rubricado, rutas y ausencia de secretos.

## Protocolo y alcance

- 70 % entrenamiento.
- 15 % validación subdividida cronológicamente.
- 15 % benchmark temporal histórico reutilizado.
- El benchmark no se describe como test ciego.
- Una promoción confirmatoria requiere una nueva cohorte etiquetada.

