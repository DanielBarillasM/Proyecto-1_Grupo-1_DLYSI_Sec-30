# Proyecto 1 — Monitoreo transaccional

**Grupo 1 · Sección 30 · Universidad del Valle de Guatemala**

- Wilson Alejandro Calderón Argueta — 22018
- Pablo Daniel Barillas Moreno — 22193

## Entregables

- `proyecto1_calderon_barillas.ipynb`: investigación ejecutada.
- `informe.tex` / `informe.pdf`: informe para comité, máximo siete páginas.
- `presentacion.html` / `presentacion.pdf`: ocho diapositivas.
- `artefactos/`: modelos A/B/C, candidato, umbrales, preprocesamiento y contrato.
- `src/proyecto1_pipeline.py`: pipeline reproducible.
- `src/download_data.py`: descarga y extracción segura de los CSV oficiales.

## Datos y reproducción

El proyecto utiliza `train_transaction.csv` y `train_identity.csv` de [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection). Los datos no se versionan.

```powershell
python -m pip install -r requirements.txt
python -c "import kagglehub; kagglehub.login()"
python src/download_data.py
python src/proyecto1_pipeline.py
python build_deliverables.py
jupyter nbconvert --to notebook --execute --inplace proyecto1_calderon_barillas.ipynb
pdflatex -interaction=nonstopmode informe.tex
```

Los CSV deben quedar en `data/raw/`. La semilla principal es 2026. El preprocesamiento se ajusta solo con entrenamiento. El test cronológico se abre después de congelar candidato y umbrales.

## Tres decisiones técnicas importantes

1. **IEEE-CIS frente a `creditcard.csv`:** se eligió IEEE-CIS porque contiene tiempo y atributos de tarjeta que permiten una clave secuencial aproximada. La alternativa europea no permite agrupar por tarjeta.
2. **GRU frente a LSTM/Transformer:** ocho eventos y CPU favorecen una GRU pequeña. La complejidad no era el objetivo; la falsificación del orden sí.
3. **Umbral económico frente a 0.5:** se minimiza `4200*FN + 180*FP` en validación. El test nunca elige el umbral.

## Uso de inteligencia artificial

Se utilizó IA para estructurar código, revisar consistencia, localizar bibliografía y diseñar HTML/CSS/LaTeX. Los integrantes verificaron datos, partición temporal, formas, ejecución, métricas, falsificaciones, umbral y archivos finales. La IA no se usó como fuente académica ni sustituyó la interpretación de resultados.

## Candidato al Proyecto Final

- **Modelo:** pieza A; se recomienda conservar como referencia `A`. Artefacto: `artefactos/modelo_candidato_A.*`.
- **Usuario:** analista de riesgo o motor de autorización; el puntaje prioriza revisión o solicita autenticación adicional.
- **Entrada preliminar:** hasta ocho eventos cronológicos con diez variables numéricas, seis categóricas y la clave compuesta documentada en `esquema_entrada.json`.
- **Salida:** `risk_score` continuo en `[0,1]`, umbral congelado y decisión sugerida.
- **Límites:** identidad aproximada, anonimización, prevalencia externa, deriva, latencia, calibración y revisión humana pendientes.

## Estructura

```text
artefactos/  modelos, umbrales, esquema y métricas
data/raw/    archivos Kaggle no versionados
figuras/     evidencia visual reproducible
src/         pipeline
```
