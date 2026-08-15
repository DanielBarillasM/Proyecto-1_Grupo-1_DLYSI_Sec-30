# Proyecto 1 — Monitoreo transaccional

**Grupo 1 · Sección 30 · Universidad del Valle de Guatemala**

- Wilson Alejandro Calderón Argueta — 22018
- Pablo Daniel Barillas Moreno — 22193

> Comparación controlada entre una línea base agregada, una GRU secuencial y una
> arquitectura híbrida sobre IEEE-CIS Fraud Detection. El objetivo es determinar
> si el orden temporal aporta valor predictivo y económico verificable.

**Repositorio:** [DanielBarillasM/Proyecto-1_Grupo-1_DLYSI_Sec-30](https://github.com/DanielBarillasM/Proyecto-1_Grupo-1_DLYSI_Sec-30)

## Entregables

- [`entregables/cuaderno/proyecto1_calderon_barillas.ipynb`](../entregables/cuaderno/proyecto1_calderon_barillas.ipynb): investigación ejecutada.
- [`entregables/informe/informe.pdf`](../entregables/informe/informe.pdf): informe ejecutivo de cuatro páginas; también se incluye su fuente LaTeX.
- [`entregables/presentacion/presentacion.html`](../entregables/presentacion/presentacion.html): presentación interactiva y autocontenida; su versión PDF contiene ocho diapositivas.
- [`entregables/ficha/Ficha_Repositorio_Proyecto1.docx`](../entregables/ficha/Ficha_Repositorio_Proyecto1.docx): ficha descriptiva editable del repositorio.
- [`artefactos/`](../artefactos): modelos A/B/C, candidato, umbrales, preprocesamiento y contrato de entrada.

## Resultado principal

| Modelo | Diseño | AUC-PR test | Recall | Costo de prueba |
|---|---|---:|---:|---:|
| A | Gradient boosting sobre agregados | 0.429 | 0.722 | Q6,193,620 |
| B | Embeddings + GRU(32) | 0.412 | 0.649 | Q6,525,960 |
| C | GRU fusionada con agregados | 0.392 | 0.671 | Q6,618,900 |

El candidato congelado es **A**. La caída de AUC-PR al permutar el
historial fue 0.002, inferior al criterio previo de 0.01. En este
experimento no se obtuvo evidencia suficiente para justificar una migración al
modelo secuencial.

## Datos y reproducción

El proyecto utiliza `train_transaction.csv` y `train_identity.csv` de [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection). Los datos no se versionan.

```powershell
python -m pip install -r configuracion/requirements.txt
python -c "import kagglehub; kagglehub.login()"
python codigo/download_data.py
python codigo/proyecto1_pipeline.py
python codigo/build_deliverables.py
jupyter nbconvert --to notebook --execute --inplace entregables/cuaderno/proyecto1_calderon_barillas.ipynb
Push-Location entregables/informe
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
Pop-Location
python codigo/crear_ficha_repositorio.py
python codigo/audit_project1.py
```

Los CSV quedan en `datos/raw/`. La semilla principal es 2026. El preprocesamiento se ajusta solo con entrenamiento. El test cronológico se abre después de congelar candidato y umbrales.

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
.github/                    README visible en GitHub
artefactos/                 modelos, umbrales, esquema y métricas
codigo/                     pipeline, descarga, construcción y auditoría
configuracion/              dependencias reproducibles
datos/raw/                  archivos Kaggle no versionados
entregables/cuaderno/       notebook ejecutado
entregables/informe/        fuente LaTeX y PDF
entregables/presentacion/   presentación HTML y PDF
entregables/ficha/          ficha DOCX del repositorio
evidencia/figuras/          evidencia visual reproducible
legal/                      licencia del repositorio
```
