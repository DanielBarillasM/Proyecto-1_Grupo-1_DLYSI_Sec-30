<div align="center">

# Proyecto 1 · Monitoreo transaccional · V5 integrada

### ¿El orden aporta información más allá de los agregados?

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-184e77)
![Estado](https://img.shields.io/badge/Candidato-A__V4-2a9d8f)

**Wilson Alejandro Calderón Argueta · 22018** · **Pablo Daniel Barillas Moreno · 22193**

Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Sección 30 · 2026

</div>

> [!IMPORTANT]
> El último 15 % de IEEE-CIS ya fue observado en iteraciones anteriores y se reporta como benchmark temporal histórico reutilizado. Todas las decisiones V5 se toman dentro de validación. Una promoción confirmatoria exige una cohorte nueva.

## Resumen ejecutivo

El proyecto estudia 590,540 transacciones IEEE-CIS, con 20,663 fraudes y prevalencia 3.50%. Compara una línea tabular competitiva sin leer eventos ordenados (A), una GRU causal sobre secuencias de hasta 16 eventos (B) y una fusión leakage-safe de sus puntajes (C). A obtiene AP interna 0.5445, mientras B obtiene 0.4245 y C 0.5425.

La permutación controlada no perjudica a B: su AP cambia de 0.4245 a 0.4355 ± 0.0021. La diferencia original−permutada es -0.0111. Con esta evidencia no se afirma que el orden aporte. C tampoco supera su criterio previo: cambio AP -0.0020 y reducción de costo -0.43%. El candidato es A.

## Datos y protocolo temporal

Se utiliza la competencia pública IEEE-CIS Fraud Detection de Kaggle. `train_transaction.csv` y `train_identity.csv` se unen por `TransactionID`, las filas se ordenan por `TransactionDT` y ambos campos se excluyen como magnitudes predictivas. La identidad secuencial es una clave aproximada formada por `card1`, `card2`, `card3`, `card5` y `addr1`.

La separación es 70 % entrenamiento, 15 % validación y 15 % benchmark histórico. Validación se subdivide cronológicamente en early stopping, ajuste de C, calibración, umbral y evaluación interna. Imputación, escalado y vocabularios se aprenden exclusivamente con entrenamiento. Las características históricas utilizan solo eventos anteriores; no se aplican particiones aleatorias.

## Modelos A/B/C

| Pieza | Diseño | Resultado interno |
|---|---|---:|
| A | LightGBM V4 con expertos `ProductCD=W/NO-W` | AP 0.5445 · costo Q1,067,460 |
| B | Embeddings + GRU(64), hasta 16 eventos | AP 0.4245 · costo Q1,320,840 |
| C | Regresión logística sobre A, B, monto, historia y producto | AP 0.5425 · costo Q1,072,020 |

A incorpora la ingeniería causal y selección V4. B utiliza 57 variables numéricas por evento, 12 categorías, BCE ponderada, AdamW, clipping y early stopping. C se entrena en un bloque independiente con puntajes A/B y no recibe el benchmark para decidir su arquitectura.

## Hipótesis y falsificaciones

**Hipótesis previa de C:** Creemos que fusionar el puntaje tabular de LightGBM con el puntaje secuencial de la GRU mejorará el AUC-PR porque ambos modelos representan información complementaria. Lo consideraremos útil si incrementa AUC-PR al menos 0.01 y reduce el costo al menos 5% frente al mejor modelo individual en la evaluación interna de validación temporal.

La hipótesis no se cumple. C pierde 0.0020 de AP y aumenta el costo 0.43 % frente a A en evaluación interna.

Las dos pruebas obligatorias son:

1. Permutación de antecedentes con cinco semillas, manteniendo la transacción objetivo al final.
2. Recorte de la historia a 3 y 8 eventos.

La historia de 3 eventos obtiene AP 0.4253 y la de 8 obtiene 0.4225. Ninguna evidencia justifica afirmar que el orden mejore el detector.

## Resultados y decisión económica

| Modelo | AUC-PR benchmark | Precisión | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|
| **A** | **0.559** | **22.15%** | 73.27% | **0.340** | **Q4,890,000** |
| B | 0.218 | 12.55% | 79.11% | 0.217 | Q5,763,000 |
| C | 0.562 | 16.06% | **79.37%** | 0.267 | Q4,972,680 |

La política de umbral minimiza $4200FN+180FP$ sujeta a recall ≥ 0.75 en selección. El umbral de A es 0.05783. En el escenario central de 12 transacciones por tarjeta al mes, A representa un costo mensual proyectado de Q927,422,359. Es una extrapolación académica, no una cifra contable.

## Tres decisiones técnicas importantes

1. **A tabular V4 frente al HistGradientBoosting V1.** Se consideró conservar el baseline antiguo. Se eligió LightGBM con expertos porque usa más variables causales y obtiene AP y costo claramente mejores.
2. **GRU frente a LSTM, TCN o Transformer.** Se eligió GRU por eficiencia en CPU, menor cantidad de parámetros y porque la rúbrica evalúa evidencia del orden, no complejidad. La permutación permitió comprobar si realmente aprovechó la secuencia.
3. **Fusión tardía frente al híbrido interno V1.** Se eligió stacking logístico con puntajes A/B porque separa el aporte de cada modelo, permite controles temporales y reduce el riesgo de que C reconstruya toda la función tabular dentro de una red opaca.

## Reproducción

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configuracion/v5/requirements-v5.txt
python -m pip install -r configuracion/v5/requirements-docs-v5.txt
python codigo/download_data.py
python -u codigo/proyecto1_v5_pipeline.py
python codigo/build_v5_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/proyecto1_calderon_barillas.ipynb
python codigo/audit_project1_v5.py
```

La descarga requiere aceptar las reglas de IEEE-CIS en Kaggle y configurar las credenciales fuera del repositorio. Los CSV de casi 700 MB no se versionan.

## Estructura

```text
codigo/                 pipeline, construcción y auditoría
configuracion/v5/       versiones exactas e instrucciones
datos/raw/              CSV locales ignorados por Git
artefactos/v5/          pesos A/B/C, calibradores, contrato y puntajes
evidencia/figuras/v5/   resultados reproducibles
entregables/cuaderno/   notebook ejecutado
entregables/informe/    fuente LaTeX y PDF
entregables/presentacion/ HTML y PDF de ocho diapositivas
entregables/ficha/       ficha del repositorio
```

## Candidato al Proyecto Final

- **Modelo conservado:** A — LightGBM V4 con expertos por `ProductCD` y calibrador V5.
- **Artefactos:** `artefactos/v4/modelo_experto_w_v4.txt`, `modelo_experto_no_w_v4.txt`, `artefactos/v5/calibradores_v5.joblib` y `contrato_entrada_salida_v5.json`.
- **Usuario:** analista de riesgo o equipo de monitoreo transaccional.
- **Decisión:** ordenar alertas y priorizar revisión; el puntaje no prueba fraude ni autoriza bloqueo autónomo.
- **Entrada preliminar:** transacción actual, variables categóricas y estadísticas históricas causales especificadas por el contrato.
- **Salida:** `risk_score` continuo en [0,1], umbral 0.05783 y política de revisión.
- **Pendientes:** nueva cohorte etiquetada, identidad bancaria fiable, costos reales, latencia, privacidad, equidad, seguridad, explicaciones y monitoreo.

## Limitaciones y uso responsable

IEEE-CIS está anonimizado y cubre aproximadamente 182 días. La clave proxy no equivale a un cliente real. El benchmark ya fue observado y ninguna conclusión se presenta como confirmación externa. Los costos y volúmenes mensuales son escenarios. El sistema debe apoyar revisión humana, no atribuir culpabilidad ni bloquear de manera autónoma.

## Declaración de uso de inteligencia artificial

Se utilizó asistencia de IA para estructurar y revisar código, diseñar documentación HTML/CSS/LaTeX, localizar bibliografía y automatizar auditorías. Los integrantes ejecutaron el pipeline y verificaron particiones, alineación de IDs, métricas, falsificaciones, umbrales y artefactos. La IA no se utilizó como fuente académica ni reemplaza la defensa de las decisiones.

## Referencias APA 7

Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP*, 1724–1734. https://doi.org/10.3115/v1/D14-1179

IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection* [Data set]. Kaggle. https://www.kaggle.com/competitions/ieee-fraud-detection

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
