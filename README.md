<div align="center">

# Proyecto 1 · Monitoreo transaccional

### ¿El orden aporta información más allá de los agregados?

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-184e77)
![Versión oficial](https://img.shields.io/badge/REVISAR-V5-2a9d8f)

**Wilson Alejandro Calderón Argueta · 22018** · **Pablo Daniel Barillas Moreno · 22193**

Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Sección 30 · 2026

</div>

> [!IMPORTANT]
> **La versión que debe revisarse y calificarse es V5.** V1–V4 se conservan únicamente como historial experimental y evidencia de la evolución del proyecto. Los entregables oficiales están en las carpetas `v5` de `entregables/`, el código reproducible está en `codigo/v5/` y las instrucciones completas están en `configuracion/v5/INSTRUCCIONES_V5.md`.

> [!NOTE]
> El último 15 % de IEEE-CIS ya fue observado en iteraciones anteriores y se reporta como benchmark temporal histórico reutilizado. Todas las decisiones V5 se toman dentro de validación. Una promoción confirmatoria exige una cohorte nueva.

## Navegación rápida

- [Versión oficial V5](#versión-oficial-v5)
- [Historial de versiones](#historial-de-versiones)
- [Resultados V5](#resultados-v5-y-decisión-económica)
- [Reproducción](#reproducción-de-v5)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Limitaciones y uso responsable](#limitaciones-y-uso-responsable)

## Versión oficial V5

| Entregable para revisión | Ruta |
|---|---|
| Notebook ejecutado | [`entregables/cuaderno/v5/proyecto1_calderon_barillas.ipynb`](entregables/cuaderno/v5/proyecto1_calderon_barillas.ipynb) |
| Informe final | [`entregables/informe/v5/informe.pdf`](entregables/informe/v5/informe.pdf) |
| Fuente LaTeX | [`entregables/informe/v5/informe.tex`](entregables/informe/v5/informe.tex) |
| Presentación HTML | [`entregables/presentacion/v5/presentacion.html`](entregables/presentacion/v5/presentacion.html) |
| Presentación PDF | [`entregables/presentacion/v5/presentacion.pdf`](entregables/presentacion/v5/presentacion.pdf) |
| Guion de exposición | [`entregables/presentacion/v5/guion_exposicion_v5.md`](entregables/presentacion/v5/guion_exposicion_v5.md) |
| Ficha del repositorio | [`entregables/ficha/v5/Ficha_Repositorio_Proyecto_1_V5.pdf`](entregables/ficha/v5/Ficha_Repositorio_Proyecto_1_V5.pdf) |
| Instrucciones de ejecución | [`configuracion/v5/INSTRUCCIONES_V5.md`](configuracion/v5/INSTRUCCIONES_V5.md) |

V5 es la integración rubricada del experimento A/B/C. Compara sobre filas, horizonte y protocolo comunes: A, un baseline tabular competitivo; B, una GRU causal; y C, una fusión leakage-safe de ambos puntajes. También incorpora las falsificaciones del orden, una hipótesis previa para C, selección de umbrales con costo y los artefactos necesarios para auditar los resultados.

## Historial de versiones

| Versión | Propósito y cambio principal | Estado | Entregables |
|---|---|---|---|
| **V1** | Primera implementación del problema de monitoreo. Introdujo el protocolo temporal y la comparación inicial A/B/C con HistGradientBoosting, GRU e híbrido. | Histórica; referencia de la rúbrica original. | [`cuaderno`](entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb) · [`informe`](entregables/informe/v1/informe.pdf) · [`presentación`](entregables/presentacion/v1/presentacion.html) |
| **V2** | Amplió la integración de IEEE-CIS, exploró correlación, poda de variables, PCA, LightGBM y stacking, además de validación walk-forward y calibración. | Histórica; mejora de datos y protocolo. | [`cuaderno`](entregables/cuaderno/v2/Proyecto_1_Monitoreo_Transaccional_V2.ipynb) · [`informe`](entregables/informe/v2/informe_proyecto1_v2.pdf) · [`presentación`](entregables/presentacion/v2/presentacion_proyecto1_v2.html) |
| **V3** | Incorporó mayor recencia, variables causales, categorías nativas, baselines de regresión logística con y sin PCA y un criterio explícito de promoción. | Histórica; versión promovida en su momento. | [`cuaderno`](entregables/cuaderno/v3/Proyecto_1_Monitoreo_Transaccional_V3.ipynb) · [`informe`](entregables/informe/v3/informe_proyecto1_v3.pdf) · [`presentación`](entregables/presentacion/v3/presentacion_proyecto1_v3.html) |
| **V4** | Extendió la señal a cientos de variables, comparó LightGBM, CatBoost y XGBoost, añadió expertos por `ProductCD`, hard negatives, calibración y selección de candidato. | Histórica; suministra el modelo tabular A reutilizado por V5. | [`cuaderno`](entregables/cuaderno/v4/Proyecto_1_Monitoreo_Transaccional_V4.ipynb) · [`informe`](entregables/informe/v4/informe_proyecto1_v4.pdf) · [`presentación`](entregables/presentacion/v4/presentacion_proyecto1_v4.html) |
| **V5** | Recuperó e integró el núcleo A/B/C exigido por la rúbrica, entrenó la GRU, evaluó el valor del orden mediante falsificaciones y documentó la decisión económica final. | **Oficial · revisar y calificar esta versión.** | [`cuaderno`](entregables/cuaderno/v5/proyecto1_calderon_barillas.ipynb) · [`informe`](entregables/informe/v5/informe.pdf) · [`presentación`](entregables/presentacion/v5/presentacion.html) |

Cada versión conserva su código, configuración, artefactos, figuras y entregables dentro de una subcarpeta homónima. Las versiones históricas no deben combinarse manualmente para calificar V5; su función es permitir trazabilidad y comparación.

## Resumen ejecutivo de V5

El proyecto estudia 590,540 transacciones IEEE-CIS, con 20,663 fraudes y prevalencia de 3.50 %. Compara una línea tabular competitiva sin leer eventos ordenados (A), una GRU causal sobre secuencias de hasta 16 eventos (B) y una fusión de sus puntajes (C). A obtiene AP interna 0.5445, B obtiene 0.4245 y C obtiene 0.5425.

La permutación controlada no perjudica a B: su AP cambia de 0.4245 a 0.4355 ± 0.0021. La diferencia original−permutada es −0.0111; por tanto, la evidencia no permite afirmar que el orden aporte señal adicional. C tampoco supera su criterio previo: el AP cambia −0.0020 y el costo aumenta 0.43 % frente a A. El candidato final es A.

## Datos y protocolo temporal

Se utiliza la competencia pública IEEE-CIS Fraud Detection de Kaggle. `train_transaction.csv` y `train_identity.csv` se unen por `TransactionID`, las filas se ordenan por `TransactionDT` y ambos campos se excluyen como magnitudes predictivas. La identidad secuencial es una aproximación formada por `card1`, `card2`, `card3`, `card5` y `addr1`.

La separación es 70 % entrenamiento, 15 % validación y 15 % benchmark histórico. Validación se subdivide cronológicamente en early stopping, ajuste de C, calibración, umbral y evaluación interna. Imputación, escalado y vocabularios se aprenden exclusivamente con entrenamiento. Las características históricas utilizan solo eventos anteriores; no se aplican particiones aleatorias.

## Modelos A/B/C

| Pieza | Diseño | Resultado interno |
|---|---|---:|
| A | LightGBM V4 con expertos `ProductCD=W/NO-W` | AP 0.5445 · costo Q1,067,460 |
| B | Embeddings + GRU(64), hasta 16 eventos | AP 0.4245 · costo Q1,320,840 |
| C | Regresión logística sobre A, B, monto, historia y producto | AP 0.5425 · costo Q1,072,020 |

A incorpora la ingeniería causal y selección V4. B utiliza 57 variables numéricas por evento, 12 categorías, BCE ponderada, AdamW, clipping y early stopping. C se entrena en un bloque independiente con puntajes A/B y no recibe el benchmark para decidir su arquitectura.

## Hipótesis y falsificaciones

**Hipótesis previa de C:** fusionar el puntaje tabular de LightGBM con el puntaje secuencial de la GRU mejoraría el AUC-PR al representar información complementaria. Se fijó como criterio de utilidad incrementar AUC-PR al menos 0.01 y reducir el costo al menos 5 % frente al mejor modelo individual en la evaluación interna temporal. La hipótesis no se cumplió.

Las dos pruebas de falsificación son:

1. Permutación de antecedentes con cinco semillas, manteniendo la transacción objetivo al final.
2. Recorte de la historia a 3 y 8 eventos.

La historia de 3 eventos obtiene AP 0.4253 y la de 8 obtiene 0.4225. Ninguna de las pruebas justifica afirmar que el orden mejora el detector.

## Resultados V5 y decisión económica

| Modelo | AUC-PR benchmark | Precisión | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|
| **A** | **0.559** | **22.15 %** | 73.27 % | **0.340** | **Q4,890,000** |
| B | 0.218 | 12.55 % | 79.11 % | 0.217 | Q5,763,000 |
| C | 0.562 | 16.06 % | **79.37 %** | 0.267 | Q4,972,680 |

La política de umbral minimiza $4200FN+180FP$ sujeta a recall ≥ 0.75 en selección. El umbral de A es 0.05783. En el escenario central de 12 transacciones por tarjeta al mes, A representa un costo mensual proyectado de Q927,422,359. Es una extrapolación académica, no una cifra contable.

## Tres decisiones técnicas importantes

1. **A tabular V4 frente al HistGradientBoosting V1.** Se eligió LightGBM con expertos porque usa más variables causales y obtiene AP y costo claramente mejores.
2. **GRU frente a LSTM, TCN o Transformer.** Se eligió GRU por eficiencia en CPU y porque la rúbrica evalúa evidencia del orden, no complejidad. La permutación comprueba si la red realmente aprovecha la secuencia.
3. **Fusión tardía frente al híbrido interno V1.** Se eligió stacking logístico con puntajes A/B porque separa el aporte de cada modelo, permite controles temporales y reduce el riesgo de reconstruir toda la función tabular dentro de una red opaca.

## Reproducción de V5

Ejecute los comandos desde la raíz del repositorio:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configuracion/v5/requirements-v5.txt
python -m pip install -r configuracion/v5/requirements-docs-v5.txt
python codigo/compartido/download_data.py
python -u codigo/v5/proyecto1_v5_pipeline.py
python codigo/v5/build_v5_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/v5/proyecto1_calderon_barillas.ipynb
python codigo/v5/audit_project1_v5.py
```

La descarga requiere aceptar las reglas de IEEE-CIS en Kaggle y configurar las credenciales fuera del repositorio. Los CSV crudos, de casi 700 MB, no se versionan. Para reproducir una versión anterior, consulte su archivo de instrucciones en `configuracion/v1` a `configuracion/v4` y ejecute los scripts de la carpeta equivalente en `codigo/`.

## Estructura del repositorio

```text
README.md                         índice general; V5 es la entrega oficial
codigo/
├── compartido/                   descarga del conjunto IEEE-CIS
├── v1/ ... v4/                   código histórico por iteración
└── v5/                           pipeline, constructor y auditoría oficiales
configuracion/
├── v1/ ... v4/                   dependencias e instrucciones históricas
└── v5/                           entorno e instrucciones de la entrega oficial
datos/
├── raw/                          CSV locales ignorados por Git
└── processed/v3 ... v5/          evidencia intermedia versionada por generación
artefactos/v1 ... v5/             modelos, predicciones, métricas y contratos
evidencia/
├── figuras/v1 ... v5/            gráficos separados por versión
└── recursos/v1 ... v5/           códigos QR separados por versión
entregables/
├── cuaderno/v1 ... v5/           notebooks
├── informe/v1 ... v5/            fuentes LaTeX y PDF
├── presentacion/v1 ... v5/       HTML y PDF
└── ficha/v1 ... v5/              fichas DOCX/PDF disponibles
legal/                            licencia del repositorio
```

`datos/raw/` es compartida porque todas las versiones utilizan la misma fuente IEEE-CIS y sus archivos no pertenecen al control de versiones. Los artefactos V5 mantienen referencias explícitas a modelos V4 cuando forman parte del candidato A; esto representa linaje experimental, no una ruta rota.

## Candidato al Proyecto Final

- **Modelo conservado:** A — LightGBM V4 con expertos por `ProductCD` y calibrador V5.
- **Artefactos:** `artefactos/v4/modelo_experto_w_v4.txt`, `artefactos/v4/modelo_experto_no_w_v4.txt`, `artefactos/v5/calibradores_v5.joblib` y `artefactos/v5/contrato_entrada_salida_v5.json`.
- **Usuario:** analista de riesgo o equipo de monitoreo transaccional.
- **Decisión:** ordenar alertas y priorizar revisión; el puntaje no prueba fraude ni autoriza bloqueo autónomo.
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
