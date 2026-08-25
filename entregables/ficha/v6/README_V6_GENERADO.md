<div align="center">

# Proyecto 1 · Monitoreo transaccional · V6 integrada

### ¿El orden aporta información más allá de los agregados?

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-184e77)
![Estado](https://img.shields.io/badge/Candidato-A__V4-2a9d8f)

**Wilson Alejandro Calderón Argueta · 22018** · **Pablo Daniel Barillas Moreno · 22193**

Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Sección 30 · 2026

</div>

> [!IMPORTANT]
> El último 15 % de IEEE-CIS ya fue observado en iteraciones anteriores y se reporta como benchmark temporal histórico reutilizado. Todas las decisiones V6 se toman dentro de validación. Una promoción confirmatoria exige una cohorte nueva.

## Resumen ejecutivo

El proyecto estudia 590,540 transacciones IEEE-CIS, con 20,663 fraudes y prevalencia 3.50%. Compara una línea tabular competitiva sin orden (A), GRU/TCN causales sobre hasta 32 eventos (B), una fusión condicionada (C) y un encoder–decoder entrenado solo con transacciones legítimas (D). A obtiene AP interna 0.5362, B 0.3915, C 0.5192 y D 0.2172.

La permutación controlada no perjudica a B: su AP cambia de 0.3915 a 0.4016 ± 0.0023. La diferencia original−permutada es -0.0101. Con esta evidencia no se afirma que el orden aporte. C tampoco supera su criterio previo: cambio AP -0.0170 y reducción de costo 0.63%. El candidato es A.

## Datos y protocolo temporal

Se utiliza la competencia pública [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) de Kaggle, con datos anonimizados suministrados por Vesta Corporation. `train_transaction.csv` y `train_identity.csv` se unen por `TransactionID`, las filas se ordenan por `TransactionDT` y ambos campos se excluyen como magnitudes predictivas. La identidad secuencial es una clave aproximada formada por `card1`, `card2`, `card3`, `card5` y `addr1`.

La separación es 70 % entrenamiento, 15 % validación y 15 % benchmark histórico. Validación se subdivide cronológicamente en early stopping, ajuste de C, calibración, umbral y evaluación interna. Imputación, escalado y vocabularios se aprenden exclusivamente con entrenamiento. Las características históricas utilizan solo eventos anteriores; no se aplican particiones aleatorias.

## EDA, correlación y PCA

La V6 entrega un cuaderno exploratorio ejecutado que examina las 434 columnas de la unión teórica, la cobertura parcial de identidad, faltantes extremos, variables constantes, cambios temporales y asociación univariada con `isFraud`. También compara varias definiciones de entidad y cuantifica qué proporción de transacciones dispone de 3, 8, 16 o 32 eventos. Esta evidencia explica por qué ampliar una secuencia sin mejorar la identidad puede añadir ruido en lugar de memoria útil.

La correlación de Spearman se usa para localizar familias redundantes, especialmente dentro de `V1–V339`, pero no para eliminar automáticamente todo par correlacionado. Los árboles pueden aprovechar umbrales e interacciones diferentes incluso entre variables similares; por ello cada reducción debe validarse temporalmente. PCA se estudia como compresión del bloque V: resume gran parte de su varianza con muchas menos componentes, pero una varianza reconstruida alta no garantiza conservar la señal de fraude minoritaria. Como la ablation V3 con PCA rindió peor, V6 conserva PCA como diagnóstico y no como transformación del candidato.

Las nuevas características priorizan significado operativo y causalidad: conteos y monto medio por entidad en 1, 6, 24 y 72 horas; tiempo desde el evento anterior; monto relativo al historial; cambios de dispositivo/dirección; cantidad de faltantes; variables `C`, `D`, `V` e identidad seleccionadas. `TransactionID` y `TransactionDT` permanecen fuera del vector predictivo.

## Modelos A/B/C y control D

| Pieza | Diseño | Resultado interno |
|---|---|---:|
| A | LightGBM V4 con expertos `ProductCD=W/NO-W` | AP 0.5362 · costo Q904,620 |
| B | GRU frente a TCN causal; seleccionada `B_TCN`, hasta 32 eventos | AP 0.3915 · costo Q1,215,540 |
| C | Regresión logística condicionada sobre A/B/D, monto, historia e identidad | AP 0.5192 · costo Q898,920 |
| D | Encoder–decoder PyTorch entrenado solo con legítimas | AP 0.2172 · costo Q1,852,080 |

A conserva `A_V4` porque el refuerzo LightGBM no fue estable en las dos subventanas de selección. B compara dos arquitecturas con BCE ponderada, AdamW, clipping y early stopping. D minimiza MSE de reconstrucción legítima; su alta tasa de anomalías demuestra que anomalía y fraude no son equivalentes. C se entrena en un bloque independiente y no recibe el benchmark para decidir su arquitectura.

## Hipótesis y falsificaciones

**Hipótesis previa de C:** Creemos que una fusión condicionada por la calidad de identidad e historial mejorará AUC-PR y costo porque el puntaje secuencial solo debería influir cuando la historia sea suficientemente fiable. La consideraremos útil si incrementa AUC-PR al menos 0.01 y reduce el costo al menos 5% frente al mejor modelo individual, manteniendo recall mayor o igual a 0.75.

La hipótesis no se cumple. C pierde 0.0170 de AP y aumenta el costo 0.63 % frente a A en evaluación interna.

Las dos pruebas obligatorias son:

1. Permutación de antecedentes con cinco semillas, manteniendo la transacción objetivo al final.
2. Recorte de la historia a 3, 8 y 16 eventos.

La historia de 3 eventos obtiene AP 0.3847, la de 8 obtiene 0.3774 y la de 16 obtiene 0.3860. Ninguna evidencia justifica afirmar que el orden mejore el detector.

## Resultados y decisión económica

| Modelo | AUC-PR benchmark | Precisión | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|
| **A** | **0.559** | **15.78%** | 80.70% | **0.264** | **Q4,889,400** |
| B | 0.465 | 12.27% | 78.49% | 0.212 | Q5,898,960 |
| C | 0.563 | 17.14% | **79.82%** | 0.282 | Q4,753,500 |
| D | 0.229 | 5.61% | 82.39% | 0.105 | Q9,977,040 |

La política de umbral minimiza $4200FN+180FP$ sujeta a recall ≥ 0.75 en selección. El umbral de A es 0.03622. En el escenario central de 12 transacciones por tarjeta al mes, A representa un costo mensual proyectado de Q927,308,565. Es una extrapolación académica, no una cifra contable.

### Cómo interpretar las métricas

AP 0.559 significa que A mantiene una relación precisión–recall muy superior a la prevalencia de 3.50%; no significa que 55.9 % de sus alertas sea correcto. Esa proporción puntual la expresa la precisión: 15.78%. El recall 80.70% indica que el umbral recupera cerca de ocho de cada diez fraudes, mientras que ROC-AUC 0.901 describe la probabilidad de ordenar un fraude por encima de una transacción legítima elegida al azar. Debido al desbalance, un ROC alto puede coexistir con miles de falsas alarmas; por eso AP, alertas/100k y costo acompañan siempre a ROC.

D ilustra el mismo punto desde otro ángulo: alcanza recall 82.39%, pero precisión de solo 5.61%. El encoder–decoder reconoce rareza, no fraude: operaciones legítimas poco frecuentes, deriva o patrones con muchos faltantes también reconstruyen mal. C aparece descriptivamente competitivo en el benchmark, pero no se promueve porque su hipótesis se rechazó en evaluación interna. Reabrir esa decisión con el período final sería seleccionar con el test reutilizado.

## Tres decisiones técnicas importantes

1. **A tabular V4 frente al HistGradientBoosting V1.** Se consideró conservar el baseline antiguo. Se eligió LightGBM con expertos porque usa más variables causales y obtiene AP y costo claramente mejores.
2. **GRU frente a TCN causal.** Ambas se entrenaron con la misma población; TCN ganó en `model_select`, aunque la permutación mostró que su ranking no depende favorablemente del orden.
3. **Fusión y anomalías.** Se añadió un encoder–decoder legítimo como D y como entrada de C. Se conserva como ablation porque su baja precisión y alto costo no justifican promoverlo.

## Reproducción

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configuracion/v6/requirements-v6.txt
python -m pip install -r configuracion/v6/requirements-docs-v6.txt
python codigo/compartido/download_data.py
python -u codigo/v6/proyecto1_v6_pipeline.py
python codigo/v6/build_v6_deliverables.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 entregables/cuaderno/v6/proyecto1_calderon_barillas.ipynb
python codigo/v6/audit_project1_v6.py
```

La descarga requiere aceptar las reglas de IEEE-CIS en Kaggle y configurar las credenciales fuera del repositorio. Los CSV de casi 700 MB no se versionan.

## Estructura

```text
codigo/v6/              pipeline, construcción y auditoría de V6
configuracion/v6/       versiones exactas e instrucciones
datos/raw/              CSV locales ignorados por Git
artefactos/v6/          pesos A/B/C/D, calibradores, contrato y puntajes
evidencia/figuras/v6/   resultados reproducibles
entregables/cuaderno/v6/ notebook ejecutado
entregables/informe/v6/  fuente LaTeX y PDF
entregables/presentacion/v6/ HTML y PDF de ocho diapositivas
entregables/ficha/v6/    ficha del repositorio
```

## Candidato al Proyecto Final

- **Modelo conservado:** A — LightGBM V4 con expertos por `ProductCD` y calibrador V6.
- **Artefactos:** `artefactos/v4/modelo_experto_w_v4.txt`, `modelo_experto_no_w_v4.txt`, `artefactos/v6/calibradores_v6.joblib` y `contrato_entrada_salida_v6.json`.
- **Usuario:** analista de riesgo o equipo de monitoreo transaccional.
- **Decisión:** ordenar alertas y priorizar revisión; el puntaje no prueba fraude ni autoriza bloqueo autónomo.
- **Entrada preliminar:** transacción actual, variables categóricas y estadísticas históricas causales especificadas por el contrato.
- **Salida:** `risk_score` continuo en [0,1], umbral 0.03622 y política de revisión.
- **Pendientes:** nueva cohorte etiquetada, identidad bancaria fiable, costos reales, latencia, privacidad, equidad, seguridad, explicaciones y monitoreo.

## Limitaciones y uso responsable

IEEE-CIS está anonimizado y cubre aproximadamente 182 días. La clave proxy no equivale a un cliente real. El benchmark ya fue observado y ninguna conclusión se presenta como confirmación externa. Los costos y volúmenes mensuales son escenarios. El sistema debe apoyar revisión humana, no atribuir culpabilidad ni bloquear de manera autónoma.

## Declaración de uso de inteligencia artificial

Se utilizó asistencia de IA para estructurar y revisar código, diseñar documentación HTML/CSS/LaTeX, localizar bibliografía y automatizar auditorías. Los integrantes ejecutaron el pipeline y verificaron particiones, alineación de IDs, métricas, falsificaciones, umbrales y artefactos. La IA no se utilizó como fuente académica ni reemplaza la defensa de las decisiones.

## Referencias APA 7

Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *Proceedings of EMNLP*, 1724–1734. https://doi.org/10.3115/v1/D14-1179

IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection* [Data set]. Kaggle. https://www.kaggle.com/competitions/ieee-fraud-detection/overview

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Zhou, C., & Paffenroth, R. C. (2017). Anomaly detection with robust deep autoencoders. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 665–674. https://doi.org/10.1145/3097983.3098052
