# Protocolo experimental congelado · Proyecto 1 V7

**Estado:** congelado antes de ejecutar los modelos V7  
**Fecha:** 25 de agosto de 2026  
**Datos:** IEEE-CIS Fraud Detection (Vesta Corporation), unión de `train_transaction.csv` y `train_identity.csv` por `TransactionID`.

## Pregunta y unidad de decisión

La pregunta central es: **¿el orden de las transacciones aporta señal incremental, bajo qué condiciones y cuánto vale económicamente frente a un baseline tabular competitivo sin orden?** La unidad de decisión es una transacción y todos los modelos producen un puntaje continuo de riesgo en `[0, 1]` sobre las mismas filas y el mismo horizonte.

## Partición y prevención de fuga

- 0–70 % temporal: entrenamiento.
- 70–85 % temporal: desarrollo V7, subdividido cronológicamente en *early stopping*, ajuste del metamodelo, selección, calibración, umbral y evaluación interna.
- 85–100 % temporal: benchmark histórico reutilizado; no es ciego y no decide arquitectura, variables, calibración, umbral ni promoción.
- Toda imputación, codificación, frecuencia, correlación, selección supervisada, escalado y PCA se ajusta únicamente con las filas de entrenamiento correspondientes. En las comprobaciones walk-forward se vuelve a ajustar dentro de cada pliegue.
- `TransactionID` se usa solo para alinear evidencia. `TransactionDT` determina orden y variables temporales causales, pero no entra como identificador/magnitud cruda.

## Modelos predeclarados

- **A0:** regresión logística como control lineal.
- **A1:** LightGBM con variables ampliadas y agregados causales.
- **A2:** LightGBM con representantes después de filtrado de redundancia por correlación.
- **A3:** LightGBM híbrido con PCA aplicado solo al bloque `V1–V339`.
- **A4:** CatBoost con categóricas nativas y selección *train-only*.
- **A5:** stacking logístico de A0–A4 y el score del control A de V6, ajustado exclusivamente en el bloque `meta_fit`. El control permanece congelado para medir valor incremental sin reentrenarlo durante V7.
- **B:** mejor modelo secuencial causal congelado entre GRU y TCN de V6, sobre hasta 32 eventos. Se conservan los pesos y se vuelven a integrar sus puntajes en el protocolo común.
- **D:** encoder–decoder PyTorch entrenado solo con transacciones legítimas; es control de anomalía y componente opcional de C, no sustituto de A/B/C.
- **C:** fusión *out-of-time* de puntajes tabular, secuencial y de anomalía.

## Hipótesis de C y controles

> Creemos que una fusión de predicciones tabulares, secuenciales y de anomalía mejorará simultáneamente AP y costo porque sus errores pueden ser complementarios. C será útil solo si mejora AP al menos `0.01`, reduce el costo al menos `5 %`, mantiene recall `>= 0.75`, no incrementa las alertas más de `10 %` y la mejora aparece en por lo menos tres de cuatro ventanas temporales internas.

Controles predeclarados:

1. `C0`: mejor A individual.
2. `C1`: A + B.
3. `C2`: A + B + D.
4. `C3`: fusión condicionada con calidad de identidad, longitud de historia, monto y faltantes.

Ninguna fila utilizada para medir C se emplea para ajustar el metamodelo. La selección se decide antes de abrir evaluación interna y benchmark.

## Correlación y PCA

Se comparan sin asumir de antemano que reducir dimensión ayudará:

1. Variables ampliadas sin reducción.
2. Representantes de pares con `|rho de Spearman| >= 0.995`; se conserva el miembro con mayor AP univariada calculada solo en train.
3. PCA del bloque V con 32, 64 y 128 componentes; el resto de variables permanece interpretable.

Una variante solo se promueve si mejora AP/costo y estabilidad temporal. Varianza explicada alta no se interpreta como evidencia suficiente de utilidad predictiva.

## Identidad, historia y falsificaciones

Se diagnostican las claves tarjeta+dirección, tarjeta+dirección+correo, tarjeta+dispositivo+producto y tarjeta+dispositivo. Para cada una se reportan entidades, mediana de eventos y cobertura de 3/8/16/32 antecedentes. La clave del modelo se selecciona sin consultar el benchmark.

Pruebas obligatorias del orden:

- Cinco permutaciones controladas de antecedentes manteniendo el evento objetivo al final.
- Recortes de historia a 3, 8, 16 y 32 eventos, declarando si hay o no reentrenamiento.

Solo se afirma valor material del orden si la AP original supera en al menos `0.01` la media permutada.

## Métricas, economía y promoción

- Primaria: Average Precision (AP/AUC-PR).
- Secundarias: ROC-AUC, precisión, recall, F1, Brier, `Precision@K`, `Recall@K`, alertas por 100 000, latencia aproximada y costo por decisión.
- Política económica: `Q4,200 × FN + Q180 × FP`; el umbral se fija únicamente en el bloque de umbral con recall mínimo `0.75`.
- Escenarios mensuales: 1.4 millones de tarjetas y 5/12/20 transacciones por tarjeta; son extrapolaciones académicas.

La V7 solo reemplaza V6 si logra, en evaluación temporal interna, `ΔAP >= 0.01`, reducción de costo `>= 5 %`, recall `>= 0.75`, crecimiento de alertas `<= 10 %`, ninguna caída por ventana mayor que `0.005` y mejora en al menos tres de cuatro ventanas. Debido a que el benchmark fue reutilizado, aun cumpliendo lo anterior la conclusión será exploratoria hasta obtener una cohorte futura etiquetada.

## Entregables posteriores al modelado

Después de finalizar las corridas se generarán en subcarpetas `v7`: notebook oficial ejecutado, notebook EDA ejecutado, pesos y preprocesamiento, resultados y predicciones, figuras, README, informe LaTeX/PDF de máximo siete páginas, presentación HTML/PDF de máximo ocho diapositivas, ficha DOCX/PDF, instrucciones, dependencias exactas, contrato de entrada/salida, manifiesto y auditoría automática de rúbrica/rutas/secretos.
