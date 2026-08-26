# Guion de exposición · Proyecto 1 V7

Duración sugerida: 10–12 minutos. El guion sigue las ocho diapositivas; presione `N` en el HTML para mostrar notas del expositor.

## 1. Decisión ejecutiva · 60 segundos

“Nuestra V7 mejora el promedio y la utilidad operativa, pero todavía no demuestra estabilidad suficiente. El candidato es A5, un stacking que conserva el baseline V6 y añade modelos nuevos. Internamente alcanza AP 0.547, precisión 19.8%, recall 77.2% y costo Q826,800. Frente a V6 mejora AP +0.011, reduce costo 8.7% y reduce alertas, pero una ventana temporal cae demasiado. Por eso decimos candidato exploratorio, no ganador definitivo.”

## 2. Datos y problema · 60 segundos

“Usamos las 590,540 transacciones de IEEE-CIS Fraud Detection, publicadas en Kaggle con datos anonimizados de Vesta Corporation. Solo 3.5 % son fraude; por eso una accuracy de 96.5 % podría lograrse sin detectar un solo caso positivo. TransactionID une las tablas, TransactionDT define el reloj y ambos se excluyen como magnitudes predictivas.”

## 3. Protocolo temporal · 70 segundos

“La separación es 70 % train, 15 % validación y 15 % benchmark histórico. Validación se subdivide en early stopping, meta_fit, model_select, calibración, umbral y evaluación. Toda imputación, correlación, selección y PCA se aprende solo con pasado. El benchmark ya fue observado y no decide nada en V7.”

## 4. EDA, correlación y PCA · 70 segundos

“El EDA muestra alta dimensionalidad, faltantes, categorías y redundancia. Spearman elimina solo 34 representantes con correlación absoluta al menos 0.995. PCA se aplica únicamente a V1–V339: 105 componentes explican 95 % de varianza, pero la variante PCA no gana AP. La lección es que conservar varianza no garantiza conservar señal de fraude.”

## 5. Diseño A/B/C/D y familia A · 110 segundos

“A0 es regresión logística; A1 LightGBM completo; A2 LightGBM reducido; A3 PCA; A4 CatBoost y A5 stacking. A es el candidato sin orden; B es la TCN causal; C integra A, B y opcionalmente D; D es el encoder-decoder entrenado con operaciones legítimas. Ningún modelo nuevo aislado generaliza mejor que V6. A5 sí mejora porque combina logits fuera de tiempo y alcanza AP de selección 0.628. C debía mejorar AP, costo, recall, alertas y estabilidad, no solo una cifra.”

## 6. Resultados y significado de métricas · 95 segundos

“A gana AP, precisión y F1. AP evalúa el ranking precisión-recall y se compara con prevalencia 0.035; no es la proporción puntual de alertas correctas. Esa proporción es precisión: A logra 19.8%, aproximadamente una alerta correcta de cada cinco. B queda en AP 0.392; C pierde AP y D logra precisión de solo 5.4%. ROC describe separación global, pero no la carga de falsas alarmas.”

## 7. Valor del orden · 75 segundos

“Probamos el orden en vez de asumirlo. Al barajar antecedentes cinco veces, AP cambia de 0.3915 a 0.4016. La diferencia es -0.0101: destruir el orden no perjudica. Recortar a 3, 8, 16 y 32 tampoco produce mejora monotónica. B aprende algo, pero no podemos atribuírselo al orden.”

## 8. Economía, estabilidad y decisión · 105 segundos

“Usamos costo Q4,200 por FN y Q180 por FP, con recall mínimo 0.75. A5 produce 1,700 FP y 124 FN internamente: Q826,800. Reduce las alertas a 11,966 por 100 mil. Sin embargo, las cuatro diferencias AP contra V6 son +0.024; +0.007; -0.014; +0.023; una cae −0.014. Conservamos A5 como candidato exploratorio, pero esa deriva bloquea la promoción confirmatoria. No promovemos B porque no demuestra orden; C porque falla la hipótesis; D porque genera demasiadas falsas alarmas. La decisión cambiaría con una cohorte futura, sin caídas por ventana, con identidad real y costos operativos.”

## Respuestas cortas ante preguntas

- **¿AP 0.55 es bajo?** No se compara con 1 de forma aislada; la base es prevalencia 0.035. Es un ranking útil, aunque no perfecto.
- **¿Por qué ROC alto y precisión moderada?** Porque hay muchísimas operaciones legítimas; una tasa pequeña de FP genera muchas alertas.
- **¿Por qué no usar accuracy?** Un clasificador “todo legítimo” tendría 96.5 % sin detectar fraude.
- **¿Por qué no hacer la TCN más grande?** La permutación muestra que el cuello de botella es identidad/representación, no capacidad.
- **¿V7 supera V6?** Mejora promedio, costo y carga, pero no cumple estabilidad; necesita cohorte futura.
- **¿Por qué C no gana si cuesta menos?** Su hipótesis exigía mejora conjunta de AP, costo y estabilidad; pierde AP y no mejora ventanas.
