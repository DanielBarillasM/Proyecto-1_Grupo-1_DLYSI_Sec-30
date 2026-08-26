<div align="center">

# Proyecto 1 · Monitoreo transaccional · V7

### ¿El orden aporta señal incremental y cuánto vale económicamente?

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-184e77)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-e9c46a)
![Versión](https://img.shields.io/badge/Revisar-V7-2a9d8f)
![Estado](https://img.shields.io/badge/Promoción-exploratoria-f4b942)

**Wilson Alejandro Calderón Argueta · 22018** · **Pablo Daniel Barillas Moreno · 22193**

Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Sección 30 · 2026

</div>

> [!IMPORTANT]
> **La versión que debe revisarse es V7.** El último 15 % de IEEE-CIS ya fue observado en versiones anteriores y se reporta como benchmark temporal histórico reutilizado. Todas las decisiones V7 se toman dentro del período de desarrollo; una conclusión confirmatoria exige una cohorte futura etiquetada.

## Resumen ejecutivo

El proyecto analiza 590,540 transacciones IEEE-CIS, 20,663 fraudes y una prevalencia de 3.50%. La pregunta no es simplemente qué arquitectura produce el número más alto, sino si el orden de eventos aporta información incremental frente a un baseline sin orden competitivo y si esa diferencia mejora una decisión de monitoreo con costos explícitos.

V7 amplía el espacio tabular a 465 columnas después de ingeniería, compara regresión logística, LightGBM completo, LightGBM reducido por correlación, PCA, CatBoost y stacking. Mantiene el experimento obligatorio A/B/C: A es el baseline sin orden; B es una TCN causal congelada de V6; C fusiona predicciones fuera de tiempo; D es un encoder–decoder PyTorch entrenado solo con operaciones legítimas y funciona como control de anomalía.

El ganador interno es `A5_ensamble_tabular`. A obtiene AP 0.5474, ROC-AUC 0.9129, precisión 19.81%, recall 77.21%, F1 0.3153, 11,966 alertas por 100,000 y costo Q826,800. Frente al control V6, aumenta AP +0.0111, reduce costo 8.68% y reduce alertas 28.62%; a cambio, recall baja de 80.33% a 77.21%.

La mejora no se declara estable. Tres ventanas son favorables, pero una cae -0.0135 AP y el límite previo era −0.005. El gate de promoción es **false**. V7 queda como candidato exploratorio mejor equilibrado, no como reemplazo confirmatorio.

## Resultado principal

| Modelo | AP interna | ROC-AUC | Precisión | Recall | F1 | Costo | Alertas/100k |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A · A5 stacking** | **0.5474** | **0.9129** | **19.81%** | 77.21% | **0.3153** | Q826,800 | **11,966** |
| B · TCN causal | 0.3915 | 0.8526 | 11.22% | 70.77% | 0.1938 | Q1,215,900 | 19,360 |
| C · A+B | 0.5426 | 0.9127 | 19.38% | **78.12%** | 0.3106 | **Q818,040** | 12,378 |
| D · autoencoder | 0.2172 | 0.7582 | 5.43% | 74.26% | 0.1012 | Q1,854,120 | 41,982 |

C cuesta ligeramente menos, pero no se promueve: cambia AP -0.0047, reduce costo solo 1.06% y no mejora ninguna de cuatro ventanas. La regla previa exigía +0.01 AP, −5 % costo, recall ≥0.75, crecimiento de alertas ≤10 % y mejora en tres ventanas. D conserva recall, pero su precisión de 5.43% confirma que rareza y fraude no son sinónimos.

## Cómo interpretar las métricas

- **AP o Average Precision** resume la curva precisión–recall a través de umbrales. AP 0.547 no significa que 54.7 % de alertas sea correcta; esa pureza puntual la expresa la precisión (19.81%). AP es principal porque la clase positiva representa solo 3.50%.
- **ROC-AUC** aproxima la probabilidad de ordenar un fraude por encima de una operación legítima elegida al azar. Un ROC 0.913 muestra buena separación global, pero puede coexistir con muchas falsas alarmas cuando hay millones de negativos.
- **Precisión** responde: “de todas las alertas, ¿cuántas son fraude?”. A logra 19.81%, aproximadamente una alerta verdadera por cada cinco.
- **Recall** responde: “de todos los fraudes, ¿cuántos se detectaron?”. A recupera 77.21%, casi ocho de diez.
- **F1** combina precisión y recall, pero no conoce el costo en quetzales. Por eso se reporta junto con `Q4,200×FN + Q180×FP`.
- **Precision@1 %** es 88.70%: si solo se revisa el 1 % de mayor riesgo, casi nueve de diez seleccionadas son fraude. **Recall@1 %** es 28.86%: esa capacidad limitada captura alrededor de tres de diez fraudes.

## Datos, orden y prevención de fuga

La fuente es [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/overview), publicada en Kaggle con datos anonimizados proporcionados por Vesta Corporation. `train_transaction.csv` y `train_identity.csv` se unen por `TransactionID`; las filas se ordenan por `TransactionDT`. Ambos campos se excluyen como magnitudes predictivas: el primero solo alinea evidencia y el segundo define el reloj.

La partición temporal es 70 % entrenamiento, 15 % validación y 15 % benchmark histórico. Validación se subdivide, en ese orden, en early stopping, `meta_fit`, `model_select`, calibración, umbral y evaluación. Toda imputación, frecuencia, asociación con fraude, correlación de Spearman, selección y PCA se ajusta exclusivamente con train. Los tres walk-forward vuelven a ajustar el preprocesamiento dentro de cada pliegue.

Las variables causales usan solo eventos anteriores: frecuencias previas de tarjeta, dirección, correo, dispositivo y producto; monto histórico; recencia; faltantes y resúmenes C/D/V/identidad. La entidad proxy se diagnostica con cuatro definiciones. Ninguna clave anonimizada equivale necesariamente a un cliente real.

## Correlación y PCA

El análisis encuentra 34 relaciones con `|ρ de Spearman| ≥ 0.995` en la muestra train-only y conserva 426 representantes. Esto reduce redundancia extrema, no “variables poco correlacionadas con fraude” de forma ciega. Una variable con baja asociación marginal todavía puede ser útil mediante interacciones.

PCA se limita al bloque `V1–V339`. Se ajustan 128 componentes con una muestra determinista contenida en train; se necesitan 70 para 90 % y 105 para 95 % de varianza. La mejor variante PCA alcanza AP de selección 0.5135, por debajo de LightGBM completo y correlación. La conclusión es predictiva: PCA comprime, pero no mejora el detector.

## Valor del orden

La prueba principal mantiene el evento objetivo al final y permuta solo antecedentes con cinco semillas. B original obtiene AP 0.3915 y la media permutada 0.4016 ± 0.0023. La diferencia original−permutada es -0.0101; se exigía una caída positiva mínima de 0.01. Por tanto, no se afirma que el orden aporte.

El segundo intento recorta la historia sin reentrenar: 3 eventos producen AP 0.3847, 8 producen 0.3774, 16 producen 0.3860 y 32 producen 0.3915. No existe patrón monotónico. B aprende señal, pero esa señal puede provenir del evento actual o la composición histórica, no del orden.

## Estabilidad y benchmark histórico

Los walk-forward del recipe reducido alcanzan AP 0.599 / 0.466 / 0.603. La dispersión evidencia deriva. En las cuatro ventanas del gate, las diferencias A5−V6 son +0.0244 / +0.0070 / -0.0135 / +0.0227. La tercera ventana impide promoción estable.

El benchmark histórico, solo descriptivo, muestra:

| Modelo | AP | Precisión | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|
| A | 0.5656 | 20.98% | 78.01% | 0.3307 | Q4,477,680 |
| B | 0.4654 | 12.47% | 78.37% | 0.2151 | Q5,854,920 |
| C | 0.5714 | 20.49% | 78.85% | 0.3252 | Q4,436,880 |
| D | 0.2287 | 5.61% | 82.35% | 0.1051 | Q9,969,180 |

## Reproducción rápida

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/v7/requirements-v7.txt
python -m pip install -r configuracion/v7/requirements-docs-v7.txt
$env:PROYECTO1_RAW=(Resolve-Path datos/raw)
python -u codigo/v7/proyecto1_v7_pipeline.py
python codigo/v7/build_notebooks_v7.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 entregables/cuaderno/v7/proyecto1_calderon_barillas.ipynb
python codigo/v7/report_v7.py
python codigo/v7/presentation_v7.py
python codigo/v7/build_documentation_v7.py
python codigo/v7/audit_project1_v7.py
```

La descarga requiere aceptar las reglas de Kaggle. Los CSV, credenciales y tokens no se versionan. Consulte [`configuracion/v7/INSTRUCCIONES_V7.md`](configuracion/v7/INSTRUCCIONES_V7.md) para la corrida completa.

## Estructura y versiones

Las carpetas principales se mantienen en la raíz; cada una contiene subcarpetas por versión.

```text
codigo/v7/                 pipeline, finalización, builders y auditoría
configuracion/v7/          protocolo, instrucciones y dependencias exactas
datos/processed/v7/        asociación, correlación y auditoría
artefactos/v7/             modelos A0–A5/C, scores, umbrales y contrato
evidencia/figuras/v7/      seis figuras reproducibles
entregables/cuaderno/v7/   notebook oficial y EDA ejecutados
entregables/informe/v7/    LaTeX y PDF de siete páginas
entregables/presentacion/v7/ HTML/PDF de ocho diapositivas y guion
entregables/ficha/v7/      ficha DOCX/PDF y copia del README
```

| Versión | Enfoque | Estado |
|---|---|---|
| V1 | Experimento A/B/C original | Histórica |
| V2 | LightGBM, walk-forward y calibración | Histórica |
| V3 | Regresión logística, PCA y controles | Histórica |
| V4 | LightGBM/CatBoost/XGBoost y expertos | Baseline fuerte heredado |
| V5 | Reintegración rubricada A/B/C | Histórica |
| V6 | EDA, TCN y encoder–decoder | Control experimental |
| **V7** | Variables completas, correlación/PCA, CatBoost y stacking con V6 | **Versión a revisar** |

## Candidato al Proyecto Final

- **Modelo:** A5, stacking logístico de A0–A4 y control V6.
- **Usuario previsto:** equipo de monitoreo o analista de riesgo.
- **Decisión:** priorizar transacciones para revisión bajo capacidad limitada.
- **Entrada:** transacción actual y agregados estrictamente causales descritos en `artefactos/v7/contrato_entrada_salida_v7.json`.
- **Salida:** `risk_score` calibrado en `[0,1]`; umbral 0.05000306; indicador binario derivado.
- **Faltantes:** medianas y frecuencias aprendidas en train; categorías nuevas reciben frecuencia cero.
- **Riesgos:** deriva, identidad proxy, falsos positivos, fraude adaptativo, costos hipotéticos y benchmark reutilizado.
- **Pendiente confirmatorio:** cohorte futura, identidad bancaria fiable, costos/capacidad reales, latencia, privacidad, equidad, explicación y monitoreo.

## Tres decisiones técnicas importantes

1. **Conservar el baseline V6 dentro de A5.** Alternativa: reemplazarlo por el mejor modelo nuevo. Evidencia: los nuevos aislados caen en evaluación; el stacking con V6 alcanza AP 0.5474 y costo Q826,800.
2. **Validar correlación y PCA como ablations.** Alternativa: borrar automáticamente columnas correlacionadas o comprimir todo el dataset. Evidencia: A2 gana selección frente a A1, pero PCA y la reducción aislada no sostienen superioridad.
3. **Rechazar C y el valor del orden según reglas previas.** Alternativa: promover C por el benchmark o asumir que TCN usa orden. Evidencia: C pierde AP interna y permutar antecedentes no perjudica B.

## Limitaciones y uso responsable

IEEE-CIS está anonimizado y cubre un período finito. La identidad es una aproximación, los costos son académicos y el benchmark no es ciego. El score mide riesgo estadístico, no certeza ni culpabilidad. Antes de cualquier despliegue se requieren pruebas con una cohorte futura, auditoría de privacidad/equidad, explicaciones, latencia, seguridad, monitoreo de deriva y un procedimiento de apelación.

## Declaración de uso de inteligencia artificial

Se utilizó asistencia de IA para estructurar y revisar código, diseñar documentación HTML/CSS/LaTeX, automatizar auditorías y mejorar la redacción. Los integrantes ejecutaron el pipeline, comprobaron alineación de IDs, particiones, transformaciones train-only, métricas, falsificaciones, gates y artefactos. La IA no se usa como fuente académica y no sustituye la defensa del proyecto.

## Referencias APA 7

IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection* [Data set]. Kaggle. https://www.kaggle.com/competitions/ieee-fraud-detection/overview

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems, 31*.

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Zhou, C., & Paffenroth, R. C. (2017). Anomaly detection with robust deep autoencoders. *Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 665–674. https://doi.org/10.1145/3097983.3098052
