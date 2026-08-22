from __future__ import annotations

import base64
import hashlib
import json
import platform
from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "artefactos" / "v1"
FIG = ROOT / "evidencia" / "figuras" / "v1"
NOTEBOOK_DIR = ROOT / "entregables" / "cuaderno" / "v1"
REPORT_DIR = ROOT / "entregables" / "informe" / "v1"
PRESENTATION_DIR = ROOT / "entregables" / "presentacion" / "v1"
README_DIR = ROOT / "entregables" / "ficha" / "v1"
CONFIG_DIR = ROOT / "configuracion" / "v1"
RESULTS = json.loads((ART / "resultados.json").read_text(encoding="utf-8"))
CANDIDATE = RESULTS["candidate"]
TEST = RESULTS["test"]
VAL = RESULTS["validation"]
FALS = RESULTS["falsification"]
ECON = RESULTS["economics"]
ORDER_DROP = FALS["order_auc_pr_drop"]
ORDER_SUPPORTED = ORDER_DROP >= 0.01
HYP = RESULTS["hypothesis"]


def metric_table(metrics: dict) -> str:
    rows = []
    for model in ["A", "B", "C"]:
        m = metrics[model]
        rows.append(
            f"<tr><td><b>{model}</b></td><td>{m['auc_pr']:.3f}</td><td>{m['precision']:.3f}</td>"
            f"<td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td><td>Q{m['cost_q']:,.0f}</td></tr>"
        )
    return "".join(rows)


def markdown_styles() -> str:
    return """
<style>
.p1{font-family:Inter,'Segoe UI',Arial,sans-serif;color:#172033;line-height:1.72}
.p1-card{margin:16px 0;padding:20px 23px;border:1px solid #c9d9e6;border-left:6px solid #2a9d8f;border-radius:14px;background:#f5fafc;box-shadow:0 5px 16px rgba(16,42,67,.07)}
.p1-warn{border-left-color:#e76f51;background:#fff8f2}.p1-proof{border-left-color:#376f9e;background:#edf5fb}
.tag{display:inline-block;margin-bottom:10px;padding:5px 11px;border-radius:999px;background:#dceef8;color:#184e77;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase}
.p1-table{width:100%;border-collapse:separate;border-spacing:0;margin:15px 0;font-size:14px;overflow:hidden;border:1px solid #cedbe5;border-radius:12px}
.p1-table th{padding:11px;background:#184e77;color:white;text-align:left}.p1-table td{padding:10px;border-top:1px solid #dde6ed}.p1-table tr:nth-child(even) td{background:#f6f9fb}
code{background:#eaf1f6;padding:2px 5px;border-radius:4px}.kpi{display:inline-block;min-width:140px;margin:5px;padding:12px;border-radius:11px;background:#fff;border:1px solid #d8e3ea}.kpi b{display:block;color:#184e77;font-size:20px}
</style>
"""


def cover() -> str:
    return f"""
<div style="box-sizing:border-box;width:100%;padding:42px 46px;border-radius:25px;color:#fff;font-family:Inter,'Segoe UI',Arial,sans-serif;background:radial-gradient(circle at 92% 10%,rgba(255,255,255,.17) 0 8%,transparent 9%),linear-gradient(125deg,#102a43 0%,#184e77 55%,#2a9d8f 100%);box-shadow:0 17px 40px rgba(16,42,67,.25)">
<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:23px"><span style="padding:6px 14px;border:1px solid rgba(255,255,255,.32);border-radius:999px;background:rgba(255,255,255,.13);font-size:12px;font-weight:800">DEEP LEARNING</span><span style="padding:6px 14px;border:1px solid rgba(255,255,255,.32);border-radius:999px;background:rgba(255,255,255,.13);font-size:12px;font-weight:800">PROYECTO 01</span><span style="padding:6px 14px;border:1px solid rgba(255,255,255,.32);border-radius:999px;background:rgba(255,255,255,.13);font-size:12px;font-weight:800">GRUPO 01 · SECCIÓN 30</span><span style="padding:6px 14px;border:1px solid rgba(255,255,255,.32);border-radius:999px;background:rgba(255,255,255,.13);font-size:12px;font-weight:800">RUTA B · KAGGLE</span></div>
<h1 style="margin:0 0 12px;border:0;color:#fff;font-size:39px;line-height:1.13">Monitoreo transaccional: detectar lo que el orden revela</h1>
<h2 style="margin:0 0 29px;border:0;color:#eaf8ff;font-size:21px;font-weight:450">Comparación temporal de agregados, GRU y fusión híbrida sobre IEEE-CIS Fraud Detection</h2>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px 22px">
{''.join(f'<div style="padding:13px 16px;border:1px solid rgba(255,255,255,.22);border-radius:11px;background:rgba(255,255,255,.09)"><span style="display:block;margin-bottom:4px;color:#d9f2f0;font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.07em">{k}</span>{v}</div>' for k,v in [
('Institución','Universidad del Valle de Guatemala'),('Facultad','Facultad de Ingeniería'),('Curso','Deep Learning y Sistemas Inteligentes'),('Docente','Kevin Recinos'),('Integrante','Wilson Alejandro Calderón Argueta · 22018'),('Integrante','Pablo Daniel Barillas Moreno · 22193'),('Modalidad','Parejas'),('Período','Semestre II · 2026'),('Ponderación','8 puntos'),('Entrega','4 de septiembre de 2026 · 23:59'),('Presentación','8 minutos + 4 de preguntas'),('Modelo candidato',f'Pieza {CANDIDATE}')])}
</div></div>
<p style="text-align:center;color:#66788a;font-size:13px">Universidad del Valle de Guatemala · Grupo 1 · Sección 30</p>
"""


def build_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": platform.python_version()},
        "project": {"title": RESULTS["project"], "group": 1, "section": 30, "route": "B", "executed": True},
    })
    cells = []
    cells.append(nbf.v4.new_markdown_cell(cover() + markdown_styles()))
    cells.append(nbf.v4.new_markdown_cell(f"""
<div class="p1 p1-card"><span class="tag">Decisión ejecutiva</span><br>
El estudio compara una línea base competitiva sin orden (A), una GRU que recibe ocho eventos ordenados (B) y una fusión de secuencia con agregados (C). El candidato congelado con validación fue <b>{CANDIDATE}</b>. La caída de AUC-PR al permutar la historia fue <b>{ORDER_DROP:.3f}</b>; por tanto, la evidencia {"respalda" if ORDER_SUPPORTED else "no alcanza para respaldar"} que el orden añade señal bajo este protocolo. Todas las decisiones se tomaron antes de abrir la partición final.</div>
"""))
    cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import json, joblib, platform, sys
import numpy as np
import pandas as pd
import torch, sklearn
from IPython.display import HTML, display, Image

def localizar_raiz(inicio: Path) -> Path:
    for candidata in (inicio, *inicio.parents):
        if (candidata/'codigo'/'proyecto1_pipeline.py').exists():
            return candidata
    raise FileNotFoundError('No se encontró la raíz del proyecto')

ROOT = localizar_raiz(Path.cwd().resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codigo.v1.proyecto1_pipeline import Config, run_experiment

results = run_experiment(Config(), force=False)
print(f\"Python {platform.python_version()} | PyTorch {torch.__version__} | scikit-learn {sklearn.__version__}\")
print(\"Resultados reproducibles cargados. Candidato:\", results[\"candidate\"])"""))
    cells.append(nbf.v4.new_markdown_cell(r"""
## 1. Pregunta, hipótesis y protocolo bloqueado

La pregunta causal-operativa no es simplemente si una GRU puede clasificar fraude, sino si el historial ordenado contiene información incremental frente a una representación invariante al orden. El puntaje de riesgo es continuo y el umbral se decide después:

$$s_i=P(y_i=1\mid x_{i-7},\ldots,x_i),\qquad \hat y_i(\tau)=\mathbb{1}[s_i\ge\tau].$$

La comparación conserva datos, particiones, horizonte y transacción objetivo. El preprocesamiento se ajusta exclusivamente con entrenamiento; validación decide arquitectura, parada y umbral; prueba se consulta una vez al final.
"""))
    cells.append(nbf.v4.new_markdown_cell(f"""
<div class="p1 p1-card p1-proof"><span class="tag">Hipótesis previa de la apuesta C</span><br>{HYP['statement']}<br><br>
<b>Veredicto:</b> cambio AUC-PR = {HYP['ap_gain']:+.3f}; reducción de costo = {HYP['cost_reduction']:.1%}; hipótesis <b>{'respaldada' if HYP['success'] else 'no respaldada'}</b>.</div>
"""))
    cells.append(nbf.v4.new_markdown_cell("""
## 2. Integridad de los datos

Se emplea IEEE-CIS Fraud Detection, publicado en Kaggle por IEEE-CIS con transacciones de comercio electrónico de Vesta. `TransactionDT` determina el orden. Debido a la anonimización, la entidad se aproxima mediante `card1 + card2 + card3 + card5 + addr1`; no se afirma que sea una identidad bancaria perfecta. Cada ejemplo incluye la transacción actual y hasta siete antecedentes de la misma clave.

La división 70/15/15 se realiza cronológicamente. Los primeros eventos entrenan, los siguientes validan y los últimos prueban. Las ventanas tempranas de validación pueden usar antecedentes de entrenamiento, porque esos eventos ya existirían en producción; nunca ocurre lo contrario.
"""))
    cells.append(nbf.v4.new_code_cell("""d = results['dataset']; s = results['splits']
assert d['rows'] == 590540 and d['frauds'] == 20663
assert s['train_population']['n'] + s['validation']['n'] + s['test']['n'] == d['rows']
table = pd.DataFrame([
    ['Entrenamiento poblacional', s['train_population']['n'], s['train_population']['fraud_rate']],
    ['Validación', s['validation']['n'], s['validation']['fraud_rate']],
    ['Prueba final', s['test']['n'], s['test']['fraud_rate']],
], columns=['Partición','n','Tasa de fraude'])
display(table.style.format({'n':'{:,.0f}','Tasa de fraude':'{:.3%}'}).hide(axis='index'))
print('OK · particiones cronológicas e integridad básica')"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'evidencia'/'figuras'/'01_integridad_temporal.png')))"))
    cells.append(nbf.v4.new_markdown_cell("""
## 3. Núcleo común: piezas A y B

| Pieza | Entrada | Modelo | Qué puede aprovechar | Limitación principal |
|---|---|---|---|---|
| **A** | Estadísticas de ocho eventos y categorías actuales | HistGradientBoosting | Nivel, dispersión, extremos, diversidad y señal actual | Es invariante a permutaciones del historial |
| **B** | Diez variables numéricas y seis categóricas por evento | Embeddings + GRU(32) | Transiciones, recencia y dependencia temporal | La clave de entidad es aproximada y el historial es corto |

La GRU usa `BCEWithLogitsLoss` ponderada y devuelve logits; el sigmoide se aplica únicamente al producir el riesgo. La línea A y la GRU se ajustan sobre la misma muestra temporal de entrenamiento y se evalúan sobre la prevalencia natural.
"""))
    cells.append(nbf.v4.new_code_cell("""required = [
    ROOT/'artefactos'/'v1'/'modelo_A_histgradientboosting.joblib',
    ROOT/'artefactos'/'v1'/'modelo_B_gru.pt',
    ROOT/'artefactos'/'v1'/'modelo_C_hibrido.pt',
    ROOT/'artefactos'/'v1'/'preprocesamiento.joblib',
    ROOT/'artefactos'/'v1'/'umbrales.json',
]
assert all(p.exists() for p in required)
print('OK · A, B, C y preprocesamiento guardados')
print('Archivos:', *[p.name for p in required], sep='\\n- ')"""))
    cells.append(nbf.v4.new_markdown_cell(r"""
## 4. Apuesta C: fusión híbrida

La pieza C conserva exactamente el codificador GRU de B y concatena su estado final con agregados normalizados:

$$z_C=h_{\mathrm{GRU}}\oplus g(x_{i-7:i}),\qquad s_C=\sigma(\mathrm{MLP}(z_C)).$$

Este control permite atribuir cualquier diferencia a la información agregada adicional, no a cambiar simultáneamente de partición u horizonte. El criterio de utilidad quedó fijado antes del test y no se reescribió después de observar los resultados.
"""))
    cells.append(nbf.v4.new_code_cell("""val = pd.DataFrame(results['validation']).T[['auc_pr','precision','recall','f1','cost_q']]
display(val.style.format({'auc_pr':'{:.3f}','precision':'{:.3f}','recall':'{:.3f}','f1':'{:.3f}','cost_q':'Q{:,.0f}'}))
print('Candidato congelado con validación:', results['candidate'])"""))
    cells.append(nbf.v4.new_markdown_cell("""
## 5. Apertura única de prueba y comparación común

Después de congelar pesos, transformaciones, umbrales y candidato, se aplican los tres modelos una sola vez a los datos más recientes. AUC-PR es la métrica de ranking principal; precisión, recall y F1 se calculan en el umbral económico propio de cada modelo. La exactitud no interviene en la selección.
"""))
    cells.append(nbf.v4.new_code_cell("""test = pd.DataFrame(results['test']).T[['auc_pr','precision','recall','f1','tp','fp','fn','tn','cost_q']]
display(test.style.format({'auc_pr':'{:.3f}','precision':'{:.3f}','recall':'{:.3f}','f1':'{:.3f}','cost_q':'Q{:,.0f}'}))
assert set(test.index) == {'A','B','C'}
print('OK · comparación final común')"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'evidencia'/'figuras'/'02_curvas_precision_recall.png')))"))
    cells.append(nbf.v4.new_markdown_cell(f"""
## 6. Valor del orden: intentos de refutación

La permutación controlada baraja únicamente los antecedentes válidos y mantiene la transacción objetivo al final. Conserva el conjunto de eventos y, por construcción, todas las estadísticas de A. Se repite con cinco semillas. La segunda prueba conserva solo los tres eventos más recientes.

<div class="p1 p1-card {'p1-proof' if ORDER_SUPPORTED else 'p1-warn'}"><span class="tag">Conclusión falsable</span><br>
AUC-PR original de B: <b>{FALS['original_B']['auc_pr']:.3f}</b>; permutada: <b>{FALS['permutation_mean_auc_pr']:.3f} ± {FALS['permutation_std_auc_pr']:.3f}</b>; caída: <b>{ORDER_DROP:.3f}</b>. Con el criterio previo de 0.01, la evidencia <b>{'sí respalda' if ORDER_SUPPORTED else 'no respalda'}</b> una contribución material del orden. Al truncar a tres eventos, AUC-PR fue <b>{FALS['truncated_to_3']['auc_pr']:.3f}</b>.</div>
"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'evidencia'/'figuras'/'03_falsificaciones_orden.png')))"))
    cells.append(nbf.v4.new_markdown_cell(rf"""
## 7. Umbral y decisión económica

El costo de validación utilizado para fijar el umbral es:

$$C(\tau)=Q4{{,}}200\,FN(\tau)+Q180\,FP(\tau).$$

Para expresar el resultado mensual se adopta el escenario de 1.4 millones de tarjetas y 12 decisiones por tarjeta al mes. Es una extrapolación, no una predicción contable: la prevalencia IEEE-CIS puede diferir del banco.

<div class="p1 p1-card"><span class="tag">Escenario candidato {CANDIDATE}</span><br>
<span class="kpi"><b>Q{ECON[CANDIDATE]['cost_per_100k_q']:,.0f}</b>costo por 100 mil</span>
<span class="kpi"><b>Q{ECON[CANDIDATE]['monthly_cost_q']:,.0f}</b>costo mensual escalado</span>
<span class="kpi"><b>Q{ECON[CANDIDATE]['monthly_savings_vs_A_q']:,.0f}</b>ahorro vs. A</span></div>
"""))
    cells.append(nbf.v4.new_code_cell("display(Image(filename=str(ROOT/'evidencia'/'figuras'/'04_curva_costo_umbral.png')))"))
    cells.append(nbf.v4.new_markdown_cell(f"""
## 8. Recomendación, errores y límites

La recomendación es **{'complementar el motor agregado con la pieza '+CANDIDATE if CANDIDATE in ['B','C'] else 'conservar por ahora la línea agregada A'}**, no realizar un reemplazo ciego. {('La degradación bajo permutación aporta evidencia de que la historia ordenada contiene señal incremental.' if ORDER_SUPPORTED else 'La permutación no produjo una caída material; por ello este experimento no demuestra que el orden justifique producción.')}

Los errores deben revisarse por producto, monto y disponibilidad de identidad. Una alerta positiva no prueba fraude y un falso negativo cuesta mucho más en la función económica suministrada. La clave compuesta puede unir personas distintas o separar una misma tarjeta; las variables están anonimizadas; solo se observaron 182 días; los costos y el volumen mensual son supuestos; y no se midieron latencia, deriva, equidad, calibración externa ni impacto de revisión humana.

La recomendación cambiaría si una clave de tarjeta confiable elimina el efecto, si una prueba cronológica posterior revierte AUC-PR/costo o si la capacidad operativa no absorbe los falsos positivos.
"""))
    cells.append(nbf.v4.new_code_cell("""errors = pd.read_csv(ROOT/'artefactos'/'v1'/'patrones_error.csv').head(12)
display(errors.style.hide(axis='index'))
print('Patrones de error más frecuentes del candidato; no son explicaciones causales.')"""))
    cells.append(nbf.v4.new_markdown_cell("""
## 9. Matriz de evidencias

| Evidencia | Figura o tabla | Conclusión | Limitación |
|---|---|---|---|
| Integridad temporal | Figura 1 y tabla de particiones | 70/15/15 cronológico, preprocesamiento train-only | Clave de tarjeta aproximada |
| Comparación A–B | Curvas PR y tabla final | Mismo horizonte, AUC-PR como métrica principal | Una sola ventana de ocho eventos |
| Valor del orden | Figura 3 | Permutación repetida y truncamiento | No equivale a intervención causal perfecta |
| Apuesta C | Tabla de validación | Hipótesis y criterio fijados antes del test | Fusión agrega complejidad |
| Decisión económica | Figura 4 | Umbral minimiza Q4,200 FN + Q180 FP | Volumen y prevalencia transferidos |
| Recomendación | Sección 8 | Decisión condicionada a evidencia y operación | Falta piloto productivo |
"""))
    cells.append(nbf.v4.new_markdown_cell("""
## 10. Declaración de uso de inteligencia artificial

Se utilizó un asistente de IA para estructurar código, revisar consistencia, mejorar la presentación HTML/CSS y localizar bibliografía. Los integrantes deben verificar y poder defender: la clave compuesta, la partición temporal, la construcción causal de ventanas, los modelos, las métricas, las falsificaciones, el umbral y la extrapolación económica. Los datos y resultados provienen de la ejecución reproducible; la IA no se considera fuente académica.
"""))
    cells.append(nbf.v4.new_markdown_cell("""
## Referencias — APA 7

Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). *Learning phrase representations using RNN encoder-decoder for statistical machine translation*. arXiv. https://arxiv.org/abs/1406.1078

Fisher, A., Rudin, C., & Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously. *Journal of Machine Learning Research, 20*(177), 1–81. https://jmlr.org/papers/v20/18-760.html

IEEE Computational Intelligence Society. (2019). *IEEE-CIS Fraud Detection* [Data set]. Kaggle. https://www.kaggle.com/competitions/ieee-fraud-detection

Kaggle. (s. f.). *KaggleHub*. GitHub. Recuperado el 14 de agosto de 2026, de https://github.com/Kaggle/kagglehub

PyTorch Contributors. (s. f.). *BCEWithLogitsLoss*. PyTorch. Recuperado el 14 de agosto de 2026, de https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Scikit-learn Developers. (s. f.). *Common pitfalls and recommended practices*. Scikit-learn. Recuperado el 14 de agosto de 2026, de https://scikit-learn.org/stable/common_pitfalls.html
"""))
    nb.cells = cells
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    output = NOTEBOOK_DIR / "proyecto1_calderon_barillas.ipynb"
    nbf.write(nb, output)
    return output


def tex_escape(text: str) -> str:
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def build_report() -> Path:
    order_sentence = (
        f"La permutación redujo AUC-PR en {ORDER_DROP:.3f}; el modelo sí mostró dependencia material del orden."
        if ORDER_SUPPORTED else
        f"La permutación solo cambió AUC-PR en {ORDER_DROP:.3f}; no se atribuye una ventaja material al orden."
    )
    recommendation = (
        f"Complementar el motor vigente con {CANDIDATE}, sujeto a piloto y monitoreo."
        if CANDIDATE in ["B", "C"] else
        "Conservar A como candidato y no migrar aún a secuencias."
    )
    rows = "\n".join(
        f"{m} & {TEST[m]['auc_pr']:.3f} & {TEST[m]['precision']:.3f} & {TEST[m]['recall']:.3f} & {TEST[m]['f1']:.3f} & Q{TEST[m]['cost_q']:,.0f} \\\\" for m in ["A", "B", "C"]
    )
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[utf8]{{inputenc}}\usepackage[T1]{{fontenc}}
\usepackage[margin=1.45cm]{{geometry}}\usepackage{{graphicx,booktabs,tabularx,xcolor,hyperref}}
\graphicspath{{{{../../../evidencia/figuras/v1/}}}}
\definecolor{{navy}}{{HTML}}{{184E77}}\definecolor{{teal}}{{HTML}}{{2A9D8F}}\definecolor{{soft}}{{HTML}}{{EDF5FB}}
\hypersetup{{colorlinks=true,linkcolor=navy,urlcolor=teal}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{4pt}}
\newcommand{{\kpi}}[1]{{\colorbox{{soft}}{{\strut\textbf{{#1}}}}}}
\begin{{document}}
{{\color{{navy}}\LARGE\bfseries Proyecto 1: Monitoreo transaccional}}\\[-1mm]
{{\large Detectar lo que el orden revela}}\hfill Grupo 1 · Sección 30\\
\textbf{{Wilson Alejandro Calderón Argueta (22018)}} · \textbf{{Pablo Daniel Barillas Moreno (22193)}}\\
Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Kevin Recinos
\vspace{{2mm}}\hrule\vspace{{2mm}}

\section*{{Resumen ejecutivo}}
Se evaluó si el orden de las transacciones añade valor frente al sistema agregado del Banco del Altiplano. Sobre 590,540 transacciones reales anonimizadas de IEEE-CIS (20,663 fraudes; 3.499\%), se compararon A: gradient boosting sobre agregados, B: GRU sobre ocho eventos y C: fusión GRU--agregados. La selección se realizó con validación cronológica y prueba se abrió una vez. El candidato fue \textbf{{{CANDIDATE}}}. {tex_escape(order_sentence)} {tex_escape(recommendation)} El costo usa Q4,200 por FN y Q180 por FP; la extrapolación mensual es un escenario, no una cifra contable.

\section*{{1. Integridad de datos y protocolo temporal}}
El origen es la competencia IEEE-CIS Fraud Detection de Kaggle, aportada por Vesta Corporation. Las transacciones cubren {RESULTS['dataset']['time_span_days']:.0f} días. La entidad se aproxima con \texttt{{card1+card2+card3+card5+addr1}}: permite historial, pero no garantiza identidad perfecta. Cada objetivo usa la transacción actual y hasta siete antecedentes; no usa futuro.

\begin{{center}}\small
\begin{{tabular}}{{lrr}}\toprule Partición & n & Fraude \\\midrule
Entrenamiento poblacional & {RESULTS['splits']['train_population']['n']:,} & {100 * RESULTS['splits']['train_population']['fraud_rate']:.3f}\% \\
Validación & {RESULTS['splits']['validation']['n']:,} & {100 * RESULTS['splits']['validation']['fraud_rate']:.3f}\% \\
Prueba final & {RESULTS['splits']['test']['n']:,} & {100 * RESULTS['splits']['test']['fraud_rate']:.3f}\% \\\bottomrule
\end{{tabular}}\end{{center}}
Los primeros 70\% del tiempo entrenan, 15\% validan y 15\% prueban. Normalización, vocabularios, parada, arquitectura y umbrales se ajustan sin prueba. El submuestreo computacional ocurre solo dentro de entrenamiento y conserva todos los positivos; ambos modelos usan la misma muestra.
\begin{{center}}\includegraphics[width=.83\linewidth]{{01_integridad_temporal.png}}\end{{center}}

\section*{{2. Núcleo A--B y apuesta C}}
\textbf{{A}} resume nivel, dispersión, extremos, recencia y diversidad, y usa HistGradientBoosting. \textbf{{B}} codifica diez variables numéricas y seis categóricas por evento mediante embeddings y GRU(32). Ambos generan riesgo continuo. \textbf{{C}} concatena el estado final de B con agregados estandarizados. La hipótesis previa exigió que C aumentara AUC-PR al menos 0.01 y redujera costo al menos 5\% frente a B en validación. El cambio observado fue {HYP['ap_gain']:+.3f} y {100 * HYP['cost_reduction']:.1f}\%; la apuesta fue \textbf{{{'útil' if HYP['success'] else 'no útil según el criterio previo'}}}.

\section*{{3. Comparación común}}
\begin{{center}}\small\begin{{tabular}}{{lrrrrr}}\toprule Modelo & AUC-PR & Prec. & Recall & F1 & Costo test \\\midrule
{rows}
\bottomrule\end{{tabular}}\end{{center}}
AUC-PR es principal por el desbalance; precisión, recall y F1 se reportan en el umbral económico propio, fijado con validación. Exactitud no se usa para decidir.
\begin{{center}}\includegraphics[width=.80\linewidth]{{02_curvas_precision_recall.png}}\end{{center}}

\section*{{4. Intentos de refutar el valor del orden}}
Se permutó cinco veces solo la historia, manteniendo la transacción objetivo al final y todos los eventos intactos. B obtuvo AUC-PR {FALS['original_B']['auc_pr']:.3f} original y {FALS['permutation_mean_auc_pr']:.3f}$\pm${FALS['permutation_std_auc_pr']:.3f} permutada. {tex_escape(order_sentence)} La segunda prueba recortó el historial a tres eventos y produjo AUC-PR {FALS['truncated_to_3']['auc_pr']:.3f}. Estas pruebas miden dependencia predictiva, no causalidad ni identidad bancaria perfecta.
\begin{{center}}\includegraphics[width=.75\linewidth]{{03_falsificaciones_orden.png}}\end{{center}}

\section*{{5. Umbral, costo y decisión}}
Se minimizó en validación $C(\tau)=4200FN(\tau)+180FP(\tau)$. Para {CANDIDATE}, el costo de prueba es Q{TEST[CANDIDATE]['cost_q']:,.0f}, equivalente a Q{ECON[CANDIDATE]['cost_per_100k_q']:,.0f} por 100 mil decisiones. Con 1.4 millones de tarjetas y 12 transacciones mensuales, el escenario escalado es Q{ECON[CANDIDATE]['monthly_cost_q']:,.0f} y el ahorro frente a A es Q{ECON[CANDIDATE]['monthly_savings_vs_A_q']:,.0f}. La prevalencia y el volumen reales deben reemplazar estos supuestos antes de una decisión financiera.
\begin{{center}}\includegraphics[width=.75\linewidth]{{04_curva_costo_umbral.png}}\end{{center}}

\section*{{6. Recomendación, errores y límites}}
\textbf{{{tex_escape(recommendation)}}} El puntaje debe priorizar revisión, no bloquear automáticamente. Los falsos negativos concentran Q4,200 y los falsos positivos deterioran experiencia y capacidad operativa. Cambiaríamos la recomendación si una clave de tarjeta confiable elimina el efecto, si una cohorte posterior revierte AUC-PR/costo o si revisión no absorbe alertas. Límites: identidad aproximada, anonimización, 182 días, una ventana, costos transferidos, ausencia de latencia, calibración externa, deriva, equidad y piloto humano.

\section*{{Matriz de evidencias}}
\scriptsize\begin{{tabularx}}{{\linewidth}}{{p{{2.4cm}}p{{2.5cm}}X X}}\toprule Evidencia & Ubicación & Conclusión & Limitación \\\midrule
Integridad & Fig. 1, Tabla 1 & 70/15/15 cronológico; ajustes train-only & Entidad aproximada \\
Comparación A--B & Fig. 2, Tabla 2 & Mismo horizonte y AUC-PR & Historial de 8 eventos \\
Valor del orden & Fig. 3 & Permutación repetida y recorte & Dependencia no implica causalidad \\
Apuesta C & Sección 2 & Criterio fijado antes del test & Mayor complejidad \\
Economía & Fig. 4 & Umbral minimiza costo suministrado & Volumen/prevalencia asumidos \\
Recomendación & Sección 6 & Decisión condicionada & Falta piloto productivo \\\bottomrule
\end{{tabularx}}

\section*{{Referencias}}
\footnotesize
Cho, K., et al. (2014). \textit{{Learning phrase representations using RNN encoder-decoder}}. \url{{https://arxiv.org/abs/1406.1078}}.\\
Fisher, A., Rudin, C., \& Dominici, F. (2019). All models are wrong, but many are useful. \textit{{JMLR, 20}}(177), 1--81.\\
IEEE Computational Intelligence Society. (2019). \textit{{IEEE-CIS Fraud Detection}} [Data set]. Kaggle.\\
Saito, T., \& Rehmsmeier, M. (2015). The precision-recall plot is more informative than ROC. \textit{{PLOS ONE, 10}}(3), e0118432.\\
Scikit-learn Developers. (s. f.). \textit{{Common pitfalls and recommended practices}}. \url{{https://scikit-learn.org/stable/common_pitfalls.html}}.
\end{{document}}
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "informe.tex"
    output.write_text(tex, encoding="utf-8")
    return output


def img_data(name: str) -> str:
    path = FIG / name
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_presentation() -> Path:
    pr_img = img_data("02_curvas_precision_recall.png")
    order_img = img_data("03_falsificaciones_orden.png")
    cost_img = img_data("04_curva_costo_umbral.png")
    data_img = img_data("01_integridad_temporal.png")
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Proyecto 1 · Monitoreo transaccional</title>
<style>
*{{box-sizing:border-box}}@page{{size:13.333in 7.5in;margin:0}}body{{margin:0;background:#071827;color:#eff8ff;font-family:Inter,'Segoe UI',Arial,sans-serif;overflow:hidden}}.deck{{width:100vw;height:100vh}}section{{display:none;width:100vw;height:100vh;padding:5.5vh 6vw;background:radial-gradient(circle at 90% 8%,rgba(42,157,143,.25),transparent 25%),linear-gradient(125deg,#071827,#102a43 65%,#184e77)}}section.active{{display:block}}h1{{font-size:5.2vw;line-height:1.05;margin:.15em 0}}h2{{font-size:3.0vw;margin:0 0 .5em;color:#9fe3d8}}h3{{font-size:1.55vw;color:#8ecae6}}p,li{{font-size:1.45vw;line-height:1.45}}.eyebrow{{font-size:1vw;letter-spacing:.16em;text-transform:uppercase;color:#8ecae6;font-weight:800}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:2vw;align-items:center}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.2vw}}.card{{padding:1.3vw;border:1px solid rgba(255,255,255,.17);border-radius:16px;background:rgba(255,255,255,.075)}}.card b{{display:block;font-size:2.2vw;color:#9fe3d8}}table{{width:100%;border-collapse:collapse;font-size:1.3vw}}th{{background:#2a9d8f;padding:.7vw;text-align:left}}td{{padding:.65vw;border-bottom:1px solid rgba(255,255,255,.16)}}img{{width:100%;max-height:62vh;object-fit:contain;border-radius:14px;background:white}}.accent{{color:#9fe3d8}}.warn{{color:#ffb38a}}.footer{{position:absolute;bottom:2.2vh;left:6vw;right:6vw;display:flex;justify-content:space-between;font-size:.85vw;color:#a9bdca}}.progress{{position:fixed;left:0;bottom:0;height:5px;background:#2a9d8f;transition:width .25s}}.nav{{position:fixed;right:2vw;bottom:1.7vh;color:#a9bdca;font-size:.9vw}}@media print{{body{{overflow:visible}}section{{display:block!important;page-break-after:always;width:13.333in;height:7.5in;padding:.45in .65in}}.progress,.nav{{display:none}}h1{{font-size:38pt}}h2{{font-size:25pt}}p,li{{font-size:13pt}}table{{font-size:11pt}}.eyebrow{{font-size:9pt}}}}
</style></head><body><div class="deck">
<section class="active"><div class="eyebrow">Universidad del Valle de Guatemala · Proyecto 1</div><h1>¿Cuánto vale<br><span class="accent">el orden</span>?</h1><p style="max-width:70%">Monitoreo transaccional con IEEE-CIS: agregados, GRU y fusión híbrida.</p><div class="cards"><div class="card"><b>590,540</b>transacciones</div><div class="card"><b>3.499%</b>fraude</div><div class="card"><b>{CANDIDATE}</b>candidato</div></div><div class="footer"><span>Wilson Calderón · Pablo Barillas</span><span>Grupo 1 · Sección 30</span></div></section>
<section><div class="eyebrow">01 · Integridad</div><h2>Pasado para aprender; futuro para comprobar</h2><div class="grid"><div><p>Partición estrictamente cronológica:</p><ul><li>70% entrenamiento</li><li>15% validación</li><li>15% prueba abierta una vez</li></ul><p class="warn">La identidad es una clave compuesta anonimizada, no una tarjeta confirmada.</p></div><img src="{data_img}"></div></section>
<section><div class="eyebrow">02 · Diseño</div><h2>Tres respuestas a la misma decisión</h2><div class="cards"><div class="card"><b>A</b><h3>Sin orden</h3><p>Gradient boosting sobre nivel, dispersión, extremos, recencia y diversidad.</p></div><div class="card"><b>B</b><h3>Secuencia</h3><p>Embeddings + GRU(32) sobre ocho eventos ordenados.</p></div><div class="card"><b>C</b><h3>Apuesta</h3><p>Estado GRU fusionado con agregados globales.</p></div></div><p>Éxito de C predefinido: ΔAUC-PR ≥ 0.01 y costo ↓ ≥ 5% en validación.</p></section>
<section><div class="eyebrow">03 · Comparación común</div><h2>AUC-PR y umbral económico</h2><div class="grid"><table><tr><th>Modelo</th><th>AUC-PR</th><th>Prec.</th><th>Recall</th><th>F1</th></tr>{''.join(f'<tr><td>{m}</td><td>{TEST[m]["auc_pr"]:.3f}</td><td>{TEST[m]["precision"]:.3f}</td><td>{TEST[m]["recall"]:.3f}</td><td>{TEST[m]["f1"]:.3f}</td></tr>' for m in ['A','B','C'])}</table><img src="{pr_img}"></div></section>
<section><div class="eyebrow">04 · Intento de refutación</div><h2>Destruir el orden sin cambiar los eventos</h2><div class="grid"><img src="{order_img}"><div><div class="card"><b>{ORDER_DROP:.3f}</b>caída AUC-PR al permutar</div><p>Promedio de cinco permutaciones; la transacción objetivo permanece al final.</p><p class="accent">Conclusión: la evidencia {'respalda' if ORDER_SUPPORTED else 'no respalda'} una contribución material del orden.</p></div></div></section>
<section><div class="eyebrow">05 · Apuesta C</div><h2>Hipótesis previa, no relato posterior</h2><div class="cards"><div class="card"><b>{HYP['ap_gain']:+.3f}</b>Δ AUC-PR</div><div class="card"><b>{HYP['cost_reduction']:.1%}</b>reducción de costo</div><div class="card"><b>{'Sí' if HYP['success'] else 'No'}</b>cumplió ambos criterios</div></div><p>Veredicto conservado incluso si la extensión falla.</p></section>
<section><div class="eyebrow">06 · Economía</div><h2>El umbral es una decisión, no 0.5 por costumbre</h2><div class="grid"><div><p>$C(τ)=Q4,200·FN+Q180·FP$</p><div class="card"><b>Q{ECON[CANDIDATE]['cost_per_100k_q']:,.0f}</b>por 100 mil decisiones</div><p>Escenario mensual: 1.4 M tarjetas × 12 transacciones.</p></div><img src="{cost_img}"></div></section>
<section><div class="eyebrow">07 · Recomendación</div><h2>{'Complementar, no reemplazar' if CANDIDATE in ['B','C'] else 'Conservar y seguir investigando'}</h2><div class="grid"><div><h3>Decisión</h3><p>Usar {CANDIDATE} para priorizar revisión, sujeto a piloto, capacidad operativa y umbral recalibrado.</p><h3>Cambiaríamos si…</h3><p>una identidad confiable elimina el efecto, una cohorte posterior revierte costo/AUC-PR o el volumen de alertas excede revisión.</p></div><div><h3>Límites</h3><ul><li>Identidad aproximada</li><li>182 días y variables anonimizadas</li><li>Costos/prevalencia transferidos</li><li>Sin latencia, deriva ni piloto humano</li></ul></div></div></section>
</div><div class="progress"></div><div class="nav">← → · espacio</div><script>const slides=[...document.querySelectorAll('section')];let i=0;function show(n){{i=Math.max(0,Math.min(slides.length-1,n));slides.forEach((s,j)=>s.classList.toggle('active',j===i));document.querySelector('.progress').style.width=((i+1)/slides.length*100)+'%'}}addEventListener('keydown',e=>{{if(['ArrowRight',' ','PageDown'].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);}});show(0);</script></body></html>"""
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    output = PRESENTATION_DIR / "presentacion.html"
    output.write_text(html, encoding="utf-8")
    return output


def build_readme() -> Path:
    recommendation = "complementar el sistema con" if CANDIDATE in ["B", "C"] else "conservar como referencia"
    text = rf"""# Proyecto 1 — Monitoreo transaccional

**Grupo 1 · Sección 30 · Universidad del Valle de Guatemala**

- Wilson Alejandro Calderón Argueta — 22018
- Pablo Daniel Barillas Moreno — 22193

> Comparación controlada entre una línea base agregada, una GRU secuencial y una
> arquitectura híbrida sobre IEEE-CIS Fraud Detection. El objetivo es determinar
> si el orden temporal aporta valor predictivo y económico verificable.

**Repositorio:** [DanielBarillasM/Proyecto-1_Grupo-1_DLYSI_Sec-30](https://github.com/DanielBarillasM/Proyecto-1_Grupo-1_DLYSI_Sec-30)

## Entregables

- [`entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb`](../entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb): investigación ejecutada.
- [`entregables/informe/v1/informe.pdf`](../entregables/informe/v1/informe.pdf): informe ejecutivo de cuatro páginas; también se incluye su fuente LaTeX.
- [`entregables/presentacion/v1/presentacion.html`](../entregables/presentacion/v1/presentacion.html): presentación interactiva y autocontenida; su versión PDF contiene ocho diapositivas.
- [`entregables/ficha/v1/Ficha_Repositorio_Proyecto1.docx`](../entregables/ficha/v1/Ficha_Repositorio_Proyecto1.docx): ficha descriptiva editable del repositorio.
- [`artefactos/`](../artefactos): modelos A/B/C, candidato, umbrales, preprocesamiento y contrato de entrada.

## Resultado principal

| Modelo | Diseño | AUC-PR test | Recall | Costo de prueba |
|---|---|---:|---:|---:|
| A | Gradient boosting sobre agregados | {TEST['A']['auc_pr']:.3f} | {TEST['A']['recall']:.3f} | Q{TEST['A']['cost_q']:,.0f} |
| B | Embeddings + GRU(32) | {TEST['B']['auc_pr']:.3f} | {TEST['B']['recall']:.3f} | Q{TEST['B']['cost_q']:,.0f} |
| C | GRU fusionada con agregados | {TEST['C']['auc_pr']:.3f} | {TEST['C']['recall']:.3f} | Q{TEST['C']['cost_q']:,.0f} |

El candidato congelado es **{CANDIDATE}**. La caída de AUC-PR al permutar el
historial fue {ORDER_DROP:.3f}, inferior al criterio previo de 0.01. En este
experimento no se obtuvo evidencia suficiente para justificar una migración al
modelo secuencial.

## Datos y reproducción

El proyecto utiliza `train_transaction.csv` y `train_identity.csv` de [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection). Los datos y las credenciales no se versionan. La guía detallada, incluidos requisitos del sistema y solución de problemas, está en [`configuracion/INSTRUCCIONES_EJECUCION.md`](../configuracion/INSTRUCCIONES_EJECUCION.md).

### Preparación del entorno en PowerShell

```powershell
python --version
python -m venv configuracion/.venv
.\configuracion\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r configuracion/requirements.txt
python -c "import torch, sklearn, pandas; print(torch.__version__, sklearn.__version__, pandas.__version__)"
```

Las versiones fueron comprobadas con Python 3.13.1. Si PowerShell bloquea la activación, ejecute antes `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

### Reproducción completa

```powershell
python -c "import kagglehub; kagglehub.login()"
python codigo/compartido/download_data.py
python codigo/v1/proyecto1_pipeline.py
python codigo/v1/build_deliverables.py
jupyter nbconvert --to notebook --execute --inplace entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb
Push-Location entregables/informe/v1
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
pdflatex -interaction=nonstopmode -halt-on-error informe.tex
Pop-Location
python codigo/v1/crear_ficha_repositorio.py
python codigo/v1/audit_project1.py
```

`codigo/v1/proyecto1_pipeline.py` fuerza un entrenamiento completo y puede tardar varios minutos en CPU. Para revisar la entrega ya ejecutada sin reentrenar, basta con activar el entorno y ejecutar `python codigo/v1/audit_project1.py`.

Los CSV quedan en `datos/raw/`. La semilla principal es 2026. El preprocesamiento se ajusta solo con entrenamiento. El test cronológico se abre después de congelar candidato y umbrales. Para abrir el cuaderno de forma interactiva:

```powershell
jupyter lab entregables/cuaderno/v1/proyecto1_calderon_barillas.ipynb
```

## Tres decisiones técnicas importantes

1. **IEEE-CIS frente a `creditcard.csv`:** se eligió IEEE-CIS porque contiene tiempo y atributos de tarjeta que permiten una clave secuencial aproximada. La alternativa europea no permite agrupar por tarjeta.
2. **GRU frente a LSTM/Transformer:** ocho eventos y CPU favorecen una GRU pequeña. La complejidad no era el objetivo; la falsificación del orden sí.
3. **Umbral económico frente a 0.5:** se minimiza `4200*FN + 180*FP` en validación. El test nunca elige el umbral.

## Uso de inteligencia artificial

Se utilizó IA para estructurar código, revisar consistencia, localizar bibliografía y diseñar HTML/CSS/LaTeX. Los integrantes verificaron datos, partición temporal, formas, ejecución, métricas, falsificaciones, umbral y archivos finales. La IA no se usó como fuente académica ni sustituyó la interpretación de resultados.

## Candidato al Proyecto Final

- **Modelo:** pieza {CANDIDATE}; se recomienda {recommendation} `{CANDIDATE}`. Artefacto: `artefactos/modelo_candidato_{CANDIDATE}.*`.
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
entregables/informe/v1/        fuente LaTeX y PDF
entregables/presentacion/   presentación HTML y PDF
entregables/ficha/          ficha DOCX del repositorio
evidencia/figuras/v1/          evidencia visual reproducible
legal/                      licencia del repositorio
```
"""
    README_DIR.mkdir(parents=True, exist_ok=True)
    output = README_DIR / "README.md"
    output.write_text(text, encoding="utf-8")
    return output


def build_requirements() -> Path:
    text = """# Entorno comprobado: Python 3.13.1 (Windows, CPU)
# Instalar desde la raíz: python -m pip install -r configuracion/v1/requirements-v1.txt

# Experimento y datos
kagglehub==0.4.1
joblib==1.5.3
matplotlib==3.10.1
numpy==2.2.2
pandas==2.2.3
scikit-learn==1.8.0
torch==2.13.0

# Notebook y ejecución reproducible
ipykernel==6.29.5
jupyterlab==4.6.1
nbconvert==7.17.0
nbformat==5.10.4

# Construcción y auditoría de entregables
lxml==6.1.0
Pillow==11.1.0
PyMuPDF==1.28.2
python-docx==1.2.0
qrcode[pil]==8.2
"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output = CONFIG_DIR / "requirements-v1.txt"
    output.write_text(text, encoding="utf-8")
    return output


def build_data_manifest() -> Path:
    manifest = {"source": "Kaggle IEEE-CIS Fraud Detection", "files": {}}
    for name in ["train_transaction.csv", "train_identity.csv"]:
        path = ROOT / "datos" / "raw" / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest["files"][name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    output = ART / "manifiesto_datos.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    outputs = [build_notebook(), build_report(), build_presentation(), build_readme(), build_requirements(), build_data_manifest()]
    print("\n".join(str(p) for p in outputs))
