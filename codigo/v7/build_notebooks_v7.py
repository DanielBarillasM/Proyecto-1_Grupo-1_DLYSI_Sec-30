"""Construye los notebooks V7 como documentos ejecutables y autocontenidos.

El cuaderno oficial ya no se limita a leer tablas finales: incorpora el código
visible de preparación temporal, modelos A/B/C/D, entrenamiento, calibración,
umbral económico y falsificaciones. La ejecución normal reutiliza artefactos
verificados; una bandera explícita permite lanzar el entrenamiento completo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "entregables" / "cuaderno" / "v7"
OUT.mkdir(parents=True, exist_ok=True)
PIPELINE = ROOT / "codigo" / "v7" / "proyecto1_v7_pipeline.py"
SEQUENTIAL = ROOT / "codigo" / "v7" / "modelos_secuenciales_v7.py"


STYLE = r"""<style>
:root{--navy:#102a43;--blue:#184e77;--teal:#2a9d8f;--gold:#e9c46a;--red:#e76f51;--ink:#172033;--muted:#5f6f7f;--pale:#edf5fb}
.section{box-sizing:border-box;margin:30px 0 15px;padding:17px 23px;border-radius:14px;background:linear-gradient(90deg,#102a43,#184e77);color:#fff;box-shadow:0 6px 18px rgba(16,42,67,.14);font-family:Inter,'Segoe UI',Arial,sans-serif}.section h2{margin:0;color:#fff;border:0;font-size:22px}.call{box-sizing:border-box;padding:18px 22px;margin:14px 0;border:1px solid #c9d9e6;border-left:6px solid #2a9d8f;border-radius:13px;background:#edf5fb;color:#172033;line-height:1.7;font-family:Inter,'Segoe UI',Arial,sans-serif}.warn{border-left-color:#e76f51;background:#fff2ed}.proof{border-left-color:#e9c46a;background:#fff9e9}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.metric{padding:13px;border:1px solid #c9e7e0;border-radius:11px;background:#e5f5f2;color:#184e77;text-align:center}.metric b{display:block;font-size:22px}.small{font-size:13px;color:#5f6f7f}.jp-RenderedHTMLCommon table{font-size:13px}.jp-RenderedHTMLCommon th{background:#184e77!important;color:#fff!important;text-align:left!important}.jp-RenderedHTMLCommon td,.jp-RenderedHTMLCommon th{padding:8px!important}.jp-RenderedHTMLCommon code{background:#102a4312;padding:2px 5px;border-radius:4px}
</style>"""


def hero(title: str, subtitle: str, label: str) -> str:
    return STYLE + f"""
<div style="box-sizing:border-box;width:100%;margin:12px 0 26px;padding:38px 42px;border:1px solid rgba(255,255,255,.18);border-radius:24px;color:#f8fbff;font-family:Inter,'Segoe UI',Arial,sans-serif;background:radial-gradient(circle at 92% 8%,rgba(255,255,255,.16) 0 8%,transparent 9%),linear-gradient(125deg,#102a43 0%,#184e77 56%,#2a9d8f 100%);box-shadow:0 16px 38px rgba(16,42,67,.28);overflow:hidden;">
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin:0 0 18px;">
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">Deep Learning</span>
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">Proyecto 1</span>
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">Versión 7</span>
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">{label}</span>
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">IEEE-CIS</span>
    <span style="display:inline-block;padding:6px 13px;border:1px solid rgba(255,255,255,.34);border-radius:999px;background:rgba(255,255,255,.13);color:#fff;font-size:11px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;">Grupo 1</span>
  </div>
  <div style="max-width:1040px;margin:0 0 10px;color:#fff;font-size:37px;font-weight:800;line-height:1.16;letter-spacing:-.02em;">{title}</div>
  <div style="max-width:980px;margin:0;color:rgba(255,255,255,.88);font-size:20px;font-weight:400;line-height:1.5;">{subtitle}</div>
  <div style="width:72px;height:4px;margin:22px 0;background:linear-gradient(90deg,#e9b949,rgba(255,255,255,.85));border-radius:999px;"></div>
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin-top:4px;">
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Institución</span>Universidad del Valle de Guatemala</div>
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Curso</span>Deep Learning y Sistemas Inteligentes</div>
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Sección y docente</span>Sección 30 · Kevin Recinos</div>
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Integrantes</span>Wilson Alejandro Calderón Argueta · 22018<br>Pablo Daniel Barillas Moreno · 22193</div>
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Datos</span>590,540 transacciones · 3.50 % fraude</div>
    <div style="padding:14px 16px;border:1px solid rgba(255,255,255,.25);border-radius:12px;background:rgba(255,255,255,.09);"><span style="display:block;margin-bottom:5px;color:rgba(255,255,255,.7);font-size:10px;font-weight:800;letter-spacing:.09em;text-transform:uppercase;">Fuente</span>IEEE-CIS · Vesta Corporation · Kaggle</div>
  </div>
  <div style="margin-top:16px;padding:12px 15px;border:1px solid rgba(233,185,73,.42);border-left:4px solid #e9b949;border-radius:10px;background:rgba(7,25,39,.22);font-size:12px;line-height:1.55;"><strong>Alcance inferencial:</strong> el 15 % final es un benchmark temporal histórico reutilizado; una afirmación confirmatoria requiere una cohorte futura etiquetada.</div>
</div>"""


def section(title: str) -> str:
    return f'<div class="section"><h2>{title}</h2></div>'


def source_nodes(path: Path, names: list[str]) -> str:
    """Extrae definiciones reales para que el notebook muestre el código fuente."""

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
    missing = set(names) - {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    if missing:
        raise ValueError(f"No se localizaron definiciones en {path.name}: {sorted(missing)}")
    return "\n\n".join(selected)


def notebook_prelude() -> str:
    return """from pathlib import Path
from dataclasses import dataclass
from typing import Any
import json, math, os, random, subprocess, sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from catboost import CatBoostClassifier
from IPython.display import HTML, Image, display
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset

def find_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate/'artefactos'/'v7'/'resultados_v7.json').exists():
            return candidate
    raise FileNotFoundError('Ejecute el notebook dentro del repositorio completo.')

ROOT = find_root()
ART = ROOT/'artefactos'/'v7'
FIG = ROOT/'evidencia'/'figuras'/'v7'
PROCESSED = ROOT/'datos'/'processed'/'v7'
RAW = ROOT/'datos'/'raw'
R = json.loads((ART/'resultados_v7.json').read_text(encoding='utf-8'))
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_colwidth', 100)
print(f'Raíz: {ROOT}')
print(f"Versión {R['version']} · {R['datos']['filas']:,} filas · fraude {R['datos']['prevalencia']:.3%}")"""


def official_notebook() -> nbf.NotebookNode:
    temporal_code = source_nodes(PIPELINE, ["ConfigV7", "temporal_split", "validation_bounds"])
    feature_code = source_nodes(PIPELINE, ["row_block_summary", "add_causal_features", "encode_train_only", "correlation_representatives", "fit_incremental_pca"])
    evaluation_code = source_nodes(PIPELINE, ["metric_set", "choose_threshold", "fit_calibrator", "apply_calibrator", "fit_meta"])
    sequence_code = source_nodes(
        SEQUENTIAL,
        ["SequenceConfigV7", "IndexedSequenceDataset", "embedding_dim", "GRURiskModel", "CausalResidualBlock", "TCNRiskModel"],
    )
    anomaly_code = source_nodes(SEQUENTIAL, ["TransactionAutoencoder", "train_autoencoder", "reconstruction_error"])
    train_sequence_code = source_nodes(SEQUENTIAL, ["train_sequence_model", "predict_sequence_model", "sequence_variant"])

    cells = [
        nbf.v4.new_markdown_cell(hero(
            "Monitoreo transaccional con evidencia temporal",
            "Notebook ejecutable con implementación visible de A/B/C/D, falsificaciones y decisión económica",
            "CUADERNO OFICIAL",
        )),
        nbf.v4.new_markdown_cell(section("1 · Pregunta, hipótesis y criterio de éxito") + r"""
<div class="call"><b>Pregunta central.</b> ¿El orden de las transacciones aporta información incremental frente a un baseline tabular competitivo, bajo qué condiciones y cuánto vale esa diferencia en quetzales?</div>
<div class="call proof"><b>Hipótesis C previa.</b> Creemos que una fusión de puntajes tabulares, secuenciales y de anomalía mejorará AP y costo porque sus errores pueden ser complementarios. C será útil únicamente si aumenta AP al menos 0.01, reduce el costo al menos 5 %, mantiene recall ≥ 0.75, no aumenta alertas más de 10 % y mejora en tres de cuatro ventanas.</div>

La métrica primaria es **AP (Average Precision)**, no “PA”. AP resume la curva precisión–recall a través de umbrales. Con prevalencia $\pi=0.035$, un ranking sin información tiene AP cercana a $0.035$. AP $=0.547$ indica una separación muy superior a esa referencia, pero **no** significa que 54.7 % de las alertas sea fraude; esa pureza puntual la mide la precisión.

$$\operatorname{Costo}(\tau)=Q4{,}200\,FN(\tau)+Q180\,FP(\tau),\qquad \operatorname{recall}(\tau)\geq0.75.$$
"""),
        nbf.v4.new_code_cell(notebook_prelude()),
        nbf.v4.new_markdown_cell(section("2 · Configuración y partición temporal") + """
El siguiente código es el mismo que utiliza el pipeline V7. La separación es cronológica y la validación se subdivide para que early stopping, stacking, selección, calibración, umbral y evaluación no compartan la misma evidencia.
"""),
        nbf.v4.new_code_cell(temporal_code),
        nbf.v4.new_code_cell("""cfg = ConfigV7()
parts = pd.DataFrame(R['datos']['particiones']).T
display(parts[['n','prevalencia','dt_min','dt_max']].style.format({'n':'{:,.0f}','prevalencia':'{:.3%}'}))
assert R['estado_benchmark'] == 'historico_reutilizado_no_ciego'
assert parts.loc['train','dt_max'] < parts.loc['validation','dt_min']
assert parts.loc['validation','dt_max'] < parts.loc['benchmark_historico','dt_min']
print('✓ Partición temporal y estado del benchmark verificados.')"""),
        nbf.v4.new_markdown_cell(section("3 · Ingeniería causal, correlación y PCA") + r"""
Para cada transacción $t$, los agregados históricos satisfacen $x_t^{hist}=f(\{x_j:t_j<t\})$. El evento actual no participa en su propia media, recencia o frecuencia. La correlación de Spearman elimina únicamente redundancia extrema dentro de train; PCA se restringe al bloque `V1–V339` y se trata como una ablation, no como una mejora garantizada.
"""),
        nbf.v4.new_code_cell(feature_code),
        nbf.v4.new_code_cell("""audit = json.loads((PROCESSED/'auditoria_variables_v7.json').read_text(encoding='utf-8'))
assoc = pd.read_csv(PROCESSED/'asociacion_variables_train_v7.csv')
corr = pd.read_csv(PROCESSED/'pares_correlacionados_train_v7.csv')
display(pd.DataFrame({
    'Indicador':['Columnas unión','Candidatas','Retenidas tras calidad','Pares redundantes','PCA para 95%'],
    'Valor':[R['datos']['columnas_union'],audit['inventario']['candidatas_sin_id_tiempo_target'],audit['inventario']['retenidas'],len(corr),audit['pca']['componentes_para_95']]
}))
display(assoc.head(15).style.format({'asociacion_point_biserial_abs_train':'{:.4f}'}).bar(subset=['asociacion_point_biserial_abs_train'],color='#2a9d8f'))
display(Image(filename=str(FIG/'04_correlacion_pca_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("4 · Modelo A: línea base sin orden") + """
A no recibe una secuencia ordenada. Se comparan regresión logística, LightGBM completo, LightGBM reducido, LightGBM con PCA y CatBoost; A5 combina sus logits con un metamodelo logístico ajustado fuera del período de entrenamiento base.
"""),
        nbf.v4.new_code_cell("""def construir_candidatos_a(seed=2026):
    return {
        'A0_logistica': Pipeline([
            ('scale', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)),
        ]),
        'A1_lightgbm': lgb.LGBMClassifier(
            objective='binary', n_estimators=1400, learning_rate=.025,
            num_leaves=63, min_child_samples=100, subsample=.85,
            colsample_bytree=.80, reg_alpha=.5, reg_lambda=1.5,
            random_state=seed, n_jobs=min(8, os.cpu_count() or 1), verbosity=-1,
        ),
        'A4_catboost': CatBoostClassifier(
            iterations=900, depth=8, learning_rate=.04, loss_function='Logloss',
            eval_metric='PRAUC', random_seed=seed, verbose=False,
        ),
    }

def ajustar_stacking_a5(scores_meta_fit, y_meta_fit, seed=2026):
    model = Pipeline([
        ('scale', StandardScaler()),
        ('meta', LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    return model.fit(scores_meta_fit, y_meta_fit)

candidatos_a = construir_candidatos_a()
print('Modelos A programados:', ', '.join(candidatos_a))
print('A3 reutiliza fit_incremental_pca mostrado en la sección anterior; A5 usa ajustar_stacking_a5.')"""),
        nbf.v4.new_code_cell("""selection = pd.Series(R['seleccion']['A']['auc_pr_model_select'],name='AP model_select').sort_values(ascending=False)
display(selection.to_frame().style.format('{:.4f}').bar(color='#184e77'))
display(Image(filename=str(FIG/'03_seleccion_modelos_a_v7.png')))
print('Seleccionado:',R['seleccion']['A']['seleccionado'])"""),
        nbf.v4.new_markdown_cell(section("5 · Modelo B: GRU y TCN causal") + """
Estas son las implementaciones ejecutables del modelo secuencial. La GRU resume el historial mediante estado oculto; la TCN usa convoluciones causales dilatadas. Congelar el control B durante la iteración tabular V7 evita atribuir a nuevas variables una mejora producida por reentrenar simultáneamente la red.
"""),
        nbf.v4.new_code_cell(sequence_code),
        nbf.v4.new_markdown_cell(section("6 · Entrenamiento de B y pruebas de falsificación") + """
B se entrena con BCE ponderada, AdamW, clipping de gradiente y early stopping por AP. La permutación conserva el evento objetivo al final y destruye únicamente el orden de sus antecedentes.
"""),
        nbf.v4.new_code_cell(train_sequence_code),
        nbf.v4.new_markdown_cell(section("7 · Modelo D: encoder–decoder de normalidad") + r"""
D se ajusta únicamente con transacciones legítimas y utiliza el error $p^{-1}\sum_j(x_j-\hat x_j)^2$ como score de rareza. Esto atiende el desbalance, pero el experimento debe comprobar si rareza realmente coincide con fraude.
"""),
        nbf.v4.new_code_cell(anomaly_code),
        nbf.v4.new_markdown_cell(section("8 · Apuesta C: fusión controlada") + """
C no es una concatenación arbitraria: combina logits de A/B/D y variables de calidad del historial. El metamodelo se ajusta en `meta_fit` y se juzga después en `model_select` y evaluación.
"""),
        nbf.v4.new_code_cell("""def matriz_fusion(score_a, score_b, score_d, quality, amount, history_length, missing):
    eps = 1e-6
    logit = lambda s: np.log(np.clip(s,eps,1-eps)/np.clip(1-s,eps,1-eps))
    la, lb, ld = logit(score_a), logit(score_b), logit(score_d)
    return np.column_stack([la,lb,ld,(lb-la)*quality,quality,amount,history_length,missing])

def ajustar_modelo_c(z_meta_fit, y_meta_fit, seed=2026):
    return Pipeline([
        ('scale',StandardScaler()),
        ('meta',LogisticRegression(max_iter=2000,random_state=seed)),
    ]).fit(z_meta_fit,y_meta_fit)

print('C1=A+B; C2=A+B+D; C3 añade calidad, monto, longitud y faltantes.')
print('Hipótesis previa:',R['hipotesis_C']['declaracion_previa'])"""),
        nbf.v4.new_markdown_cell(section("9 · Métricas, calibración y umbral económico")),
        nbf.v4.new_code_cell(evaluation_code),
        nbf.v4.new_code_cell("""metrics = ['auc_pr','roc_auc','precision','recall','f1','cost_q','alertas_por_100k']
internal = pd.DataFrame(R['evaluacion_interna']).T.loc[['A','B','C','D'],metrics]
display(internal.style.format({'auc_pr':'{:.4f}','roc_auc':'{:.4f}','precision':'{:.2%}','recall':'{:.2%}','f1':'{:.4f}','cost_q':'Q{:,.0f}','alertas_por_100k':'{:,.0f}'}))
display(Image(filename=str(FIG/'01_comparacion_interna_v7.png')))
display(Image(filename=str(FIG/'05_costos_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("10 · Ejecución reproducible: completa o desde artefactos") + """
El notebook se entrega ejecutado en modo de verificación para no volver a entrenar accidentalmente al abrirlo. Cambiar la bandera a `True` ejecuta el pipeline completo con los CSV locales. El código de los modelos permanece visible arriba aun cuando se reutilizan artefactos comprobados.
"""),
        nbf.v4.new_code_cell("""REENTRENAR_DESDE_CERO = False

if REENTRENAR_DESDE_CERO:
    env = os.environ.copy()
    env['PROYECTO1_ROOT'] = str(ROOT)
    env['PROYECTO1_RAW'] = str(RAW)
    subprocess.run([sys.executable,'-u',str(ROOT/'codigo'/'v7'/'proyecto1_v7_pipeline.py')],check=True,env=env)
    R = json.loads((ART/'resultados_v7.json').read_text(encoding='utf-8'))
    print('✓ Entrenamiento completo finalizado.')
else:
    required = [ART/'modelo_A5_ensamble_tabular_v7.joblib',ART/'modelos_C_fusion_v7.joblib',ART/'resultados_v7.json']
    assert all(path.exists() and path.stat().st_size>0 for path in required)
    print('Modo verificación: se reutilizan artefactos V7 ya entrenados y auditados.')"""),
        nbf.v4.new_markdown_cell(section("11 · Resultados de C y valor del orden")),
        nbf.v4.new_code_cell("""c=R['hipotesis_C']; fals=R['falsificaciones']
display(HTML(f'''<div class="call {'proof' if c['success'] else 'warn'}"><b>C:</b> ΔAP={c['delta_ap']:+.4f}; reducción de costo={c['reduccion_costo']:+.2%}; ventanas positivas={sum(v['delta_ap_C_vs_A']>0 for v in c['ventanas'])}/4. Veredicto: {'útil' if c['success'] else 'no cumple'}.</div>'''))
order=pd.DataFrame([
 ['Original',fals['original_internal']['auc_pr']],['Permutada',fals['permutation_mean_auc_pr']],
 ['Historia 3',fals['historia_3']['auc_pr']],['Historia 8',fals['historia_8']['auc_pr']],
 ['Historia 16',fals['historia_16']['auc_pr']],['Historia 32',fals['historia_32']['auc_pr']],
],columns=['Prueba','AP'])
display(order.style.format({'AP':'{:.4f}'}).bar(subset=['AP'],color='#e9c46a'))
print(f"Caída original−permutada: {fals['order_auc_pr_drop']:+.4f}; orden material={fals['orden_material']}")"""),
        nbf.v4.new_markdown_cell(section("12 · Estabilidad, benchmark y conclusión")),
        nbf.v4.new_code_cell("""gate=R['promocion_V7']
display(pd.DataFrame(gate['ventanas']).style.format({'delta_ap_V7_vs_V6':'{:+.4f}'}).bar(subset=['delta_ap_V7_vs_V6'],align='zero',color=['#e76f51','#2a9d8f']))
benchmark=pd.DataFrame(R['benchmark_historico']).T.loc[['A','B','C','D'],['auc_pr','precision','recall','f1','cost_q']]
display(benchmark.style.format({'auc_pr':'{:.4f}','precision':'{:.2%}','recall':'{:.2%}','f1':'{:.4f}','cost_q':'Q{:,.0f}'}))
display(Image(filename=str(FIG/'02_benchmark_historico_v7.png')))"""),
        nbf.v4.new_markdown_cell("""<div class="call proof"><b>Conclusión final.</b> A5 es el candidato exploratorio porque mejora AP, precisión, F1, costo y alertas en promedio frente al control. No se declara reemplazo confirmatorio: una ventana temporal incumple el gate. B contiene señal, pero permutar el orden no la reduce; C y D no satisfacen sus reglas. La próxima evidencia útil es una cohorte futura con identidad bancaria más fiable y costos operativos reales.</div>"""),
        nbf.v4.new_code_cell("""assert len(R['falsificaciones']['permutaciones'])==5
assert set(['A','B','C','D']).issubset(R['evaluacion_interna'])
assert R['datos']['particiones']['train']['dt_max'] < R['datos']['particiones']['validation']['dt_min']
print('✓ Notebook oficial completo: A/B/C/D, dos falsificaciones, costo, límites y artefactos verificados.')"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    return notebook


def eda_notebook() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(hero(
            "Análisis exploratorio causal de IEEE-CIS",
            "Calidad, desbalance, deriva, identidad proxy, asociación, correlación y PCA antes del modelado",
            "EDA CAUSAL",
        )),
        nbf.v4.new_markdown_cell(section("1 · Alcance y fuente") + """
<div class="call"><b>Fuente:</b> <a href="https://www.kaggle.com/competitions/ieee-fraud-detection/overview">IEEE-CIS Fraud Detection</a>, datos anonimizados proporcionados por Vesta Corporation. El EDA distingue descripción global de decisiones train-only: cualquier cálculo que pueda cambiar variables o modelos se ajusta exclusivamente con entrenamiento.</div>
"""),
        nbf.v4.new_code_cell(notebook_prelude()),
        nbf.v4.new_markdown_cell(section("2 · Código para reconstruir el inventario desde los CSV") + """
La ejecución entregada utiliza la auditoría cacheada para ser rápida. La función siguiente permite comprobar directamente encabezados, tamaños y unión cuando los CSV están disponibles.
"""),
        nbf.v4.new_code_cell("""def auditar_archivos_raw(raw_dir=RAW):
    tx=raw_dir/'train_transaction.csv'; identity=raw_dir/'train_identity.csv'
    if not tx.exists() or not identity.exists():
        raise FileNotFoundError('Descargue IEEE-CIS en datos/raw antes de recalcular.')
    tx_head=pd.read_csv(tx,nrows=5); id_head=pd.read_csv(identity,nrows=5)
    return pd.DataFrame({
        'archivo':[tx.name,identity.name],
        'columnas':[tx_head.shape[1],id_head.shape[1]],
        'clave_presente':['TransactionID' in tx_head,'TransactionID' in id_head],
        'tamaño_MB':[tx.stat().st_size/2**20,identity.stat().st_size/2**20],
    })

display(auditar_archivos_raw().style.format({'tamaño_MB':'{:.1f}'}))"""),
        nbf.v4.new_markdown_cell(section("3 · Inventario, exclusiones y control de IDs")),
        nbf.v4.new_code_cell("""audit=json.loads((PROCESSED/'auditoria_variables_v7.json').read_text(encoding='utf-8'))
inv=audit['inventario']
display(pd.Series(inv,name='valor').to_frame())
contract=json.loads((ART/'contrato_entrada_salida_v7.json').read_text(encoding='utf-8'))
assert 'TransactionID' not in contract['entrada']['variables']
assert 'TransactionDT' not in contract['entrada']['variables']
print('✓ IDs y tiempo crudo excluidos como magnitudes predictivas.')"""),
        nbf.v4.new_markdown_cell(section("4 · Desbalance y deriva temporal") + r"""
La prevalencia se compara cronológicamente. Una diferencia entre períodos puede cambiar AP, calibración y el umbral aun cuando la arquitectura sea idéntica.
"""),
        nbf.v4.new_code_cell("""parts=pd.DataFrame(R['datos']['particiones']).T
display(parts[['n','prevalencia','dt_min','dt_max']].style.format({'n':'{:,.0f}','prevalencia':'{:.3%}'}))
fig,axes=plt.subplots(1,2,figsize=(12,4))
axes[0].bar(parts.index,parts['n'],color=['#184e77','#2a9d8f','#e9c46a']); axes[0].set_title('Filas por período'); axes[0].tick_params(axis='x',rotation=10)
axes[1].bar(parts.index,100*parts['prevalencia'],color=['#184e77','#2a9d8f','#e9c46a']); axes[1].set_title('Fraude por período'); axes[1].set_ylabel('%'); axes[1].tick_params(axis='x',rotation=10)
plt.tight_layout(); plt.show()
print(f"Baseline AP aproximado por prevalencia total: {R['datos']['prevalencia']:.4f}")"""),
        nbf.v4.new_markdown_cell(section("5 · Identidad proxy y cobertura secuencial")),
        nbf.v4.new_code_cell("""ident=pd.DataFrame(R['identidades']).T
cols=['entidades','mediana_transacciones_entidad','p90_transacciones_entidad','porcentaje_con_3','porcentaje_con_8','porcentaje_con_16','porcentaje_con_32','porcentaje_campos_faltantes']
display(ident[cols].style.format('{:.2f}').background_gradient(cmap='Blues',subset=['porcentaje_con_3','porcentaje_con_8','porcentaje_con_16','porcentaje_con_32']))
display(HTML('<div class="call warn"><b>Límite:</b> cobertura no equivale a identidad correcta. Una clave puede mezclar personas o fragmentar una misma tarjeta.</div>'))"""),
        nbf.v4.new_markdown_cell(section("6 · Asociación con fraude calculada solo en train")),
        nbf.v4.new_code_cell("""assoc=pd.read_csv(PROCESSED/'asociacion_variables_train_v7.csv')
display(assoc.head(30).style.format({'asociacion_point_biserial_abs_train':'{:.4f}'}).bar(subset=['asociacion_point_biserial_abs_train'],color='#2a9d8f'))
top=assoc.head(20).sort_values('asociacion_point_biserial_abs_train')
fig,ax=plt.subplots(figsize=(9,6)); ax.barh(top['variable'],top['asociacion_point_biserial_abs_train'],color='#2a9d8f'); ax.set(title='Asociación absoluta train-only',xlabel='|r point-biserial|'); plt.show()"""),
        nbf.v4.new_markdown_cell(section("7 · Correlación: reducir ruido sin borrar señal marginal útil")),
        nbf.v4.new_code_cell("""corr=pd.read_csv(PROCESSED/'pares_correlacionados_train_v7.csv')
display(corr.head(25).style.format({'rho_spearman_abs_train':'{:.4f}'}))
print(f"Pares extremos: {len(corr)} · umbral: {R['variables']['correlacion']['umbral']}")
print('La variable conservada es la de mayor asociación train-only dentro del par, no la de nombre más cómodo.')"""),
        nbf.v4.new_markdown_cell(section("8 · PCA: compresión frente a señal predictiva") + r"""
PCA maximiza varianza explicada, no separación de fraude. Se ajusta solo con train y únicamente sobre el bloque anónimo V.
"""),
        nbf.v4.new_code_cell("""pca=audit['pca']; cumulative=np.asarray(pca['varianza_acumulada'])
display(pd.DataFrame({'corte':['90%','95%','99%','máximo'],'componentes':[pca['componentes_para_90'],pca['componentes_para_95'],pca['componentes_para_99'],pca['componentes_ajustados']]}))
fig,ax=plt.subplots(figsize=(9,4)); ax.plot(np.arange(1,len(cumulative)+1),cumulative,color='#184e77'); ax.axhline(.95,ls='--',color='#e76f51'); ax.set(xlabel='Componentes',ylabel='Varianza acumulada',title='PCA train-only del bloque V'); ax.grid(alpha=.2); plt.show()
display(Image(filename=str(FIG/'04_correlacion_pca_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("9 · Diagnóstico que orienta el modelado") + """
<table><thead><tr><th>Hallazgo</th><th>Qué descarta</th><th>Decisión experimental</th></tr></thead><tbody>
<tr><td>Clase positiva 3.50 %</td><td>Accuracy como métrica principal</td><td>AP, recall, precisión, costo y alertas</td></tr>
<tr><td>465 columnas y faltantes</td><td>Aumentar red sin revisar representación</td><td>LightGBM, CatBoost y controles de calidad</td></tr>
<tr><td>34 redundancias extremas</td><td>Borrado ciego por correlación</td><td>A2 como ablation train-only</td></tr>
<tr><td>PCA-128 pierde AP</td><td>Suponer que varianza equivale a fraude</td><td>No promover PCA como candidato</td></tr>
<tr><td>Identidad proxy incierta</td><td>Atribuir éxito a memoria por arquitectura</td><td>Permutación y longitudes 3/8/16/32</td></tr>
<tr><td>Deriva entre ventanas</td><td>Promover por una media</td><td>Walk-forward y gate de estabilidad</td></tr>
</tbody></table>
<div class="call proof"><b>Conclusión del EDA.</b> La oportunidad principal está en representación tabular causal y control temporal. El orden debe demostrar valor mediante falsificación; una red más grande no resuelve una identidad ruidosa.</div>
"""),
        nbf.v4.new_code_cell("""required=[PROCESSED/'auditoria_variables_v7.json',PROCESSED/'asociacion_variables_train_v7.csv',PROCESSED/'pares_correlacionados_train_v7.csv']
assert all(path.exists() and path.stat().st_size>0 for path in required)
print('✓ EDA reproducible y evidencia train-only presentes.')"""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3.13"}
    return notebook


def main() -> None:
    nbf.write(official_notebook(), OUT / "proyecto1_calderon_barillas.ipynb")
    nbf.write(eda_notebook(), OUT / "EDA_IEEE_CIS_Diagnostico_Datos_V7.ipynb")
    print(f"Cuadernos V7 creados en {OUT}")


if __name__ == "__main__":
    main()
