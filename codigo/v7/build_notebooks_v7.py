"""Construye los dos cuadernos V7 estéticos a partir de artefactos ejecutados."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "entregables" / "cuaderno" / "v7"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = r"""<style>
:root{--navy:#102a43;--blue:#184e77;--teal:#2a9d8f;--gold:#e9c46a;--red:#e76f51;--ink:#172033;--muted:#5f6f7f;--pale:#edf5fb}
.hero{padding:38px 42px;border-radius:24px;color:#f8fbff;background:radial-gradient(circle at 92% 8%,rgba(255,255,255,.16) 0 8%,transparent 9%),linear-gradient(125deg,var(--navy),var(--blue) 55%,var(--teal));box-shadow:0 16px 38px #102a433d;font-family:Inter,'Segoe UI',sans-serif}.hero h1{font-size:38px;margin:18px 0 10px;color:white;border:0}.hero h2{font-size:20px;font-weight:450;color:white;margin:0}.chips{display:flex;gap:8px;flex-wrap:wrap}.chips span{padding:6px 13px;border:1px solid #ffffff55;border-radius:999px;background:#ffffff20;font-size:11px;font-weight:800}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:13px;margin-top:24px}.card{padding:14px 16px;border:1px solid #ffffff44;border-radius:12px;background:#ffffff12}.section{margin:28px 0 14px;padding:16px 22px;border-radius:14px;background:linear-gradient(90deg,var(--navy),var(--blue));color:white}.section h2{margin:0;color:white;border:0}.call{padding:18px 22px;margin:14px 0;border:1px solid #c9d9e6;border-left:6px solid var(--teal);border-radius:13px;background:var(--pale);color:var(--ink);line-height:1.7}.warn{border-left-color:var(--red);background:#fff2ed}.proof{border-left-color:var(--gold);background:#fff9e9}.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.metric{padding:13px;border-radius:11px;background:#e5f5f2;color:#184e77;text-align:center}.metric b{display:block;font-size:22px}.small{font-size:13px;color:var(--muted)}table{font-size:13px}th{background:#184e77!important;color:white!important;text-align:left!important}td,th{padding:8px!important}code{background:#102a4312;padding:2px 5px;border-radius:4px}
</style>"""


def hero(title: str, subtitle: str, label: str) -> str:
    return STYLE + f"""
<div class="hero">
  <div class="chips"><span>DEEP LEARNING</span><span>PROYECTO 1</span><span>VERSIÓN 7</span><span>{label}</span></div>
  <h1>{title}</h1><h2>{subtitle}</h2>
  <div class="grid">
    <div class="card"><b>Institución</b><br>Universidad del Valle de Guatemala</div>
    <div class="card"><b>Curso</b><br>Deep Learning y Sistemas Inteligentes</div>
    <div class="card"><b>Sección</b><br>30</div>
    <div class="card"><b>Integrante</b><br>Wilson Alejandro Calderón Argueta · 22018</div>
    <div class="card"><b>Integrante</b><br>Pablo Daniel Barillas Moreno · 22193</div>
    <div class="card"><b>Fuente</b><br>IEEE-CIS · Vesta Corporation · Kaggle</div>
  </div>
</div>
<p class="small" style="text-align:center">Versión exploratoria: el benchmark final es histórico y reutilizado.</p>"""


def section(title: str) -> str:
    return f'<div class="section"><h2>{title}</h2></div>'


def code_prelude() -> str:
    return """from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, display, Image

def find_root(start=Path.cwd()):
    for candidate in [start, *start.parents]:
        if (candidate / 'artefactos' / 'v7' / 'resultados_v7.json').exists():
            return candidate
    raise FileNotFoundError('Ejecute el notebook dentro del repositorio con artefactos/v7 disponibles.')

ROOT = find_root()
ART = ROOT / 'artefactos' / 'v7'
FIG = ROOT / 'evidencia' / 'figuras' / 'v7'
PROCESSED = ROOT / 'datos' / 'processed' / 'v7'
R = json.loads((ART / 'resultados_v7.json').read_text(encoding='utf-8'))
pd.set_option('display.max_colwidth', 90)
print(f'Raíz verificada: {ROOT}')
print(f"Versión: {R['version']} · filas: {R['datos']['filas']:,} · prevalencia: {R['datos']['prevalencia']:.3%}")"""


def official_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.13"}
    cells = [
        nbf.v4.new_markdown_cell(hero("Monitoreo transaccional con evidencia temporal", "Comparación A/B/C, reducción dimensional y decisión económica", "CUADERNO OFICIAL")),
        nbf.v4.new_markdown_cell(section("1 · Propósito, alcance y regla de honestidad") + r"""
<div class="call"><b>Pregunta:</b> ¿el orden de las transacciones aporta señal incremental frente a un baseline tabular competitivo y cuánto vale económicamente? V7 amplía las variables, contrasta regresión logística, LightGBM, CatBoost, correlación, PCA y stacking; conserva B secuencial y D encoder–decoder como controles congelados.</div>
<div class="call warn"><b>Lectura correcta:</b> el último 15 % ya fue observado en V1–V6. Por ello ninguna decisión V7 se toma con ese período y sus resultados se denominan <i>benchmark histórico reutilizado</i>. La conclusión confirmatoria exige una cohorte futura.</div>

La métrica primaria es Average Precision (AP), apropiada para fraude desbalanceado. El costo académico del umbral es

$$\operatorname{Costo}(\tau)=4200\,FN(\tau)+180\,FP(\tau),$$

sujeto a $\operatorname{recall}(\tau)\geq 0.75$."""),
        nbf.v4.new_code_cell(code_prelude()),
        nbf.v4.new_markdown_cell(section("2 · Integridad temporal y datos")),
        nbf.v4.new_code_cell("""parts = pd.DataFrame(R['datos']['particiones']).T
parts['prevalencia'] = parts['prevalencia'].map(lambda x: f'{x:.3%}')
display(HTML('<div class="call proof"><b>Comprobación:</b> train precede a validation y validation precede al benchmark; no existe partición aleatoria.</div>'))
display(parts)
assert R['estado_benchmark'] == 'historico_reutilizado_no_ciego'
assert R['datos']['particiones']['train']['dt_max'] < R['datos']['particiones']['validation']['dt_min']
assert R['datos']['particiones']['validation']['dt_max'] < R['datos']['particiones']['benchmark_historico']['dt_min']
print('✓ Integridad temporal y estado del benchmark verificados.')"""),
        nbf.v4.new_markdown_cell(section("3 · Variables, correlación y PCA") + r"""
<div class="call">La correlación no se utiliza como borrado automático. Solo identifica redundancia extrema ($|\rho_s|\geq0.995$) dentro de train. PCA se limita a `V1–V339`; las variables operativas y causales permanecen interpretables. Se comparan 32, 64 y 128 componentes porque conservar varianza no garantiza conservar la señal minoritaria.</div>"""),
        nbf.v4.new_code_cell("""audit = json.loads((PROCESSED / 'auditoria_variables_v7.json').read_text(encoding='utf-8'))
assoc = pd.read_csv(PROCESSED / 'asociacion_variables_train_v7.csv')
corr = pd.read_csv(PROCESSED / 'pares_correlacionados_train_v7.csv')
summary = pd.DataFrame({
    'Indicador': ['Columnas unión', 'Variables retenidas', 'Pares redundantes eliminados', 'Componentes PCA ajustados', 'Varianza con 128 componentes'],
    'Valor': [R['datos']['columnas_union'], audit['inventario']['retenidas'], len(corr), audit['pca']['componentes_ajustados'], f"{sum(audit['pca']['varianza_acumulada']):.2%}"]
})
display(summary)
display(assoc.head(15).style.bar(subset=['asociacion_point_biserial_abs_train'], color='#2a9d8f'))
display(Image(filename=str(FIG / '04_correlacion_pca_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("4 · Selección de A y controles")),
        nbf.v4.new_code_cell("""sel = pd.DataFrame({
    'Modelo': list(R['seleccion']['A']['auc_pr_model_select']),
    'AP model_select': list(R['seleccion']['A']['auc_pr_model_select'].values())
}).sort_values('AP model_select', ascending=False)
display(sel.style.format({'AP model_select':'{:.4f}'}).bar(subset=['AP model_select'], color='#184e77'))
display(HTML(f'''<div class="call proof"><b>A seleccionado:</b> {R['seleccion']['A']['seleccionado']}. El stacking incluye explícitamente el control V6; cada modelo base fue entrenado antes del bloque usado para ajustar el metamodelo.</div>'''))
display(Image(filename=str(FIG / '03_seleccion_modelos_a_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("5 · Comparación común A/B/C/D") + """
<div class="call"><b>A</b> es el baseline sin orden; <b>B</b> es la TCN causal congelada; <b>C</b> fusiona puntajes fuera de tiempo; <b>D</b> es un encoder–decoder entrenado solo con operaciones legítimas. Los cuatro reciben las mismas transacciones y producen puntajes continuos.</div>"""),
        nbf.v4.new_code_cell("""metric_cols = ['auc_pr','roc_auc','precision','recall','f1','cost_q','alertas_por_100k','precision_at_1pct','recall_at_1pct']
internal = pd.DataFrame(R['evaluacion_interna']).T.loc[['A','B','C','D'], metric_cols]
display(internal.style.format({c:'{:.4f}' for c in metric_cols if c not in ['cost_q','alertas_por_100k']} | {'cost_q':'Q{:,.0f}','alertas_por_100k':'{:,.0f}'}).highlight_max(subset=['auc_pr','precision','recall','f1'], color='#dff3ec').highlight_min(subset=['cost_q','alertas_por_100k'], color='#fff0cc'))
display(Image(filename=str(FIG / '01_comparacion_interna_v7.png')))
display(Image(filename=str(FIG / '05_costos_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("6 · Hipótesis C y falsificación del orden")),
        nbf.v4.new_code_cell("""c = R['hipotesis_C']; f = R['falsificaciones']
display(HTML(f'''<div class="call {'proof' if c['success'] else 'warn'}"><b>Veredicto C:</b> {'cumple' if c['success'] else 'no cumple'}. ΔAP={c['delta_ap']:+.4f}; reducción de costo={c['reduccion_costo']:+.2%}; crecimiento de alertas={c['crecimiento_alertas']:+.2%}.</div>'''))
order = pd.DataFrame([
    ['Original (32)', f['original_internal']['auc_pr']],
    ['Permutada · media', f['permutation_mean_auc_pr']],
    ['Historia 3', f['historia_3']['auc_pr']],
    ['Historia 8', f['historia_8']['auc_pr']],
    ['Historia 16', f['historia_16']['auc_pr']],
    ['Historia 32', f['historia_32']['auc_pr']],
], columns=['Prueba','AP'])
display(order.style.format({'AP':'{:.4f}'}).bar(subset=['AP'], color='#e9c46a'))
print(f"ΔAP original − permutada = {f['order_auc_pr_drop']:+.4f}; orden material = {f['orden_material']}")
assert len(f['permutaciones']) == 5"""),
        nbf.v4.new_markdown_cell(section("7 · Gate V7, estabilidad y benchmark histórico")),
        nbf.v4.new_code_cell("""gate = R['promocion_V7']
display(pd.DataFrame(gate['ventanas']).style.format({'delta_ap_V7_vs_V6':'{:+.4f}'}).bar(subset=['delta_ap_V7_vs_V6'], align='zero', color=['#e76f51','#2a9d8f']))
display(HTML(f'''<div class="call warn"><b>Gate de promoción:</b> {'APROBADO' if gate['success'] else 'NO APROBADO'}. ΔAP global={gate['delta_ap']:+.4f}; reducción de costo={gate['reduccion_costo']:+.2%}; cambio de alertas={gate['crecimiento_alertas']:+.2%}. La ventana negativa impide afirmar superioridad estable.</div>'''))
benchmark = pd.DataFrame(R['benchmark_historico']).T.loc[['A','B','C','D'], ['auc_pr','precision','recall','f1','cost_q','alertas_por_100k']]
display(benchmark.style.format({'auc_pr':'{:.4f}','precision':'{:.2%}','recall':'{:.2%}','f1':'{:.4f}','cost_q':'Q{:,.0f}','alertas_por_100k':'{:,.0f}'}))
display(Image(filename=str(FIG / '02_benchmark_historico_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("8 · Conclusión") + """
<div class="call proof"><b>Conclusión principal.</b> El ensamble A5 mejora el promedio interno y el perfil operativo frente a V6: eleva AP y precisión, reduce falsas alertas y costo, manteniendo recall por encima de 0.75. Sin embargo, la mejora no es uniforme en cuatro ventanas y por ello V7 no se declara confirmatoriamente superior.</div>
<div class="call"><b>Sobre el orden.</b> La permutación no reduce AP en al menos 0.01; B queda por debajo de A y C no satisface el criterio previo. Por tanto, con esta identidad proxy, el valor incremental del orden no está respaldado. La próxima evidencia útil sería una cohorte temporal nueva y una identidad bancaria más fiable, no una red más grande.</div>"""),
        nbf.v4.new_code_cell("""required = [ART/'resultados_v7.json', ART/'modelo_A5_ensamble_tabular_v7.joblib', ART/'modelos_C_fusion_v7.joblib', ART/'contrato_entrada_salida_v7.json']
assert all(path.exists() and path.stat().st_size > 0 for path in required)
print('✓ Artefactos críticos V7 presentes.')
print(f"Candidato exploratorio: {R['candidato']['modelo']} · {R['candidato']['detalle']}")"""),
    ]
    nb.cells = cells
    return nb


def eda_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.13"}
    cells = [
        nbf.v4.new_markdown_cell(hero("EDA causal de IEEE‑CIS", "Diagnóstico de variables, identidad, correlación, PCA y deriva", "ANÁLISIS EXPLORATORIO")),
        nbf.v4.new_markdown_cell(section("1 · Alcance del EDA") + """
<div class="call"><b>Fuente:</b> <a href="https://www.kaggle.com/competitions/ieee-fraud-detection/overview">IEEE-CIS Fraud Detection</a>, datos anonimizados proporcionados por Vesta Corporation. El EDA global se limita a estructura y calidad; toda conclusión que puede cambiar el modelo —asociación con `isFraud`, correlación, selección, imputación y PCA— se calcula solo en train.</div>
<div class="call warn"><b>Riesgo evitado:</b> observar correlaciones con el target o ajustar PCA en validación/test contaminaría la estimación temporal aunque no se entrene una red con esas etiquetas.</div>"""),
        nbf.v4.new_code_cell(code_prelude()),
        nbf.v4.new_markdown_cell(section("2 · Inventario y exclusiones")),
        nbf.v4.new_code_cell("""audit = json.loads((PROCESSED / 'auditoria_variables_v7.json').read_text(encoding='utf-8'))
inv = audit['inventario']
display(pd.DataFrame({'Indicador':['Columnas de la unión','Candidatas sin target/ID/tiempo','Retenidas','Numéricas','Categóricas'], 'Valor':[inv['columnas_crudas_union'],inv['candidatas_sin_id_tiempo_target'],inv['retenidas'],inv['numericas'],inv['categoricas']]}))
print('Eliminadas por >99.5% faltantes:', ', '.join(inv['eliminadas_faltantes_gt_99_5']) or 'ninguna')
print('Eliminadas por constancia en train:', ', '.join(inv['eliminadas_constantes_train']) or 'ninguna')
assert 'TransactionID' not in json.loads((ART/'contrato_entrada_salida_v7.json').read_text(encoding='utf-8'))['entrada']['variables']"""),
        nbf.v4.new_markdown_cell(section("3 · Desbalance y deriva temporal")),
        nbf.v4.new_code_cell("""parts = pd.DataFrame(R['datos']['particiones']).T
display(parts[['n','prevalencia','dt_min','dt_max']].style.format({'n':'{:,.0f}','prevalencia':'{:.3%}','dt_min':'{:,.0f}','dt_max':'{:,.0f}'}))
fig, ax = plt.subplots(figsize=(8,4)); ax.bar(parts.index, parts['prevalencia']*100, color=['#184e77','#2a9d8f','#e9c46a']); ax.set(ylabel='Fraude (%)', title='Prevalencia por período temporal'); ax.tick_params(axis='x', rotation=10); plt.show()
print('La prevalencia baja confirma que accuracy no debe ser la métrica principal.')"""),
        nbf.v4.new_markdown_cell(section("4 · Identidad proxy y cobertura de historia")),
        nbf.v4.new_code_cell("""ident = pd.DataFrame(R['identidades']).T
cols = ['entidades','mediana_transacciones_entidad','p90_transacciones_entidad','porcentaje_con_3','porcentaje_con_8','porcentaje_con_16','porcentaje_con_32','porcentaje_campos_faltantes']
display(ident[cols].style.format({c:'{:.2f}' for c in cols}).background_gradient(cmap='Blues', subset=['porcentaje_con_3','porcentaje_con_8','porcentaje_con_16','porcentaje_con_32']))
display(HTML('<div class="call"><b>Lectura:</b> más cobertura no garantiza identidad correcta. Las claves con dispositivo pueden fragmentar un mismo cliente y las claves de tarjeta/dirección pueden mezclar personas. Esto limita la capacidad de B para aprender orden auténtico.</div>'))"""),
        nbf.v4.new_markdown_cell(section("5 · Asociación train-only") + """
La asociación point-biserial absoluta se usa como diagnóstico y para limitar costos de CatBoost/logística. No se interpreta causalmente: una variable puede correlacionarse con fraude por mezcla de segmentos, faltantes o cambios temporales."""),
        nbf.v4.new_code_cell("""assoc = pd.read_csv(PROCESSED/'asociacion_variables_train_v7.csv')
display(assoc.head(25).style.format({'asociacion_point_biserial_abs_train':'{:.4f}'}).bar(subset=['asociacion_point_biserial_abs_train'], color='#2a9d8f'))
top = assoc.head(20).sort_values('asociacion_point_biserial_abs_train')
fig, ax = plt.subplots(figsize=(8,6)); ax.barh(top['variable'], top['asociacion_point_biserial_abs_train'], color='#2a9d8f'); ax.set(xlabel='|asociación| en train', title='Variables más asociadas dentro de train'); plt.show()"""),
        nbf.v4.new_markdown_cell(section("6 · Redundancia por correlación")),
        nbf.v4.new_code_cell("""corr = pd.read_csv(PROCESSED/'pares_correlacionados_train_v7.csv')
print(f"Pares con |rho_s| >= {R['variables']['correlacion']['umbral']}: {len(corr):,}")
display(corr.head(20).style.format({'rho_spearman_abs_train':'{:.4f}'}))
display(HTML('<div class="call proof"><b>Resultado:</b> eliminar redundancia extrema mejora AP en selección frente al LightGBM completo, pero esa ventaja no se sostiene por sí sola en evaluación. Correlación sirve para compactar; no garantiza generalización.</div>'))"""),
        nbf.v4.new_markdown_cell(section("7 · PCA: varianza frente a señal") + r"""
PCA encuentra componentes $\mathbf z=\mathbf W^\top(\mathbf x-\boldsymbol\mu)$ que maximizan varianza, no separación entre fraude y no fraude. Por eso se restringe al bloque anónimo V y se valida como ablation."""),
        nbf.v4.new_code_cell("""pca = audit['pca']; cumulative = np.array(pca['varianza_acumulada'])
summary = pd.DataFrame({'Corte':['90%','95%','99%','máximo ajustado'], 'Componentes':[pca['componentes_para_90'],pca['componentes_para_95'],pca['componentes_para_99'],pca['componentes_ajustados']]})
display(summary)
fig, ax = plt.subplots(figsize=(8,4)); ax.plot(np.arange(1,len(cumulative)+1), cumulative, color='#184e77'); ax.axhline(.95,ls='--',color='#e76f51'); ax.set(xlabel='Componentes',ylabel='Varianza explicada acumulada',title='PCA train-only del bloque V'); ax.grid(alpha=.2); plt.show()
display(Image(filename=str(FIG/'04_correlacion_pca_v7.png')))"""),
        nbf.v4.new_markdown_cell(section("8 · Recomendaciones derivadas del EDA") + """
<table><thead><tr><th>Hallazgo</th><th>Implicación</th><th>Decisión V7</th></tr></thead><tbody>
<tr><td>Alta dimensionalidad y categóricas</td><td>Una red mayor no corrige representación pobre</td><td>LightGBM, CatBoost y stacking</td></tr>
<tr><td>Redundancia extrema localizada</td><td>Puede reducir costo computacional</td><td>Ablation Spearman, sin borrado ciego</td></tr>
<tr><td>PCA conserva varianza, no necesariamente fraude</td><td>Debe validarse contra AP/costo</td><td>32/64/128 componentes solo en bloque V</td></tr>
<tr><td>Identidad proxy imperfecta</td><td>Historia puede contener ruido</td><td>Falsificación de orden y longitudes 3/8/16/32</td></tr>
<tr><td>Deriva entre ventanas</td><td>Una media oculta inestabilidad</td><td>Tres walk-forward y gate de cuatro ventanas</td></tr>
</tbody></table>
<div class="call"><b>Conclusión del EDA:</b> la mejora más defendible es combinar representaciones tabulares complementarias con el baseline fuerte. PCA y CatBoost no ganan individualmente; el orden sigue sin aportar señal material. La siguiente inversión debe ser una cohorte nueva e identidad más fiable.</div>"""),
    ]
    nb.cells = cells
    return nb


def main() -> None:
    nbf.write(official_notebook(), OUT / "proyecto1_calderon_barillas.ipynb")
    nbf.write(eda_notebook(), OUT / "EDA_IEEE_CIS_Diagnostico_Datos_V7.ipynb")
    print(f"Cuadernos V7 creados en {OUT}")


if __name__ == "__main__":
    main()
