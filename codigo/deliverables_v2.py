from __future__ import annotations

import html
import json
import shutil
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "artefactos/v2/resultados_v2.json").read_text(encoding="utf-8"))


def f(x, d=3):
    return f"{float(x):,.{d}f}"


def pct(x):
    return f"{100 * float(x):.2f}%"


def money(x):
    return f"Q{float(x):,.0f}"


def styles():
    return """<style>
.hero{padding:38px 42px;border-radius:24px;color:#f8fbff;background:linear-gradient(125deg,#102a43,#184e77 55%,#2a9d8f);box-shadow:0 16px 38px #102a433d;font-family:Inter,'Segoe UI',sans-serif}.hero h1{font-size:38px;margin:12px 0;color:white}.chips span{display:inline-block;padding:6px 12px;margin:3px;border:1px solid #ffffff55;border-radius:999px;background:#ffffff20;font-size:12px;font-weight:700}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.card{padding:16px;border-radius:12px;background:#ffffff16;border:1px solid #ffffff38}.callout{margin:18px 0;padding:20px 24px;border-left:6px solid #2a9d8f;border-radius:12px;background:#edf7f6;color:#172033}.warn{border-left-color:#e9a23b;background:#fff8e8}.section{margin-top:28px;padding:15px 22px;border-radius:13px;background:linear-gradient(90deg,#102a43,#184e77);color:white}.metric{display:inline-block;min-width:150px;padding:15px;margin:5px;border-radius:12px;background:#edf5fb;border:1px solid #c9d9e6;text-align:center}.metric b{display:block;font-size:24px;color:#184e77}table{width:100%}th{background:#184e77!important;color:white!important;text-align:left!important}td,th{padding:9px!important}code{background:#102a4310;padding:2px 5px;border-radius:4px}</style>"""


def readme():
    walk = R["validacion_walk_forward"]["resumen"]
    wr = "\n".join(
        f"| {x['modelo']} | {f(x['auc_pr_media'], 4)} | {f(x['auc_pr_sd'], 4)} | {f(x['roc_auc_media'], 4)} | {f(x['segundos_media'], 1)} s |"
        for x in walk
    )
    tab = R["modelo_tabular_v2"]["benchmark_historico"]
    ens = R["ensamble_v2"]["benchmark_historico"]
    ids = "\n".join(
        f"| {k.replace('_', ' ').title()} | {v['entidades']:,} | {f(v['mediana_transacciones'], 1)} | {f(v['porcentaje_con_3'], 1)}% | {f(v['porcentaje_con_8'], 1)}% | {f(v['porcentaje_con_16'], 1)}% |"
        for k, v in R["identidad_secuencial"].items()
    )
    text = rf"""<div align="center">

# Proyecto 1 · Monitoreo transaccional con aprendizaje profundo

### Variables causales, validación temporal y decisión económica

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-184e77)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-f4a261)
![Estado](https://img.shields.io/badge/Estado-V2%20auditada-2a9d8f)

**Wilson Alejandro Calderón Argueta · 22018** · **Pablo Daniel Barillas Moreno · 22193**

Universidad del Valle de Guatemala · Deep Learning y Sistemas Inteligentes · Sección 30 · 2026

</div>

> [!IMPORTANT]
> El último 15% cronológico es un **benchmark histórico reutilizado**: ya fue observado en V1 y no se presenta como test ciego. La selección y comparación se realizan en ventanas anteriores.

## Contenido

- [Resumen ejecutivo](#resumen-ejecutivo)
- [Datos y fuga temporal](#datos-y-fuga-temporal)
- [De V1 a V2](#de-v1-a-v2)
- [Correlación, selección y PCA](#correlación-selección-y-pca)
- [Protocolo y resultados](#protocolo-y-resultados)
- [Economía, calibración y capacidad](#economía-calibración-y-capacidad)
- [Reproducción y estructura](#reproducción-y-estructura)
- [Limitaciones, ética y referencias](#limitaciones-ética-y-referencias)

## Resumen ejecutivo

El proyecto estudia detección de fraude en {R["datos"]["filas"]:,} transacciones de IEEE-CIS, distribuidas durante {f(R["datos"]["dias"], 1)} días, con prevalencia {pct(R["datos"]["prevalencia"])}. Debido al desbalance, AUC-PR es la métrica principal de ranking. La decisión operativa usa un supuesto académico de Q4,200 por falso negativo y Q180 por falso positivo; omitir fraude pesa 23.3 veces más que generar una alerta innecesaria.

La V2 amplía la lectura desde 24 columnas en V1 hasta todas las variables transaccionales y de identidad disponibles, pero evita incorporarlas ciegamente. Cada candidata se evalúa por ausencia, cardinalidad, varianza, Pearson, Spearman, información mutua y redundancia. `TransactionID` solo une tablas y `TransactionDT` solo ordena. Los códigos de tarjeta, dirección y dispositivo se tratan como categorías o componentes de una identidad aproximada, no como magnitudes continuas.

El ganador interno fue **{R["validacion_walk_forward"]["ganador"]}**. LightGBM V2 alcanza AUC-PR {f(tab["auc_pr"], 4)} en el benchmark histórico; el ensamble calibrado, {f(ens["auc_pr"], 4)}. El intervalo descriptivo por bloques temporales de LightGBM V2 es [{f(R["intervalo_auc_pr_benchmark"]["li95"], 4)}, {f(R["intervalo_auc_pr_benchmark"]["ls95"], 4)}]. Estas cifras facilitan continuidad con V1, pero una afirmación confirmatoria requiere una cohorte nueva.

## Datos y fuga temporal

`train_transaction.csv` y `train_identity.csv` se integran uno-a-uno mediante `TransactionID`. El pipeline ordena por `TransactionDT` y después por la llave. Para el evento actual $t$, toda variable histórica cumple:

$$x_t^{{hist}}=f\left(\{{x_j:t_j<t\}}\right).$$

Se construyen conteo previo, promedio y desviación histórica de monto, razón monto/promedio, tiempo desde la transacción anterior y actividad en 1, 6, 24 y 72 horas. La estadística se emite antes de incorporar el evento actual. Imputación, codificación, reducción, modelo, calibración y umbral se ajustan con pasado únicamente.

```mermaid
flowchart LR
 A[IEEE-CIS] --> B[Unión y orden]
 B --> C[Variables causales]
 C --> D[Auditoría en pasado]
 D --> E[Walk-forward × 3]
 E --> F[LightGBM · CatBoost · PCA]
 F --> G[Calibración y costo]
 G --> H[Benchmark histórico]
 H --> I[resultados_v2.json]
 I --> J[Todos los entregables]
```

### Cobertura de identidades aproximadas

| Proxy | Entidades | Mediana | ≥3 | ≥8 | ≥16 |
|---|---:|---:|---:|---:|---:|
{ids}

Estas claves pueden mezclar personas o fragmentar una misma persona. La tabla informa si una arquitectura secuencial realmente dispone de historia; más longitud nominal no ayuda cuando la mayoría de entidades carece de antecedentes.

## De V1 a V2

| Dimensión | V1 preservada | V2 |
|---|---|---|
| Variables leídas | 21 transacción + 3 identidad | 394 transacción + 41 identidad |
| Muestra tabular | máximo 180,000 | 180k, 300k y todo el 70% |
| Validación | corte único | tres ventanas walk-forward |
| Reducción | selección manual | correlación, MI, redundancia y PCA |
| Baseline | HistGradientBoosting | LightGBM y CatBoost |
| Riesgo | umbral por costo | calibración, Brier, ECE y holdout |
| Evidencia | AP, costo, permutación | además CI, top-k, deriva y segmentos |

V1 no se borra ni reescribe: modelos, figuras, código y resultados se congelan en carpetas `v1`. La permutación del orden en V1 redujo AUC-PR apenas 0.0017 y usar tres eventos rindió casi igual que ocho; por eso una GRU más profunda no fue la primera intervención.

## Correlación, selección y PCA

Pearson detecta asociación lineal; Spearman, monotonía; información mutua, dependencia no lineal. Ninguna prueba causalidad. La correlación se usa para comprender señal y quitar redundancia, no para eliminar automáticamente todo predictor con correlación marginal pequeña: interacciones y no linealidades pueden volverlo útil.

Se retuvieron {len(R["seleccion_variables"]["numericas_seleccionadas"])} numéricas y {len(R["seleccion_variables"]["categoricas_seleccionadas"])} categóricas. Se descartaron {len(R["seleccion_variables"]["redundantes_eliminadas"])} por $|\rho_s|\geq {R["seleccion_variables"]["umbral_redundancia"]}$. Cada exclusión queda registrada en `datos/processed/v2`.

PCA se ajusta dentro de cada fold, únicamente en variables numéricas elegibles, después de imputar y escalar. Nunca incluye etiqueta, orden, IDs o categorías. Conservar 95% de varianza no equivale a conservar señal de fraude; PCA solo se adopta si mejora establemente fuera de tiempo.

![Relevancia](../evidencia/figuras/v2/03_relevancia_variables.png)

## Protocolo y resultados

Tres ventanas simulan reentrenamientos consecutivos: siempre entrenan con el pasado y evalúan el bloque posterior. CatBoost conserva categorías nativas; LightGBM modela la representación depurada; LightGBM+PCA mide compresión. El presupuesto CatBoost de 240 iteraciones es transparente y no pretende representar una búsqueda exhaustiva.

| Modelo | AP media | Desv. | ROC-AUC | Tiempo |
|---|---:|---:|---:|---:|
{wr}

![Walk-forward](../evidencia/figuras/v2/01_validacion_walk_forward.png)

### Benchmark histórico reutilizado

| Modelo | AUC-PR | Precisión | Recall | F1 | Costo | Alertas/100k |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM V2 | {f(tab["auc_pr"])} | {pct(tab["precision"])} | {pct(tab["recall"])} | {f(tab["f1"])} | {money(tab["costo_q"])} | {f(tab["alertas_por_100k"], 0)} |
| Ensamble calibrado | {f(ens["auc_pr"])} | {pct(ens["precision"])} | {pct(ens["recall"])} | {f(ens["f1"])} | {money(ens["costo_q"])} | {f(ens["alertas_por_100k"], 0)} |

![Curvas PR](../evidencia/figuras/v2/02_curvas_pr_v2.png)

El ensamble combina el score tabular con agregados causales mediante regresión logística. No se denomina A+B-GRU, porque no usa predicciones out-of-fold de una GRU V2. Esta distinción evita presentar como realizado un experimento pendiente.

La comparación reproducible completa con A, B y C de V1 se encuentra en [`artefactos/v2/comparacion_v1_v2.csv`](../artefactos/v2/comparacion_v1_v2.csv). Frente a A-V1, LightGBM V2 gana 0.0251 de AUC-PR y reduce Q115,500 de costo (1.86%), pero pierde 0.0188 de recall; por eso la mejora no se presenta como dominancia en todas las métricas.

## Economía, calibración y capacidad

El umbral minimiza $C(\tau)=4200FN(\tau)+180FP(\tau)$ en un holdout cronológico. Mejor F1 no garantiza menor costo: perder recall puede ser más grave que reducir falsos positivos. Brier y Expected Calibration Error evalúan si la escala del score representa riesgo de manera razonable; calibración no necesariamente cambia AUC-PR, que depende del orden.

![Calibración](../evidencia/figuras/v2/04_calibracion_v2.png)

`Precision@K` y `Recall@K` traducen ranking a capacidad de revisión para 0.1%, 0.5%, 1% y 2% de transacciones. También se guardan resultados por producto, dispositivo y cuartil de monto, junto con deriva de prevalencia, monto y ausencia. En operación se debe inspeccionar especialmente falsos negativos de alto monto y estabilidad del umbral.

## Reproducción y estructura

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r configuracion/v2/requirements-v2.txt
python -m pip install -r configuracion/v2/requirements-docs-v2.txt
python codigo/download_data.py
python -u codigo/proyecto1_v2_pipeline.py
python codigo/postprocess_v2.py
python codigo/compare_versions.py
python codigo/deliverables_v2.py
python codigo/crear_ficha_repositorio_v2.py
python codigo/audit_project1_v2.py
```

```text
.github/                 README visible en GitHub
artefactos/v1|v2/        evidencia congelada y fuente única
codigo/v1/               scripts originales preservados
codigo/                  pipeline y constructores V2
configuracion/v2/        dependencias exactas y guía
datos/raw/               CSV locales ignorados por Git
datos/processed/v2/      auditorías y selección
entregables/             notebook, informe, slides y ficha
evidencia/figuras/v1|v2/ gráficos separados por versión
legal/                   licencia
```

## Limitaciones, ética y referencias

- El benchmark reutilizado no es test ciego.
- La identidad proxy puede colisionar o fragmentar usuarios.
- Las columnas anonimizadas limitan interpretación semántica.
- Correlación e información mutua no demuestran causalidad.
- El presupuesto CatBoost es acotado y el costo es académico.
- El prototipo prioriza revisión humana; no debe rechazar operaciones ni atribuir culpabilidad.

### Model Card

**Uso previsto:** priorización académica de alertas. **No previsto:** decisión financiera automática. **Salida:** score y alerta según umbral. **Monitoreo:** AP, recall, costo, Brier/ECE, alertas/100k, deriva y segmentos. Antes de producción se requieren datos recientes, privacidad, seguridad, explicabilidad, análisis de sesgo y costos reales.

### Referencias APA 7

Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling*. arXiv. https://doi.org/10.48550/arXiv.1803.01271

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*. https://papers.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems, 31*. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

Scikit-learn developers. (2026). *Probability calibration*. https://scikit-learn.org/stable/modules/calibration.html

### Declaración de IA

Se utilizó IA para estructurar código, redacción, visualización y controles de consistencia. Los autores ejecutan el experimento, verifican cifras y asumen responsabilidad por interpretación, seguridad y defensa. La IA no se considera fuente académica.
"""
    path = ROOT / ".github/README.md"
    path.write_text(text, encoding="utf-8", newline="\n")


def notebook():
    hero = (
        styles()
        + """<div class="hero"><div class="chips"><span>DEEP LEARNING</span><span>PROYECTO 1</span><span>VERSIÓN 2</span><span>SECCIÓN 30</span></div><h1>Monitoreo transaccional y detección de fraude</h1><p style="font-size:20px">Variables causales, validación temporal, boosting, PCA y costo</p><div class="grid"><div class="card"><b>Universidad</b><br>Universidad del Valle de Guatemala</div><div class="card"><b>Curso</b><br>Deep Learning y Sistemas Inteligentes</div><div class="card"><b>Docente</b><br>Kevin Recinos</div><div class="card"><b>Integrantes</b><br>Wilson Calderón · 22018<br>Pablo Barillas · 22193</div><div class="card"><b>Grupo</b><br>Grupo 1 · Sección 30</div><div class="card"><b>Período</b><br>Semestre II · 2026</div></div></div><div class="callout"><b>Propósito.</b> Mejorar datos, protocolo y decisión antes de aumentar complejidad. V1 permanece congelada y los resultados negativos se conservan.</div>"""
    )
    sections = [
        (
            "1 · Pregunta y protocolo",
            "La pregunta es si el historial aporta señal incremental frente a una representación tabular fuerte. La V1 mostró una caída de AUC-PR de solo 0.0017 al permutar el orden; por eso V2 prioriza variables, identidad y validación. El costo es $$C(\\tau)=4200FN(\\tau)+180FP(\\tau).$$ El último 15% es benchmark histórico reutilizado, no test ciego.",
        ),
        (
            "2 · Datos y causalidad",
            "`TransactionID` solo une; `TransactionDT` solo ordena. Tarjetas y direcciones son códigos, no magnitudes. Cada agregado satisface $$x_t^{hist}=f(\\{x_j:t_j<t\\}),$$ e incluye conteos previos, monto histórico, razón de monto, tiempo anterior y actividad 1/6/24/72 h.",
        ),
        (
            "3 · Correlación, ruido y PCA",
            "Pearson, Spearman e información mutua miden formas distintas de asociación; ninguna demuestra causalidad. Constantes, ausencia extrema e IDs operativos se excluyen. PCA se ajusta por fold solo en numéricas elegibles: retener varianza no garantiza retener fraude.",
        ),
        (
            "4 · Walk-forward",
            "Tres ventanas entrenan con pasado y evalúan futuro inmediato. Se comparan LightGBM depurado, CatBoost con categorías nativas y LightGBM+PCA95. El presupuesto CatBoost es acotado y se reporta honestamente.",
        ),
        (
            "5 · Tamaño de muestra",
            "Se comparan 180k, 300k y todo el 70% disponible. Más datos no se declara mejor por definición: la deriva puede reducir utilidad de eventos antiguos.",
        ),
        (
            "6 · Calibración y ensamble",
            "El ensamble logístico combina score tabular y agregados causales. Reserva el final de validación para calibrar y escoger umbral. No se llama A+B-GRU porque no contiene predicciones OOF de una GRU V2.",
        ),
        (
            "7 · Resultados y capacidad",
            "AUC-PR mide ranking; recall, fraude recuperado; precisión, carga; costo, consecuencia del umbral. Precision@K y Recall@K conectan el modelo con límites de revisión.",
        ),
        (
            "8 · Conclusión",
            "Los datos preceden a la arquitectura. Correlación quita redundancia pero no decide sola; PCA se acepta solo si gana fuera de tiempo. Para una V3: cohorte nueva, predicciones OOF tabular+GRU, claves proxy, ventanas 3/8/16, Adam/AdamW y focal loss antes de TCN o atención.",
        ),
    ]
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(hero))
    nb.cells.append(
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport json, pandas as pd\nROOT=Path.cwd()\nif not (ROOT/'artefactos').exists(): ROOT=ROOT.parents[1]\nR=json.loads((ROOT/'artefactos/v2/resultados_v2.json').read_text(encoding='utf-8'))\npd.DataFrame(R['datos']['particiones']).T"
        )
    )
    code = [
        "pd.DataFrame(R['identidad_secuencial']).T",
        "pd.read_csv(ROOT/'datos/processed/v2/auditoria_variables.csv').sort_values('puntaje_relevancia',ascending=False).head(20)",
        "pd.read_csv(ROOT/'artefactos/v2/validacion_walk_forward.csv').pivot(index='fold',columns='modelo',values='auc_pr')",
        "pd.read_csv(ROOT/'artefactos/v2/ablacion_tamano_entrenamiento.csv')",
        "pd.DataFrame({'LightGBM':R['modelo_tabular_v2']['benchmark_historico'],'Ensamble':R['ensamble_v2']['benchmark_historico']}).T[['auc_pr','precision','recall','f1','costo_q','alertas_por_100k']]",
        "pd.DataFrame(R['metricas_top_k'])",
        "pd.DataFrame(R['intervalo_auc_pr_benchmark'],index=['valor']).T",
        "R['limitaciones']",
    ]
    for (title, body), snippet in zip(sections, code):
        nb.cells.append(
            nbf.v4.new_markdown_cell(
                f'<div class="section"><h2>{title}</h2></div>\n\n{body}'
            )
        )
        nb.cells.append(nbf.v4.new_code_cell(snippet))
    nb.cells.append(
        nbf.v4.new_markdown_cell(
            '<div class="callout warn"><b>Uso responsable.</b> Prototipo académico para priorizar revisión humana. No debe bloquear transacciones ni atribuir culpabilidad. Requiere privacidad, explicabilidad, seguridad, sesgo, monitoreo y costos reales antes de producción.</div>'
        )
    )
    nb.cells.append(
        nbf.v4.new_markdown_cell("""<div class="section"><h2>9 · Referencias APA 7 y declaración de IA</h2></div>

Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling*. arXiv. https://doi.org/10.48550/arXiv.1803.01271

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*. https://papers.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems, 31*. https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE, 10*(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

**Declaración de IA.** Se utilizó asistencia para estructurar código, redacción, visualización y auditoría. Los autores ejecutaron el experimento, verificaron las cifras y asumen responsabilidad por interpretación, seguridad y defensa.""")
    )
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    }
    out = ROOT / "entregables/cuaderno/Proyecto_1_Monitoreo_Transaccional_V2.ipynb"
    nbf.write(nb, out)


def slides():
    tab = R["modelo_tabular_v2"]["benchmark_historico"]
    ens = R["ensamble_v2"]["benchmark_historico"]
    data = [
        (
            "¿El historial mejora la detección?",
            "DECISIÓN",
            f'<div class="big">{f(tab["auc_pr"])}</div><p>AUC-PR LightGBM V2</p><div class="call">Primero datos y protocolo; V1 mostró señal de orden débil.</div>',
            "Aclarar benchmark reutilizado.",
        ),
        (
            "590,540 eventos, futuro fuera",
            "DATOS",
            '<div class="cols"><div><h3>434 columnas</h3><p>Integradas y auditadas.</p></div><div><h3>70 · 15 · 15</h3><p>Train, validación, benchmark.</p></div></div><p class="formula">xᵗʰⁱˢᵗ=f({xⱼ:tⱼ&lt;t})</p>',
            "TransactionID une; TransactionDT ordena.",
        ),
        (
            "La señal se audita",
            "CORRELACIÓN · PCA",
            f'<div class="big small">{len(R["seleccion_variables"]["numericas_seleccionadas"])}</div><p>variables numéricas retenidas</p><img src="../../evidencia/figuras/v2/03_relevancia_variables.png">',
            "Correlación no implica causalidad.",
        ),
        (
            "Tres futuros simulados",
            "WALK-FORWARD",
            '<img src="../../evidencia/figuras/v2/01_validacion_walk_forward.png"><div class="call">Ganador: '
            + html.escape(R["validacion_walk_forward"]["ganador"])
            + "</div>",
            "Comparar promedio y dispersión.",
        ),
        (
            "Ranking y costo difieren",
            "RESULTADOS",
            f'<div class="cols"><div class="model"><h3>LightGBM</h3><b>AP {f(tab["auc_pr"])}</b><p>Recall {pct(tab["recall"])}</p><p>{money(tab["costo_q"])}</p></div><div class="model accent"><h3>Ensamble</h3><b>AP {f(ens["auc_pr"])}</b><p>Recall {pct(ens["recall"])}</p><p>{money(ens["costo_q"])}</p></div></div>',
            "FN cuesta 23.3 veces FP.",
        ),
        (
            "El umbral es operativo",
            "CALIBRACIÓN",
            '<img src="../../evidencia/figuras/v2/04_calibracion_v2.png"><p>Precision@K y Recall@K conectan riesgo y capacidad.</p>',
            "Explicar Brier y ECE.",
        ),
        (
            "Lo que no sabemos",
            "LIMITACIONES",
            "<ul><li>Benchmark reutilizado.</li><li>Identidad proxy.</li><li>CatBoost acotado.</li><li>Costos académicos.</li><li>Señal secuencial V1 pequeña.</li></ul>",
            "Resultados negativos son evidencia.",
        ),
        (
            "Congelar V2; obtener datos nuevos",
            "RECOMENDACIÓN",
            '<div class="road"><span>cohorte nueva</span><span>OOF tabular+GRU</span><span>ventanas 3/8/16</span><span>calibrar</span><span>monitorear</span></div><div class="call">Solo revisión humana.</div>',
            "TCN/atención después de demostrar orden.",
        ),
    ]
    cards = []
    for i, (title, eye, body, notes) in enumerate(data, 1):
        cards.append(
            f'<section class="slide" data-index="{i}"><div class="eye">{eye}</div><h2>{title}</h2><div class="content">{body}</div><footer>Grupo 1 · Sección 30 <b>{i}/8</b></footer><aside>{notes}</aside></section>'
        )
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Proyecto 1 V2</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#081a2a;color:#eef7ff;font-family:Inter,'Segoe UI',sans-serif;overflow:hidden}}.slide{{display:none;width:100vw;height:100vh;padding:6vh 7vw;background:radial-gradient(circle at 90% 10%,#2a9d8f44,transparent 30%),linear-gradient(135deg,#102a43,#091b2b);position:relative}}.active{{display:block}}.eye{{color:#65d3c3;font-weight:800;letter-spacing:.16em;font-size:1.2vw}}h2{{font-size:4vw;margin:2vh 0 4vh}}p,li{{font-size:1.55vw;line-height:1.5}}.content{{height:68vh}}.big{{font-size:12vw;font-weight:900;color:#65d3c3;line-height:1}}.small{{font-size:7vw}}.call{{padding:1.3vw;margin-top:2vh;border-left:.5vw solid #2a9d8f;background:#ffffff10;border-radius:.8vw;font-size:1.5vw}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:3vw}}.model{{padding:2vw;background:#ffffff0d;border:1px solid #ffffff25;border-radius:1.3vw}}.model b{{font-size:3vw}}.accent{{border-color:#2a9d8f}}img{{display:block;max-width:82%;max-height:50vh;margin:auto;border-radius:1vw;background:white}}.formula{{font-size:3vw;text-align:center;color:#65d3c3}}.road{{display:flex;flex-wrap:wrap;gap:1vw}}.road span{{padding:1.4vw;border-radius:999px;background:#184e77;border:1px solid #65d3c3;font-size:1.4vw}}footer{{position:absolute;bottom:3vh;left:7vw;right:7vw;display:flex;justify-content:space-between;color:#9eb6c8}}aside{{display:none}}body.notes aside{{display:block;position:absolute;right:2vw;bottom:7vh;width:30vw;padding:1vw;background:white;color:#172033;border-radius:.7vw}}@page{{size:13.333in 7.5in;margin:0}}@media print{{html,body{{width:13.333in;height:7.5in;overflow:visible}}.slide{{display:block!important;width:13.333in;height:7.5in;page-break-after:always}}aside{{display:none!important}}}}</style></head><body>{"".join(cards)}<script>const s=[...document.querySelectorAll('.slide')];let i=0;function show(n){{i=Math.max(0,Math.min(7,n));s.forEach((x,j)=>x.classList.toggle('active',j===i))}}document.onkeydown=e=>{{if(['ArrowRight',' ','PageDown'].includes(e.key))show(i+1);if(['ArrowLeft','PageUp'].includes(e.key))show(i-1);if(e.key.toLowerCase()==='n')document.body.classList.toggle('notes')}};show(0)</script></body></html>"""
    (ROOT / "entregables/presentacion/presentacion_proyecto1_v2.html").write_text(
        page, encoding="utf-8", newline="\n"
    )


def report():
    tab = R["modelo_tabular_v2"]["benchmark_historico"]
    ens = R["ensamble_v2"]["benchmark_historico"]
    winner_tex = R["validacion_walk_forward"]["ganador"].replace("_", r"\_")
    tex = rf"""\documentclass[10pt]{{article}}
\usepackage[letterpaper,margin=1.6cm]{{geometry}}\usepackage{{graphicx,booktabs,xcolor,amsmath,hyperref}}
\definecolor{{navy}}{{HTML}}{{102A43}}\definecolor{{teal}}{{HTML}}{{2A9D8F}}\hypersetup{{colorlinks=true,urlcolor=teal}}\pagestyle{{plain}}\setlength{{\parindent}}{{0pt}}\setlength{{\parskip}}{{5pt}}
\begin{{document}}\begin{{titlepage}}\pagecolor{{navy}}\color{{white}}\raggedright\vspace*{{1cm}}{{\Large DEEP LEARNING Y SISTEMAS INTELIGENTES\par}}\vspace{{1cm}}{{\Huge\bfseries Proyecto 1\\[.25cm]Monitoreo transaccional\par}}\vspace{{.7cm}}{{\LARGE Variables causales, boosting y validación temporal\par}}\vfill{{\Large Universidad del Valle de Guatemala\\Kevin Recinos · Grupo 1 · Sección 30\\Semestre II 2026\par}}\vspace{{1cm}}{{\Large Wilson Alejandro Calderón Argueta · 22018\\Pablo Daniel Barillas Moreno · 22193\par}}\vfill\textbf{{Nota:}} el 15\% final es un benchmark histórico reutilizado, no test ciego.\end{{titlepage}}\nopagecolor
\section*{{Resumen ejecutivo}} Se analizaron {R["datos"]["filas"]:,} transacciones IEEE-CIS con prevalencia {pct(R["datos"]["prevalencia"])}. V2 preserva V1, amplía variables y audita ausencia, correlación, información mutua y redundancia. El ganador interno fue \textbf{{{winner_tex}}}. En benchmark histórico, LightGBM alcanzó AP {tab["auc_pr"]:.4f} y el ensamble calibrado {ens["auc_pr"]:.4f}. La decisión combina ranking, recall, costo y calibración.
\section{{Datos y causalidad}} Las tablas se unieron por TransactionID y ordenaron por TransactionDT. Ninguna de ambas se interpreta como predictor continuo. Los códigos de tarjeta/dirección/dispositivo son categorías o proxies. Para cada $t$, $x_t^{{hist}}=f(\{{x_j:t_j<t\}})$. Se calculan conteo y monto previos, razón de monto, tiempo anterior y actividad de 1/6/24/72 horas. El protocolo usa 70\% entrenamiento, 15\% validación y 15\% benchmark reutilizado; tres ventanas internas simulan reentrenamientos.

La clave tarjeta--dirección produce {R["identidad_secuencial"]["tarjeta_direccion"]["entidades"]:,} entidades y una mediana de {R["identidad_secuencial"]["tarjeta_direccion"]["mediana_transacciones"]:.1f} transacciones. {R["identidad_secuencial"]["tarjeta_direccion"]["porcentaje_con_8"]:.1f}\% de los eventos pertenece a proxies con al menos ocho observaciones y {R["identidad_secuencial"]["tarjeta_direccion"]["porcentaje_con_16"]:.1f}\% a proxies con al menos dieciséis. Se compararon además tarjeta--dirección--correo y tarjeta--dispositivo--producto. Esta cobertura establece si una ventana secuencial nominal tiene antecedentes reales; una clave demasiado específica fragmenta y una demasiado amplia mezcla usuarios.

\section{{Selección, correlación y PCA}} Pearson captura linealidad, Spearman monotonía e información mutua dependencia no lineal; ninguna implica causalidad. Se excluyen constantes, alta ausencia e IDs. Pares con $|\rho_s|\ge {R["seleccion_variables"]["umbral_redundancia"]}$ se depuran por relevancia. Quedan {len(R["seleccion_variables"]["numericas_seleccionadas"])} numéricas y {len(R["seleccion_variables"]["categoricas_seleccionadas"])} categóricas. PCA se ajusta por fold tras imputar/escalar solo numéricas elegibles. Explicar 95\% de varianza no garantiza separar fraude.

La selección se aprendió únicamente en el primer 55\% temporal. Esto impide que el patrón de ausencia o la distribución de un período posterior determine qué variable se conserva. Variables con baja asociación marginal no se eliminan automáticamente: pueden resultar informativas por interacción. El umbral alto de redundancia solo retira sustitutos casi monotónicos y registra para cada exclusión la variable retenida y su coeficiente.
\begin{{figure}}[h]\centering\includegraphics[width=.78\linewidth]{{../../evidencia/figuras/v2/03_relevancia_variables.png}}\caption{{Señal estimada en el pasado inicial.}}\end{{figure}}
\newpage
\section{{Modelos y validación}} LightGBM usa variables depuradas; CatBoost conserva categorías nativas con 240 iteraciones; LightGBM+PCA mide compresión. El ensamble logístico combina score tabular y agregados causales, y reserva el final de validación para calibración y umbral. No se denomina A+B-GRU porque no usa predicciones OOF GRU V2.

La media walk-forward fue 0.4727 para LightGBM depurado, 0.4580 para CatBoost y 0.4494 para PCA95. Sus tiempos medios fueron aproximadamente 24.8, 398.6 y 33.4 segundos, respectivamente. Por tanto, PCA redujo dimensionalidad pero perdió señal supervisada, y CatBoost no compensó su costo computacional bajo el presupuesto documentado. Esto no demuestra inferioridad universal de ambos enfoques; delimita su rendimiento bajo este protocolo.

La ablación de muestra obtuvo AP de validación 0.4661 con 180,000 filas, 0.4682 con 300,000 y 0.4664 con las {R["datos"]["particiones"]["train"]["n"]:,} filas. La ventaja pequeña de 300k sobre el conjunto completo es compatible con deriva o menor utilidad de observaciones antiguas. Se mantiene el entrenamiento completo como modelo principal para representar toda la diversidad, pero el hallazgo se conserva como evidencia negativa.
\begin{{figure}}[h]\centering\includegraphics[width=.82\linewidth]{{../../evidencia/figuras/v2/01_validacion_walk_forward.png}}\caption{{Comparación walk-forward.}}\end{{figure}}
\newpage
\section{{Resultados y economía}}\begin{{center}}\begin{{tabular}}{{lrrrrr}}\toprule Modelo&AUC-PR&Precisión&Recall&F1&Costo\\\midrule LightGBM V2&{tab["auc_pr"]:.3f}&{tab["precision"]:.3f}&{tab["recall"]:.3f}&{tab["f1"]:.3f}&Q{tab["costo_q"]:,.0f}\\Ensamble&{ens["auc_pr"]:.3f}&{ens["precision"]:.3f}&{ens["recall"]:.3f}&{ens["f1"]:.3f}&Q{ens["costo_q"]:,.0f}\\\bottomrule\end{{tabular}}\end{{center}} El costo $4200FN+180FP$ hace que recall tenga gran peso. Mejor F1 no implica menor costo. Para LightGBM V2, el intervalo AP por bloques es [{R["intervalo_auc_pr_benchmark"]["li95"]:.4f}, {R["intervalo_auc_pr_benchmark"]["ls95"]:.4f}]. Precision@K/Recall@K traducen ranking a capacidad.

Frente a V1, LightGBM V2 aumenta AP de 0.4285 a {tab["auc_pr"]:.4f} y reduce el costo de Q6,193,620 a Q{tab["costo_q"]:,.0f}. El recall baja de 0.7217 a {tab["recall"]:.4f}; la reducción de falsos positivos compensa económicamente bajo los costos fijados, pero un operador que privilegie fraude recuperado podría elegir otro umbral. La decisión debe presentarse con esta tensión, no solamente con la mejora de AP.

El ensamble tiene AP {ens["auc_pr"]:.4f} y costo Q{ens["costo_q"]:,.0f}, peores que LightGBM. Se rechaza como candidato final y se conserva el resultado: agregar señales causales a un metamodelo no garantiza información incremental si el árbol ya explota esas variables.
\begin{{figure}}[h]\centering\includegraphics[width=.65\linewidth]{{../../evidencia/figuras/v2/02_curvas_pr_v2.png}}\end{{figure}}
\newpage
\section{{Calibración, operación y errores}} Brier y ECE evalúan calidad probabilística. Se reportan producto, dispositivo, monto, deriva, alertas/100k y falsos negativos. Calibrar no necesariamente altera AUC-PR, pero estabiliza decisiones de riesgo. Antes de producción: cohorte nueva, privacidad, explicabilidad, sesgo, seguridad, costos reales y monitoreo.

En validación, el Brier tabular fue {R["ensamble_v2"]["calibracion"]["brier_tabular_validacion"]:.4f}; el del ensamble calibrado, {R["ensamble_v2"]["calibracion"]["brier_ensamble_validacion"]:.4f}. ECE pasó de {R["ensamble_v2"]["calibracion"]["ece_tabular_validacion"]:.4f} a {R["ensamble_v2"]["calibracion"]["ece_ensamble_validacion"]:.4f}. Estas cifras se interpretan junto con AP: una escala probabilística mejor no rescata un ranking inferior.

La tasa de alertas LightGBM es {tab["alertas_por_100k"]:.0f} por 100,000 transacciones en el umbral económico. Se guardan Precision@K y Recall@K para capacidades de 0.1\%, 0.5\%, 1\% y 2\%, además de desglose por ProductCD, DeviceType y cuartil de monto. La operación debe vigilar cambios de prevalencia, ausencia, monto, costo y volumen, pues un umbral fijo puede degradarse aun sin cambios de arquitectura.
\begin{{figure}}[h]\centering\includegraphics[width=.58\linewidth]{{../../evidencia/figuras/v2/04_calibracion_v2.png}}\end{{figure}}
\newpage
\section{{Conclusión y hoja de ruta}} La evidencia respalda mejorar datos antes de arquitectura. Correlación quita redundancia; PCA se acepta solo si gana temporalmente. V1 mostró orden débil. V3 debe usar cohorte nueva, OOF tabular+GRU, proxies alternativos, ventanas 3/8/16, Adam/AdamW, 6/12/20 épocas y focal loss; TCN/atención solo si la falsificación revela señal.

La recomendación actual es LightGBM V2 con representación depurada, acompañado por calibración y monitoreo, no el ensamble. Antes de congelar una siguiente versión se debe definir una cohorte nunca observada, producir predicciones out-of-fold de A y B, comparar stacking logístico, calibrar en un bloque separado y fijar el umbral antes de abrir el nuevo test. Una arquitectura secuencial mayor solo se justifica si permutar la historia produce una caída clara y repetible.
\section*{{Limitaciones y ética}} Benchmark reutilizado; identidad aproximada; anonimización; CatBoost acotado; costo académico. Uso exclusivo para priorizar revisión humana, no rechazo automático ni atribución de culpa.
\section*{{Referencias}}\small Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., \& Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. \textit{{NeurIPS, 30}}.\\Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., \& Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. \textit{{NeurIPS, 31}}.\\Saito, T., \& Rehmsmeier, M. (2015). Precision-recall plots for imbalanced data. \textit{{PLOS ONE, 10}}(3), e0118432.\\Bai, S., Kolter, J. Z., \& Koltun, V. (2018). Sequence modeling. arXiv.\\Scikit-learn developers. (2026). Probability calibration. \url{{https://scikit-learn.org/stable/modules/calibration.html}}
\section*{{Declaración de IA}} IA apoyó código, redacción y auditoría. Los autores ejecutan, verifican y asumen responsabilidad.
\end{{document}}"""
    out = ROOT / "entregables/informe/informe_proyecto1_v2.tex"
    out.write_text(tex, encoding="utf-8", newline="\n")
    exe = shutil.which("pdflatex")
    if exe:
        for _ in range(2):
            subprocess.run(
                [
                    exe,
                    "-interaction=nonstopmode",
                    "-halt-on-error",
                    "-output-directory",
                    str(out.parent),
                    str(out),
                ],
                cwd=out.parent,
                capture_output=True,
            )


def main():
    readme()
    notebook()
    slides()
    report()
    print("Entregables V2 generados desde artefactos/v2/resultados_v2.json")


if __name__ == "__main__":
    main()
