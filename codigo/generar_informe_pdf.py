from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "entregables" / "informe" / "informe.pdf"
ART = ROOT / "artefactos"
FIG = ROOT / "evidencia" / "figuras"
RESULTS = json.loads((ART / "resultados.json").read_text(encoding="utf-8"))
ERRORS = pd.read_csv(ART / "patrones_error.csv")

NAVY = colors.HexColor("#184E77")
TEAL = colors.HexColor("#2A9D8F")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#526576")
SOFT = colors.HexColor("#EDF5FB")
WARN = colors.HexColor("#FFF3EA")


def register_fonts() -> tuple[str, str]:
    regular = Path(r"C:\Windows\Fonts\arial.ttf")
    bold = Path(r"C:\Windows\Fonts\arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ProjectSans", regular))
        pdfmetrics.registerFont(TTFont("ProjectSans-Bold", bold))
        return "ProjectSans", "ProjectSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def money(value: float) -> str:
    return f"Q{value:,.0f}"


def footer(canvas, doc):
    canvas.saveState()
    width, _ = letter
    canvas.setStrokeColor(colors.HexColor("#CAD8E3"))
    canvas.line(0.58 * inch, 0.43 * inch, width - 0.58 * inch, 0.43 * inch)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.58 * inch, 0.25 * inch, "UVG - Deep Learning 2026 - Grupo 1")
    canvas.drawRightString(width - 0.58 * inch, 0.25 * inch, f"{doc.page}")
    canvas.restoreState()


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=0.58 * inch,
            rightMargin=0.58 * inch,
            topMargin=0.52 * inch,
            bottomMargin=0.55 * inch,
            title="Proyecto 1 - Monitoreo transaccional",
            author="Wilson Alejandro Calderón Argueta; Pablo Daniel Barillas Moreno",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=footer))


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleP", fontName=FONT_BOLD, fontSize=19, leading=21, textColor=NAVY, spaceAfter=3))
styles.add(ParagraphStyle(name="SubtitleP", fontName=FONT, fontSize=10.2, leading=12, textColor=MUTED, spaceAfter=7))
styles.add(ParagraphStyle(name="H1P", fontName=FONT_BOLD, fontSize=12.2, leading=14, textColor=NAVY, spaceBefore=3, spaceAfter=4))
styles.add(ParagraphStyle(name="H2P", fontName=FONT_BOLD, fontSize=9.5, leading=11.5, textColor=TEAL, spaceBefore=2, spaceAfter=2))
styles.add(ParagraphStyle(name="BodyP", fontName=FONT, fontSize=8.45, leading=10.6, textColor=INK, spaceAfter=3.5))
styles.add(ParagraphStyle(name="SmallP", fontName=FONT, fontSize=7.3, leading=8.7, textColor=INK))
styles.add(ParagraphStyle(name="TinyP", fontName=FONT, fontSize=6.45, leading=7.6, textColor=INK))
styles.add(ParagraphStyle(name="CalloutP", fontName=FONT, fontSize=8.7, leading=11, textColor=INK, borderColor=TEAL, borderWidth=0, borderPadding=7, backColor=SOFT, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="WarnP", fontName=FONT, fontSize=8.5, leading=10.7, textColor=INK, borderPadding=7, backColor=WARN, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="CenterP", fontName=FONT, fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=MUTED))


def p(text: str, style: str = "BodyP") -> Paragraph:
    return Paragraph(text, styles[style])


def table(data, widths, font_size=7.3, header=True) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD8E3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(data)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F5F8FA")))
    t.setStyle(TableStyle(commands))
    return t


def figure(name: str, width: float, height: float) -> KeepTogether:
    image = Image(str(FIG / name), width=width, height=height, kind="proportional")
    image.hAlign = "CENTER"
    return KeepTogether([image])


def build_story():
    d = RESULTS["dataset"]
    s = RESULTS["splits"]
    val = RESULTS["validation"]
    test = RESULTS["test"]
    fals = RESULTS["falsification"]
    hyp = RESULTS["hypothesis"]
    econ = RESULTS["economics"]
    candidate = RESULTS["candidate"]
    top_fn = ERRORS.loc[ERRORS["error"].eq("FN")].iloc[0]
    top_fp = ERRORS.loc[ERRORS["error"].eq("FP")].iloc[0]
    fn_total = int(test[candidate]["fn"])
    fn_share = int(top_fn["n"]) / fn_total

    story = [
        p("Proyecto 1: Monitoreo transaccional", "TitleP"),
        p("Detectar lo que el orden revela", "SubtitleP"),
        p("Wilson Alejandro Calderón Argueta (22018) - Pablo Daniel Barillas Moreno (22193)<br/>Universidad del Valle de Guatemala - Deep Learning y Sistemas Inteligentes - Kevin Recinos", "SmallP"),
        Spacer(1, 5),
        p("Resumen ejecutivo", "H1P"),
        p(
            f"Se analizaron <b>{d['rows']:,}</b> transacciones IEEE-CIS, con {d['frauds']:,} fraudes ({d['fraud_rate']:.3%}). "
            "La comparación controlada enfrentó A (agregados + HistGradientBoosting), B (ocho eventos ordenados + GRU) y C (GRU + agregados). "
            f"El candidato congelado fue <b>{candidate}</b>. B alcanzó AUC-PR {test['B']['auc_pr']:.3f}; al permutar la historia obtuvo {fals['permutation_mean_auc_pr']:.3f}. "
            "La caída de 0.002 no demuestra valor material del orden. Recomendamos conservar A para priorizar revisión y no migrar todavía a secuencias.",
            "CalloutP",
        ),
        p("1. Integridad de datos y protocolo temporal", "H1P"),
        p(
            "Ruta B con IEEE-CIS Fraud Detection (Vesta Corporation). TransactionDT define el orden. La entidad se aproxima con "
            "card1+card2+card3+card5+addr1; puede mezclar o fragmentar clientes. Cada objetivo usa su transacción actual y hasta siete antecedentes, nunca eventos futuros. "
            "Imputación, escalado y vocabularios se ajustan exclusivamente con entrenamiento. El submuestreo computacional conserva todos los positivos y ocurre solo dentro del 70% inicial."
        ),
        table([
            ["Partición", "n", "Tasa de fraude"],
            ["Entrenamiento poblacional", f"{s['train_population']['n']:,}", f"{s['train_population']['fraud_rate']:.3%}"],
            ["Validación", f"{s['validation']['n']:,}", f"{s['validation']['fraud_rate']:.3%}"],
            ["Prueba final", f"{s['test']['n']:,}", f"{s['test']['fraud_rate']:.3%}"],
        ], [2.4 * inch, 1.4 * inch, 1.5 * inch]),
        Spacer(1, 5),
        figure("01_integridad_temporal.png", 6.4 * inch, 2.6 * inch),
        PageBreak(),
        p("2. Núcleo común A-B y apuesta C", "H1P"),
        table([
            ["Pieza", "Entrada y modelo", "Control / propósito"],
            ["A - sin orden", p("Agregados invariantes a la permutación + HistGradientBoosting", "SmallP"), p("Representa el sistema competitivo sin leer la secuencia", "SmallP")],
            ["B - secuencial", p("10 variables numéricas, 6 categóricas, embeddings + GRU(32)", "SmallP"), p("Puede explotar transiciones y dependencia temporal", "SmallP")],
            ["C - apuesta", p("Estado final GRU concatenado con agregados estandarizados", "SmallP"), p("Prueba si la información global complementa la secuencia", "SmallP")],
        ], [1.2 * inch, 3.0 * inch, 2.55 * inch]),
        Spacer(1, 5),
        p("Hipótesis previa", "H2P"),
        p(hyp["statement"] + f" Resultado en validación: Delta AUC-PR {hyp['ap_gain']:+.3f}; reducción de costo {hyp['cost_reduction']:.1%}. <b>No cumplió ambos criterios.</b>", "WarnP"),
        p("3. Comparación común en prueba cronológica", "H1P"),
        table([
            ["Modelo", "AUC-PR", "Precisión", "Recall", "F1", "Costo"],
            *[[m, f"{test[m]['auc_pr']:.3f}", f"{test[m]['precision']:.3f}", f"{test[m]['recall']:.3f}", f"{test[m]['f1']:.3f}", money(test[m]['cost_q'])] for m in ["A", "B", "C"]],
        ], [0.72 * inch, 0.78 * inch, 0.88 * inch, 0.72 * inch, 0.68 * inch, 1.24 * inch]),
        Spacer(1, 4),
        p("AUC-PR es la métrica principal por el desbalance. Precisión, recall y F1 se calculan con el umbral económico elegido solo en validación; exactitud no interviene.", "SmallP"),
        figure("02_curvas_precision_recall.png", 6.25 * inch, 3.65 * inch),
        PageBreak(),
        p("4. Valor del orden: intentos de refutación", "H1P"),
        p(
            f"Permutación controlada: se barajó cinco veces solo la historia válida y se mantuvo la transacción objetivo al final. B obtuvo AUC-PR {fals['original_B']['auc_pr']:.3f} original y "
            f"{fals['permutation_mean_auc_pr']:.3f} +/- {fals['permutation_std_auc_pr']:.3f} permutada. La caída fue {fals['order_auc_pr_drop']:.3f}, inferior al criterio previo de 0.01. "
            f"Segunda prueba: al conservar solo los tres eventos más recientes, AUC-PR fue {fals['truncated_to_3']['auc_pr']:.3f}. <b>Conclusión honesta: este experimento no demuestra aporte material del orden.</b>",
            "CalloutP",
        ),
        figure("03_falsificaciones_orden.png", 6.05 * inch, 3.05 * inch),
        p("5. Umbral y decisión económica", "H1P"),
        p(
            f"Se minimizó C(tau)=Q4,200*FN+Q180*FP en validación. En prueba, A cuesta {money(test['A']['cost_q'])} ({money(econ['A']['cost_per_100k_q'])} por 100 mil decisiones). "
            f"Con 1.4 millones de tarjetas y 12 transacciones mensuales, A escala a {money(econ['A']['monthly_cost_q'])}/mes. Frente a A, B perdería "
            f"{money(abs(econ['B']['monthly_savings_vs_A_q']))}/mes y C {money(abs(econ['C']['monthly_savings_vs_A_q']))}/mes. Es un escenario académico; prevalencia, volumen y costos reales deben sustituir estos supuestos.",
            "WarnP",
        ),
        figure("04_curva_costo_umbral.png", 5.85 * inch, 2.45 * inch),
        PageBreak(),
        p("6. Recomendación, patrón de error y límites", "H1P"),
        p(
            "<b>Conservar A y usar el puntaje para priorizar revisión, no para bloquear automáticamente.</b> No hay evidencia suficiente para reemplazar el motor agregado con B o C. "
            f"Patrón concreto: {int(top_fn['n'])} de {fn_total} falsos negativos de A ({fn_share:.1%}) pertenecen a ProductCD={top_fn['ProductCD']} y al intervalo de monto {top_fn['amount_band']}. "
            "La concentración es descriptiva, no causal. La recomendación cambiaría con una identidad de tarjeta confiable, una cohorte posterior donde B/C reduzcan costo y una capacidad operativa capaz de absorber alertas.",
            "CalloutP",
        ),
        p("Límites principales", "H2P"),
        p("Identidad aproximada; atributos anonimizados; solo 182 días; ventana máxima de ocho eventos; costos transferidos; sin piloto humano, latencia, deriva, equidad ni calibración externa. El benchmark no debe interpretarse como causalidad ni como autorización para acusar o bloquear clientes."),
        p("Patrones de error observados", "H2P"),
        table([
            ["Tipo", "Segmento descriptivo", "n", "Lectura operativa"],
            ["FN", f"ProductCD={top_fn['ProductCD']}; monto {top_fn['amount_band']}", f"{int(top_fn['n']):,}", f"{fn_share:.1%} de los FN del candidato"],
            ["FP", f"ProductCD={top_fp['ProductCD']}; monto {top_fp['amount_band']}", f"{int(top_fp['n']):,}", "Mayor presión sobre revisión manual"],
        ], [0.55 * inch, 2.35 * inch, 0.55 * inch, 2.75 * inch]),
        Spacer(1, 6),
        p("Uso de IA", "H2P"),
        p("Se usó IA para estructurar código, revisar consistencia, apoyar redacción y diseño. Los autores verificaron datos, partición, construcción causal de ventanas, métricas, falsificaciones, umbral y archivos finales. La IA no se considera fuente académica.", "SmallP"),
        PageBreak(),
        p("Matriz de evidencias", "H1P"),
        table([
            ["Evidencia", "Figura o tabla", "Conclusión", "Limitación"],
            [p("Integridad de datos", "TinyP"), p("Fig. 1 y tabla de particiones", "TinyP"), p("70/15/15 cronológico; ajustes train-only", "TinyP"), p("Entidad aproximada", "TinyP")],
            [p("Comparación A-B", "TinyP"), p("Fig. 2 y tabla A/B/C", "TinyP"), p("A supera a B en AUC-PR y costo", "TinyP"), p("Historial máximo de 8 eventos", "TinyP")],
            [p("Valor del orden", "TinyP"), p("Fig. 3", "TinyP"), p("Permutación y recorte no causan caída material", "TinyP"), p("Proxy de identidad puede diluir señal", "TinyP")],
            [p("Apuesta C", "TinyP"), p("Hipótesis y tabla de validación", "TinyP"), p("C no cumple DeltaAP ni reducción de costo", "TinyP"), p("Mayor complejidad y una semilla", "TinyP")],
            [p("Economía", "TinyP"), p("Fig. 4 y comparación mensual", "TinyP"), p("A minimiza el costo bajo Q4,200/Q180", "TinyP"), p("Volumen, prevalencia y costos asumidos", "TinyP")],
            [p("Recomendación", "TinyP"), p("Sección 6", "TinyP"), p("Conservar A; secuencias quedan en investigación", "TinyP"), p("Falta piloto productivo", "TinyP")],
        ], [1.2 * inch, 1.45 * inch, 2.65 * inch, 1.45 * inch], font_size=6.5),
        Spacer(1, 10),
        p("Referencias", "H1P"),
        p("Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. arXiv:1406.1078.", "SmallP"),
        p("IEEE Computational Intelligence Society. (2019). IEEE-CIS Fraud Detection [Data set]. Kaggle.", "SmallP"),
        p("Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.", "SmallP"),
        p("Scikit-learn Developers. Common pitfalls and recommended practices.", "SmallP"),
        Spacer(1, 12),
        p("Decisión para el comité", "H1P"),
        p("El orden es una hipótesis valiosa, pero con esta identidad aproximada y este horizonte no genera valor incremental demostrable. Conservar A, recopilar una clave de tarjeta confiable y repetir la prueba en una cohorte futura.", "CalloutP"),
    ]
    return story


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDoc(str(OUT))
    doc.build(build_story())
    print(OUT)


if __name__ == "__main__":
    main()
