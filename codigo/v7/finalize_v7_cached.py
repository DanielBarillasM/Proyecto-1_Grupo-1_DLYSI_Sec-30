"""Reintegra A5 con el control V6 usando puntajes base ya entrenados.

Este paso no reabre ni reentrena A0--A4: corrige la omisión de A_V6_control en
el stacking predeclarado y repite desde meta-fit hasta evaluación/benchmark.
"""

from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proyecto1_v7_pipeline import (  # noqa: E402
    ROOT, ART, FIG, PROCESSED, RAW, V6_ART, ConfigV7, HYPOTHESIS_C,
    apply_calibrator, choose_threshold, fit_calibrator, fit_meta, logit,
    metric_set, plot_pr, ready, resolve_raw, validation_bounds, write_json,
)


def read_context() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recupera contexto causal/operativo sin mantener el CSV completo en RAM."""
    tx_ids: list[np.ndarray] = []
    amounts: list[np.ndarray] = []
    tx_missing: list[np.ndarray] = []
    for chunk in pd.read_csv(resolve_raw("train_transaction.csv"), chunksize=50_000, low_memory=True):
        tx_ids.append(chunk["TransactionID"].to_numpy(np.int64))
        amounts.append(chunk["TransactionAmt"].fillna(0).to_numpy(np.float32))
        tx_missing.append(chunk.drop(columns=["isFraud"], errors="ignore").isna().sum(axis=1).to_numpy(np.int16))
    ids = np.concatenate(tx_ids)
    amount = np.concatenate(amounts)
    missing = np.concatenate(tx_missing).astype(np.int16)
    identity_missing: dict[int, int] = {}
    identity_present: dict[int, float] = {}
    identity_width = 0
    for chunk in pd.read_csv(resolve_raw("train_identity.csv"), chunksize=50_000, low_memory=True):
        identity_width = len(chunk.columns) - 1
        counts = chunk.drop(columns=["TransactionID"]).isna().sum(axis=1).to_numpy(np.int16)
        present = chunk[[c for c in ("DeviceType", "id_01") if c in chunk]].notna().any(axis=1).to_numpy(float)
        for transaction_id, count, flag in zip(chunk["TransactionID"].to_numpy(np.int64), counts, present):
            identity_missing[int(transaction_id)] = int(count)
            identity_present[int(transaction_id)] = float(flag)
    id_missing = np.fromiter((identity_missing.get(int(t), identity_width) for t in ids), dtype=np.int16, count=len(ids))
    id_present = np.fromiter((identity_present.get(int(t), 0.0) for t in ids), dtype=np.float32, count=len(ids))
    return ids, amount, (missing + id_missing).astype(np.float32), id_present


def main() -> None:
    cfg = ConfigV7()
    result_path = ART / "resultados_v7.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    v6_val = pd.read_csv(V6_ART / "predicciones_validacion_v6.csv")
    v6_bench = pd.read_csv(V6_ART / "predicciones_benchmark_v6.csv")
    y_val = v6_val["y"].to_numpy(np.int8)
    y_bench = v6_bench["y"].to_numpy(np.int8)
    val_indices = v6_val["indice"].to_numpy(np.int64)
    bench_indices = v6_bench["indice"].to_numpy(np.int64)
    bounds = validation_bounds(len(y_val))

    cache = np.load(ART / "puntajes_crudos_candidatos_a_v7.npz")
    base_names = sorted(k[5:] for k in cache.files if k.startswith("val__") and k[5:] != "A5_ensamble_tabular")
    a_val = {name: cache[f"val__{name}"] for name in base_names}
    a_bench = {name: cache[f"bench__{name}"] for name in base_names}
    stack_inputs = [*base_names, "A_V6_control"]
    stack_val = np.column_stack([*[logit(a_val[n]).ravel() for n in base_names], logit(v6_val["score_A"].to_numpy(float)).ravel()])
    stack_bench = np.column_stack([*[logit(a_bench[n]).ravel() for n in base_names], logit(v6_bench["score_A"].to_numpy(float)).ravel()])
    a5 = fit_meta(stack_val, y_val, bounds["meta_fit"], cfg.seed)
    a_val["A5_ensamble_tabular"] = a5.predict_proba(stack_val)[:, 1]
    a_bench["A5_ensamble_tabular"] = a5.predict_proba(stack_bench)[:, 1]
    joblib.dump({"model": a5, "inputs": stack_inputs}, ART / "modelo_A5_ensamble_tabular_v7.joblib")
    np.savez_compressed(ART / "puntajes_crudos_candidatos_a_v7.npz", **{f"val__{k}": v for k, v in a_val.items()}, **{f"bench__{k}": v for k, v in a_bench.items()})

    select_ap = {name: float(average_precision_score(y_val[bounds["model_select"]], score[bounds["model_select"]])) for name, score in a_val.items()}
    select_windows = np.array_split(bounds["model_select"], 2)
    window_ap = {name: [float(average_precision_score(y_val[w], score[w])) for w in select_windows] for name, score in a_val.items()}
    best_a_name = max(select_ap, key=select_ap.get)
    a_selected_val, a_selected_bench = a_val[best_a_name], a_bench[best_a_name]

    ids, amount, missing, identity = read_context()
    assert np.array_equal(ids[val_indices], v6_val["TransactionID"].to_numpy(np.int64))
    assert np.array_equal(ids[bench_indices], v6_bench["TransactionID"].to_numpy(np.int64))
    lengths = np.load(PROCESSED.parent / "v6" / "esquema_indices_secuencia_v6.npz")["lengths"]
    quality_val = np.clip(np.log1p(lengths[val_indices]) / np.log1p(32), 0, 1) * (.6 + .4 * identity[val_indices])
    quality_bench = np.clip(np.log1p(lengths[bench_indices]) / np.log1p(32), 0, 1) * (.6 + .4 * identity[bench_indices])
    b_val, b_bench = v6_val["score_B"].to_numpy(float), v6_bench["score_B"].to_numpy(float)
    d_val, d_bench = v6_val["score_D"].to_numpy(float), v6_bench["score_D"].to_numpy(float)
    c_inputs = {
        "C1_A_B": (np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel()]), np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel()])),
        "C2_A_B_D": (np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel(), logit(d_val).ravel()]), np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel(), logit(d_bench).ravel()])),
        "C3_condicionada": (
            np.column_stack([logit(a_selected_val).ravel(), logit(b_val).ravel(), logit(d_val).ravel(), (logit(b_val).ravel() - logit(a_selected_val).ravel()) * quality_val, quality_val, np.log1p(amount[val_indices]), missing[val_indices]]),
            np.column_stack([logit(a_selected_bench).ravel(), logit(b_bench).ravel(), logit(d_bench).ravel(), (logit(b_bench).ravel() - logit(a_selected_bench).ravel()) * quality_bench, quality_bench, np.log1p(amount[bench_indices]), missing[bench_indices]]),
        ),
    }
    c_models = {}; c_val_options = {}; c_bench_options = {}
    for i, (name, (z_val, z_bench)) in enumerate(c_inputs.items()):
        model = fit_meta(z_val, y_val, bounds["meta_fit"], cfg.seed + i)
        c_models[name] = model
        c_val_options[name] = model.predict_proba(z_val)[:, 1]
        c_bench_options[name] = model.predict_proba(z_bench)[:, 1]
    c_select_ap = {name: float(average_precision_score(y_val[bounds["model_select"]], score[bounds["model_select"]])) for name, score in c_val_options.items()}
    best_c_name = max(c_select_ap, key=c_select_ap.get)
    c_val, c_bench = c_val_options[best_c_name], c_bench_options[best_c_name]
    joblib.dump({"models": c_models, "selected": best_c_name, "hypothesis": HYPOTHESIS_C}, ART / "modelos_C_fusion_v7.joblib")

    raw_val = {"A": a_selected_val, "B": b_val, "C": c_val, "D": d_val, "A_V6_control": v6_val["score_A"].to_numpy(float)}
    raw_bench = {"A": a_selected_bench, "B": b_bench, "C": c_bench, "D": d_bench, "A_V6_control": v6_bench["score_A"].to_numpy(float)}
    calibrated_val = {}; calibrated_bench = {}; calibrators = {}; calibration_info = {}
    for name in raw_val:
        calibrator = fit_calibrator(raw_val[name][bounds["calibration"]], y_val[bounds["calibration"]])
        calibrators[name] = calibrator
        calibrated_val[name] = apply_calibrator(calibrator, raw_val[name])
        calibrated_bench[name] = apply_calibrator(calibrator, raw_bench[name])
        calibration_info[name] = {"brier_raw": brier_score_loss(y_val[bounds["calibration"]], raw_val[name][bounds["calibration"]]), "brier_calibrado": brier_score_loss(y_val[bounds["calibration"]], calibrated_val[name][bounds["calibration"]])}
    joblib.dump(calibrators, ART / "calibradores_v7.joblib")
    thresholds = {}; internal = {}; benchmark = {}; curves = []
    for name, score in calibrated_val.items():
        threshold, curve = choose_threshold(y_val[bounds["threshold"]], score[bounds["threshold"]], cfg)
        thresholds[name] = threshold; curve.insert(0, "modelo", name); curves.append(curve)
        internal[name] = metric_set(y_val[bounds["evaluation"]], score[bounds["evaluation"]], threshold, cfg)
        benchmark[name] = metric_set(y_bench, calibrated_bench[name], threshold, cfg)
    pd.concat(curves, ignore_index=True).to_csv(ART / "curvas_umbral_v7.csv", index=False)
    write_json(ART / "umbrales_v7.json", thresholds)

    eval_windows = np.array_split(bounds["evaluation"], 4)
    c_deltas = [{"ventana": i, "delta_ap_C_vs_A": float(average_precision_score(y_val[w], calibrated_val["C"][w]) - average_precision_score(y_val[w], calibrated_val["A"][w]))} for i, w in enumerate(eval_windows, 1)]
    promotion_deltas = [{"ventana": i, "delta_ap_V7_vs_V6": float(average_precision_score(y_val[w], calibrated_val["A"][w]) - average_precision_score(y_val[w], calibrated_val["A_V6_control"][w]))} for i, w in enumerate(eval_windows, 1)]
    c_ap_gain = internal["C"]["auc_pr"] - internal["A"]["auc_pr"]
    c_cost_reduction = (internal["A"]["cost_q"] - internal["C"]["cost_q"]) / max(1, internal["A"]["cost_q"])
    c_alert_growth = internal["C"]["alertas_por_100k"] / internal["A"]["alertas_por_100k"] - 1
    c_success = bool(c_ap_gain >= cfg.hypothesis_ap_gain and c_cost_reduction >= cfg.hypothesis_cost_reduction and internal["C"]["recall"] >= cfg.recall_floor and c_alert_growth <= cfg.alert_growth_tolerance and sum(r["delta_ap_C_vs_A"] > 0 for r in c_deltas) >= 3)
    promotion_ap_gain = internal["A"]["auc_pr"] - internal["A_V6_control"]["auc_pr"]
    promotion_cost_reduction = (internal["A_V6_control"]["cost_q"] - internal["A"]["cost_q"]) / max(1, internal["A_V6_control"]["cost_q"])
    promotion_alert_growth = internal["A"]["alertas_por_100k"] / internal["A_V6_control"]["alertas_por_100k"] - 1
    promotion_success = bool(promotion_ap_gain >= .01 and promotion_cost_reduction >= .05 and internal["A"]["recall"] >= cfg.recall_floor and promotion_alert_growth <= cfg.alert_growth_tolerance and sum(r["delta_ap_V7_vs_V6"] > 0 for r in promotion_deltas) >= 3 and min(r["delta_ap_V7_vs_V6"] for r in promotion_deltas) >= -.005)
    candidate = "C" if c_success else "A"

    selection = {"A": {"auc_pr_model_select": select_ap, "auc_pr_subventanas": window_ap, "seleccionado": best_a_name, "pca_auc_pr_model_select": result["seleccion"]["A"]["pca_auc_pr_model_select"], "correccion": "A5 incluye A_V6_control como exigía el protocolo"}, "C": {"auc_pr_model_select": c_select_ap, "seleccionado": best_c_name}}
    result["seleccion"] = selection
    result["calibracion"] = calibration_info
    result["umbrales"] = thresholds
    result["evaluacion_interna"] = internal
    result["benchmark_historico"] = benchmark
    result["hipotesis_C"] = {"declaracion_previa": HYPOTHESIS_C, "control": "A", "delta_ap": c_ap_gain, "reduccion_costo": c_cost_reduction, "crecimiento_alertas": c_alert_growth, "ventanas": c_deltas, "success": c_success}
    result["promocion_V7"] = {"control": "A_V6_control", "delta_ap": promotion_ap_gain, "reduccion_costo": promotion_cost_reduction, "crecimiento_alertas": promotion_alert_growth, "ventanas": promotion_deltas, "success": promotion_success, "confirmatoria": False}
    result["candidato"] = {"modelo": candidate, "detalle": best_c_name if candidate == "C" else best_a_name, "threshold": thresholds[candidate], "confirmatorio": False}
    economics = {}
    for name in ("A", "B", "C", "D"):
        economics[name] = {}
        for tx_per_card in cfg.monthly_transactions_scenarios:
            decisions = cfg.monthly_cards * tx_per_card
            economics[name][str(tx_per_card)] = {"decisiones_mensuales": decisions, "costo_mensual_q": benchmark[name]["cost_per_decision_q"] * decisions, "diferencia_vs_A_q": (benchmark[name]["cost_per_decision_q"] - benchmark["A"]["cost_per_decision_q"]) * decisions}
    result["economia_mensual"] = economics
    result["decision"] = "Conservar el mejor A seleccionado internamente; C no se promueve salvo que cumpla todos los gates. La mejora V7 continúa siendo exploratoria hasta una cohorte futura."
    write_json(result_path, result)
    write_json(ART / "seleccion_modelos_v7.json", selection)

    pd.DataFrame({"indice": val_indices, "TransactionID": v6_val["TransactionID"], "y": y_val, **{f"score_{k}": v for k, v in calibrated_val.items()}}).to_csv(ART / "predicciones_validacion_v7.csv", index=False)
    pd.DataFrame({"indice": bench_indices, "TransactionID": v6_bench["TransactionID"], "y": y_bench, **{f"score_{k}": v for k, v in calibrated_bench.items()}}).to_csv(ART / "predicciones_benchmark_v7.csv", index=False)

    plot_pr(y_val[bounds["evaluation"]], {name: calibrated_val[name][bounds["evaluation"]] for name in ("A", "B", "C", "D", "A_V6_control")}, FIG / "01_comparacion_interna_v7.png", "V7 · evaluación temporal interna")
    plot_pr(y_bench, {name: calibrated_bench[name] for name in ("A", "B", "C", "D")}, FIG / "02_benchmark_historico_v7.png", "V7 · benchmark histórico reutilizado")
    fig, ax = plt.subplots(figsize=(9, 5)); names = list(select_ap); ax.barh(names, [select_ap[n] for n in names], color="#184e77"); ax.set(xlabel="AP en model_select", title="Selección train-only de A"); fig.tight_layout(); fig.savefig(FIG / "03_seleccion_modelos_a_v7.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); table = pd.DataFrame(internal).T.loc[["A", "B", "C", "D"]]; ax.bar(table.index, table["cost_q"] / 1e6, color=["#184e77", "#e9c46a", "#2a9d8f", "#e76f51"]); ax.set(ylabel="Costo interno (millones Q)", title="Costo bajo umbral predefinido"); fig.tight_layout(); fig.savefig(FIG / "05_costos_v7.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in ("A", "B", "C", "D"):
        observed, predicted = calibration_curve(y_val[bounds["calibration"]], calibrated_val[name][bounds["calibration"]], n_bins=8, strategy="quantile")
        ax.plot(predicted, observed, marker="o", label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#6b7280"); ax.set(xlabel="Probabilidad predicha", ylabel="Frecuencia observada", title="Calibración independiente V7"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "06_calibracion_v7.png", dpi=180); plt.close(fig)
    manifest = [{"archivo": str(path.relative_to(ROOT)).replace("\\", "/"), "bytes": path.stat().st_size} for path in sorted(ART.rglob("*")) if path.is_file() and path.name != "manifiesto_v7.json"]
    write_json(ART / "manifiesto_v7.json", manifest)
    gc.collect()
    print(json.dumps(ready({"A": best_a_name, "C": best_c_name, "candidato": candidate, "interno": internal[candidate], "promueve_v7": promotion_success, "C_util": c_success}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
