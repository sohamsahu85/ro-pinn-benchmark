"""
compare_best_two.py — head-to-head: Physics-only vs Gaussian Process.

WHY THIS SCRIPT
---------------
compare_models.py ran 15 models across accuracy + anomaly-detection and found
two standouts (see results_compare/comparison.json):
  * Physics-only  — the six transport equations fit by regression, no network.
                    Best accuracy AND best anomaly detector, and it recovers a
                    physically-interpretable defect fraction (beta).
  * GaussianProcess — the best model with NO physics assumptions at all, close
                    behind Physics-only, and it comes with predictive
                    uncertainty for free (every other black-box model here
                    only gives a point estimate).

This script does NOT repeat that broad sweep. It takes just these two
finalists and puts them through a much deeper, apples-to-apples head-to-head
than the 9-column heatmap could show:

  1. ACCURACY on all four datasets (sst, sss, sin, ssr), not just sin/ssr —
     shows in-sample fit quality too, and whether either model overfits the
     small (47-point) training pool.
  2. ANOMALY DETECTION via residuals (ssr/sin MSE ratio) for all three
     outputs, exactly as in compare_models.py, for direct comparability.
  3. ANOMALY DETECTION via predictive UNCERTAINTY — a GP gives a std alongside
     every prediction; does that std also spike on the degraded set, or does
     only the residual notice? Physics-only has no equivalent (it has no
     concept of "uncertainty," only a point fit), so this axis is GP-only and
     is reported as a bonus, not a competing column.
  4. INTERPRETABILITY: Physics-only is refit separately on EACH dataset (not
     just sst+sss) to recover beta per dataset, reproducing the paper's own
     Table 2 diagnostic (beta should jump on ssr, the degraded set). This is
     the one thing the GP cannot do at all — it has no beta.

On top of that head-to-head, run_paper_style() puts both models through
EXACTLY the protocol reproduce_ro_pinn.py used to build its own Table 1 /
Table 2 / Fig. 1 (same fmt() formatting, same mean +/- sd over seeds, same
four-dataset evaluation, same parity-plot code) so Physics-only and
GaussianProcess land in directly comparable tables/figures to the paper's own
NN/PINN results — merging with results/table1.json when it exists to print
one side-by-side table with all four models.

WHAT THIS SCRIPT DOES / DOES NOT DO
------------------------------------
It imports and reuses the paper's data loader and physics model from
reproduce_ro_pinn.py, unchanged. It does not modify compare_models.py or
baseline_models.py — this is a focused follow-up once the broad sweep has
already picked its two finalists.

USAGE
-----
    python compare_best_two.py --data data --runs 5 --steps 4000 --out results_best2

Requires: torch, numpy, pandas, matplotlib, scikit-learn, openpyxl
(same as compare_models.py — no new dependency).
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")  # silence sklearn/GP convergence chatter

from reproduce_ro_pinn import (
    load_datasets, to_arrays, LABELS, TARGETS,
    fit_physics_only, element_forward, ElementConfig,
    fmt, save_json, plot_fig1,
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# =============================================================================
# Physics-only: fit + predict + per-dataset beta recovery
# =============================================================================
def fit_physics(Xtr, Ytr, cfg=None, steps=4000, seed=0):
    """Fit the six transport parameters on the healthy training pool
    (sst+sss), exactly as compare_models.py does for model C."""
    cfg = cfg or ElementConfig()
    vals, loss = fit_physics_only(Xtr, Ytr, cfg=cfg, fixed=("pi0", "n"),
                                  bs_min=0.04, steps=steps, seed=seed)
    return vals, cfg, loss


def physics_predict(vals, cfg, X):
    p = {k: torch.tensor(float(v)) for k, v in vals.items()}
    out = element_forward(torch.as_tensor(X[:, 0]), torch.as_tensor(X[:, 1]), p, cfg)
    return np.column_stack([out["Qp"].detach().numpy(),
                            out["Cpo"].detach().numpy(),
                            out["Pb"].detach().numpy()])


def beta_per_dataset(data, cfg, steps=4000, seed=0):
    """Refit the physics model SEPARATELY on each of the four datasets (not
    the sst+sss pool used for prediction) to recover how beta, the defect
    fraction, differs by dataset -- this is the paper's Table 2 diagnostic,
    and only makes sense per-dataset since beta is a property of the specific
    membrane state at the time that data was collected."""
    betas = {}
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        vals, _ = fit_physics_only(X, Y, cfg=cfg, fixed=("pi0", "n"),
                                   bs_min=0.04, steps=steps, seed=seed)
        betas[lab] = float(vals["beta"])
    return betas


# =============================================================================
# Gaussian Process: fit + predict with per-output uncertainty
# =============================================================================
def fit_gp(Xtr, Ytr, seed=0):
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(1e-3)
    model = make_pipeline(
        StandardScaler(),
        MultiOutputRegressor(GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, alpha=1e-6, random_state=seed)))
    model.fit(Xtr, Ytr)
    return model


def gp_predict_with_std(model, X):
    """Mean AND std per output. Pipeline/MultiOutputRegressor don't expose
    return_std end-to-end, so this reaches into the per-output GPs directly."""
    scaler = model.named_steps["standardscaler"]
    moreg = model.named_steps["multioutputregressor"]
    Xs = scaler.transform(X)
    means, stds = [], []
    for est in moreg.estimators_:
        m, s = est.predict(Xs, return_std=True)
        means.append(m)
        stds.append(s)
    return np.column_stack(means), np.column_stack(stds)


# =============================================================================
# shared helpers
# =============================================================================
def anomaly_ratio(ssr_val, sin_val):
    """How many times larger is the degraded-set value than the healthy one."""
    return float(ssr_val / max(sin_val, 1e-12))


def r2_per_output(P, Y):
    """R^2 = 1 - SS_res/SS_tot, computed independently per output column and
    per dataset (SS_tot uses THAT dataset's own mean, the standard held-out-R^2
    definition). R^2 = 1 is a perfect fit, 0 means "no better than predicting
    the mean", and it can go arbitrarily negative when a model is worse than
    that -- which is exactly what should happen on the degraded (ssr) set for
    a model trained only on healthy data, and is a cleaner way to say that
    than an MSE ratio on an arbitrary unit scale."""
    ss_res = ((P - Y) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot == 0, 1e-12, ss_tot)


# =============================================================================
# main head-to-head
# =============================================================================
def run_head_to_head(data, runs=5, steps=4000, out="results_best2"):
    os.makedirs(out, exist_ok=True)

    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])
    print(f"Train (sst+sss): {len(Xtr)} pts. Evaluating on all four datasets: "
          f"{', '.join(f'{lab}={len(data[lab])}' for lab in LABELS)}\n")

    summary = {"Physics-only": {"mse": {}}, "GaussianProcess": {"mse": {}, "std": {}}}

    # ---- Physics-only ----------------------------------------------------
    print("Fitting Physics-only on sst+sss ...")
    vals, cfg, floss = fit_physics(Xtr, Ytr, steps=steps)
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        P = physics_predict(vals, cfg, X)
        summary["Physics-only"]["mse"][lab] = ((P - Y) ** 2).mean(0).tolist()
    summary["Physics-only"]["params_fit_on_train"] = vals
    summary["Physics-only"]["final_train_loss"] = floss

    print("Refitting Physics-only separately on each dataset for beta (Table-2 style) ...")
    summary["Physics-only"]["beta_per_dataset"] = beta_per_dataset(data, cfg, steps=steps)

    # ---- Gaussian Process (averaged over `runs` seeds) --------------------
    print(f"Fitting GaussianProcess on sst+sss ({runs} seeds) ...")
    mse_runs = {lab: [] for lab in LABELS}
    std_runs = {lab: [] for lab in LABELS}
    for seed in range(runs):
        gp = fit_gp(Xtr, Ytr, seed=seed)
        for lab in LABELS:
            X, Y = to_arrays(data[lab])
            mean, std = gp_predict_with_std(gp, X)
            mse_runs[lab].append(((mean - Y) ** 2).mean(0))
            std_runs[lab].append(std.mean(0))
    for lab in LABELS:
        summary["GaussianProcess"]["mse"][lab] = np.mean(mse_runs[lab], axis=0).tolist()
        summary["GaussianProcess"]["std"][lab] = np.mean(std_runs[lab], axis=0).tolist()

    # ---- derived metrics: anomaly ratios (residual-based, both models) ----
    for name in ("Physics-only", "GaussianProcess"):
        sin_mse = summary[name]["mse"]["sin"]
        ssr_mse = summary[name]["mse"]["ssr"]
        summary[name]["anomaly_ratio_mse"] = {
            out_lab: anomaly_ratio(ssr_mse[i], sin_mse[i])
            for i, out_lab in enumerate(TARGETS)
        }

    # ---- derived metric: anomaly via UNCERTAINTY, GP only ------------------
    sin_std = summary["GaussianProcess"]["std"]["sin"]
    ssr_std = summary["GaussianProcess"]["std"]["ssr"]
    summary["GaussianProcess"]["anomaly_ratio_std"] = {
        out_lab: anomaly_ratio(ssr_std[i], sin_std[i])
        for i, out_lab in enumerate(TARGETS)
    }

    # ---- print head-to-head tables ----
    print_tables(summary)

    json.dump(summary, open(os.path.join(out, "best_two.json"), "w"), indent=2)
    make_figures(summary, out)
    return summary


# =============================================================================
# PAPER-STYLE REPRODUCTION — run Physics-only and GaussianProcess through the
# exact same protocol reproduce_ro_pinn.py used for NN/PINN: Table 1 (MSE mean
# +/- sd over `runs` seeds, all four datasets, same fmt()/spacing), a Table-2-
# style physics-parameter cross-check, and the paper's own Fig. 1 parity plot
# (plot_fig1 is reused unmodified — it hard-codes a 2-column layout, which is
# exactly what two models need).
#
# This does not replace run_head_to_head() above -- that digs into accuracy,
# anomaly ratios and uncertainty; this instead makes the two winners directly
# comparable to the paper's own NN/PINN numbers, in the paper's own format.
# =============================================================================
def run_paper_style(data, runs=5, steps=4000, out="results_best2"):
    os.makedirs(out, exist_ok=True)
    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])
    print(f"\n[paper-style] Training on sst+sss ({len(Xtr)} pts); Table 1 protocol: "
          f"evaluate on sst, sss, sin, ssr, {runs} seeds each.")

    results, preds = {}, {}

    # ---- Physics-only, run like the paper trains NN/PINN: fit once on
    # sst+sss per seed, evaluate on all four datasets ----
    per_run_mse = {lab: [] for lab in LABELS}
    per_run_pred = {lab: [] for lab in LABELS}
    for seed in range(runs):
        vals, cfg, _ = fit_physics(Xtr, Ytr, steps=steps, seed=seed)
        for lab in LABELS:
            X, Y = to_arrays(data[lab])
            P = physics_predict(vals, cfg, X)
            per_run_mse[lab].append(((P - Y) ** 2).mean(0))
            per_run_pred[lab].append(P)
    results["Physics-only"] = {lab: np.array(v) for lab, v in per_run_mse.items()}
    preds["Physics-only"] = {lab: np.array(v) for lab, v in per_run_pred.items()}

    # ---- GaussianProcess, same protocol ----
    per_run_mse = {lab: [] for lab in LABELS}
    per_run_pred = {lab: [] for lab in LABELS}
    for seed in range(runs):
        gp = fit_gp(Xtr, Ytr, seed=seed)
        for lab in LABELS:
            X, Y = to_arrays(data[lab])
            mean, _ = gp_predict_with_std(gp, X)
            per_run_mse[lab].append(((mean - Y) ** 2).mean(0))
            per_run_pred[lab].append(mean)
    results["GaussianProcess"] = {lab: np.array(v) for lab, v in per_run_mse.items()}
    preds["GaussianProcess"] = {lab: np.array(v) for lab, v in per_run_pred.items()}

    # both models fit deterministically from a fixed initial point (Physics-only
    # never randomises its parameter init; the GP kernel optimiser takes 0
    # restarts by default) -- so sd collapses to ~0 across seeds. That's a real
    # finding, not a bug: neither model has the NN/PINN's seed-to-seed training
    # instability that table1.json shows for the paper's own two models.
    for tag in results:
        sds = [results[tag][lab].std(0).max() for lab in LABELS]
        if max(sds) < 1e-9:
            print(f"  note: {tag} is deterministic across seeds (sd ~ 0) -- "
                  f"unlike the paper's NN/PINN, which vary run to run.")

    # ---- Table 1, in the paper's own layout ----
    print("\nTable 1 (extended) — MSE, mean ± sd over %d runs "
          "(Qp L/min, Cpo ppm, Pb bar)" % runs)
    head = f"{'Dataset':8s}" + "".join(f"{t:>66s}" for t in results)
    print(head)
    print(f"{'':8s}" + "".join(f"{'Qp':>22s}{'Cpo':>22s}{'Pb':>22s}" for _ in results))
    for lab in LABELS:
        row = f"{lab:8s}"
        for tag in results:
            A = results[tag][lab]
            for j in range(3):
                row += f"{fmt(A[:, j].mean(), A[:, j].std()):>22s}"
        print(row)

    save_json(Path(out) / "table1_best_two.json",
              {t: {l: v.tolist() for l, v in d.items()} for t, d in results.items()})

    # ---- if the original NN/PINN table1.json is available, print + save the
    # FULL four-model side-by-side table (Cpo column only, for width) ----
    orig_path = Path("results") / "table1.json"
    if orig_path.exists():
        orig = json.loads(orig_path.read_text())
        combined = dict(orig)
        combined.update({t: {l: v.tolist() for l, v in d.items()} for t, d in results.items()})
        print(f"\nTable 1 (FULL, Cpo only) — all four models "
              f"({orig_path} + our two winners)")
        print(f"{'Dataset':8s}" + "".join(f"{t:>26s}" for t in combined))
        for lab in LABELS:
            row = f"{lab:8s}"
            for tag, d in combined.items():
                A = np.array(d[lab])
                row += f"{fmt(A[:, 1].mean(), A[:, 1].std()):>26s}"
            print(row)
        save_json(Path(out) / "table1_full_four_models.json", combined)
        nn_arr = np.array(orig["NN (no physics)"]["sin"])
        pinn_arr = np.array(orig["PINN (with physics)"]["sin"])
        if np.array_equal(nn_arr, pinn_arr):
            print("  note: NN and PINN rows are bit-identical here. Under the paper's "
                "Eq. (7) read literally (physics_target=\"data\"), the physics loss "
                "never touches the network's gradient -- lambda only fits a parallel, "
                "non-interacting physics-parameter regression. So 'PINN vs NN accuracy' "
                "is not actually a comparison of two different predictors.")
    else:
        print(f"\n(note: {orig_path} not found — run "
              f"'python reproduce_ro_pinn.py main --data <folder>' first for the "
              f"full four-model side-by-side table)")

    # ---- Table 2 cross-check style: physical parameters per dataset, exactly
    # as exp_individual()'s "no network" row -- Physics-only only, GP has no
    # physical parameters to report ----
    print("\nTable 2 cross-check — Physics-only parameters fit per dataset "
          "(pi0=25 bar, n=1.5 fixed, Bs >= 0.04 lmh)")
    print(f"{'Dataset':8s}{'Lp':>10s}{'Bs':>12s}{'k':>12s}{'beta':>12s}{'NMSE':>12s}")
    table2 = {}
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        v, loss = fit_physics_only(X, Y, fixed=("pi0", "n"), bs_min=0.04, steps=steps)
        table2[lab] = {**v, "nmse": loss}
        print(f"{lab:8s}{v['Lp']:>10.3f}{v['Bs']:>12.4f}{v['k']:>12.4f}"
              f"{v['beta']:>12.2e}{loss:>12.2e}")
    print("  (GaussianProcess has no physical parameters — it has no row here.)")
    save_json(Path(out) / "table2_physics_only.json", table2)

    # ---- Fig. 1 style parity plot, reusing the paper's own plotting code ----
    plot_fig1(data, preds, Path(out))

    return results, table2


# =============================================================================
# R^2 + honest improvement percentages
# =============================================================================
# MSE is scale-dependent (a Cpo MSE of 30 vs 60 means nothing without knowing
# Cpo's own spread) and the earlier "% improvement" numbers reported for this
# analysis were computed against a buggy PINN reproduction and later retracted.
# This function redoes both properly: R^2 per output (scale-free, "how much
# better than predicting the mean") on the SAME corrected models, and
# improvement percentages stated against the corrected NN(=PINN) baseline.
# =============================================================================
def run_r2_report(data, runs=5, steps=4000, out="results_best2"):
    os.makedirs(out, exist_ok=True)
    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])

    from reproduce_ro_pinn import train as train_nn, predict as predict_nn

    r2 = {name: {} for name in ("NN(=PINN)", "Physics-only", "GaussianProcess")}
    mse = {name: {} for name in r2}

    # ---- NN(=PINN): physics_target="data" makes lambda irrelevant to the
    # network, so training once at lam=0 IS the PINN's predictive model too ----
    nn_pred = {lab: [] for lab in LABELS}
    for seed in range(runs):
        m = train_nn(Xtr, Ytr, lam=0.0, seed=seed, epochs=5000, physics_freq="epoch")
        for lab in LABELS:
            X, _ = to_arrays(data[lab])
            nn_pred[lab].append(predict_nn(m, X))
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        P = np.mean(nn_pred[lab], axis=0)
        r2["NN(=PINN)"][lab] = r2_per_output(P, Y).tolist()
        mse["NN(=PINN)"][lab] = ((P - Y) ** 2).mean(0).tolist()

    # ---- Physics-only ----
    vals, cfg, _ = fit_physics(Xtr, Ytr, steps=steps, seed=0)
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        P = physics_predict(vals, cfg, X)
        r2["Physics-only"][lab] = r2_per_output(P, Y).tolist()
        mse["Physics-only"][lab] = ((P - Y) ** 2).mean(0).tolist()

    # ---- GaussianProcess ----
    gp_pred = {lab: [] for lab in LABELS}
    for seed in range(runs):
        gp = fit_gp(Xtr, Ytr, seed=seed)
        for lab in LABELS:
            X, _ = to_arrays(data[lab])
            mean, _ = gp_predict_with_std(gp, X)
            gp_pred[lab].append(mean)
    for lab in LABELS:
        X, Y = to_arrays(data[lab])
        P = np.mean(gp_pred[lab], axis=0)
        r2["GaussianProcess"][lab] = r2_per_output(P, Y).tolist()
        mse["GaussianProcess"][lab] = ((P - Y) ** 2).mean(0).tolist()

    # ---- print R^2 table ----
    print("\n" + "=" * 88)
    print("R^2 per output, per dataset  (1 = perfect, 0 = no better than the mean, "
          "negative = worse than the mean)")
    print("=" * 88)
    print(f"{'Model':16s}{'Dataset':10s}{'Qp':>10s}{'Cpo':>10s}{'Pb':>10s}")
    print("-" * 88)
    for name in r2:
        for lab in LABELS:
            q, c, p = r2[name][lab]
            print(f"{name:16s}{lab:10s}{q:>10.3f}{c:>10.3f}{p:>10.3f}")
        print("-" * 88)
    print("Reading: ssr R^2 is expected to be strongly negative for every model here --")
    print("that IS the anomaly signal (none of these models were trained on degraded")
    print("membrane behaviour, so all fail worse than a naive mean-predictor on ssr).")

    # ---- honest improvement percentages, vs the corrected NN(=PINN) baseline ----
    print("\nIMPROVEMENT vs. corrected NN(=PINN) baseline, on sin (held-out healthy)")
    print(f"{'Model':16s}{'Cpo MSE reduction':>20s}{'Cpo R^2 (baseline -> model)':>32s}")
    base_mse = mse["NN(=PINN)"]["sin"][1]
    base_r2 = r2["NN(=PINN)"]["sin"][1]
    for name in ("Physics-only", "GaussianProcess"):
        m_new = mse[name]["sin"][1]
        r_new = r2[name]["sin"][1]
        pct = 100.0 * (base_mse - m_new) / base_mse
        print(f"{name:16s}{pct:>19.1f}%{f'{base_r2:.3f} -> {r_new:.3f}':>32s}")

    save_json(Path(out) / "r2_report.json", {"r2": r2, "mse": mse})
    return r2, mse


# =============================================================================
# Bootstrap confidence interval on the accuracy gap
# =============================================================================
# The R^2/MSE numbers above are point estimates on one fixed 45-point test set
# (sin). This resamples that test set (with replacement, B times) to ask: does
# Physics-only/GaussianProcess's advantage over NN(=PINN) survive resampling,
# or could it be driven by a handful of points? NN(=PINN)'s own training-seed
# variability is folded in by drawing a random one of its 5 already-trained
# seeds on each bootstrap replicate, alongside the (deterministic) physics-only
# and GPR predictions -- this is a paired bootstrap, so each replicate compares
# all three models on the SAME resampled points.
# =============================================================================
def run_bootstrap_ci(data, runs=5, steps=4000, n_boot=5000, out="results_best2", seed=0):
    from reproduce_ro_pinn import train as train_nn, predict as predict_nn

    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])
    Xsin, Ysin = to_arrays(data["sin"])
    n = len(Xsin)

    print(f"\nBootstrap CI: fitting NN(=PINN) ({runs} seeds), Physics-only, GPR once each ...")
    nn_preds = []
    for s in range(runs):
        m = train_nn(Xtr, Ytr, lam=0.0, seed=s, epochs=5000, physics_freq="epoch")
        nn_preds.append(predict_nn(m, Xsin))

    vals, cfg, _ = fit_physics(Xtr, Ytr, steps=steps, seed=0)
    phys_pred = physics_predict(vals, cfg, Xsin)

    gp = fit_gp(Xtr, Ytr, seed=0)
    gp_pred, _ = gp_predict_with_std(gp, Xsin)

    rng = np.random.default_rng(seed)
    icpo = TARGETS.index("Cpo")
    diff_phys, diff_gp = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y = Ysin[idx, icpo]
        nn_p = nn_preds[rng.integers(0, runs)][idx, icpo]
        ph_p = phys_pred[idx, icpo]
        gp_p = gp_pred[idx, icpo]
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2_nn = 1 - ((nn_p - y) ** 2).sum() / ss_tot
        r2_ph = 1 - ((ph_p - y) ** 2).sum() / ss_tot
        r2_gp = 1 - ((gp_p - y) ** 2).sum() / ss_tot
        diff_phys.append(r2_ph - r2_nn)
        diff_gp.append(r2_gp - r2_nn)
    diff_phys, diff_gp = np.array(diff_phys), np.array(diff_gp)

    def summarize(name, d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_not_better = float((d <= 0).mean())
        print(f"  {name}: R^2 advantage over NN(=PINN) = {d.mean():+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(not better) = {p_not_better:.4f}  (n_boot={n_boot})")
        return {"mean_r2_advantage": float(d.mean()), "ci95": [float(lo), float(hi)],
                "p_not_better": p_not_better}

    print(f"\nBootstrap results (Cpo R^2 on sin, {n_boot} resamples of the {n}-point test set):")
    result = {
        "Physics-only": summarize("Physics-only", diff_phys),
        "GaussianProcess": summarize("GaussianProcess", diff_gp),
    }
    save_json(Path(out) / "bootstrap_ci.json", result)
    return result


def print_tables(summary):
    print("\n" + "=" * 88)
    print("ACCURACY  —  MSE per output, per dataset  (lower = better)")
    print("=" * 88)
    print(f"{'Model':16s}{'Dataset':10s}{'Qp':>14s}{'Cpo':>14s}{'Pb':>14s}")
    print("-" * 88)
    for name in ("Physics-only", "GaussianProcess"):
        for lab in LABELS:
            q, c, p = summary[name]["mse"][lab]
            print(f"{name:16s}{lab:10s}{q:>14.3e}{c:>14.1f}{p:>14.3e}")
        print("-" * 88)

    print("\nANOMALY DETECTION  (ssr / sin ratio — higher = better detector)")
    print(f"{'Model':16s}{'signal':16s}{'Qp':>10s}{'Cpo':>10s}{'Pb':>10s}")
    for name in ("Physics-only", "GaussianProcess"):
        r = summary[name]["anomaly_ratio_mse"]
        print(f"{name:16s}{'residual (MSE)':16s}"
              f"{r['Qp']:>10.0f}{r['Cpo']:>10.0f}{r['Pb']:>10.0f}")
    r = summary["GaussianProcess"]["anomaly_ratio_std"]
    print(f"{'GaussianProcess':16s}{'uncertainty (std)':16s}"
          f"{r['Qp']:>10.1f}{r['Cpo']:>10.1f}{r['Pb']:>10.1f}")
    print("  (Physics-only has no uncertainty estimate -- it's a point fit, not a")
    print("   probabilistic model -- so that row only exists for the GP.)")

    print("\nINTERPRETABILITY  —  Physics-only beta, refit per dataset")
    print(f"{'Dataset':10s}{'beta':>14s}")
    for lab in LABELS:
        print(f"{lab:10s}{summary['Physics-only']['beta_per_dataset'][lab]:>14.2e}")
    print("  (beta rising sharply on ssr vs the healthy sets is the paper's own")
    print("   degradation signature; GaussianProcess has no analogue at all.)")


# =============================================================================
# figures
# =============================================================================
def make_figures(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    # (1) accuracy across all 4 datasets, one subplot per output
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    width = 0.35
    xpos = np.arange(len(LABELS))
    for j, (ax, out_lab) in enumerate(zip(axes, TARGETS)):
        phys = [summary["Physics-only"]["mse"][lab][j] for lab in LABELS]
        gp = [summary["GaussianProcess"]["mse"][lab][j] for lab in LABELS]
        ax.bar(xpos - width / 2, phys, width, label="Physics-only", color="#2e7d32",
              edgecolor="black", linewidth=0.5)
        ax.bar(xpos + width / 2, gp, width, label="GaussianProcess", color="#1565c0",
              edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels(LABELS)
        ax.set_title(f"{out_lab} MSE (log scale)")
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.set_axisbelow(True)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Physics-only vs GaussianProcess — accuracy across all four datasets")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "best2_accuracy.png"), dpi=150)
    plt.close(fig)

    # (2) anomaly detection: residual ratio (both) vs uncertainty ratio (GP only)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    outs = TARGETS
    xpos = np.arange(len(outs))
    w = 0.25
    phys_r = [summary["Physics-only"]["anomaly_ratio_mse"][o] for o in outs]
    gp_r = [summary["GaussianProcess"]["anomaly_ratio_mse"][o] for o in outs]
    gp_std_r = [summary["GaussianProcess"]["anomaly_ratio_std"][o] for o in outs]
    ax.bar(xpos - w, phys_r, w, label="Physics-only (residual)", color="#2e7d32",
          edgecolor="black", linewidth=0.5)
    ax.bar(xpos, gp_r, w, label="GaussianProcess (residual)", color="#1565c0",
          edgecolor="black", linewidth=0.5)
    ax.bar(xpos + w, gp_std_r, w, label="GaussianProcess (uncertainty)", color="#90caf9",
          edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels(outs)
    ax.set_ylabel("ssr / sin ratio  (higher = stronger anomaly signal)")
    ax.set_title("Anomaly detection: residual-based (both models) vs uncertainty-based (GP only)")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "best2_anomaly.png"), dpi=150)
    plt.close(fig)

    # (3) beta per dataset (Physics-only's unique interpretability output)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    betas = [summary["Physics-only"]["beta_per_dataset"][lab] for lab in LABELS]
    colors = ["#2e7d32" if lab != "ssr" else "#c62828" for lab in LABELS]
    ax.bar(LABELS, betas, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("beta (defect fraction)")
    ax.set_title("Physics-only interpretability: beta jumps on the degraded set (ssr)\n"
                "GaussianProcess has no analogue -- this row is Physics-only only")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "best2_beta.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {figdir}/")
    print("  best2_accuracy.png   MSE per output, all four datasets, both models")
    print("  best2_anomaly.png    residual anomaly (both) vs uncertainty anomaly (GP only)")
    print("  best2_beta.png       Physics-only's recovered beta per dataset")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--runs", type=int, default=5, help="GP seeds to average over")
    ap.add_argument("--steps", type=int, default=4000, help="physics-fit optimisation steps")
    ap.add_argument("--out", default="results_best2")
    ap.add_argument("--skip-paper-style", dest="paper_style", action="store_false",
                    help="skip the Table-1/Table-2/Fig-1 paper-format reproduction")
    args = ap.parse_args()
    data = load_datasets(args.data, verbose=True)
    run_head_to_head(data, runs=args.runs, steps=args.steps, out=args.out)
    if args.paper_style:
        run_paper_style(data, runs=args.runs, steps=args.steps, out=args.out)
    run_r2_report(data, runs=args.runs, steps=args.steps, out=args.out)
    run_bootstrap_ci(data, runs=args.runs, steps=args.steps, out=args.out)


if __name__ == "__main__":
    main()
