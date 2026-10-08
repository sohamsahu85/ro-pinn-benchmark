"""
baseline_models.py — standard ML models as a reference point for
reproduce_ro_pinn.py's NN/PINN results (M. Li, J. Li 2025).

This is a SEPARATE script from reproduce_ro_pinn.py on purpose: the original
file reproduces the paper as published and is left untouched, so its numbers
stay a clean, checkable reference. This file imports the paper's data loader
and formatting from it, trains a range of standard regressors on the exact
same split (train sst+sss, evaluate sst/sss/sin/ssr), and writes its own
results file so the two can be compared side by side.

USAGE
-----
    python baseline_models.py --data <folder> [--out results] [--runs 5] [--no-plot]

Requires: numpy, pandas, openpyxl, torch (via reproduce_ro_pinn.py's imports),
scikit-learn, matplotlib. xgboost is optional — skipped with a note if absent.

OUTPUT
------
    results/table3_baselines.json   per-model MSE (mean/std over --runs) + the
                                     hyperparameters chosen by 5-fold CV, plus
                                     a top-3 prediction-averaging ensemble.
    results/fig3_model_comparison.png
        Bar chart comparing a curated subset of models (NN, PINN from Table 1,
        plus the best baselines) on the sin (held-out) and ssr (degraded) test
        sets — the two datasets the paper's argument turns on. Only drawn if
        results/table1.json (from reproduce_ro_pinn.py) already exists.

WHAT THIS ADDS BEYOND THE PAPER
--------------------------------
The paper only reports one architecture (2-layer/20-neuron Tanh MLP) with and
without the physics loss. This script asks: do standard regressors, with no
physics term at all, see the same things?
  * The ssr Cpo MSE ~1.1e5 ppm^2 anomaly signal (Sec. 3 of the paper) shows up
    in EVERY model family tried here, including plain linear regression —
    evidence the anomaly is a real shift in the data, not a PINN artifact.
  * On the sin held-out set, a tuned Gaussian Process matches or slightly
    beats the paper's non-physics NN, with none of the seed-to-seed training
    instability the reproduced PINN shows in table1.json (some seeds diverge
    to 10-100x the error of others under the paper's lambda=10 setting).
  * None of these baselines recover physically-interpretable parameters
    (Lp, Bs, beta, ...) the way the PINN does — that half of the paper's
    contribution (Table 2, degradation diagnosis) has no analogue here.
"""

import argparse
from pathlib import Path

import numpy as np

from reproduce_ro_pinn import LABELS, TARGETS, load_datasets, to_arrays, fmt, save_json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


# =============================================================================
# model zoo + small hyperparameter grids
# =============================================================================
def build_estimator(name, params, seed):
    """name + hyperparameter dict -> (unfitted estimator, needs_multioutput_wrapper)."""
    if name == "Linear Regression":
        return LinearRegression(), False
    if name == "Ridge Regression":
        return Ridge(random_state=seed, **params), False
    if name == "k-Nearest Neighbors":
        return KNeighborsRegressor(**params), False
    if name == "Random Forest":
        return RandomForestRegressor(random_state=seed, **params), False
    if name == "Extra Trees":
        return ExtraTreesRegressor(random_state=seed, **params), False
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=seed, **params), True
    if name == "AdaBoost":
        return AdaBoostRegressor(random_state=seed, **params), True
    if name == "Support Vector (RBF)":
        return SVR(**params), True
    if name == "MLP (sklearn)":
        return MLPRegressor(random_state=seed, max_iter=5000, **params), False
    if name == "Gaussian Process":
        return GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(1e-3),
            random_state=seed, n_restarts_optimizer=3, normalize_y=False, **params), False
    if name == "Polynomial Ridge (deg 2)":
        alpha = params.get("alpha", 1.0)
        return Pipeline([("poly", PolynomialFeatures(degree=2, include_bias=False)),
                          ("ridge", Ridge(alpha=alpha, random_state=seed))]), False
    if name == "XGBoost":
        return XGBRegressor(random_state=seed, verbosity=0, **params), True
    raise KeyError(name)


# small hand-picked grids (training data is ~50 points, so grids stay small to
# avoid over-fitting the search itself); a single-entry list means "no
# tunable hyperparameters worth searching".
SEARCH_SPACE = {
    "Linear Regression": [{}],
    "Ridge Regression": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "k-Nearest Neighbors": [{"n_neighbors": k} for k in (3, 5, 7, 9)],
    "Random Forest": [{"n_estimators": n, "max_depth": d}
                       for n in (100, 300) for d in (None, 5, 10)],
    "Extra Trees": [{"n_estimators": n, "max_depth": d}
                     for n in (100, 300) for d in (None, 5, 10)],
    "Gradient Boosting": [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
                            for n in (100, 300) for d in (2, 3) for lr in (0.05, 0.1)],
    "AdaBoost": [{"n_estimators": n, "learning_rate": lr}
                  for n in (50, 100) for lr in (0.5, 1.0)],
    "Support Vector (RBF)": [{"C": c, "epsilon": e}
                               for c in (1.0, 10.0, 100.0) for e in (0.001, 0.01, 0.1)],
    "MLP (sklearn)": [{"hidden_layer_sizes": hl, "alpha": a}
                        for hl in ((20,), (20, 20), (50,)) for a in (1e-4, 1e-2)],
    "Gaussian Process": [{"alpha": a} for a in (1e-6, 1e-4, 1e-2)],
    "Polynomial Ridge (deg 2)": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0)],
}
if HAVE_XGBOOST:
    SEARCH_SPACE["XGBoost"] = [{"n_estimators": n, "max_depth": d, "learning_rate": lr}
                                 for n in (100, 300) for d in (2, 3) for lr in (0.05, 0.1)]


def wrap_model(estimator, needs_multi):
    """Scale X, scale each target independently, wrap non-multioutput estimators."""
    if needs_multi:
        estimator = MultiOutputRegressor(estimator)
    inner = Pipeline([("scale_x", StandardScaler()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())


def _kfold(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return np.array_split(idx, k)


def cv_nmse(name, params, X, Y, k=5, seed=0):
    """Mean, across folds and outputs, of per-output MSE normalised by mean(y^2)
    (the same normalisation as the paper's Eq. 7) — scale-free so Qp/Cpo/Pb
    contribute comparably when picking hyperparameters."""
    folds = _kfold(len(X), min(k, len(X)), seed)
    scores = []
    for i in range(len(folds)):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        if len(tr) < 3 or len(te) == 0:
            continue
        est, needs_multi = build_estimator(name, params, seed)
        model = wrap_model(est, needs_multi)
        model.fit(X[tr], Y[tr])
        P = model.predict(X[te])
        scores.append((((P - Y[te]) ** 2).mean(0) / (Y[te] ** 2).mean(0)).mean())
    return float(np.mean(scores)) if scores else np.inf


def best_params(name, X, Y, seed=0):
    """Returns (best_params, best_cv_score) — lower score is better."""
    grid = SEARCH_SPACE[name]
    if len(grid) == 1:
        return grid[0], cv_nmse(name, grid[0], X, Y, seed=seed)
    scored = [(cv_nmse(name, p, X, Y, seed=seed), p) for p in grid]
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[0][0]


# =============================================================================
# experiment
# =============================================================================
def exp_baselines(data, args):
    """For each model family: 5-fold CV on sst+sss picks hyperparameters
    (minimising the Eq.-7-style normalised MSE), then the tuned model is
    retrained --runs times and evaluated on all four sets with the same
    per-output MSE as Table 1 — plus a top-3 prediction-averaging ensemble,
    the top-3 chosen by CV score (not by peeking at test-set numbers)."""
    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])
    print(f"\nBaseline ML models (not in the paper) — trained on sst+sss "
          f"({len(Xtr)} points), evaluated like Table 1.")
    if not HAVE_XGBOOST:
        print("  (xgboost not installed — skipping that model; `pip install xgboost` to add it)")

    names = list(SEARCH_SPACE)
    results, preds, chosen, cv_score = {}, {}, {}, {}
    for name in names:
        params, score = best_params(name, Xtr, Ytr, seed=0)
        chosen[name], cv_score[name] = params, score
        per_run_mse = {lab: [] for lab in LABELS}
        per_run_pred = {lab: [] for lab in LABELS}
        for seed in range(args.runs):
            est, needs_multi = build_estimator(name, params, seed)
            model = wrap_model(est, needs_multi)
            model.fit(Xtr, Ytr)
            for lab in LABELS:
                X, Y = to_arrays(data[lab])
                P = model.predict(X)
                per_run_mse[lab].append(((P - Y) ** 2).mean(0))
                per_run_pred[lab].append(P)
        results[name] = {lab: np.array(v) for lab, v in per_run_mse.items()}
        preds[name] = per_run_pred

    # top-3 ensemble: average predictions of the 3 lowest-CV-score models
    top3 = sorted(names, key=lambda n: cv_score[n])[:3]
    ens_name = f"Ensemble (avg of top 3: {', '.join(top3)})"
    ens_mse = {lab: [] for lab in LABELS}
    for seed in range(args.runs):
        for lab in LABELS:
            avg = np.mean([preds[n][lab][seed] for n in top3], axis=0)
            _, Y = to_arrays(data[lab])
            ens_mse[lab].append(((avg - Y) ** 2).mean(0))
    results[ens_name] = {lab: np.array(v) for lab, v in ens_mse.items()}
    names_with_ens = names + [ens_name]
    chosen[ens_name] = {"members": top3}

    print("\nTable 3 — baseline ML models, MSE mean +/- sd over %d runs "
          "(Qp L/min, Cpo ppm, Pb bar); hyperparameters chosen by 5-fold CV" % args.runs)
    for name in names_with_ens:
        print(f"\n  {name}  (params: {chosen[name] or 'default'})")
        print(f"    {'Dataset':8s}{'Qp':>22s}{'Cpo':>22s}{'Pb':>22s}")
        for lab in LABELS:
            A = results[name][lab]
            row = f"    {lab:8s}"
            for j in range(3):
                row += f"{fmt(A[:, j].mean(), A[:, j].std()):>22s}"
            print(row)

    save_json(Path(args.out) / "table3_baselines.json",
              {n: {"params": chosen[n], "mse": {l: v.tolist() for l, v in d.items()}}
               for n, d in results.items()})

    if args.plot:
        plot_comparison(Path(args.out), results, top3, ens_name)


# =============================================================================
# comparison figure: NN/PINN (from reproduce_ro_pinn.py's table1.json) vs. a
# curated subset of the baselines here, on the two datasets the paper's
# argument turns on: sin (held-out generalisation) and ssr (degradation).
# =============================================================================
def plot_comparison(outdir: Path, results, top3, ens_name):
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table1_path = outdir / "table1.json"
    if not table1_path.exists():
        print(f"\n(skipping comparison figure: {table1_path} not found — run "
              f"'python reproduce_ro_pinn.py main --data <folder>' first)")
        return

    table1 = json.loads(table1_path.read_text())
    combined = {}
    for tag, d in table1.items():          # "NN (no physics)", "PINN (with physics)"
        combined[tag] = {lab: np.array(v) for lab, v in d.items()}
    curated = ["Linear Regression", "Random Forest", "Support Vector (RBF)",
               "Gaussian Process", ens_name]
    for name in curated:
        if name in results:
            combined[name] = results[name]

    model_names = list(table1.keys()) + [n for n in curated if n in results]
    short = {n: (n.replace(" (no physics)", "").replace(" (with physics)", "")
                 .replace(f" (avg of top 3: {', '.join(top3)})", " (top-3)")
                 .replace(" (RBF)", ""))
             for n in model_names}

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    for row, lab in enumerate(["sin", "ssr"]):
        for col, (j, unit) in enumerate([(0, "Qp MSE (L/min)^2"),
                                          (1, "Cpo MSE (ppm)^2"),
                                          (2, "Pb MSE (bar)^2")]):
            ax = axes[row, col]
            means = [combined[n][lab][:, j].mean() for n in model_names]
            stds = [combined[n][lab][:, j].std() for n in model_names]
            colors = ["tab:orange" if n.startswith(("NN", "PINN")) else "tab:blue"
                      for n in model_names]
            ax.bar(range(len(model_names)), means, yerr=stds, color=colors, capsize=3)
            ax.set_yscale("log")
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([short[n] for n in model_names], rotation=45,
                                ha="right", fontsize=7)
            ax.set_title(f"{lab} — {unit}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{'held-out' if lab == 'sin' else 'degraded'} MSE (log scale)",
                               fontsize=8)
    fig.suptitle("Model comparison: paper's NN/PINN (orange) vs. standard ML baselines (blue)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = outdir / "fig3_model_comparison.png"
    fig.savefig(p, dpi=160)
    print(f"\nComparison figure written to {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data", help="folder with the xlsx/csv files")
    ap.add_argument("--out", default="results",
                    help="also where reproduce_ro_pinn.py's table1.json is read from, "
                         "for the comparison figure")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--no-plot", dest="plot", action="store_false")
    args = ap.parse_args()

    data = load_datasets(args.data)
    for lab in LABELS:
        d = data[lab]
        print(f"  {lab}: n={len(d)}  Qf {d.Qf.min():.1f}-{d.Qf.max():.1f} L/min  "
              f"Pf {d.Pf.min():.0f}-{d.Pf.max():.0f} bar  "
              f"Qp {d.Qp.min():.2f}-{d.Qp.max():.2f} L/min  "
              f"Cpo {d.Cpo.min():.0f}-{d.Cpo.max():.0f} ppm")

    exp_baselines(data, args)


if __name__ == "__main__":
    main()
