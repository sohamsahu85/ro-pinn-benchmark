"""
compare_models.py — Extension 3: a like-for-like comparison of many models on the
RO dataset, on the two axes the paper cares about but never tests across models:
predictive accuracy and degradation/anomaly detection.

WHY THIS EXTENSION
------------------
The paper is usually described as "comparing a data-driven NN against a PINN", but
architecturally those are ONE network at two settings of the physics weight lambda
(lambda=0 vs lambda=10). It compares its method against essentially no other model.
This script fills that gap: it holds the task and the train/test split fixed exactly
as the paper does (train on sst+sss, test on sin+ssr) and runs a spread of models
through the same evaluation.

MODELS COMPARED
---------------
From the paper (reused from reproduce_ro_pinn.py):
  A  NN (lambda=0)          the plain neural network
  B  PINN (lambda=10)       the network + physics loss
  C  Physics-only           the six transport equations fitted by regression, no net

Added baselines (the obvious "is the neural net necessary?" competitors):
  D  Linear regression      simplest possible baseline
  E  Ridge regression       linear + L2, guards against collinearity
  F  Polynomial (deg 2)     linear on quadratic features
  G  Random forest          nonparametric, no scaling needed
  H  Extra trees            more randomised, usually lower variance than RF
  I  Gradient boosting      strong tabular baseline
  J  AdaBoost               boosted shallow trees, different bias/variance than GB
  K  Support vector (RBF)   max-margin regression, kernelised
  L  MLP (sklearn)          black-box net with no physics loss, for comparison to A
  M  Gaussian process       gives predictive uncertainty for free
  N  k-nearest neighbours   pure local interpolation
  O  XGBoost                (only if installed) another strong tabular baseline

TWO EVALUATION AXES
-------------------
(1) ACCURACY.  MSE per output (Qp, Cpo, Pb) on the held-out healthy set (sin).
    Question: does the neural network earn its complexity, or do simpler models
    match it? (The paper's own results hint the physics buys no accuracy.)

(2) ANOMALY DETECTION.  Each model is trained on healthy data only; we then measure
    how strongly it flags the degraded set (ssr) via its prediction residual on Cpo.
    A good anomaly detector shows a LARGE ratio of ssr-residual to healthy-residual.
    Question: do you even need the PINN to detect degradation, or does any decent
    regressor's residual spike on ssr too?

INTERPRETABILITY is tracked as a third, qualitative column: only the physics-bearing
models (B, C) yield the defect fraction beta and can say *why* ssr is anomalous.

WHAT THIS SCRIPT DOES / DOES NOT DO
-----------------------------------
It PRINTS a comparison table and writes figures + JSON. It does not change any of
the paper-reproduction code; it imports and reuses it. Run it after you can already
run `reproduce_ro_pinn.py main`.

USAGE
-----
    python compare_models.py --data <folder> --runs 3 --out results_compare

Requires: torch, numpy, pandas, matplotlib, scikit-learn, openpyxl
"""

from __future__ import annotations
import argparse, json, os, warnings
import numpy as np

warnings.filterwarnings("ignore")   # silence sklearn convergence chatter for a clean log

# --- reuse the paper's models and data from the reproduction file ---------------
from reproduce_ro_pinn import (
    load_datasets, to_arrays, LABELS, TARGETS,
    train as train_nn,            # the paper's NN / PINN (lambda switch)
    mse_per_output, predict as nn_predict,
    fit_physics_only, element_forward, ElementConfig, PhysicalParams,
)

# --- sklearn baselines ----------------------------------------------------------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False


# ============================================================================
# sklearn model factory — each returns a fresh, untrained estimator
# ============================================================================
def make_sklearn_models(seed=0):
    """Return {name: estimator}. All are multi-output (predict Qp, Cpo, Pb at once)."""
    gp_kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0]) + WhiteKernel(1e-3)
    models = {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed)),
        "Poly-2": make_pipeline(StandardScaler(), PolynomialFeatures(2), LinearRegression()),
        "RandomForest": MultiOutputRegressor(
            RandomForestRegressor(n_estimators=300, random_state=seed)),
        "ExtraTrees": MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=300, random_state=seed)),
        "GradBoost": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, random_state=seed)),
        "AdaBoost": MultiOutputRegressor(
            AdaBoostRegressor(n_estimators=100, random_state=seed)),
        "SVR": make_pipeline(StandardScaler(),
            MultiOutputRegressor(SVR(C=10.0, epsilon=0.01))),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(20, 20), alpha=1e-4, max_iter=5000, random_state=seed)),
        "GaussianProcess": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(GaussianProcessRegressor(
                kernel=gp_kernel, normalize_y=True, alpha=1e-6, random_state=seed))),
        "kNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
    }
    if HAVE_XGBOOST:
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                        random_state=seed, verbosity=0))
    return models


def sklearn_mse_per_output(model, X, Y):
    P = model.predict(X)
    return ((P - Y) ** 2).mean(axis=0)      # [Qp, Cpo, Pb]


# ============================================================================
# anomaly score: train on healthy, measure residual inflation on ssr
# ============================================================================
def anomaly_ratio(mse_ssr_cpo, mse_healthy_cpo):
    """How many times larger is the degraded-set salinity error than the healthy?
    A strong anomaly detector gives a large ratio."""
    return float(mse_ssr_cpo / max(mse_healthy_cpo, 1e-12))


# ============================================================================
# main comparison
# ============================================================================
def run_comparison(data, runs=3, epochs=3000, out="results_compare"):
    os.makedirs(out, exist_ok=True)

    # fixed split, exactly as the paper: train on healthy sst+sss; test on sin, ssr
    Xtr = np.vstack([to_arrays(data["sst"])[0], to_arrays(data["sss"])[0]])
    Ytr = np.vstack([to_arrays(data["sst"])[1], to_arrays(data["sss"])[1]])
    Xsin, Ysin = to_arrays(data["sin"])     # held-out HEALTHY (accuracy test)
    Xssr, Yssr = to_arrays(data["ssr"])     # held-out DEGRADED (anomaly test)
    icpo = TARGETS.index("Cpo")

    print(f"Train (sst+sss): {len(Xtr)} pts | test-healthy sin: {len(Xsin)} | "
          f"test-degraded ssr: {len(Xssr)}\n")

    results = {}   # name -> dict of metrics
    OUT_LABELS = ["Qp", "Cpo", "Pb"]

    # ---- helper to record one model's metrics from its per-output MSEs ----
    # keeps every output (not just Cpo) for both accuracy and anomaly ratio,
    # so the heatmap has something to show beyond the single Cpo column.
    def record(name, sin_mse, ssr_mse, interpretable, notes=""):
        row = {"interpretable": interpretable, "notes": notes}
        for i, lab in enumerate(OUT_LABELS):
            row[f"sin_{lab}"] = float(sin_mse[i])
            row[f"ssr_{lab}"] = float(ssr_mse[i])
            row[f"anomaly_ratio_{lab}"] = anomaly_ratio(ssr_mse[i], sin_mse[i])
        results[name] = row

    # ======================================================================
    # A + B: the paper's NN (lambda=0) and PINN (lambda=10)
    # ======================================================================
    # physics_target="data" reads Eq. (7) literally (physics loss compared
    # against measurements, not the network's own output) -- this is the
    # setting that actually reproduces the paper's Table 1 (PINN sin Cpo
    # MSE ~44, close to NN's ~46). The other option, physics_target="nn"
    # (this script's old default), lets the physics term distort the network
    # weights and produces a PINN that never matches the paper's numbers --
    # confirmed by directly comparing both against the published Table 1.
    for name, lam, interp in [("A: NN (lambda=0)", 0.0, False),
                              ("B: PINN (lambda=10)", 10.0, True)]:
        sin_runs, ssr_runs = [], []
        for s in range(runs):
            m = train_nn(Xtr, Ytr, lam, seed=s, epochs=epochs, physics_freq="epoch",
                        physics_target="data")
            sin_runs.append(mse_per_output(m, Xsin, Ysin))
            ssr_runs.append(mse_per_output(m, Xssr, Yssr))
        record(name, np.mean(sin_runs, 0), np.mean(ssr_runs, 0), interp,
               "learns beta" if interp else "")
        print(f"  done {name}")

    # ======================================================================
    # C: physics-only inverse fit (no network) — the implicit third model
    # ======================================================================
    # fit parameters on the healthy training pool, then predict via the element model
    cfg = ElementConfig()
    params_c, _ = None, None
    # fit on sst (representative healthy); evaluate physics predictions on sin/ssr
    vals, _loss = fit_physics_only(Xtr, Ytr, cfg=cfg, fixed=("pi0", "n"),
                                   bs_min=0.04, steps=4000)
    def physics_predict(X):
        import torch
        p = {k: torch.tensor(float(v)) for k, v in vals.items()}
        out = element_forward(torch.as_tensor(X[:, 0]), torch.as_tensor(X[:, 1]), p, cfg)
        return np.column_stack([out["Qp"].detach().numpy(),
                                out["Cpo"].detach().numpy(),
                                out["Pb"].detach().numpy()])
    sin_c = ((physics_predict(Xsin) - Ysin) ** 2).mean(0)
    ssr_c = ((physics_predict(Xssr) - Yssr) ** 2).mean(0)
    record("C: Physics-only", sin_c, ssr_c, True, "learns beta; no network")
    print("  done C: Physics-only")

    # ======================================================================
    # D..O: sklearn baselines (+ XGBoost if installed)
    # ======================================================================
    if not HAVE_XGBOOST:
        print("  (xgboost not installed — skipping O: XGBoost; `pip install xgboost` to add it)")
    STOCHASTIC = {"RandomForest", "ExtraTrees", "GradBoost", "AdaBoost", "MLP", "XGBoost"}
    for name, model in make_sklearn_models(seed=0).items():
        sin_runs, ssr_runs = [], []
        # tree ensembles / MLP / boosting have randomness; average a few seeds.
        # deterministic ones (Linear, Ridge, Poly-2, SVR, GP, kNN): 1 pass.
        nseed = runs if name in STOCHASTIC else 1
        for s in range(nseed):
            mdl = make_sklearn_models(seed=s)[name]
            mdl.fit(Xtr, Ytr)
            sin_runs.append(sklearn_mse_per_output(mdl, Xsin, Ysin))
            ssr_runs.append(sklearn_mse_per_output(mdl, Xssr, Yssr))
        record(name, np.mean(sin_runs, 0), np.mean(ssr_runs, 0), False)
        print(f"  done {name}")

    # ---- print the comparison table ----
    print("\n" + "=" * 92)
    print("MODEL COMPARISON  (train: sst+sss healthy;  Cpo = permeate salinity)")
    print("=" * 92)
    print(f"{'Model':22s}{'sin Cpo MSE':>14s}{'ssr Cpo MSE':>16s}"
          f"{'anomaly x':>12s}{'sin Qp MSE':>14s}{'interp?':>9s}")
    print("-" * 92)
    # order: accuracy on healthy sin (lower = better)
    for name in results:
        r = results[name]
        print(f"{name:22s}{r['sin_Cpo']:>14.1f}{r['ssr_Cpo']:>16.1f}"
              f"{r['anomaly_ratio_Cpo']:>12.0f}{r['sin_Qp']:>14.2e}"
              f"{'yes' if r['interpretable'] else 'no':>9s}")
    print("-" * 92)
    print("Reading: low 'sin Cpo MSE' = accurate on healthy held-out data.")
    print("         high 'anomaly x'   = residual spikes on the degraded set (good detector).")
    print("         interp? = does the model yield the physical defect fraction beta.")

    json.dump(results, open(os.path.join(out, "comparison.json"), "w"), indent=2)
    make_figures(results, out)
    make_heatmap(results, out)
    return results


# ============================================================================
# figures
# ============================================================================
def make_figures(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    names = list(results)
    short = [n.split(":")[-1].strip() if ":" in n else n for n in names]
    sin_cpo = [results[n]["sin_Cpo"] for n in names]
    anom = [results[n]["anomaly_ratio_Cpo"] for n in names]
    interp = [results[n]["interpretable"] for n in names]
    colors = ["#2e7d32" if i else "#1565c0" for i in interp]
    # figure width scales with model count so long names never collide
    barw = max(9.0, 0.62 * len(names))

    # (1) accuracy bar — sin Cpo MSE (lower better)
    fig, ax = plt.subplots(figsize=(barw, 5))
    ax.bar(short, sin_cpo, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Held-out healthy (sin) $C_{po}$ MSE  (lower = better)")
    ax.set_title("Predictive accuracy across models — does the neural net earn its complexity?")
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "compare_accuracy.png"), dpi=150)
    plt.close(fig)

    # (2) anomaly ratio bar (higher better) — log scale
    fig, ax = plt.subplots(figsize=(barw, 5))
    ax.bar(short, anom, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Anomaly ratio  (ssr / sin $C_{po}$ MSE)  — higher = better detector")
    ax.set_title("Degradation detection across models — is the PINN needed to flag ssr?")
    ax.set_xticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "compare_anomaly.png"), dpi=150)
    plt.close(fig)

    # (3) the two-axis scatter: accuracy vs interpretability story
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    xs = [results[n]["sin_Cpo"] for n in names]
    ys = [results[n]["anomaly_ratio_Cpo"] for n in names]
    for x, y, i in zip(xs, ys, interp):
        ax.scatter(x, y, s=90, color="#2e7d32" if i else "#1565c0",
                   edgecolor="black", zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("held-out healthy $C_{po}$ MSE  (left = more accurate)")
    ax.set_ylabel("anomaly ratio  (up = better detector)")
    ax.set_title("Accuracy vs anomaly detection\n(green = interpretable / yields $\\beta$; blue = black box)")
    ax.grid(alpha=0.3, which="both")

    # greedy label placement: try offsets in order, skip any that overlap an
    # already-placed label (points cluster tightly in log-space with 15 models)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    candidate_offsets = [(7, 4), (7, -14), (-70, 4), (-70, -14),
                        (7, 22), (-70, 22), (7, -32), (-70, -32)]
    placed = []
    for sn, x, y in zip(short, xs, ys):
        chosen = None
        for dx, dy in candidate_offsets:
            ann = ax.annotate(sn, (x, y), textcoords="offset points",
                              xytext=(dx, dy), fontsize=8)
            bbox = ann.get_window_extent(renderer=renderer)
            if not any(bbox.overlaps(p) for p in placed):
                chosen = bbox
                break
            ann.remove()
        placed.append(chosen if chosen is not None else bbox)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#2e7d32", label="interpretable (physics)"),
                       Patch(color="#1565c0", label="black box")],
              frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "compare_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\nFigures written to {figdir}/")
    print("  compare_accuracy.png   accuracy across models")
    print("  compare_anomaly.png    anomaly-detection strength across models")
    print("  compare_scatter.png    the two-axis trade-off (accuracy vs detection)")


# ============================================================================
# heatmap — every output (Qp, Cpo, Pb) x both metrics (accuracy, anomaly),
# one cell per model x metric, colour = rank within that column (not raw
# value), because MSE and anomaly ratio live on wildly different scales.
# ============================================================================
def make_heatmap(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(out, "figures")
    os.makedirs(figdir, exist_ok=True)

    names = list(results)
    short = [n.split(":")[-1].strip() if ":" in n else n for n in names]

    # (json key, column label, higher_is_better)
    metrics = [
        ("sin_Qp",  "sin Qp\nMSE",  False),
        ("sin_Cpo", "sin Cpo\nMSE", False),
        ("sin_Pb",  "sin Pb\nMSE",  False),
        ("ssr_Qp",  "ssr Qp\nMSE",  False),
        ("ssr_Cpo", "ssr Cpo\nMSE", False),
        ("ssr_Pb",  "ssr Pb\nMSE",  False),
        ("anomaly_ratio_Qp",  "anomaly x\n(Qp)",  True),
        ("anomaly_ratio_Cpo", "anomaly x\n(Cpo)", True),
        ("anomaly_ratio_Pb",  "anomaly x\n(Pb)",  True),
    ]
    raw = np.array([[results[n][k] for k, _, _ in metrics] for n in names])

    # normalise each column independently to [0, 1] with 1 always = "best in
    # this column" (inverted for MSE columns, where lower is better)
    norm = np.zeros_like(raw)
    for j, (_, _, higher_better) in enumerate(metrics):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        span = (hi - lo) or 1.0
        scaled = (col - lo) / span
        norm[:, j] = scaled if higher_better else 1 - scaled

    fig, ax = plt.subplots(figsize=(1.25 * len(metrics) + 1.5, 0.42 * len(names) + 2))
    im = ax.imshow(norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(short, fontsize=8)

    # annotate with the ACTUAL value, not the colour-mapped rank
    for i in range(len(names)):
        for j, (_, _, higher_better) in enumerate(metrics):
            v = raw[i, j]
            txt = f"{v:.0f}x" if higher_better else f"{v:.2g}"
            color = "black" if 0.25 < norm[i, j] < 0.85 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)

    ax.set_title("Full metric comparison  (color = rank within column: green = best, red = worst)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("rank within metric  (1 = best)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "compare_heatmap.png"), dpi=150)
    plt.close(fig)
    print("  compare_heatmap.png    all outputs x accuracy+anomaly, colour-ranked")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data_real")
    ap.add_argument("--runs", type=int, default=3, help="seeds for stochastic models")
    ap.add_argument("--epochs", type=int, default=3000, help="epochs for the NN/PINN")
    ap.add_argument("--out", default="results_compare")
    args = ap.parse_args()
    data = load_datasets(args.data, verbose=True)
    run_comparison(data, runs=args.runs, epochs=args.epochs, out=args.out)


if __name__ == "__main__":
    main()
