"""
Nested repeated-resampling pipeline for small-n moire classification.

For iteration i = 1 ... N:
  1. Seed = i. Stratified random split of all 42 samples -> 30 train / 12 test.
  2. Inner 5-fold CV on the 30 training samples: grid search over param_grid,
     mean validation loss/accuracy/AUC per combination, select best combination.
  3. Refit best combination on the FULL 30 training samples.
  4. Evaluate on training set (30) and held-out test set (12).
  5. Save per-sample test predictions.

Outputs:
  - iteration_results.csv       (one row per outer iteration: train/val/test metrics + best params)
  - test_predictions.csv        (one row per test sample per iteration)
  - fig_boxplot_metrics.png     (train/val/test distributions, 3 subplots: loss, accuracy, AUC)
  - fig_running_mean_auc.png    (cumulative mean test AUC vs iteration count)
  - fig_hyperparam_freq.png     (frequency each hyperparameter combination was selected as best)
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ------------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------------
N_ITERATIONS = 300
N_TEST = 12
INNER_FOLDS = 5

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.05, 0.1],
    "gamma": [0, 0.1, 0.3],
    "reg_lambda": [1, 5, 10],
}

FIXED_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    subsample=1.0,
    colsample_bytree=1.0,
    verbosity=0,
)

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
dataset_path = "processed_dataset.csv"
df = pd.read_csv(dataset_path)

feature_cols = [
    col for col in df.columns
    if (col.startswith("Edge_") or col.startswith("Wrinkle_"))
]

X_all = df[feature_cols].reset_index(drop=True)
y_all = df["Label"].astype(int).reset_index(drop=True)
sample_ids_all = df["Sample_ID"].reset_index(drop=True)

print(f"Loaded {len(df)} samples, {len(feature_cols)} features.")
print(f"Class balance: {y_all.value_counts().to_dict()}")

# All hyperparameter combinations (Cartesian product)
param_keys = list(param_grid.keys())
param_combos = list(itertools.product(*param_grid.values()))
print(f"Hyperparameter grid: {len(param_combos)} combinations.")


def combo_to_dict(combo):
    return dict(zip(param_keys, combo))


def combo_to_label(combo):
    d = combo_to_dict(combo)
    return f"n{d['n_estimators']}_d{d['max_depth']}_lr{d['learning_rate']}_g{d['gamma']}_rl{d['reg_lambda']}"


def safe_auc(y_true, y_prob):
    # AUC is undefined if only one class is present
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def evaluate_combo_inner_cv(X_tr, y_tr, combo, seed):
    """Run INNER_FOLDS-fold CV for one hyperparameter combination.
    Returns mean (logloss, accuracy, auc) across inner folds."""
    params = {**FIXED_PARAMS, **combo_to_dict(combo), "random_state": seed}
    inner_kfold = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)

    losses, accs, aucs = [], [], []
    for tr_idx, va_idx in inner_kfold.split(X_tr, y_tr):
        X_f_tr, X_f_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
        y_f_tr, y_f_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_f_tr, y_f_tr)

        y_prob = model.predict_proba(X_f_va)[:, 1]
        y_pred = model.predict(X_f_va)

        # clip probs to avoid log_loss(0) blowing up on tiny folds
        y_prob_clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)

        losses.append(log_loss(y_f_va, y_prob_clipped, labels=[0, 1]))
        accs.append(accuracy_score(y_f_va, y_pred))
        aucs.append(safe_auc(y_f_va, y_prob))

    return np.mean(losses), np.mean(accs), np.nanmean(aucs)


# ------------------------------------------------------------------
# 2. Outer loop
# ------------------------------------------------------------------
iteration_rows = []
test_prediction_rows = []

for i in range(1, N_ITERATIONS + 1):
    seed = i

    # --- Step 1: stratified 30/12 split of ALL 42 samples ---
    train_idx, test_idx = train_test_split(
        np.arange(len(X_all)),
        test_size=N_TEST,
        # Class balance in training and test sets.
        stratify=y_all,
        random_state=seed,
    )

    X_train, X_test = X_all.iloc[train_idx].reset_index(drop=True), X_all.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y_all.iloc[train_idx].reset_index(drop=True), y_all.iloc[test_idx].reset_index(drop=True)
    ids_test = sample_ids_all.iloc[test_idx].reset_index(drop=True)

    # --- Step 2: inner CV grid search ---
    best_combo = None
    best_val_auc = -np.inf
    best_val_loss = None
    best_val_acc = None

    for combo in param_combos:
        val_loss, val_acc, val_auc = evaluate_combo_inner_cv(X_train, y_train, combo, seed)
        # selection criterion: highest mean inner-CV AUC
        # (fallback to lowest loss if AUC ties/undefined across the board)
        score = val_auc if not np.isnan(val_auc) else -val_loss
        if best_combo is None or score > (best_val_auc if not np.isnan(best_val_auc) else -best_val_loss):
            best_combo = combo
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_val_acc = val_acc

    best_params = {**FIXED_PARAMS, **combo_to_dict(best_combo), "random_state": seed}

    # --- Step 3: refit on FULL 30 training samples ---
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # --- Step 4a: evaluate on training set ---
    train_prob = final_model.predict_proba(X_train)[:, 1]
    train_pred = final_model.predict(X_train)
    train_prob_clipped = np.clip(train_prob, 1e-6, 1 - 1e-6)

    train_loss = log_loss(y_train, train_prob_clipped, labels=[0, 1])
    train_acc = accuracy_score(y_train, train_pred)
    train_auc = safe_auc(y_train, train_prob)

    # --- Step 4b: evaluate on held-out test set ---
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = final_model.predict(X_test)
    test_prob_clipped = np.clip(test_prob, 1e-6, 1 - 1e-6)

    test_loss = log_loss(y_test, test_prob_clipped, labels=[0, 1])
    test_acc = accuracy_score(y_test, test_pred)
    test_auc = safe_auc(y_test, test_prob)

    # --- Step 5: save predictions ---
    for k in range(len(X_test)):
        test_prediction_rows.append({
            "iteration": i,
            "sample_id": ids_test.iloc[k],
            "true_label": int(y_test.iloc[k]),
            "pred_prob": float(test_prob[k]),
            "pred_label": int(test_pred[k]),
            "correct": bool(test_pred[k] == y_test.iloc[k]),
        })

    iteration_rows.append({
        "iteration": i,
        "train_loss": train_loss, "train_acc": train_acc, "train_auc": train_auc,
        "val_loss": best_val_loss, "val_acc": best_val_acc, "val_auc": best_val_auc,
        "test_loss": test_loss, "test_acc": test_acc, "test_auc": test_auc,
        "best_params_label": combo_to_label(best_combo),
        "best_params_json": json.dumps(combo_to_dict(best_combo)),
    })

    if i % 10 == 0 or i == 1:
        print(f"Iteration {i}/{N_ITERATIONS}: test_auc={test_auc:.3f}, "
              f"val_auc={best_val_auc:.3f}, best={combo_to_label(best_combo)}")

results_df = pd.DataFrame(iteration_rows)
predictions_df = pd.DataFrame(test_prediction_rows)

results_df.to_csv("iteration_results.csv", index=False)
predictions_df.to_csv("test_predictions.csv", index=False)

print("\nSaved iteration_results.csv and test_predictions.csv")
print(results_df[["train_auc", "val_auc", "test_auc"]].describe())

# ------------------------------------------------------------------
# 3. FIGURE 1 — Box plot: train / val / test distributions (loss, acc, AUC)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

metric_specs = [
    ("loss", "Log Loss"),
    ("acc", "Accuracy"),
    ("auc", "ROC-AUC"),
]

for ax, (metric_key, metric_label) in zip(axes, metric_specs):
    data = [
        results_df[f"train_{metric_key}"].dropna(),
        results_df[f"val_{metric_key}"].dropna(),
        results_df[f"test_{metric_key}"].dropna(),
    ]
    bp = ax.boxplot(data, labels=["Train", "Validation", "Test"], patch_artist=True, widths=0.5)
    colors = ["#2ca02c", "#1f77b4", "#d62728"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
    for element in ["whiskers", "caps", "medians"]:
        for line, color in zip(bp[element], colors * 2):
            line.set_color(color if element == "medians" else "#555555")
    ax.set_title(metric_label)
    ax.set_ylabel(metric_label)

fig.suptitle(f"Train / Validation / Test Metric Distributions ({N_ITERATIONS} Iterations)", fontsize=13)
fig.tight_layout()
fig.savefig("fig_boxplot_metrics.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# 4. FIGURE 2 — Running-mean convergence of test AUC
# ------------------------------------------------------------------
test_auc_series = results_df["test_auc"].values
running_mean = np.cumsum(np.nan_to_num(test_auc_series, nan=np.nanmean(test_auc_series))) / np.arange(1, len(test_auc_series) + 1)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(np.arange(1, len(running_mean) + 1), running_mean, color="#2a78d6", linewidth=2)
ax.axhline(running_mean[-1], color="#898781", linestyle="--", linewidth=1,
           label=f"Final mean = {running_mean[-1]:.3f}")
ax.set_xlabel("Number of iterations included")
ax.set_ylabel("Cumulative mean test AUC")
ax.set_title("Running-Mean Convergence of Test AUC")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig("fig_running_mean_auc.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------------
# 5. FIGURE 3 — Hyperparameter selection frequency
# ------------------------------------------------------------------
freq = results_df["best_params_label"].value_counts()

fig, ax = plt.subplots(figsize=(7, max(3.5, 0.35 * len(freq))))
ax.barh(freq.index[::-1], freq.values[::-1], color="#2a78d6")
ax.set_xlabel(f"Number of iterations selected as best (of {N_ITERATIONS})")
ax.set_title("Hyperparameter Combination Selection Frequency")
fig.tight_layout()
fig.savefig("fig_hyperparam_freq.png", bbox_inches="tight")
plt.close(fig)

print("\nSaved figures: fig_boxplot_metrics.png, fig_running_mean_auc.png, fig_hyperparam_freq.png")