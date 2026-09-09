"""
NLR_PLR_LRM_avg_stack_platelet.py
=========================
PRODUCTION FILE — Weighted average of two stacking ensembles.
(Bản rút gọn: chỉ tính AUC + vẽ ROC curve. Đã bỏ UMAP plot và
feature-importance plot để chạy nhanh hơn. Biomarker set có
thêm 3 cột platelet lấy từ blood.csv.)

══════════════════════════════════════════════════════════════
 WHAT IS STACK-A?
══════════════════════════════════════════════════════════════
  Base learners:
    1. RandomForestClassifier  (RF)
       - Bagging ensemble: trains N trees on bootstrap samples
       - Each tree uses random feature subsets (max_features)
       - Final prediction = average of all tree probabilities
       - Strength: low variance, robust to noise, stable on ID

    2. ExtraTreesClassifier    (ET)
       - Like RF but splits are RANDOM (not best-split search)
       - Higher variance than RF → explores more of feature space
       - Decorrelated from RF → adds genuine diversity
       - Strength: generalisation on shifted distributions (OOD/Opa)

  Meta-learner:
    LogisticRegression (via StandardScaler → LR pipeline)
    - Reads 4 columns: [P_RF(0), P_RF(1), P_ET(0), P_ET(1)]
    - Learns optimal LINEAR combination of the two base learner probs
    - Trained on out-of-fold (OOF) predictions → no data leakage

  Stack-A excels at: OOD and Oropharynx (ET diversity helps
  generalise under covariate shift)
  Stack-A weakness:  ID — RF and ET make correlated errors (both
  bagging), meta-LR has limited new signal to learn from

══════════════════════════════════════════════════════════════
 WHAT IS STACK-B?
══════════════════════════════════════════════════════════════
  Base learners:
    1. RandomForestClassifier  (RF)  — same as Stack-A
       Shared anchor for stability across both stacks.

    2. CatBoostClassifier      (CB)
       - Ordered boosting: learns on sequentially ordered subsets
         to avoid target leakage during boosting (unlike vanilla GBDT)
       - Asymmetric trees: can model different feature interactions
         per branch → richer representations than symmetric trees
       - Built-in L2 regularisation (l2_leaf_reg) → less overfitting
         on small oncology cohorts (100-300 patients)
       - Better probability calibration than ET → meta-LR reads
         more reliable probability inputs

  Meta-learner: same StandardScaler → LR pipeline

  Stack-B excels at: ID (CB ordered boosting anchors the dense
  clinical feature space without overfitting); OOD also good
  Stack-B weakness:  slightly less exploration diversity than ET
  for heavily shifted distributions

══════════════════════════════════════════════════════════════
 WHY WEIGHTED AVERAGE WORKS
══════════════════════════════════════════════════════════════
  The two stacks make PARTIALLY INDEPENDENT errors:
    - Stack-A errors: ID correlation between RF and ET
    - Stack-B errors: CB may miss random-feature interactions
      that ET finds

  Averaging two partially independent predictors ALWAYS reduces
  variance without increasing bias (bias-variance decomposition).

  Weights are split-specific (fitted from prior experiments):
    ID:       A=0.35, B=0.65  → lean toward CB stack (ID stability)
    OOD:      A=0.50, B=0.50  → equal (both generalise well OOD)
    Opa:      A=0.50, B=0.50  → equal (both generalise well Opa)

══════════════════════════════════════════════════════════════
 PIPELINE OVERVIEW
══════════════════════════════════════════════════════════════
  Input features → [FIX-2] Biomarker clipping (train quantile)
               → [S2]    Interaction features (NLR×pN, PLR×lymph, …)
               → [FIX-3] Preprocessor (OHE + scaling, fit on train)
               → [S1]    SelectPercentile (per-split %, fit on SMOTE)
               → SMOTE (×5 iterations for variance estimation)
               →  Stack-A (RF+ET→LR)  ──┐
                                          ├─ weighted avg → final P(y=1)
               →  Stack-B (RF+CB→LR)  ──┘

  Output: chỉ gồm
    - roc_testsets_{target}_avg_stack_platelet.svg  (biểu đồ ROC/AUC)
    - auc_summary_{target}_platelet.csv             (bảng AUC theo split)
  (Không còn UMAP plot, không còn feature-importance plot.)
"""

from argparse import ArgumentParser
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from pathlib import Path

from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline


# ── Config ────────────────────────────────────────────────────────────────────
BIOMARKER_COLS      = ["NLR", "PLR", "LMR"]
CLIP_LOWER_QUANTILE = 0.01
CLIP_UPPER_QUANTILE = 0.99
N_SMOTE_ITERATIONS  = 5

PERCENTILE_PER_SPLIT = {
    "In distribution":     80,
    "Out of distribution": 55,
    "Oropharynx":          65,
}

# Weighted average: (weight_A, weight_B)
# Stack-A = RF+ET→LR  |  Stack-B = RF+CB→LR
AVG_WEIGHTS = {
    "In distribution":     (0.35, 0.65),
    "Out of distribution": (0.50, 0.50),
    "Oropharynx":          (0.50, 0.50),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def clip_biomarkers_on_train(X_train_df, X_test_df, cols):
    Xtr, Xte = X_train_df.copy(), X_test_df.copy()
    for col in cols:
        if col not in X_train_df.columns:
            continue
        q_lo = X_train_df[col].quantile(CLIP_LOWER_QUANTILE)
        q_hi = X_train_df[col].quantile(CLIP_UPPER_QUANTILE)
        Xtr[col] = X_train_df[col].clip(lower=q_lo, upper=q_hi)
        Xte[col] = X_test_df[col].clip(lower=q_lo, upper=q_hi)
        print(f"  [CLIP] {col}: [{q_lo:.3f}, {q_hi:.3f}]")
    return Xtr, Xte


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)

    def find_col(keywords):
        for kw in keywords:
            m = [c for c in cols if kw.lower() in str(c).lower()]
            if m: return m[0]
        return None

    nlr = find_col(["NLR", "nlr"]); plr = find_col(["PLR", "plr"])
    lmr = find_col(["LMR", "lmr"]); pn  = find_col(["pN_stage", "pn_stage", "numeric_pN"])
    lym = find_col(["positive_lymph", "lymph_node"])
    pni = find_col(["perineural", "perineural_invasion"])

    if nlr and pn:  df["interact_NLR_x_pNstage"] = df[nlr] * df[pn].fillna(0)
    if plr and lym: df["interact_PLR_x_lymph"]   = df[plr] * df[lym].fillna(0)
    if lmr and pni: df["interact_LMR_x_PNI"]     = df[lmr] * df[pni].fillna(0)
    if nlr and plr: df["interact_NLR_div_PLR"]   = df[nlr] / (df[plr].replace(0, np.nan)).fillna(df[plr].median())
    if nlr and lmr: df["interact_NLR_x_LMR"]     = df[nlr] * df[lmr]
    return df


def _rf_params(target, split_label):
    if target == "recurrence":
        if split_label == "In distribution":
            return dict(n_estimators=1600, max_features="log2", max_depth=30,
                        min_samples_split=2, min_samples_leaf=1,
                        max_leaf_nodes=1000, criterion="gini")
        elif split_label == "Out of distribution":
            return dict(n_estimators=800, max_features="sqrt", max_depth=80,
                        min_samples_split=2, min_samples_leaf=1,
                        max_leaf_nodes=100, criterion="log_loss")
        else:
            return dict(n_estimators=1200, max_features="sqrt", max_depth=20,
                        min_samples_split=2, min_samples_leaf=1,
                        max_leaf_nodes=1000, criterion="gini")
    else:
        return dict(n_estimators=1000, max_features="log2",
                    min_samples_split=5, min_samples_leaf=1,
                    max_leaf_nodes=1000, criterion="entropy")


def _et_params(target, split_label):
    p = _rf_params(target, split_label).copy()
    if p.get("criterion") == "log_loss":
        p["criterion"] = "gini"
    return p


def _cb_params(target, split_label):
    base = dict(eval_metric="AUC", verbose=0,
                allow_writing_files=False, task_type="CPU")
    if target == "recurrence":
        if split_label == "In distribution":
            base.update(iterations=500, depth=6, learning_rate=0.05,
                        l2_leaf_reg=5, border_count=64, bagging_temperature=0.5)
        elif split_label == "Out of distribution":
            base.update(iterations=400, depth=5, learning_rate=0.05,
                        l2_leaf_reg=7, border_count=64, bagging_temperature=1.0)
        else:
            base.update(iterations=400, depth=6, learning_rate=0.05,
                        l2_leaf_reg=5, border_count=64, bagging_temperature=0.5)
    else:
        base.update(iterations=400, depth=6, learning_rate=0.05,
                    l2_leaf_reg=5, border_count=64, bagging_temperature=0.5)
    return base


def _meta():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))])


def _cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


def build_stack_A(target, split_label, rng):
    """Stack-A: RF + ExtraTrees → LogisticRegression"""
    rf = RandomForestClassifier(**_rf_params(target, split_label), random_state=rng, n_jobs=-1)
    et = ExtraTreesClassifier(**_et_params(target, split_label),  random_state=rng, n_jobs=-1)
    return StackingClassifier(estimators=[("rf", rf), ("et", et)],
                              final_estimator=_meta(), cv=_cv(),
                              stack_method="predict_proba", n_jobs=-1, passthrough=False)


def build_stack_B(target, split_label, rng):
    """Stack-B: RF + CatBoost → LogisticRegression"""
    cb_seed = int(rng.randint(0, 2**16))
    rf = RandomForestClassifier(**_rf_params(target, split_label), random_state=rng, n_jobs=-1)
    cb = CatBoostClassifier(**_cb_params(target, split_label), random_seed=cb_seed)
    return StackingClassifier(estimators=[("rf", rf), ("cb", cb)],
                              final_estimator=_meta(), cv=_cv(),
                              stack_method="predict_proba", n_jobs=-1, passthrough=False)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory")
    parser.add_argument("features_directory")
    parser.add_argument("results_directory")
    parser.add_argument("target", choices=["recurrence", "survival_status"])
    args = parser.parse_args()

    data_dir    = Path(args.features_directory)
    results_dir = Path(args.results_directory)
    results_dir.mkdir(parents=True, exist_ok=True)
    split_dir   = Path(args.datasplit_directory)
    target      = args.target
    rng         = np.random.RandomState(42)

    # [FIX-1] Load — biomarkers file gồm NLR/PLR/LMR + 3 cột platelet
    clinical     = pd.read_csv(data_dir/"clinical.csv",     dtype={"patient_id": str})
    patho        = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    icd          = pd.read_csv(data_dir/"icd_codes.csv",    dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})
    biomarkers   = pd.read_csv(data_dir/"calculated_biomarkers_test.csv", dtype={"patient_id": str})

    if "recurrence" in biomarkers.columns:
        biomarkers = biomarkers.drop(columns=["recurrence"])

    df = (clinical.merge(patho, on="patient_id", how="outer")
          .merge(icd,          on="patient_id", how="outer")
          .merge(cell_density, on="patient_id", how="outer")
          .merge(biomarkers,   on="patient_id", how="outer")
          .reset_index(drop=True))
    print(f"Merged: {df.shape[0]} patients, {df.shape[1]} columns")

    x_linspace = np.linspace(0, 1, 100)
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"

    data_split_paths  = [split_dir/"dataset_split_in.json",
                         split_dir/"dataset_split_out.json",
                         split_dir/"dataset_split_Oropharynx.json"]
    data_split_labels = ["In distribution", "Out of distribution", "Oropharynx"]

    tpr_list = [[] for _ in range(3)]
    auc_list = [[] for _ in range(3)]

    for i, (split_path, split_label) in enumerate(zip(data_split_paths, data_split_labels)):
        print(f"\n{'='*65}\nSPLIT: {split_label}\n{'='*65}")
        assert os.path.exists(split_path)

        df_split   = pd.read_json(split_path, dtype={"patient_id": str})[["patient_id","dataset"]]
        df_targets = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
        df_split   = df_split.merge(df_targets, on="patient_id", how="inner")

        if target == "recurrence":
            df_split = df_split[
                ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
                ((df_split.recurrence == "no")  & ((df_split.days_to_last_information > 365*3) |
                                                    (df_split.survival_status == "living")))]
            df_split = df_split.copy()
            df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})
        else:
            df_split = df_split[
                ~(df_split.survival_status_with_cause == "deceased not tumor specific")]
            df_split = df_split.copy()
            df_split.survival_status = df_split.survival_status.replace(
                {"living": 0, "deceased": 1})

        df_train = (df_split[df_split.dataset=="training"][["patient_id", target]]
                    .copy().rename(columns={target: "target"})
                    .merge(df, on="patient_id", how="inner"))
        df_test  = (df_split[df_split.dataset=="test"][["patient_id", target]]
                    .copy().rename(columns={target: "target"})
                    .merge(df, on="patient_id", how="inner"))

        print(f"  Train: {len(df_train)} | {df_train['target'].value_counts().to_dict()}")
        print(f"  Test : {len(df_test)}  | {df_test['target'].value_counts().to_dict()}")

        X_tr_raw = df_train.drop(["patient_id","target"], axis=1)
        X_te_raw = df_test.drop(["patient_id","target"],  axis=1)
        X_tr, X_te = clip_biomarkers_on_train(X_tr_raw, X_te_raw, BIOMARKER_COLS)

        X_tr = add_interaction_features(X_tr)
        X_te = add_interaction_features(X_te)
        print(f"  [S2] Interactions: {[c for c in X_tr.columns if c.startswith('interact_')]}")

        preprocessor = setup_preprocessing_pipeline(X_tr.columns)
        X_tr_proc    = preprocessor.fit_transform(X_tr)
        X_te_proc    = preprocessor.transform(X_te)

        y_train = df_train["target"].to_numpy()
        y_test  = df_test["target"].to_numpy()

        pct    = PERCENTILE_PER_SPLIT[split_label]
        w_a, w_b = AVG_WEIGHTS[split_label]
        print(f"  [S1] Percentile={pct}%  |  Weights A={w_a}, B={w_b}")

        for iteration in range(N_SMOTE_ITERATIONS):
            print(f"\n  Iter {iteration+1}/{N_SMOTE_ITERATIONS}")

            smote = SMOTE(random_state=rng)
            X_sm, y_sm = smote.fit_resample(X_tr_proc, y_train)

            sel = SelectPercentile(score_func=f_classif, percentile=pct)
            X_sm_sel = sel.fit_transform(X_sm, y_sm)
            X_te_sel = sel.transform(X_te_proc)
            print(f"    [S1] {X_sm_sel.shape[1]}/{X_sm.shape[1]} features")

            # Stack-A: RF + ET → LR
            sa = build_stack_A(target, split_label, rng)
            sa.fit(X_sm_sel, y_sm)
            p_a = sa.predict_proba(X_te_sel)[:, 1]

            # Stack-B: RF + CB → LR
            sb = build_stack_B(target, split_label, rng)
            sb.fit(X_sm_sel, y_sm)
            p_b = sb.predict_proba(X_te_sel)[:, 1]

            # Weighted average
            y_pred = w_a * p_a + w_b * p_b

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            tpr_i = np.interp(x_linspace, fpr, tpr)
            tpr_i[0] = 0.0; tpr_i[-1] = 1.0
            tpr_list[i].append(tpr_i)
            auc_score = roc_auc_score(y_test, y_pred)
            auc_list[i].append(auc_score)
            auc_a = roc_auc_score(y_test, p_a)
            auc_b = roc_auc_score(y_test, p_b)
            print(f"    AUC: avg={auc_score:.4f}  A={auc_a:.4f}  B={auc_b:.4f}")

        print(f"\n  [{split_label}] Mean AUC = "
              f"{np.mean(auc_list[i]):.4f} ± {np.std(auc_list[i]):.4f}")

    # ── ROC (biểu đồ AUC) ──────────────────────────────────────────────────
    colors = sns.color_palette("Set2")
    plt.figure(figsize=(2.2, 1.75))
    for i in range(3):
        mean_tpr = np.mean(tpr_list[i], axis=0)
        std_tpr  = np.std(tpr_list[i],  axis=0)
        plt.plot(x_linspace, mean_tpr, linewidth=0.8, color=colors[i],
                 label=f"AUC={np.mean(auc_list[i]):.2f}±{np.std(auc_list[i]):.2f}")
        plt.fill_between(x_linspace,
                         np.maximum(mean_tpr - std_tpr, 0),
                         np.minimum(mean_tpr + std_tpr, 1),
                         color=colors[i], alpha=0.3, lw=0)
    plt.plot([0,1],[0,1],"--",color="black",linewidth=1,label="Random")
    plt.xticks(np.arange(0,1.2,0.5)); plt.yticks(np.arange(0,1.2,0.5))
    plt.xlabel("FPR", fontsize=6); plt.ylabel("TPR", fontsize=6)
    plt.title(f"{target} — Avg(Stack-A RF+ET + Stack-B RF+CB) [+platelet]")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5), fontsize=5)
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    roc_fname = f"roc_testsets_{target}_avg_stack_platelet.svg"
    plt.savefig(results_dir/roc_fname, bbox_inches="tight")
    plt.close()
    print(f"\n[PLOT] {roc_fname}")

    # ── Summary + lưu AUC ra CSV ───────────────────────────────────────────
    print(f"\n{'='*65}\nFINAL RESULTS SUMMARY\n{'='*65}")
    baselines = {"In distribution": 0.79, "Out of distribution": 0.71, "Oropharynx": 0.69}
    auc_summary = []
    for i, label in enumerate(data_split_labels):
        m = np.mean(auc_list[i]); s = np.std(auc_list[i])
        b = baselines[label]; d = m - b
        print(f"  {label:<24}: {m:.4f} ± {s:.4f}  "
              f"({'↑' if d>0 else '↓'}{abs(d):.4f} vs baseline {b:.2f})")
        auc_summary.append({
            "split": label,
            "mean_auc": m,
            "std_auc": s,
            "baseline": b,
            "delta_vs_baseline": d,
        })

    auc_fname = f"auc_summary_{target}_platelet.csv"
    pd.DataFrame(auc_summary).to_csv(results_dir / auc_fname, index=False)
    print(f"[SAVE] {auc_fname}")

    print(f"\nDone. Results → {results_dir}")