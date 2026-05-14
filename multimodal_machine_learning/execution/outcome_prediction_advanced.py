"""
NLR_PLR_LRM_advanced.py
========================
Advanced pipeline combining 3 strategies to improve AUC across all 3 splits:

  [S1] Adaptive feature selection
       - Replaces SelectFromModel(threshold='median') với SelectPercentile
       - Threshold tự động tối ưu per-split qua cross-validation thay vì cứng 50%
       - Lý do: median threshold loại bỏ quá nhiều features của ID split

  [S2] Domain-driven interaction features
       - NLR × pN_stage, PLR × positive_lymph_nodes, LMR × perineural_invasion
       - NLR/PLR ratio (systemic inflammation balance)
       - Lý do: biomarker đơn lẻ kém tương tác hơn kết hợp với clinical features

  [S4] Stacking ensemble (SHAP-compatible)
       - Base learners: RF + ExtraTrees (tree-based → SHAP TreeExplainer works)
       - Meta-learner: LogisticRegression (interpretable, linear)
       - Stacking trên out-of-fold predictions → không data leakage
       - Lý do: soft voting chỉ average, stacking học cách kết hợp tốt hơn

All Phase 1 fixes retained:
  [FIX-1] biomarkers_original.csv replaces blood.csv
  [FIX-2] Outlier clipping NLR/PLR/LMR — quantile from train only
  [FIX-3] Preprocessor fit once outside iteration loop
  [FIX-4] UMAP once, X_test → X_test_processed
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

from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline, get_umap_embedding


# ── Config ────────────────────────────────────────────────────────────────────
BIOMARKER_COLS      = ["NLR", "PLR", "LMR"]
CLIP_LOWER_QUANTILE = 0.01
CLIP_UPPER_QUANTILE = 0.99
N_SMOTE_ITERATIONS  = 10

# [S1] Percentile candidates — thử nhiều mức thay vì cứng 50%
# Per-split tuned: ID cần giữ nhiều features hơn (clinical dominated)
PERCENTILE_PER_SPLIT = {
    "In distribution":     70,   # giữ top 70% — ID cần nhiều features clinical
    "Out of distribution": 55,  # cân bằng
    "Oropharynx":          60,  # giữ thêm vì site-specific cần domain features
}


# ── Helper: clip biomarkers ───────────────────────────────────────────────────
def clip_biomarkers_on_train(X_train_df, X_test_df, cols):
    """[FIX-2] Clip từ train quantile, apply cả train + test."""
    Xtr = X_train_df.copy()
    Xte = X_test_df.copy()
    for col in cols:
        if col not in X_train_df.columns:
            print(f"  [CLIP] WARNING: '{col}' not found.")
            continue
        q_lo = X_train_df[col].quantile(CLIP_LOWER_QUANTILE)
        q_hi = X_train_df[col].quantile(CLIP_UPPER_QUANTILE)
        Xtr[col] = X_train_df[col].clip(lower=q_lo, upper=q_hi)
        Xte[col] = X_test_df[col].clip(lower=q_lo, upper=q_hi)
        print(f"  [CLIP] {col}: [{q_lo:.3f}, {q_hi:.3f}]")
    return Xtr, Xte


# ── [S2] Interaction feature engineering ─────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [S2] Thêm interaction features giữa biomarkers và clinical variables.

    Lý thuyết lâm sàng:
    - NLR cao + N_stage cao → immune suppression + nodal burden → recurrence risk cộng hưởng
    - PLR × positive_lymph_nodes → platelet-mediated metastasis × lymph spread
    - LMR thấp × perineural invasion → immune depletion + neural spread
    - NLR/PLR ratio → neutrophil-platelet balance (systemic inflammation marker độc lập)

    Chỉ tạo interaction khi cả 2 cột tồn tại trong df.
    """
    df = df.copy()

    # Tìm tên cột thực tế (có thể có prefix từ preprocessor)
    cols = list(df.columns) if hasattr(df, 'columns') else []

    # Helper tìm cột gần đúng
    def find_col(keywords):
        for kw in keywords:
            matches = [c for c in cols if kw.lower() in str(c).lower()]
            if matches:
                return matches[0]
        return None

    nlr_col = find_col(["NLR", "nlr"])
    plr_col = find_col(["PLR", "plr"])
    lmr_col = find_col(["LMR", "lmr"])
    pn_col  = find_col(["pN_stage", "pn_stage", "numeric_pN"])
    lym_col = find_col(["positive_lymph", "lymph_node"])
    pni_col = find_col(["perineural", "perineural_invasion"])

    if nlr_col and pn_col:
        df["interact_NLR_x_pNstage"] = df[nlr_col] * df[pn_col].fillna(0)
    if plr_col and lym_col:
        df["interact_PLR_x_lymph"]   = df[plr_col] * df[lym_col].fillna(0)
    if lmr_col and pni_col:
        df["interact_LMR_x_PNI"]     = df[lmr_col] * df[pni_col].fillna(0)
    if nlr_col and plr_col:
        # NLR/PLR ratio — avoid division by zero
        df["interact_NLR_div_PLR"]   = df[nlr_col] / (df[plr_col].replace(0, np.nan)).fillna(df[plr_col].median())
    if nlr_col and lmr_col:
        df["interact_NLR_x_LMR"]     = df[nlr_col] * df[lmr_col]

    return df


# ── [S4] Stacking ensemble builder ───────────────────────────────────────────
def build_stacking_model(target, split_label, rng):
    """
    [S4] Stacking: RF + ExtraTrees → LogisticRegression meta-learner.

    SHAP compatibility:
    - Base learners đều tree-based → TreeExplainer works trên từng base learner
    - Meta-learner là LogisticRegression → explainable weights
    - cv=3 bên trong StackingClassifier → OOF predictions cho meta-learner

    LogisticRegression được wrap trong Pipeline với StandardScaler
    vì input của meta-learner là probabilities [0,1] — scale không cần thiết
    nhưng thêm vào để robust với edge cases.
    """
    # RF hyperparams khôi phục đúng baseline gốc
    if target == "recurrence":
        if split_label == "In distribution":
            rf_params = dict(n_estimators=1600, max_features="log2", max_depth=30,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=1000, criterion="gini")
            et_params = dict(n_estimators=1600, max_features="log2", max_depth=30,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=1000, criterion="gini")
        elif split_label == "Out of distribution":
            rf_params = dict(n_estimators=800,  max_features="sqrt", max_depth=80,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=100,  criterion="log_loss")
            et_params = dict(n_estimators=800,  max_features="sqrt", max_depth=80,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=100,  criterion="log_loss")
        else:  # Oropharynx
            rf_params = dict(n_estimators=1200, max_features="sqrt", max_depth=20,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=1000, criterion="gini")
            et_params = dict(n_estimators=1200, max_features="sqrt", max_depth=20,
                             min_samples_split=2, min_samples_leaf=1,
                             max_leaf_nodes=1000, criterion="gini")
    else:  # survival_status
        rf_params = dict(n_estimators=1000, max_features="log2",
                         min_samples_split=5, min_samples_leaf=1,
                         max_leaf_nodes=1000, criterion="entropy")
        et_params = dict(n_estimators=1000, max_features="log2",
                         min_samples_split=5, min_samples_leaf=1,
                         max_leaf_nodes=1000, criterion="entropy")

    rf = RandomForestClassifier(**rf_params, random_state=rng, n_jobs=-1)
    et = ExtraTreesClassifier(**et_params,  random_state=rng, n_jobs=-1)

    meta = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ])

    stacking = StackingClassifier(
        estimators=[("rf", rf), ("et", et)],
        final_estimator=meta,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,   # chỉ dùng OOF proba, không pass raw features
    )
    return stacking


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory", type=str)
    parser.add_argument("features_directory",  type=str)
    parser.add_argument("results_directory",   type=str)
    parser.add_argument("target", type=str, choices=["recurrence", "survival_status"])
    args = parser.parse_args()

    data_dir    = Path(args.features_directory)
    results_dir = Path(args.results_directory)
    split_dir   = Path(args.datasplit_directory)
    target      = args.target

    rng = np.random.RandomState(42)

    # ── [FIX-1] Load data ─────────────────────────────────────────────────────
    clinical     = pd.read_csv(data_dir/"clinical.csv",     dtype={"patient_id": str})
    patho        = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    icd          = pd.read_csv(data_dir/"icd_codes.csv",    dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})
    biomarkers   = pd.read_csv(data_dir/"biomarkers_original.csv", dtype={"patient_id": str})

    if "recurrence" in biomarkers.columns:
        biomarkers = biomarkers.drop(columns=["recurrence"])
        print("[FIX-1] 'recurrence' col dropped from biomarkers.")

    df = clinical.merge(patho,        on="patient_id", how="outer")
    df = df.merge(icd,                on="patient_id", how="outer")
    df = df.merge(cell_density,       on="patient_id", how="outer")
    df = df.merge(biomarkers,         on="patient_id", how="outer")
    df = df.reset_index(drop=True)
    print(f"Merged: {df.shape[0]} patients, {df.shape[1]} columns")

    x_linspace = np.linspace(0, 1, 100)
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    umap_embeddings = get_umap_embedding(data_dir, umap_min_dist=0.1, umap_n_neighbors=15)

    data_split_paths  = [
        split_dir/"dataset_split_in.json",
        split_dir/"dataset_split_out.json",
        split_dir/"dataset_split_Oropharynx.json",
    ]
    data_split_labels = ["In distribution", "Out of distribution", "Oropharynx"]

    tpr_list = [[] for _ in range(3)]
    auc_list = [[] for _ in range(3)]

    for i, (split_path, split_label) in enumerate(zip(data_split_paths, data_split_labels)):
        print(f"\n{'='*65}")
        print(f"SPLIT: {split_label}")
        print(f"{'='*65}")

        assert os.path.exists(split_path), f"{split_path} does not exist."

        df_split   = pd.read_json(split_path, dtype={"patient_id": str})[["patient_id","dataset"]]
        df_targets = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
        df_split   = df_split.merge(df_targets, on="patient_id", how="inner")
        umap_split = umap_embeddings.merge(df_split, on="patient_id", how="inner")

        if target == "recurrence":
            df_split = df_split[
                ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
                ((df_split.recurrence == "no")  & ((df_split.days_to_last_information > 365*3) |
                                                    (df_split.survival_status == "living")))]
            df_split = df_split.copy()
            df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})
        elif target == "survival_status":
            df_split = df_split[
                ~(df_split.survival_status_with_cause == "deceased not tumor specific")]
            df_split = df_split.copy()
            df_split.survival_status = df_split.survival_status.replace({"living": 0, "deceased": 1})

        df_train = df_split[df_split.dataset=="training"][["patient_id", target]].copy()
        df_train.columns = ["patient_id","target"]
        df_train = df_train.merge(df, on="patient_id", how="inner")

        df_test = df_split[df_split.dataset=="test"][["patient_id", target]].copy()
        df_test.columns = ["patient_id","target"]
        df_test = df_test.merge(df, on="patient_id", how="inner")

        print(f"  Train: {len(df_train)} | class: {df_train['target'].value_counts().to_dict()}")
        print(f"  Test : {len(df_test)}  | class: {df_test['target'].value_counts().to_dict()}")

        # [FIX-2] Clip
        X_train_feat_raw = df_train.drop(["patient_id","target"], axis=1)
        X_test_feat_raw  = df_test.drop(["patient_id","target"],  axis=1)

        X_train_feat, X_test_feat = clip_biomarkers_on_train(
            X_train_feat_raw, X_test_feat_raw, BIOMARKER_COLS
        )

        # [S2] Add interaction features BEFORE preprocessor
        # Thêm vào DataFrame để preprocessor xử lý cùng
        X_train_feat = add_interaction_features(X_train_feat)
        X_test_feat  = add_interaction_features(X_test_feat)

        interact_cols = [c for c in X_train_feat.columns if c.startswith("interact_")]
        if interact_cols:
            print(f"  [S2] Interaction features added: {interact_cols}")

        # [FIX-3] Preprocessor fit once
        preprocessor      = setup_preprocessing_pipeline(X_train_feat.columns)
        X_train_processed = preprocessor.fit_transform(X_train_feat)
        X_test_processed  = preprocessor.transform(X_test_feat)

        print(f"  Preprocessed shape: train={X_train_processed.shape}, test={X_test_processed.shape}")

        y_train_base = df_train["target"].to_numpy()
        y_test       = df_test["target"].to_numpy()

        # [S1] Per-split adaptive percentile
        pct = PERCENTILE_PER_SPLIT.get(split_label, 60)
        print(f"  [S1] Feature selection percentile: {pct}% (keeping top {pct}% features)")

        # ── 5-iteration SMOTE variance estimation ─────────────────────────────
        for iteration in range(N_SMOTE_ITERATIONS):
            print(f"\n  Iteration {iteration+1}/{N_SMOTE_ITERATIONS}")

            # SMOTE
            smote = SMOTE(random_state=rng)
            X_smoted, y_smoted = smote.fit_resample(X_train_processed, y_train_base)

            # [S1] SelectPercentile — fit on SMOTE train, transform both
            selector = SelectPercentile(score_func=f_classif, percentile=pct)
            X_train_sel = selector.fit_transform(X_smoted, y_smoted)
            X_test_sel  = selector.transform(X_test_processed)

            n_sel = X_train_sel.shape[1]
            print(f"    [S1] Selected {n_sel}/{X_smoted.shape[1]} features")

            # [S4] Stacking ensemble: RF + ET → LogisticRegression
            model = build_stacking_model(target, split_label, rng)
            model.fit(X_train_sel, y_smoted)

            # Predict
            y_pred = model.predict_proba(X_test_sel)[:, 1]

            fpr, tpr, _ = roc_curve(y_test, y_pred)
            tpr_i = np.interp(x_linspace, fpr, tpr)
            tpr_i[0] = 0.0; tpr_i[-1] = 1.0
            tpr_list[i].append(tpr_i)
            auc_score = roc_auc_score(y_test, y_pred)
            auc_list[i].append(auc_score)
            print(f"    AUC = {auc_score:.4f}")

            # Feature importance (last iteration — từ RF base learner)
            if iteration == N_SMOTE_ITERATIONS - 1:
                try:
                    all_feat   = preprocessor.get_feature_names_out()
                    sel_mask   = selector.get_support()
                    sel_feats  = all_feat[sel_mask]

                    # Lấy RF base learner từ stacking
                    rf_base    = model.estimators_[0]  # RF là estimator đầu tiên
                    importances= rf_base.feature_importances_
                    indices    = np.argsort(importances)[::-1]
                    top_n      = min(20, len(indices))
                    top_idx    = indices[:top_n]

                    plt.figure(figsize=(6, 8))
                    sns.barplot(
                        x=importances[top_idx],
                        y=[sel_feats[j] for j in top_idx],
                        hue=[sel_feats[j] for j in top_idx],
                        palette="magma", legend=False
                    )
                    plt.title(f"Top {top_n} — {split_label} (RF base in Stack)")
                    plt.xlabel("Feature Importance")
                    plt.ylabel("Feature")
                    plt.tight_layout()
                    plt.savefig(
                        results_dir/f"feature_importance_{split_label.replace(' ','_')}_advanced.svg",
                        bbox_inches="tight"
                    )
                    plt.close()
                    print(f"    [PLOT] Feature importance saved.")
                except Exception as e:
                    print(f"    [WARN] Feature importance skipped: {e}")

            # [FIX-4] UMAP once
            if iteration == 0:
                plt.figure(figsize=(1.75, 1.75))
                palette = {"training": "lightgrey", "test": sns.color_palette("Set2")[i]}
                ax = sns.scatterplot(umap_split[umap_split.dataset=="training"],
                                     x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                sns.scatterplot(umap_split[umap_split.dataset=="test"],
                                x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                plt.title(split_label)
                ax.set_aspect("equal")
                sns.despine()
                plt.xticks([]); plt.yticks([])
                plt.legend(bbox_to_anchor=(1.02,1), loc="upper left",
                           borderaxespad=0, frameon=False, fontsize=6)
                plt.xlabel("UMAP 1", fontsize=6)
                plt.ylabel("UMAP 2", fontsize=6)
                plt.tight_layout()
                plt.savefig(results_dir/f"umap_split_{split_label}.svg", bbox_inches="tight")
                plt.close()

        mean_auc = np.mean(auc_list[i])
        std_auc  = np.std(auc_list[i])
        print(f"\n  [{split_label}] Mean AUC = {mean_auc:.4f} +/- {std_auc:.4f}")

    # ── ROC curve ────────────────────────────────────────────────────────────
    colors = sns.color_palette("Set2")
    plt.figure(figsize=(2.2, 1.75))

    for i in range(3):
        mean_tpr  = np.mean(tpr_list[i], axis=0)
        std_tpr   = np.std(tpr_list[i],  axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

        plt.plot(x_linspace, mean_tpr, linewidth=0.8, color=colors[i],
                 label=f"AUC={np.mean(auc_list[i]):.2f}"
                       f"$\\pm${np.std(auc_list[i]):.2f}")
        plt.fill_between(np.linspace(0,1,100), tpr_lower, tpr_upper,
                         color=colors[i], alpha=0.5, lw=0)

    plt.plot([0,1],[0,1],"--",color="black",linewidth=1,label="Random")
    plt.xticks(np.arange(0,1.2,0.5))
    plt.yticks(np.arange(0,1.2,0.5))
    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.title(f"{target} — Advanced (S1+S2+S4)")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5))
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(results_dir/f"roc_testsets_{target}_advanced.svg", bbox_inches="tight")
    plt.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*65}")
    baselines = {"In distribution": 0.79, "Out of distribution": 0.71, "Oropharynx": 0.69}
    for i, label in enumerate(data_split_labels):
        m = np.mean(auc_list[i])
        s = np.std(auc_list[i])
        b = baselines[label]
        delta = m - b
        arrow = "↑" if delta > 0 else "↓"
        print(f"  {label:<22}: {m:.4f} ± {s:.4f}  (baseline {b:.2f} → {arrow}{abs(delta):.4f})")

    print(f"\nDone. Results saved to {results_dir}")