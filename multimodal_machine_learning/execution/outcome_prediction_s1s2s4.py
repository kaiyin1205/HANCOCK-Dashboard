from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline, get_umap_embedding

# ── Outlier clipping 設定 ──────────────────────────────────
BIOMARKER_COLS = ["NLR", "PLR", "LMR", "SII", "Invasion_Score"]
CLIP_LOWER_QUANTILE = 0.01
CLIP_UPPER_QUANTILE = 0.99

# ── S1: Per-split percentile threshold 設定 ────────────────
PERCENTILE_THRESHOLDS = {
    "In distribution"    : 50, //70
    "Out of distribution": 60,
    "Oropharynx"         : 55,
}


def clip_biomarkers_on_train(X_train_df, X_test_df, cols):
    """Clip outliers based on training set quantiles only (no data leakage)"""
    X_train_clipped = X_train_df.copy()
    X_test_clipped  = X_test_df.copy()
    for col in cols:
        if col not in X_train_df.columns:
            continue
        q_low  = X_train_df[col].quantile(CLIP_LOWER_QUANTILE)
        q_high = X_train_df[col].quantile(CLIP_UPPER_QUANTILE)
        X_train_clipped[col] = X_train_df[col].clip(lower=q_low, upper=q_high)
        X_test_clipped[col]  = X_test_df[col].clip(lower=q_low, upper=q_high)
        print(f"  [CLIP] {col}: train range [{q_low:.3f}, {q_high:.3f}]")
    return X_train_clipped, X_test_clipped


def add_feature_interactions(df, feature_cols):
    """S2: Add interaction features between inflammatory markers and pathological features"""
    df = df.copy()

    # ── NLR × pT_stage ────────────────────────────────────
    if "NLR" in df.columns and "pT_stage" in df.columns:
        df["NLR_x_pT_stage"] = df["NLR"] * pd.to_numeric(df["pT_stage"], errors="coerce")

    # ── PLR × number_of_positive_lymph_nodes ──────────────
    if "PLR" in df.columns and "number_of_positive_lymph_nodes" in df.columns:
        df["PLR_x_N_nodes"] = df["PLR"] * pd.to_numeric(df["number_of_positive_lymph_nodes"], errors="coerce")

    # ── LMR × perineural_invasion_Pn ──────────────────────
    if "LMR" in df.columns and "perineural_invasion_Pn" in df.columns:
        df["LMR_x_perineural"] = df["LMR"] * pd.to_numeric(df["perineural_invasion_Pn"], errors="coerce")

    # ── SII × pN_stage ────────────────────────────────────
    if "SII" in df.columns and "pN_stage" in df.columns:
        df["SII_x_pN_stage"] = df["SII"] * pd.to_numeric(df["pN_stage"], errors="coerce")

    return df


def build_stacking_model(random_state):
    """S4: Build stacking ensemble with RF + ET + LR as base, LR as meta-learner"""
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=800, min_samples_split=2,
            min_samples_leaf=1, max_leaf_nodes=1000,
            max_features="sqrt", max_depth=15,
            criterion="gini", random_state=random_state
        )),
        ("et", ExtraTreesClassifier(
            n_estimators=400, min_samples_split=2,
            min_samples_leaf=2, max_leaf_nodes=500,
            max_features="sqrt", max_depth=20,
            criterion="entropy", random_state=random_state
        )),
        ("lr", LogisticRegression(
            max_iter=1000, random_state=random_state
        )),
    ]
    meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory", type=str)
    parser.add_argument("features_directory",  type=str)
    parser.add_argument("results_directory",   type=str)
    parser.add_argument("target", type=str, choices=["recurrence", "survival_status"])
    parser.add_argument("--biomarkers", type=str, default="calculated_biomarkers",
                        help="Biomarkers filename (without .csv)")
    args = parser.parse_args()

    data_dir    = Path(args.features_directory)
    results_dir = Path(args.results_directory)
    split_dir   = Path(args.datasplit_directory)
    target      = args.target

    # ── 建立結果資料夾 ─────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 設定隨機種子 ───────────────────────────────────────
    rng = np.random.RandomState(42)

    # ── 載入資料 ───────────────────────────────────────────
    clinical     = pd.read_csv(data_dir/"clinical.csv",          dtype={"patient_id": str})
    patho        = pd.read_csv(data_dir/"pathological.csv",      dtype={"patient_id": str})
    icd          = pd.read_csv(data_dir/"icd_codes.csv",         dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv",  dtype={"patient_id": str})
    biomarkers   = pd.read_csv(data_dir/f"{args.biomarkers}.csv", dtype={"patient_id": str})

    # ── 避免 data leakage ──────────────────────────────────
    if "recurrence" in biomarkers.columns:
        biomarkers = biomarkers.drop(columns=["recurrence"])

    # ── 合併所有模態資料 ───────────────────────────────────
    df = clinical.merge(patho,        on="patient_id", how="outer")
    df = df.merge(icd,                on="patient_id", how="outer")
    df = df.merge(cell_density,       on="patient_id", how="outer")
    df = df.merge(biomarkers,         on="patient_id", how="outer")
    df = df.reset_index(drop=True)

    # ── S2: 加入交互特徵 ───────────────────────────────────
    df = add_feature_interactions(df, df.columns.tolist())
    print(f"Total features after interaction: {len(df.columns)}")

    # ── 準備繪圖設定 ───────────────────────────────────────
    x_linspace      = np.linspace(0, 1, 100)
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    umap_embeddings = get_umap_embedding(data_dir, umap_min_dist=0.1, umap_n_neighbors=15)

    data_split_paths = [
        split_dir/"dataset_split_in.json",
        split_dir/"dataset_split_out.json",
        split_dir/"dataset_split_Oropharynx.json"
    ]
    data_split_labels = ["In distribution", "Out of distribution", "Oropharynx"]

    tpr_list = [[] for _ in range(len(data_split_paths))]
    auc_list = [[] for _ in range(len(data_split_paths))]

    for i in range(len(data_split_paths)):
        print(f"\nTraining and testing models on {data_split_labels[i]} data...")

        assert os.path.exists(data_split_paths[i]), f"{data_split_paths[i]} does not exist."

        # ── 載入資料切分 ───────────────────────────────────
        df_split   = pd.read_json(data_split_paths[i], dtype={"patient_id": str})[["patient_id", "dataset"]]
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
            df_split = df_split[~(df_split.survival_status_with_cause == "deceased not tumor specific")]
            df_split = df_split.copy()
            df_split.survival_status = df_split.survival_status.replace({"living": 0, "deceased": 1})

        df_train = df_split[df_split.dataset == "training"][["patient_id", target]].copy()
        df_train.columns = ["patient_id", "target"]
        df_train = df_train.merge(df, on="patient_id", how="inner")

        df_test = df_split[df_split.dataset == "test"][["patient_id", target]].copy()
        df_test.columns = ["patient_id", "target"]
        df_test = df_test.merge(df, on="patient_id", how="inner")

        print(f"  Train: {len(df_train)} | Test: {len(df_test)}")

        # ── Clip biomarkers ────────────────────────────────
        X_train_raw = df_train.drop(["patient_id", "target"], axis=1)
        X_test_raw  = df_test.drop(["patient_id", "target"],  axis=1)
        X_train_raw, X_test_raw = clip_biomarkers_on_train(
            X_train_raw, X_test_raw, BIOMARKER_COLS
        )

        # ── 前處理 ─────────────────────────────────────────
        preprocessor      = setup_preprocessing_pipeline(X_train_raw.columns)
        X_train_processed = preprocessor.fit_transform(X_train_raw)
        X_test_processed  = preprocessor.transform(X_test_raw)

        y_train_base = df_train["target"].to_numpy()
        y_test       = df_test["target"].to_numpy()

        for iteration in range(5):
            print(f"  Iteration {iteration + 1}/5 ... ", end="", flush=True)

            # ── SMOTE ──────────────────────────────────────
            smote = SMOTE(random_state=rng)
            X_train, y_train = smote.fit_resample(X_train_processed, y_train_base)

            # ── S1: SelectPercentile + per-split threshold ─
            percentile = PERCENTILE_THRESHOLDS[data_split_labels[i]]
            selector   = SelectPercentile(f_classif, percentile=percentile)
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected  = selector.transform(X_test_processed)

            # ── S4: Stacking Ensemble ──────────────────────
            model = build_stacking_model(random_state=42)
            model.fit(X_train_selected, y_train)

            # ── Feature Importance 圖（最後一次迭代）────────
            if iteration == 4:
                all_feature_names      = preprocessor.get_feature_names_out()
                selected_feature_names = all_feature_names[selector.get_support()]

                # 從 RF base estimator 取 feature importance
                rf_model    = model.estimators_[0]
                importances = rf_model.feature_importances_
                indices     = np.argsort(importances)[::-1]
                top_n       = min(20, len(indices))
                top_indices = indices[:top_n]

                plt.figure(figsize=(6, 8))
                sns.barplot(
                    x=importances[top_indices],
                    y=[selected_feature_names[idx] for idx in top_indices],
                    palette="magma",
                    hue=[selected_feature_names[idx] for idx in top_indices],
                    legend=False
                )
                plt.title(f"Top {top_n} - {data_split_labels[i]} Feature Importance (S1+S2+S4)")
                plt.xlabel("Feature Importance")
                plt.tight_layout()
                plt.savefig(results_dir/f"feature_importance_{data_split_labels[i].replace(' ', '_')}.svg", bbox_inches="tight")
                plt.close()

            # ── 預測與 ROC ─────────────────────────────────
            y_test_predicted = model.predict_proba(X_test_selected)[:, 1]
            fpr, tpr, _      = roc_curve(y_test, y_test_predicted)
            tpr              = np.interp(x_linspace, fpr, tpr)
            tpr[0]           = 0.0
            tpr[-1]          = 1.0
            tpr_list[i].append(tpr)
            auc_score = roc_auc_score(y_test, y_test_predicted)
            auc_list[i].append(auc_score)
            print(f"AUC = {auc_score:.4f}")

            # ── UMAP 圖（第一次迭代）──────────────────────
            if iteration == 0:
                plt.figure(figsize=(1.75, 1.75))
                palette = {"training": "lightgrey", "test": sns.color_palette("Set2")[i]}
                ax = sns.scatterplot(umap_split[umap_split.dataset=="training"], x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                ax = sns.scatterplot(umap_split[umap_split.dataset=="test"],     x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                plt.title(data_split_labels[i])
                ax.set_aspect("equal")
                sns.despine()
                plt.xticks([])
                plt.yticks([])
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)
                plt.xlabel("UMAP 1", fontsize=6)
                plt.ylabel("UMAP 2", fontsize=6)
                plt.tight_layout()
                plt.savefig(results_dir/f"umap_split_{data_split_labels[i]}.svg", bbox_inches="tight")
                plt.close()

        print(f"\n  [{data_split_labels[i]}] Mean AUC = {np.mean(auc_list[i]):.4f} +/- {np.std(auc_list[i]):.4f}")

    # ── ROC 曲線圖 ─────────────────────────────────────────
    colors = sns.color_palette("Set2")
    plt.figure(figsize=(2.2, 1.75))
    for i in range(len(auc_list)):
        mean_tpr  = np.mean(tpr_list[i], axis=0)
        std_tpr   = np.std(tpr_list[i],  axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_fpr  = np.linspace(0, 1, 100)
        plt.plot(x_linspace, mean_tpr, linewidth=0.8, color=colors[i],
                 label=f"AUC = {np.mean(auc_list[i]):.2f}$\\pm${np.std(auc_list[i]):.2f}")
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.5, lw=0)

    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Random")
    plt.xticks(np.arange(0, 1.2, 0.5))
    plt.yticks(np.arange(0, 1.2, 0.5))
    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.title(f"{target} - S1+S2+S4")
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(results_dir/f"roc_testsets_{target}.svg", bbox_inches="tight")
    plt.close()

    print(f"\nDone. Saved results to {results_dir}")