from argparse import ArgumentParser
import os
from random import Random

import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline, get_umap_embedding


# ── Config cho outlier clipping ───────────────────────────────────────────────
BIOMARKER_COLS = ["NLR", "PLR", "LMR"]  # cột trong calculated_biomarkers.csv
CLIP_LOWER_QUANTILE = 0.01              # 1st percentile
CLIP_UPPER_QUANTILE = 0.99              # 99th percentile


def clip_biomarkers_on_train(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    cols: list
) -> tuple:
    """
    [FIX-2] Clip outliers cho NLR/PLR/LMR — tránh data leakage.

    Quantile ngưỡng được tính CHỈ từ X_train_df, sau đó apply lên cả
    train và test. Test set không tham gia tính ngưỡng clip.

    Args:
        X_train_df : DataFrame train features (chưa qua preprocessor)
        X_test_df  : DataFrame test features  (chưa qua preprocessor)
        cols       : list tên cột cần clip

    Returns:
        (X_train_clipped, X_test_clipped)
    """
    X_train_clipped = X_train_df.copy()
    X_test_clipped  = X_test_df.copy()

    for col in cols:
        if col not in X_train_df.columns:
            print(f"  [CLIP] WARNING: '{col}' không tìm thấy trong data, bỏ qua.")
            continue
        q_low  = X_train_df[col].quantile(CLIP_LOWER_QUANTILE)
        q_high = X_train_df[col].quantile(CLIP_UPPER_QUANTILE)
        X_train_clipped[col] = X_train_df[col].clip(lower=q_low, upper=q_high)
        X_test_clipped[col]  = X_test_df[col].clip(lower=q_low, upper=q_high)
        print(f"  [CLIP] {col}: ngưỡng train [{q_low:.3f}, {q_high:.3f}]")

    return X_train_clipped, X_test_clipped


def return_optimal_random_forest(
        target: str = 'recurrence', random_state: np.random.RandomState = np.random.RandomState(42),
        data_split: str = 'In distribution'
) -> RandomForestClassifier:
    """
    Returns the optimal random forest classifier that is hyperparameter tuned with random search on 100 iterations
    with the evaluation metric F1-Score.

    Args:
        target (str, optional): The target that should be predicted. Available are ['recurrence', 'survival_status'].
            Defaults to 'recurrence'.
        random_state (np.random.RandomState, optional): The random state used for the random forest classifier.
            Defaults to np.random.RandomState(42).
        data_split (str, optional): The type of data split that will be used for determining the test and training split.
            Available are ["In distribution", "Out of distribution", "Oropharynx"]. Defaults to 'In'.

    Raises:
        KeyError: When either the target or the data_split is not supported

        "In distribution"
        "Out of distribution"
        "Oropharynx"
    """
    if target == 'recurrence':
        if data_split == "In distribution":
            return RandomForestClassifier(
                n_estimators=1600, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=15,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Oropharynx":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Out of distribution":
            return RandomForestClassifier(
                n_estimators=800, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=100,
                max_features='sqrt', max_depth=80,
                criterion='log_loss',
                random_state=random_state
            )
    elif target == 'survival_status':
        if data_split == "In distribution":
            return RandomForestClassifier(
                n_estimators=1000, min_samples_split=5,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='log2', # max_depth=null,
                criterion='entropy',
                random_state=random_state
            )
        elif data_split == "Oropharynx":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
        elif data_split == "Out of distribution":
            return RandomForestClassifier(
                n_estimators=1200, min_samples_split=2,
                min_samples_leaf=1, max_leaf_nodes=1000,
                max_features='sqrt', max_depth=20,
                criterion='gini',
                random_state=random_state
            )
    raise KeyError(f'Target {target} or data split {data_split} not recognized')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("datasplit_directory", type=str, help="Path to directory that contains data splits as JSON files")
    parser.add_argument("features_directory", type=str, help="Path to directory with extracted features")
    parser.add_argument("results_directory", type=str, help="Path to directory where results will be saved")
    parser.add_argument("target", type=str, help="Target class", choices=["recurrence", "survival_status"])
    args = parser.parse_args()

    data_dir = Path(args.features_directory)
    results_dir = Path(args.results_directory)
    split_dir = Path(args.datasplit_directory)
    target = args.target

    # Set seed for reproducibility
    rng = np.random.RandomState(42)

    # Load extracted features
    clinical = pd.read_csv(data_dir/"clinical.csv", dtype={"patient_id": str})
    patho = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
    # blood = pd.read_csv(data_dir/"blood.csv", dtype={"patient_id": str})
    icd = pd.read_csv(data_dir/"icd_codes.csv", dtype={"patient_id": str})
    cell_density = pd.read_csv(data_dir/"tma_cell_density.csv", dtype={"patient_id": str})
    biomarkers = pd.read_csv(data_dir/"calculated_biomarkers.csv", dtype={"patient_id": str})

    # Lưu ý: Phải xóa cột 'recurrence' trong file này trước khi gộp để tránh rò rỉ dữ liệu (data leakage)
    if "recurrence" in biomarkers.columns:
        biomarkers = biomarkers.drop(columns=["recurrence"])

    # Merge modalities (Gộp 6 file)
    df = clinical.merge(patho, on="patient_id", how="outer")
    # df = df.merge(blood, on="patient_id", how="outer")
    df = df.merge(icd, on="patient_id", how="outer")
    df = df.merge(cell_density, on="patient_id", how="outer")
    df = df.merge(biomarkers, on="patient_id", how="outer") # Dòng mới thêm
    df = df.reset_index(drop=True)

    # Prepare for plots
    x_linspace = np.linspace(0, 1, 100)
    rcParams.update({"font.size": 6})
    rcParams["svg.fonttype"] = "none"
    umap_embeddings = get_umap_embedding(data_dir, umap_min_dist=0.1, umap_n_neighbors=15)

    data_split_paths = [
        split_dir/"dataset_split_in.json",
        split_dir/"dataset_split_out.json",
        split_dir/"dataset_split_Oropharynx.json"
    ]
    data_split_labels = [
        "In distribution",
        "Out of distribution",
        "Oropharynx",
    ]

    tpr_list = [[] for _ in range(len(data_split_paths))]
    auc_list = [[] for _ in range(len(data_split_paths))]

    for i in range(len(data_split_paths)):

        print(f"Training and testing models on {data_split_labels[i]} data...")

        assert_text = f"{data_split_paths[i]} does not exist. Please run genetic_algorithm.py or " \
                      f"split_by_tumor_site.py to generate the corresponding split"
        assert os.path.exists(data_split_paths[i]), assert_text

        # Load patient IDs with dataset split and target classes
        df_split = pd.read_json(data_split_paths[i], dtype={"patient_id": str})[["patient_id", "dataset"]]
        df_targets = pd.read_csv(data_dir/"targets.csv", dtype={"patient_id": str})
        df_split = df_split.merge(df_targets, on="patient_id", how="inner")
        umap_split = umap_embeddings.merge(df_split, on="patient_id", how="inner")

        if target == "recurrence":
            # Only include patients who had a recurrence within 3 years
            # or who survived at least 3 years without recurrence
            df_split = df_split[
                ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
                ((df_split.recurrence == "no") & ((df_split.days_to_last_information > 365*3) |
                                                  (df_split.survival_status == "living")))]
            # Strings to class labels
            df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})

        elif target == "survival_status":
            # Exclude not tumor specific deaths
            df_split = df_split[~(df_split.survival_status_with_cause == "deceased not tumor specific")]
            # Strings to class labels
            df_split.survival_status = df_split.survival_status.replace({"living": 0, "deceased": 1})

        df_train = df_split[df_split.dataset == "training"][["patient_id", target]].copy()
        df_train.columns = ["patient_id", "target"]
        df_train = df_train.merge(df, on="patient_id", how="inner")

        df_test = df_split[df_split.dataset == "test"][["patient_id", target]].copy()
        df_test.columns = ["patient_id", "target"]
        df_test = df_test.merge(df, on="patient_id", how="inner")

        print(f"  Train: {len(df_train)} patients | "
              f"class balance: {df_train['target'].value_counts().to_dict()}")
        print(f"  Test : {len(df_test)} patients  | "
              f"class balance: {df_test['target'].value_counts().to_dict()}")

        # [FIX-2] Clip biomarkers TRUOC preprocessor
        # Quantile tinh tu training set -> apply len ca train + test (khong leakage)
        X_train_feat_raw = df_train.drop(["patient_id", "target"], axis=1)
        X_test_feat_raw  = df_test.drop(["patient_id", "target"],  axis=1)

        X_train_feat, X_test_feat = clip_biomarkers_on_train(
            X_train_feat_raw, X_test_feat_raw, BIOMARKER_COLS
        )

        # [FIX-3] Preprocessor fit 1 lan ngoai vong lap — data khong thay doi
        # giua cac iteration nen khong can fit lai moi lan
        preprocessor      = setup_preprocessing_pipeline(X_train_feat.columns)
        X_train_processed = preprocessor.fit_transform(X_train_feat)
        X_test_processed  = preprocessor.transform(X_test_feat)

        y_train_base = df_train["target"].to_numpy()
        y_test       = df_test["target"].to_numpy()

        # Vong lap 5 lan: SMOTE variance estimation
        # Moi lan SMOTE dung random seed khac (rng stateful) -> do AUC stability
        # KHONG phai k-fold CV, train/test split co dinh tu JSON
        for iteration in range(5):
            print(f"  Iteration {iteration + 1}/5 ... ", end="", flush=True)

            # Can bang du lieu — SMOTE chi apply tren X_train_processed
            smote = SMOTE(random_state=rng)
            X_train, y_train = smote.fit_resample(X_train_processed, y_train_base)

            # LỌC ĐẶC TRƯNG — selector fit trên X_train (đã SMOTE), transform X_test_processed
            # X_test_processed đã được clip + preprocessor đúng cách (không leakage)
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=rng),
                threshold='median'
            )
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected  = selector.transform(X_test_processed)  # [FIX] dùng X_test_processed

            # CHÚ Ý ĐIỂM NÀY: Huấn luyện trên tập ĐÃ LỌC (X_train_selected)
            model = return_optimal_random_forest(target=target, data_split=data_split_labels[i])
            model.fit(X_train_selected, y_train)

            # ================= VẼ BIỂU ĐỒ TOP TỪ CAO XUỐNG THẤP =================
            if iteration == 4:  
                # 1. Lấy toàn bộ tên cột
                all_feature_names = preprocessor.get_feature_names_out()
                
                # 2. CHỈ lọc ra tên của các cột được giữ lại
                selected_feature_names = all_feature_names[selector.get_support()]
                
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Lấy tối đa top 20 (hoặc ít hơn nếu số cột giữ lại < 20)
                top_n = min(20, len(indices)) 
                top_indices = indices[:top_n]
                
                plt.figure(figsize=(6, 8))
                sns.barplot(x=importances[top_indices], y=[selected_feature_names[idx] for idx in top_indices], palette="magma")
                plt.title(f"Top {top_n} - {data_split_labels[i]} Feature Important")
                plt.xlabel("Feature Importance")
                plt.ylabel("Index")
                plt.tight_layout()
                plt.savefig(results_dir/f"feature_importance_{data_split_labels[i].replace(' ', '_')}.svg", bbox_inches="tight")
                plt.close()
            # ====================================================================

            # Dự đoán
            y_test_predicted = model.predict_proba(X_test_selected)[:, 1]

            # ROC curve
            fpr, tpr, thresh = roc_curve(y_test, y_test_predicted)
            tpr = np.interp(x_linspace, fpr, tpr)
            tpr[0] = 0.0
            tpr[-1] = 1.0
            tpr_list[i].append(tpr)
            auc_score = roc_auc_score(y_test, y_test_predicted)
            auc_list[i].append(auc_score)
            print(f"AUC = {auc_score:.4f}")

            # [FIX-4] UMAP plot: chỉ vẽ 1 lần ở iteration đầu
            # data không thay đổi giữa các iteration → vẽ lại 5 lần là thừa
            if iteration == 0:
                plt.figure(figsize=(1.75, 1.75))
                palette = {"training": "lightgrey", "test": sns.color_palette("Set2")[i]}
                ax = sns.scatterplot(umap_split[umap_split.dataset=="training"], x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                ax = sns.scatterplot(umap_split[umap_split.dataset=="test"], x="UMAP 1", y="UMAP 2", hue="dataset", palette=palette, s=2)
                plt.title(data_split_labels[i])
                ax.set_aspect("equal")
                plt.legend()
                sns.despine()
                plt.xticks([])
                plt.yticks([])
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False, fontsize=6)
                plt.xlabel("UMAP 1", fontsize=6)
                plt.ylabel("UMAP 2", fontsize=6)
                plt.tight_layout()
                plt.savefig(results_dir/f"umap_split_{data_split_labels[i]}.svg", bbox_inches="tight")
                plt.close()

        print(f"\n  [{data_split_labels[i]}] Mean AUC = "
              f"{np.mean(auc_list[i]):.4f} +/- {np.std(auc_list[i]):.4f}")

    # Plot ROC curve
    colors = sns.color_palette("Set2")
    plt.figure(figsize=(2.2, 1.75))
    for i in range(len(auc_list)):
        mean_tpr = np.mean(tpr_list[i], axis=0)
        std_tpr = np.std(tpr_list[i], axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        mean_fpr = np.linspace(0, 1, 100)

        plt.plot(x_linspace, mean_tpr, linewidth=0.8, color=colors[i],
                 label=f"AUC = {np.mean(auc_list[i]):.2f}$\pm${np.std(auc_list[i]):.2f}")
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=colors[i], alpha=0.5, lw=0)

    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Random")
    plt.xticks(np.arange(0, 1.2, 0.5))
    plt.yticks(np.arange(0, 1.2, 0.5))
    plt.xlabel("FPR", fontsize=6)
    plt.ylabel("TPR", fontsize=6)
    plt.title(f"{target}")
    plt.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(results_dir/f"roc_testsets_{target}.svg", bbox_inches="tight")
    plt.close()

    print(f"Done. Saved results to {results_dir}")