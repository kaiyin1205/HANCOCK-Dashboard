import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline

# ── 設定路徑 ──────────────────────────────────────────────
data_dir  = Path("features")
split_dir = Path("results")

# ── 設定隨機種子 ───────────────────────────────────────────
rng = np.random.RandomState(42)

# ── 載入資料 ───────────────────────────────────────────────
clinical     = pd.read_csv(data_dir/"clinical.csv",              dtype={"patient_id": str})
patho        = pd.read_csv(data_dir/"pathological.csv",          dtype={"patient_id": str})
icd          = pd.read_csv(data_dir/"icd_codes.csv",             dtype={"patient_id": str})
cell_density = pd.read_csv(data_dir/"tma_cell_density.csv",      dtype={"patient_id": str})
biomarkers   = pd.read_csv(data_dir/"biomarkers_v3_sii_invasion.csv", dtype={"patient_id": str})

# ── 避免 data leakage ──────────────────────────────────────
if "recurrence" in biomarkers.columns:
    biomarkers = biomarkers.drop(columns=["recurrence"])

# ── 合併資料 ───────────────────────────────────────────────
df = clinical.merge(patho,        on="patient_id", how="outer")
df = df.merge(icd,                on="patient_id", how="outer")
df = df.merge(cell_density,       on="patient_id", how="outer")
df = df.merge(biomarkers,         on="patient_id", how="outer")
df = df.reset_index(drop=True)

# ── 只用 In distribution 切分來調整超參數 ─────────────────
df_split   = pd.read_json(split_dir/"dataset_split_in.json", dtype={"patient_id": str})[["patient_id", "dataset"]]
df_targets = pd.read_csv(data_dir/"targets.csv",             dtype={"patient_id": str})
df_split   = df_split.merge(df_targets, on="patient_id", how="inner")

# ── 篩選病人 ───────────────────────────────────────────────
df_split = df_split[
    ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
    ((df_split.recurrence == "no")  & ((df_split.days_to_last_information > 365*3) |
                                       (df_split.survival_status == "living")))]
df_split = df_split.copy()
df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})

# ── 只用訓練集來調整超參數 ─────────────────────────────────
df_train = df_split[df_split.dataset == "training"][["patient_id", "recurrence"]].copy()
df_train.columns = ["patient_id", "target"]
df_train = df_train.merge(df, on="patient_id", how="inner")

print(f"Train: {len(df_train)} patients")

# ── 前處理 ─────────────────────────────────────────────────
X_train_raw = df_train.drop(["patient_id", "target"], axis=1)
y_train     = df_train["target"].to_numpy()

preprocessor = setup_preprocessing_pipeline(X_train_raw.columns)
X_train_proc = preprocessor.fit_transform(X_train_raw)

# ── SMOTE ──────────────────────────────────────────────────
smote = SMOTE(random_state=rng)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# ── Feature Selection ──────────────────────────────────────
selector = SelectFromModel(
    ExtraTreesClassifier(n_estimators=100, random_state=rng),
    threshold="median"
)
X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
print(f"Selected features: {X_train_sel.shape[1]}")

# ── 超參數搜尋範圍 ─────────────────────────────────────────
param_dist = {
    "n_estimators"    : [200, 400, 800, 1200, 1600, 2000],
    "max_depth"       : [10, 15, 20, 30, 50, 80, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "max_features"    : ["sqrt", "log2", 0.3, 0.5],
    "max_leaf_nodes"  : [100, 500, 1000, None],
    "criterion"       : ["gini", "entropy"],
}

# ── Random Search ──────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator  = ExtraTreesClassifier(random_state=42),
    param_distributions = param_dist,
    n_iter     = 50,          # 測試 50 種組合
    scoring    = "roc_auc",
    cv         = cv,
    n_jobs     = -1,          # 用所有 CPU 加速
    random_state = 42,
    verbose    = 2
)

print("\nStarting Random Search (50 iterations)...")
print("This may take 20-40 minutes, please wait...\n")
search.fit(X_train_sel, y_train_bal)

# ── 輸出最佳結果 ───────────────────────────────────────────
print("\n" + "="*50)
print(f"Best AUC (CV): {search.best_score_:.4f}")
print(f"Best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")
print("="*50)