import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline

# ── 設定路徑 ──────────────────────────────────────────────
data_dir   = Path("features")
split_dir  = Path("results")
model_dir  = Path("dashboard\models")
model_dir.mkdir(parents=True, exist_ok=True)

# ── 設定隨機種子 ───────────────────────────────────────────
rng = np.random.RandomState(42)

# ── 載入資料 ───────────────────────────────────────────────
clinical     = pd.read_csv(data_dir/"clinical.csv",          dtype={"patient_id": str})
patho        = pd.read_csv(data_dir/"pathological.csv",      dtype={"patient_id": str})
icd          = pd.read_csv(data_dir/"icd_codes.csv",         dtype={"patient_id": str})
cell_density = pd.read_csv(data_dir/"tma_cell_density.csv",  dtype={"patient_id": str})
biomarkers   = pd.read_csv(data_dir/"biomarkers_original.csv", dtype={"patient_id": str})

if "recurrence" in biomarkers.columns:
    biomarkers = biomarkers.drop(columns=["recurrence"])

# ── 合併資料 ───────────────────────────────────────────────
df = clinical.merge(patho,        on="patient_id", how="outer")
df = df.merge(icd,                on="patient_id", how="outer")
df = df.merge(cell_density,       on="patient_id", how="outer")
df = df.merge(biomarkers,         on="patient_id", how="outer")
df = df.reset_index(drop=True)

# ── 載入 In distribution 資料切分 ─────────────────────────
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

# ── 只用訓練集訓練模型 ─────────────────────────────────────
df_train = df_split[df_split.dataset == "training"][["patient_id", "recurrence"]].copy()
df_train.columns = ["patient_id", "target"]
df_train = df_train.merge(df, on="patient_id", how="inner")

print(f"Training on {len(df_train)} patients...")

# ── 前處理 ─────────────────────────────────────────────────
X_train_raw = df_train.drop(["patient_id", "target"], axis=1)
y_train     = df_train["target"].to_numpy()

# ── Clip NLR/PLR/LMR ──────────────────────────────────────
for col in ["NLR", "PLR", "LMR"]:
    if col in X_train_raw.columns:
        q_low  = X_train_raw[col].quantile(0.01)
        q_high = X_train_raw[col].quantile(0.99)
        X_train_raw[col] = X_train_raw[col].clip(lower=q_low, upper=q_high)

# ── Interaction features ───────────────────────────────────
def add_interaction_features(df):
    df = df.copy()
    cols = list(df.columns)
    def find_col(keywords):
        for kw in keywords:
            matches = [c for c in cols if kw.lower() in str(c).lower()]
            if matches:
                return matches[0]
        return None

    nlr_col = find_col(["NLR"])
    plr_col = find_col(["PLR"])
    lmr_col = find_col(["LMR"])
    pn_col  = find_col(["pN_stage"])
    lym_col = find_col(["positive_lymph"])
    pni_col = find_col(["perineural"])

    if nlr_col and pn_col:
        df["interact_NLR_x_pNstage"] = df[nlr_col] * df[pn_col].fillna(0)
    if plr_col and lym_col:
        df["interact_PLR_x_lymph"]   = df[plr_col] * df[lym_col].fillna(0)
    if lmr_col and pni_col:
        df["interact_LMR_x_PNI"]     = df[lmr_col] * df[pni_col].fillna(0)
    if nlr_col and plr_col:
        df["interact_NLR_div_PLR"]   = df[nlr_col] / (df[plr_col].replace(0, np.nan)).fillna(df[plr_col].median())
    if nlr_col and lmr_col:
        df["interact_NLR_x_LMR"]     = df[nlr_col] * df[lmr_col]
    return df

X_train_raw = add_interaction_features(X_train_raw)

# ── Preprocessor ───────────────────────────────────────────
preprocessor  = setup_preprocessing_pipeline(X_train_raw.columns)
X_train_proc  = preprocessor.fit_transform(X_train_raw)

# ── SMOTE ──────────────────────────────────────────────────
smote = SMOTE(random_state=rng)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# ── Feature Selection ──────────────────────────────────────
selector = SelectPercentile(score_func=f_classif, percentile=70)
X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
print(f"Selected {X_train_sel.shape[1]} features")

# ── Stacking Model ─────────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=1600, max_features="log2", max_depth=30,
    min_samples_split=2, min_samples_leaf=1,
    max_leaf_nodes=1000, criterion="gini",
    random_state=rng, n_jobs=-1
)
et = ExtraTreesClassifier(
    n_estimators=1600, max_features="log2", max_depth=30,
    min_samples_split=2, min_samples_leaf=1,
    max_leaf_nodes=1000, criterion="gini",
    random_state=rng, n_jobs=-1
)
meta = Pipeline([
    ("scaler", StandardScaler()),
    ("lr",     LogisticRegression(C=1.0, max_iter=2000, random_state=42))
])
model = StackingClassifier(
    estimators=[("rf", rf), ("et", et)],
    final_estimator=meta,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    stack_method="predict_proba",
    n_jobs=-1
)

print("Training model...")
model.fit(X_train_sel, y_train_bal)
print("Done!")

# ── 儲存所有需要的物件 ─────────────────────────────────────
joblib.dump(model,        model_dir/"model.pkl")
joblib.dump(preprocessor, model_dir/"preprocessor.pkl")
joblib.dump(selector,     model_dir/"selector.pkl")

# ── 儲存特徵名稱 ───────────────────────────────────────────
all_feature_names      = preprocessor.get_feature_names_out()
selected_feature_names = all_feature_names[selector.get_support()]
joblib.dump(selected_feature_names, model_dir/"feature_names.pkl")

# ── 儲存 clip 的閾值（給 dashboard 用）────────────────────
clip_bounds = {}
for col in ["NLR", "PLR", "LMR"]:
    if col in df_train.columns:
        q_low  = df_train[col].quantile(0.01)
        q_high = df_train[col].quantile(0.99)
        clip_bounds[col] = (q_low, q_high)
joblib.dump(clip_bounds, model_dir/"clip_bounds.pkl")
joblib.dump(list(X_train_raw.columns), model_dir/"train_columns.pkl")

print(f"\nAll files saved to {model_dir}")
print(f"Feature names: {len(selected_feature_names)} features")