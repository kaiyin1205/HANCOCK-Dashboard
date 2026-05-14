import numpy as np
import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).parents[2]))
from data_exploration.umap_embedding import setup_preprocessing_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# ── 設定路徑 ──────────────────────────────────────────────
data_dir    = Path("features")
results_dir = Path("results/shap")
split_dir   = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

# ── 設定隨機種子 ───────────────────────────────────────────
rng = np.random.RandomState(42)

# ── 載入資料 ───────────────────────────────────────────────
clinical     = pd.read_csv(data_dir/"clinical.csv",              dtype={"patient_id": str})
patho        = pd.read_csv(data_dir/"pathological.csv",          dtype={"patient_id": str})
icd          = pd.read_csv(data_dir/"icd_codes.csv",             dtype={"patient_id": str})
cell_density = pd.read_csv(data_dir/"tma_cell_density.csv",      dtype={"patient_id": str})
biomarkers   = pd.read_csv(data_dir/"calculated_biomarkers.csv", dtype={"patient_id": str})

# ── 避免 data leakage ──────────────────────────────────────
if "recurrence" in biomarkers.columns:
    biomarkers = biomarkers.drop(columns=["recurrence"])

# ── 合併所有模態資料 ───────────────────────────────────────
df = clinical.merge(patho,        on="patient_id", how="outer")
df = df.merge(icd,                on="patient_id", how="outer")
df = df.merge(cell_density,       on="patient_id", how="outer")
df = df.merge(biomarkers,         on="patient_id", how="outer")
df = df.reset_index(drop=True)

# ── 載入 In distribution 資料切分 ─────────────────────────
df_split   = pd.read_json(split_dir/"dataset_split_in.json", dtype={"patient_id": str})[["patient_id", "dataset"]]
df_targets = pd.read_csv(data_dir/"targets.csv",             dtype={"patient_id": str})
df_split   = df_split.merge(df_targets, on="patient_id", how="inner")

# ── 篩選病人：三年內復發 or 存活超過三年 ──────────────────
df_split = df_split[
    ((df_split.recurrence == "yes") & (df_split.days_to_recurrence <= 365*3)) |
    ((df_split.recurrence == "no")  & ((df_split.days_to_last_information > 365*3) |
                                       (df_split.survival_status == "living")))]
df_split = df_split.copy()
df_split.recurrence = df_split.recurrence.replace({"no": 0, "yes": 1})

# ── 分訓練集和測試集 ───────────────────────────────────────
df_train = df_split[df_split.dataset == "training"][["patient_id", "recurrence"]].copy()
df_train.columns = ["patient_id", "target"]
df_train = df_train.merge(df, on="patient_id", how="inner")

df_test = df_split[df_split.dataset == "test"][["patient_id", "recurrence"]].copy()
df_test.columns = ["patient_id", "target"]
df_test = df_test.merge(df, on="patient_id", how="inner")

print(f"Train: {len(df_train)} patients | Test: {len(df_test)} patients")

# ── 前處理 ─────────────────────────────────────────────────
X_train_raw = df_train.drop(["patient_id", "target"], axis=1)
X_test_raw  = df_test.drop(["patient_id", "target"],  axis=1)
y_train     = df_train["target"].to_numpy()
y_test      = df_test["target"].to_numpy()

preprocessor = setup_preprocessing_pipeline(X_train_raw.columns)
X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

# ── SMOTE：處理類別不平衡 ──────────────────────────────────
smote = SMOTE(random_state=rng)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# ── 特徵篩選 ───────────────────────────────────────────────
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=rng),
    threshold="median"
)
X_train_sel = selector.fit_transform(X_train_bal, y_train_bal)
X_test_sel  = selector.transform(X_test_proc)

# ── 取得篩選後的特徵名稱 ───────────────────────────────────
all_feature_names      = preprocessor.get_feature_names_out()
selected_feature_names = all_feature_names[selector.get_support()]
print(f"Number of selected features: {len(selected_feature_names)}")

# ── 訓練隨機森林 ───────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=1600, min_samples_split=2,
    min_samples_leaf=1, max_leaf_nodes=1000,
    max_features="sqrt", max_depth=15,
    criterion="gini", random_state=rng
)
model.fit(X_train_sel, y_train_bal)

# ── SHAP 分析 ──────────────────────────────────────────────
print("Calculating SHAP values, please wait...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_sel)

# ── 取有復發（class=1）的 SHAP 值 ─────────────────────────
shap_vals_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

# ── 取有復發（class=1）的 SHAP 值 ─────────────────────────
if len(np.array(shap_values).shape) == 3:
    shap_vals_class1 = shap_values[:, :, 1]
else:
    shap_vals_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

# ── 圖1：Summary Plot（每個特徵對每個病人的影響） ──────────
plt.figure()
shap.summary_plot(
    shap_vals_class1,
    X_test_sel,
    feature_names=selected_feature_names,
    show=False
)
plt.title("SHAP Summary Plot - Recurrence Prediction")
plt.tight_layout()
plt.savefig(results_dir/"shap_summary_plot.svg", bbox_inches="tight")
plt.close()
print("Saved: shap_summary_plot.svg")

# ── 圖2：Bar Plot（各特徵的平均絕對 SHAP 值） ─────────────
plt.figure()
shap.summary_plot(
    shap_vals_class1,
    X_test_sel,
    feature_names=selected_feature_names,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance - Recurrence Prediction")
plt.tight_layout()
plt.savefig(results_dir/"shap_bar_plot.svg", bbox_inches="tight")
plt.close()
print("Saved: shap_bar_plot.svg")

# ── 圖2：Bar Plot（各特徵的平均絕對 SHAP 值） ─────────────
plt.figure()
shap.summary_plot(
    shap_vals_class1,
    X_test_sel,
    feature_names=selected_feature_names,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance - Recurrence Prediction")
plt.tight_layout()
plt.savefig(results_dir/"shap_bar_plot.svg", bbox_inches="tight")
plt.close()
print("Saved: shap_bar_plot.svg")

# ── 圖3：Waterfall Plot（單一高風險病人的風險歸因） ─────────
# 找出指定病患 ID 的資料
target_id = "555"  # 輸入想查詢的病患 ID
patient_ids = df_test["patient_id"].values
if target_id not in patient_ids:
    print(f"Patient ID {target_id} not found in test set!")
else:
    high_risk_idx = np.where(patient_ids == target_id)[0][0]
    proba = model.predict_proba(X_test_sel)[:, 1]

# ── 修正維度抓取邏輯 ──
# 如果 shap_values 是 3D (samples, features, classes)，取 class 1 (復發)
if len(shap_values.shape) == 3:
    patient_shap_values = shap_values[high_risk_idx, :, 1]
    base_value = explainer.expected_value[1]
else:
    # 如果是舊版 list 格式
    patient_shap_values = shap_vals_class1[high_risk_idx]
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

explanation = shap.Explanation(
    values       = patient_shap_values,
    base_values  = base_value,
    data         = X_test_sel[high_risk_idx],
    feature_names= selected_feature_names
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
plt.title(f"SHAP Waterfall - Highest Risk Patient (Prob={proba[high_risk_idx]:.2f})")
plt.tight_layout()
plt.savefig(results_dir/"shap_waterfall_high_risk.svg", bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall_high_risk.svg")
# 取得該病患在測試集中的原始 ID
target_patient_id = df_test.iloc[high_risk_idx]['patient_id']
print(f"Highest risk patient ID: {target_patient_id}")