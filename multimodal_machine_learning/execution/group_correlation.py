import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ─────────────────────────────────────────────
data_dir    = Path("features")
results_dir = Path("results/group_correlation")
results_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
clinical     = pd.read_csv(data_dir/"clinical.csv",          dtype={"patient_id": str})
patho        = pd.read_csv(data_dir/"pathological.csv",      dtype={"patient_id": str})
blood        = pd.read_csv(data_dir/"blood.csv",             dtype={"patient_id": str})
targets      = pd.read_csv(data_dir/"targets.csv",           dtype={"patient_id": str})
biomarkers   = pd.read_csv(data_dir/"biomarkers_original.csv", dtype={"patient_id": str})

if "recurrence" in biomarkers.columns:
    biomarkers = biomarkers.drop(columns=["recurrence"])

# ── Merge all modalities ───────────────────────────────────
df = clinical.merge(patho,      on="patient_id", how="outer")
df = df.merge(blood,            on="patient_id", how="outer")
df = df.merge(biomarkers,       on="patient_id", how="outer")
df = df.merge(targets,          on="patient_id", how="outer")
df = df.reset_index(drop=True)

print(f"Total patients: {len(df)}")

# ── Keep only deceased patients ────────────────────────────
# Reason: living patients have censored survival time
df_deceased = df[df["survival_status"] == "deceased"].copy()
print(f"Deceased patients: {len(df_deceased)}")

# ── Compute median survival per recurrence subgroup ────────
rec_yes = df_deceased[df_deceased["recurrence"] == "yes"]
rec_no  = df_deceased[df_deceased["recurrence"] == "no"]

median_rec_yes = rec_yes["days_to_last_information"].median()
median_rec_no  = rec_no["days_to_last_information"].median()

print(f"Median survival (Recurrence YES, deceased): {median_rec_yes:.1f} days")
print(f"Median survival (Recurrence NO,  deceased): {median_rec_no:.1f} days")

# ── Assign groups ──────────────────────────────────────────
def assign_group(row):
    if row["recurrence"] == "yes":
        if row["days_to_last_information"] >= median_rec_yes:
            return "G1: Recurrence + Long"
        else:
            return "G2: Recurrence + Short"
    else:
        if row["days_to_last_information"] >= median_rec_no:
            return "G4: No Recurrence + Long"
        else:
            return "G3: No Recurrence + Short"

df_deceased["group"] = df_deceased.apply(assign_group, axis=1)

print("\nGroup distribution:")
print(df_deceased["group"].value_counts().sort_index())

# ── Select numeric features ────────────────────────────────
exclude_cols = [
    "days_to_last_information", "days_to_recurrence",
    "days_to_progress", "days_to_metastasis",
    "days_to_rfs_event"
]
numeric_cols  = df_deceased.select_dtypes(include=[np.number]).columns.tolist()
feature_cols  = [c for c in numeric_cols if c not in exclude_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# ── Step 1: Compute median per group per feature ───────────
group_medians = df_deceased.groupby("group")[feature_cols].median()
group_medians.to_csv(results_dir/"group_medians.csv")
print("Saved: group_medians.csv")

# ── Step 2: Compute Spearman Correlation Coefficient ───────
# Encode groups as ordinal numbers for correlation
group_order = {
    "G1: Recurrence + Long"    : 1,
    "G2: Recurrence + Short"   : 2,
    "G3: No Recurrence + Short": 3,
    "G4: No Recurrence + Long" : 4,
}
df_deceased["group_num"] = df_deceased["group"].map(group_order)

correlation_results = []
for col in feature_cols:
    valid = df_deceased[[col, "group_num"]].dropna()
    if len(valid) < 10:
        continue
    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(valid[col], valid["group_num"])
    # Pearson correlation
    pearson_r,  pearson_p  = stats.pearsonr(valid[col], valid["group_num"])

    correlation_results.append({
        "feature"          : col,
        "spearman_r"       : spearman_r,
        "spearman_p"       : spearman_p,
        "pearson_r"        : pearson_r,
        "pearson_p"        : pearson_p,
        "abs_spearman"     : abs(spearman_r),
        "significant_0.05" : spearman_p < 0.05
    })

corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values("abs_spearman", ascending=False)
corr_df.to_csv(results_dir/"feature_correlation.csv", index=False)

print("\nTop 20 features by Spearman correlation:")
print(corr_df.head(20)[["feature", "spearman_r", "spearman_p", "significant_0.05"]].to_string(index=False))

# ── Plot 1: Top 20 Correlation Bar Chart ──────────────────
top20 = corr_df.head(20).copy()
colors = ["#d73027" if r > 0 else "#4575b4" for r in top20["spearman_r"]]

plt.figure(figsize=(10, 8))
bars = plt.barh(top20["feature"], top20["spearman_r"], color=colors)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.xlabel("Spearman Correlation Coefficient")
plt.title("Top 20 Features: Spearman Correlation with Patient Group\n(Red = positive, Blue = negative)")
plt.tight_layout()
plt.savefig(results_dir/"top20_correlation.svg", bbox_inches="tight")
plt.close()
print("\nSaved: top20_correlation.svg")

# ── Plot 2: Heatmap of group medians ──────────────────────
top20_features = corr_df.head(20)["feature"].tolist()
heatmap_data   = group_medians[top20_features]
heatmap_norm   = (heatmap_data - heatmap_data.mean()) / (heatmap_data.std() + 1e-8)

plt.figure(figsize=(16, 5))
sns.heatmap(
    heatmap_norm,
    annot=True, fmt=".2f",
    cmap="RdBu_r", center=0,
    linewidths=0.5
)
plt.title("Normalized Median Feature Values Across Four Patient Groups (Top 20 Features)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(results_dir/"group_heatmap.svg", bbox_inches="tight")
plt.close()
print("Saved: group_heatmap.svg")

# ── Plot 3: NLR / PLR / LMR Boxplot ──────────────────────
biomarker_cols = [c for c in ["NLR", "PLR", "LMR"] if c in df_deceased.columns]
if biomarker_cols:
    fig, axes = plt.subplots(1, len(biomarker_cols), figsize=(5*len(biomarker_cols), 5))
    if len(biomarker_cols) == 1:
        axes = [axes]
    group_order_list = [
        "G1: Recurrence + Long",
        "G2: Recurrence + Short",
        "G3: No Recurrence + Short",
        "G4: No Recurrence + Long"
    ]
    for ax, col in zip(axes, biomarker_cols):
        sns.boxplot(
            data=df_deceased, x="group", y=col,
            order=group_order_list,
            palette="Set2", ax=ax
        )
        ax.set_title(f"{col} across Patient Groups")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(results_dir/"biomarker_boxplot.svg", bbox_inches="tight")
    plt.close()
    print("Saved: biomarker_boxplot.svg")

print(f"\nDone! All results saved to {results_dir}")