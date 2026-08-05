import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ── Path setup ─────────────────────────────────────────────
data_dir    = Path("features")
results_dir = Path("results/group_correlation_v2")
results_dir.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────
clinical   = pd.read_csv(data_dir/"clinical.csv",           dtype={"patient_id": str})
patho      = pd.read_csv(data_dir/"pathological.csv",       dtype={"patient_id": str})
blood      = pd.read_csv(data_dir/"blood.csv",              dtype={"patient_id": str})
targets    = pd.read_csv(data_dir/"targets.csv",            dtype={"patient_id": str})
biomarkers = pd.read_csv(data_dir/"biomarkers_original.csv",dtype={"patient_id": str})

if "recurrence" in biomarkers.columns:
    biomarkers = biomarkers.drop(columns=["recurrence"])

# ── Merge ──────────────────────────────────────────────────
df = clinical.merge(patho,      on="patient_id", how="outer")
df = df.merge(blood,            on="patient_id", how="outer")
df = df.merge(biomarkers,       on="patient_id", how="outer")
df = df.merge(targets,          on="patient_id", how="outer")
df = df.reset_index(drop=True)

# ── Keep only deceased patients ────────────────────────────
df_deceased = df[df["survival_status"] == "deceased"].copy()
print(f"Deceased patients: {len(df_deceased)}")

# ── Compute median survival per recurrence subgroup ────────
rec_yes = df_deceased[df_deceased["recurrence"] == "yes"]
rec_no  = df_deceased[df_deceased["recurrence"] == "no"]

median_rec_yes = rec_yes["days_to_last_information"].median()
median_rec_no  = rec_no["days_to_last_information"].median()

print(f"Median survival (Recurrence YES): {median_rec_yes:.1f} days")
print(f"Median survival (Recurrence NO):  {median_rec_no:.1f} days")

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

# ── Compute median survival days per group ─────────────────
group_survival = df_deceased.groupby("group")["days_to_last_information"].median()
print("\nMedian survival days per group:")
print(group_survival)

# ── Select numeric features ────────────────────────────────
exclude_cols = [
    "days_to_last_information", "days_to_recurrence",
    "days_to_progress", "days_to_metastasis", "days_to_rfs_event"
]
numeric_cols = df_deceased.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]
print(f"\nNumber of features: {len(feature_cols)}")

# ── Compute median feature value per group ─────────────────
group_medians = df_deceased.groupby("group")[feature_cols].median()

# ── Compute correlation between group feature medians
#    and group survival medians ─────────────────────────────
# Both have 4 values (one per group) → correlation is based on 4 data points

group_order = [
    "G1: Recurrence + Long",
    "G2: Recurrence + Short",
    "G3: No Recurrence + Short",
    "G4: No Recurrence + Long"
]

# Survival median for each group in the same order
survival_values = group_survival[group_order].values

correlation_results = []
for col in feature_cols:
    feature_values = group_medians.loc[group_order, col].values

    # Skip if all values are the same
    if np.std(feature_values) == 0:
        continue

    # Pearson correlation (4 data points: one per group)
    pearson_r,  pearson_p  = stats.pearsonr(feature_values,  survival_values)

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(feature_values, survival_values)

    correlation_results.append({
        "feature"    : col,
        "pearson_r"  : pearson_r,
        "pearson_p"  : pearson_p,
        "spearman_r" : spearman_r,
        "spearman_p" : spearman_p,
        "abs_pearson": abs(pearson_r)
    })

corr_df = pd.DataFrame(correlation_results)
corr_df = corr_df.sort_values("abs_pearson", ascending=False)

print("\nTop 20 features by Pearson correlation (group medians vs survival):")
print(corr_df.head(20)[["feature", "pearson_r", "pearson_p", "spearman_r"]].to_string(index=False))

# ── Save results ───────────────────────────────────────────
corr_df.to_csv(results_dir/"feature_correlation_v2.csv", index=False)
group_medians.to_csv(results_dir/"group_medians.csv")
print("\nSaved: feature_correlation_v2.csv")

# ── Plot 1: Top 20 Correlation Bar Chart ──────────────────
top20  = corr_df.head(20).copy()
colors = ["#d73027" if r > 0 else "#4575b4" for r in top20["pearson_r"]]

plt.figure(figsize=(10, 8))
plt.barh(top20["feature"], top20["pearson_r"], color=colors)
plt.axvline(x=0, color="black", linewidth=0.8)
plt.xlabel("Pearson Correlation Coefficient\n(Group Median Feature Value vs Group Median Survival Days)")
plt.title("Top 20 Features: Correlation between Group Feature Medians\nand Group Survival Medians (n=4 groups)")
plt.tight_layout()
plt.savefig(results_dir/"top20_correlation_v2.svg", bbox_inches="tight")
plt.close()
print("Saved: top20_correlation_v2.svg")

# ── Plot 2: Scatter plots for top 5 features ──────────────
top5_features = corr_df.head(5)["feature"].tolist()

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, col in zip(axes, top5_features):
    x = group_medians.loc[group_order, col].values
    y = survival_values
    r = corr_df[corr_df["feature"] == col]["pearson_r"].values[0]

    ax.scatter(x, y, color="steelblue", s=100, zorder=3)

    # Add group labels
    for xi, yi, label in zip(x, y, ["G1", "G2", "G3", "G4"]):
        ax.annotate(label, (xi, yi), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    # Trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=1)

    ax.set_xlabel(col, fontsize=8)
    ax.set_ylabel("Median Survival Days", fontsize=8)
    ax.set_title(f"r = {r:.3f}", fontsize=10)

plt.suptitle("Top 5 Features: Group Median vs Survival Days", fontsize=12)
plt.tight_layout()
plt.savefig(results_dir/"top5_scatter.svg", bbox_inches="tight")
plt.close()
print("Saved: top5_scatter.svg")

# ── Plot 3: NLR / PLR / LMR scatter ──────────────────────
biomarker_cols = [c for c in ["NLR", "PLR", "LMR"] if c in feature_cols]
if biomarker_cols:
    fig, axes = plt.subplots(1, len(biomarker_cols), figsize=(5*len(biomarker_cols), 4))
    if len(biomarker_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, biomarker_cols):
        x = group_medians.loc[group_order, col].values
        y = survival_values
        r = corr_df[corr_df["feature"] == col]["pearson_r"].values[0]

        ax.scatter(x, y, color="steelblue", s=100, zorder=3)
        for xi, yi, label in zip(x, y, ["G1", "G2", "G3", "G4"]):
            ax.annotate(label, (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)

        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=1)

        ax.set_xlabel(f"Median {col} per Group", fontsize=9)
        ax.set_ylabel("Median Survival Days", fontsize=9)
        ax.set_title(f"{col}  r = {r:.3f}", fontsize=10)

    plt.suptitle("NLR / PLR / LMR: Group Median vs Survival Days", fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir/"biomarker_scatter.svg", bbox_inches="tight")
    plt.close()
    print("Saved: biomarker_scatter.svg")

print(f"\nDone! All results saved to {results_dir}")