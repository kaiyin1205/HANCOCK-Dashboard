import pandas as pd
import numpy as np
from pathlib import Path

# ── 設定路徑 ──────────────────────────────────────────────
data_dir    = Path("features")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_name", type=str, help="Output filename (without .csv)")
args = parser.parse_args()
output_path = data_dir / f"{args.output_name}.csv"

# ── 載入資料 ───────────────────────────────────────────────
blood  = pd.read_csv(data_dir/"blood.csv",        dtype={"patient_id": str})
patho  = pd.read_csv(data_dir/"pathological.csv", dtype={"patient_id": str})
target = pd.read_csv(data_dir/"targets.csv",      dtype={"patient_id": str})

# ── 重新命名血液欄位（方便計算） ───────────────────────────
blood = blood.rename(columns={
    "Leukocytes [#/volume] in Blood"                              : "WBC",
    "Hemoglobin [Mass/volume] in Blood"                           : "Hemoglobin",
    "Platelets [#/volume] in Blood"                               : "Platelets",
    "Erythrocytes [#/volume] in Blood"                            : "RBC",
    "Hematocrit [Volume Fraction] of Blood"                       : "Hematocrit",
    "Erythrocyte mean corpuscular hemoglobin [Entitic mass]"      : "MCH",
    "Erythrocyte mean corpuscular volume [Entitic volume]"        : "MCV",
    "Erythrocyte mean corpuscular hemoglobin concentration [Mass/volume]": "MCHC",
    "Erythrocyte distribution width [Ratio]"                      : "RDW",
    "Platelet mean volume [Entitic volume] in Blood"              : "MPV",
    "Granulocytes [#/volume] in Blood"                            : "Granulocytes",
    "Eosinophils [#/volume] in Blood"                             : "Eosinophils",
    "Basophils [#/volume] in Blood"                               : "Basophils",
    "Lymphocytes [#/volume] in Blood"                             : "Lymphocytes",
    "Monocytes [#/volume] in Blood"                               : "Monocytes",
    "Platelet distribution width [Entitic volume] in Blood by Automated count": "PDW",
})

# ── 重建 Neutrophils（嗜中性球）─────────────────────────────
# 因為 blood.csv 沒有直接提供 Neutrophils，用公式反推
blood["Neutrophils"] = blood["Granulocytes"] - (blood["Eosinophils"] + blood["Basophils"])
blood["Neutrophils"] = blood["Neutrophils"].clip(lower=0)  # 避免負值

# ── 合併病理資料 ───────────────────────────────────────────
df = blood.merge(patho, on="patient_id", how="outer")

# ── 合併復發標籤（避免 data leakage，最後會移除） ──────────
df = df.merge(target[["patient_id", "recurrence"]], on="patient_id", how="left")

# ── 計算指標（優先五項放前面） ─────────────────────────────

# [優先 1] SII — Systemic Immune-Inflammation Index
# 文獻支持最強，整合血小板、嗜中性球、淋巴球
df["SII"] = (df["Platelets"] * df["Neutrophils"]) / df["Lymphocytes"]

# [優先 2] SIRI — Systemic Inflammation Response Index
# 跟 NLR 互補，加入單核球的影響
df["SIRI"] = (df["Neutrophils"] * df["Monocytes"]) / df["Lymphocytes"]

# [優先 3] PIV — Pan-Immune-Inflammation Value
# 最新最全面的綜合發炎指標
df["PIV"] = (df["Neutrophils"] * df["Platelets"] * df["Monocytes"]) / df["Lymphocytes"]

# [優先 4] Invasion_Score — 腫瘤侵犯綜合分數（來自病理資料）
# 整合四種侵犯指標，分數 0~4
invasion_cols = ["perinodal_invasion", "lymphovascular_invasion_L",
                 "vascular_invasion_V", "perineural_invasion_Pn"]
df["Invasion_Score"] = df[invasion_cols].apply(
    lambda row: pd.to_numeric(row, errors="coerce"), axis=1
).sum(axis=1, min_count=1)

# [優先 5] ELR — Eosinophil-to-Lymphocyte Ratio
# 嗜酸性球跟免疫反應相關
df["ELR"] = df["Eosinophils"] / df["Lymphocytes"]

# ── 其他指標 ───────────────────────────────────────────────

# NLR — Neutrophil-to-Lymphocyte Ratio（原有）
df["NLR"] = df["Neutrophils"] / df["Lymphocytes"]

# PLR — Platelet-to-Lymphocyte Ratio（原有）
df["PLR"] = df["Platelets"] / df["Lymphocytes"]

# LMR — Lymphocyte-to-Monocyte Ratio（原有）
df["LMR"] = df["Lymphocytes"] / df["Monocytes"]

# MLR — Monocyte-to-Lymphocyte Ratio
df["MLR"] = df["Monocytes"] / df["Lymphocytes"]

# BLR — Basophil-to-Lymphocyte Ratio
df["BLR"] = df["Basophils"] / df["Lymphocytes"]

# GLR — Granulocyte-to-Lymphocyte Ratio
df["GLR"] = df["Granulocytes"] / df["Lymphocytes"]

# PLT_WBC — Platelet-to-Leukocyte Ratio
df["PLT_WBC"] = df["Platelets"] / df["WBC"]

# RDW（直接使用原始欄位）
df["RDW_index"] = df["RDW"]

# MCV（直接使用原始欄位）
df["MCV_index"] = df["MCV"]

# ── 根據版本選擇輸出欄位 ───────────────────────────────────
version = args.output_name
if "original" in version:
    output_cols = ["patient_id", "NLR", "PLR", "LMR", "recurrence"]
elif "priority5" in version:
    output_cols = ["patient_id", "SII", "SIRI", "PIV", "Invasion_Score", "ELR", "NLR", "PLR", "LMR", "recurrence"]
elif "sii_invasion" in version:
    output_cols = ["patient_id", "SII", "Invasion_Score", "NLR", "PLR", "LMR", "recurrence"]
else:
    output_cols = ["patient_id", "SII", "SIRI", "PIV", "Invasion_Score", "ELR", "NLR", "PLR", "LMR", "MLR", "BLR", "GLR", "PLT_WBC", "RDW_index", "MCV_index", "recurrence"]

result = df[output_cols].copy()

# ── 儲存結果 ───────────────────────────────────────────────
result.to_csv(output_path, index=False)
print(f"Done! Saved {len(result)} patients to {output_path}")
print(f"\nColumns: {result.columns.tolist()}")
print(f"\nPreview:")
print(result.head())