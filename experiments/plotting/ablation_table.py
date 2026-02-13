import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#paths = glob.glob("results/ablations/ablation_loss/*.csv")
paths = glob.glob("results/ablation_loss/*.csv")
data = []
for path in paths:
    tmp = pd.read_csv(path)
    tmp = tmp[tmp["queries"] == tmp["queries"].max()]
    data.append(tmp)

df = pd.concat(data)
df = df[['dataset', 'model', 'ari', 'accuracy', 'fnmi']]

df_group = df.groupby(['dataset', 'model'])
stats = df_group[["ari"]].agg(['mean', 'std'])
# build a table with models as columns and dataset as rows
stats = stats.copy()
stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
stats = stats.reset_index()

pivot_mean = stats.pivot(index='dataset', columns='model', values='ari_mean')
pivot_std = stats.pivot(index='dataset', columns='model', values='ari_std')

# print(pivot_mean.describe())

# format cells as "mean (std)" and bold best model(s) per dataset
formatted = pivot_mean.copy().astype(object)

for idx in pivot_mean.index:
    row = pivot_mean.loc[idx]
    if row.dropna().empty:
        for col in pivot_mean.columns:
            formatted.at[idx, col] = '-'
        continue

    maxv = row.max()
    if pd.isna(maxv):
        for col in pivot_mean.columns:
            formatted.at[idx, col] = '-'
        continue

    # std corresponding to the first max mean model
    try:
        max_model = row.idxmax()
        max_std = pivot_std.at[idx, max_model]
    except Exception:
        max_std = 0.0

    threshold = maxv - (max_std if not pd.isna(max_std) else 0.0)

    for col in pivot_mean.columns:
        m = pivot_mean.at[idx, col]
        s = pivot_std.at[idx, col]
        if pd.isna(m):
            formatted.at[idx, col] = '-'
            continue
        cell = f"{m:.3f} ({s:.3f})"
        # bold if mean+std >= (max_mean - std_of_max_model)
        if (m + (0.0 if pd.isna(s) else s)) >= (threshold - 1e-12):
            cell = "\\textbf{" + cell + "}"
        formatted.at[idx, col] = cell

# print and save LaTeX
latex = formatted.to_latex(escape=False, na_rep='-')
print(latex)
# with open('results/ablations/ablation_ari_table.tex', 'w') as f:
#     f.write(latex)

for data in df.dataset.unique():
    print(data)
    subset = stats[stats["dataset"] == data]
    max_idx = np.argmax(subset["ari_mean"])
    max_tr = subset["ari_mean"].values[max_idx] - subset["ari_std"].values[max_idx]
    upper_values = subset["ari_mean"].values + subset["ari_std"].values
    bold_mask = max_tr <= upper_values
    print("bold: ", subset["model"][bold_mask].values)
    print("=====================")
    
    