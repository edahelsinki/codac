import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv(path) for path in glob.glob("results/constrained_clustering/*.csv")])
df = df[["dataset", "model", "ari", "accuracy"]]
df2 = pd.concat([pd.read_csv(path) for path in glob.glob("results/constrained_clustering_ver2/*.csv")])

# codac with active queries
df3 = pd.concat([pd.read_csv(path) for path in glob.glob("results/constrained_clustering_active/*.csv")])
df3 = df3[["dataset", "model", "ari", "accuracy"]]

df2 = df2[["dataset", "model", "ari", "accuracy"]]
df2 = df2[df2["model"] == "DCC"]
df2.loc[:,"model"] = value = "DCC2"

df = pd.concat([df, df2, df3])


df_group = df.groupby(['dataset', 'model'])
stats = df_group[["ari", "accuracy"]].agg(['mean', 'std'])
print(stats)

# 1. Standard Aggregation
df_group = df.groupby(['dataset', 'model'])
stats = df_group[["ari", "accuracy"]].agg(['mean', 'std'])

# 2. Reset index and flatten MultiIndex columns
stats = stats.reset_index()
stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in stats.columns]

# 3. Create "mean (std)" strings
for metric in ["ari", "accuracy"]:
    stats[metric] = stats.apply(
        lambda row: f"{row[metric+'_mean']:.2f} ({row[metric+'_std']:.2f})", axis=1
    )

# 4. Melt the metrics into rows
final_df = stats[["dataset", "model", "ari", "accuracy"]]
melted = final_df.melt(id_vars=["dataset", "model"], var_name="metric")

# 5. Pivot: Swap 'dataset' and 'metric' in the index list
# This puts 'dataset' in the first column and 'metric' in the second
pivot_table = melted.pivot(index=["dataset", "metric"], columns="model", values="value")

# 6. Export to LaTeX
latex_table = pivot_table.to_latex(
    index=True,
    column_format="llccc", 
    caption="Clustering Results grouped by Dataset",
    escape=False # Useful if your metric names or values have underscores
)

print(latex_table)