import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def read_results(dir_path):
    search = Path(dir_path) / "*fashion*.csv"
    csv_paths = glob.glob(str(search))
    if len(csv_paths) == 0:
        return None
    df = pd.concat([pd.read_csv(path) for path in csv_paths])
    return df

#df_deep = read_results("results/aclust_compare_deep/")
df_codac = read_results("results/runtime_codac/")
df_shallow = read_results("results/aclust_compare_shallow/")
df_a3s_shallow = read_results("results/aclust_compare_a3s_shallow/")
# df_a3s_deep = read_results("results/aclust_compare_a3s_deep/")
df_ffqs_shallow = read_results("results/aclust_compare_ffqs_shallow/")
# df_ffqs_deep = read_results("results/aclust_compare_ffqs_deep/")

df = pd.concat([df_codac, df_shallow, df_a3s_shallow, df_ffqs_shallow])

df = df[df["queries"]==0]
df = df[['model', 'runtime_total', 'runtime_querying', 'runtime_optimizing']]
df["runtime_init"] = df["runtime_total"] - df["runtime_optimizing"] - df["runtime_querying"]

variables = ["runtime_init", "runtime_querying", "runtime_optimizing", "runtime_total"]
df = df[["model"] + variables]

df_group = df.groupby(['model'])
stats = df_group[variables].agg(['mean', 'std'])
print(stats)

formatted_df = pd.DataFrame(index=stats.index)

for var in variables:
    formatted_df[var] = stats[var].apply(
        lambda x: f"{x['mean']:.0f} ± {x['std']:.0f}", axis=1
    )

print(formatted_df.to_latex())