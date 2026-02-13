import glob
import os
import os.path as osp
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# font sizes
title_size = 18
tick_size = 14
label_size = 16


df = pd.concat([pd.read_csv(csv_path) for csv_path in glob.glob("results/heatmap_part*/*.csv")])

df = df[df["queries"]==600]
df = df[['model', 'run-index', 'ari',]]


for model in df["model"].unique():
    n_reps = len(df[df["model"] == model])
    print("model {} : n={}".format(model, n_reps))

result = df.groupby(['model'])['ari'].mean()

constraint_w = []
ssl_w = []

for s in result.index.to_list():
    weights = s.split("con=")[-1].split(",ssl=")
    constraint_w.append(float(weights[0]))
    ssl_w.append(float(weights[1]))

data_df = pd.DataFrame({"const": constraint_w, "ssl": ssl_w, "ari": result.to_numpy()})

# Assuming your dataframe is named 'df'
# index = Y-axis, columns = X-axis, values = Color Intensity
heatmap_data = data_df.pivot(index='const', columns='ssl', values='ari')

# create the heatmap figure
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, annot_kws={"size": label_size})

plt.title(r'Interaction of $\lambda_1$ and $\lambda_2$', fontsize=title_size)
plt.xlabel('$\lambda_1$', fontsize=label_size)
plt.ylabel('$\lambda_2$', fontsize=label_size)
plt.xticks(fontsize=title_size)
plt.yticks(fontsize=title_size)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=title_size)
cbar.set_label('ARI', fontsize=label_size)
plt.savefig('figures/lambda_heatmap.pdf', bbox_inches='tight')