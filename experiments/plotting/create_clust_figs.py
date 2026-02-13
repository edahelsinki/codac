import argparse
import glob
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clustpy.metrics import (
    fair_normalized_mutual_information,
    unsupervised_clustering_accuracy,
)
from sklearn.metrics.cluster import adjusted_rand_score

plt.rcParams["figure.figsize"] = (6, 4)

import matplotlib as mpl

OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#999999",  # gray (optional extra)
]

mpl.rcParams.update({
    "figure.figsize": (4.5, 3.2),
    "figure.dpi": 150,

    # Fonts
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "font.family": "DejaVu Sans",

    # Lines & markers
    "lines.linewidth": 0.8,
    "lines.markersize": 2,

    # Grid & spines
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,

    # Save & fonts embedding
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Positional argument: one or more CSV file paths
    parser.add_argument(
        'files',
        metavar='FILE',
        type=str,
        nargs='+',
        help='Path(s) to directories containing experimental results'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="figures",
        help="Output directory for results.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="all",
        help="What type of plot is created, i.e., which methods are plotted, options [compare, abl-loss, abl-sampling, abl-clusterer, abl-margin, all]",
    )
    parser.add_argument(
        "-l",
        "--legend",
        type=str,
        default="all",
        help="Which dataset figures should have a legend (default all)",
    )

    args = parser.parse_args()
    return args


def write_row_to_csv(results, save_path):
    df = pd.DataFrame([results])  # dataframe with one row
    # append results to a file, creates a new file if one does not exists
    df.to_csv(
        save_path,
        mode="a",
        header=not os.path.exists(save_path),
        index=False,
    )


def compute_metrics(y, preds):
    ari = adjusted_rand_score(y, preds)
    acc = unsupervised_clustering_accuracy(y, preds)
    fnmi = fair_normalized_mutual_information(y, preds)
    return ari, acc, fnmi


def create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str):
    res = {
        "dataset": dataset_name,    
        "model": model_name,
        "queries": n_queries,
        "run-index": run_index,
        "ari": ari,
        "accuracy": acc,
        "fnmi": fnmi,
        "queries_for_verifying": np.nan,
        "different_medoids_at_init": np.nan,
        "model_params": params_str
        }
    return res


def cobra_init_clusters_bound(Q, K):
    # Q max number of queries, K number of true clusters
    return math.ceil(Q / K + (K + 1) / 2)


def process_results(
    df, q_low=0.1, q_high=0.9
):
    """
    Process raw results dataframe for plotting

    - select results for a given dataset and model
    - compute medians and quantiles over repeats
    """
    df0 = df.groupby(["dataset", "model", "queries"]).agg(
            ari_median=pd.NamedAgg(column="ari", aggfunc="median"),
            ari_mean=pd.NamedAgg(column="ari", aggfunc="mean"),
            ari_std=pd.NamedAgg(column="ari", aggfunc="std"),
            ari_q_low=pd.NamedAgg(
                column="ari", aggfunc=lambda x: np.quantile(x, q=q_low)
            ),
            ari_q_high=pd.NamedAgg(
                column="ari", aggfunc=lambda x: np.quantile(x, q=q_high)
            )
        ).reset_index()
    return df0

def interpolate_results(df, query_batch_size, max_queries):

    results = []
    for run_index in df["run-index"].unique():
        df_subset = df[df["run-index"] == run_index]
        # interpolate the results to get a regular grid of values
        x = np.array(df_subset["queries"], dtype=float)
        x_grid = np.arange(0, max_queries+1, query_batch_size, dtype=float)
        df_results = pd.DataFrame({"queries": x_grid.astype(int)})

        for metric in ["ari", "accuracy", "fnmi"]:
            y_real = np.array(df_subset[metric], dtype=float)
            y_grid = np.interp(x_grid, x, y_real, left=y_real[0], right=y_real[-1])
            df_results[metric] = y_grid

        model = df_subset["model"].iloc[0]
        dataset = df_subset["dataset"].iloc[0]
        run_index = df_subset["run-index"].iloc[0]

        df_results["model"] = model
        df_results["dataset"] = dataset
        df_results["run-index"] = run_index
        results.append(df_results)
    
    return pd.concat(results)


def plot_results(df, save_dir, add_legend, plot_type="all"):
    colors = OKABE_ITO

    # Define specific color mapping
    linestyle_map = {
        "CODAC": (7, "-", 1.0),
        "CODAC-DCN": (7, "-", 1.0),
        "CODAC-DKM": (0, "-", 1.0),
        "CODAC-IDEC": (1, "-", 1.0),
        "CODAC-deepkmeans": (2, "-", 1.0),
        "CODAC-train": (0, "-", 1.0),
        "CODAC-test": (1, "-", 1.0),
        "CODAC-guess0.5": (1, "-", 1.0),
        "CODAC-guess2.0": (2, "-", 1.0),
        "CODAC-nomust": (0, "-", 1.0),
        "CODAC-betamax": (1, "-", 1.0),
        "CODAC-beta1000": (2, "-", 1.0),
        "CODAC-nopush": (0, "-", 1.0),
        "CODAC-nosoftmin": (1, "-", 1.0),
        "CODAC-nossl": (2, "-", 1.0),
        "CODAC-noclust": (3, "-", 1.0),
        "CODAC-nodensity": (0, "-", 1.0),
        "CODAC-uncertainty": (1, "-", 1.0),
        "CODAC-diversity": (2, "-", 1.0),
        "CODAC-random-sample": (3, "-", 1.0),
        "CODAC-0.4": (5, "-", 1.0),
        "CODAC-0.5": (0, "-", 1.0),
        "CODAC-0.6": (7, "-", 1.0),
        "CODAC-0.7": (1, "-", 1.0),
        "CODAC-0.8": (2, "-", 1.0),
        "CODAC-w1": (0, "-", 1.0),
        "CODAC-n=2": (1, "-", 1.0),
        "CODAC-n=10": (2, "-", 1.0),
        "CODAC-n=20": (7, "-", 1.0),
        "CODAC-n=100": (3, "-", 1.0),
        "CODAC-n=1000": (4, "-", 1.0),
        "CODAC-omega-0.01": (0, "-", 1.0),
        "CODAC-omega-0.1": (1, "-", 1.0),
        "CODAC-omega-1": (2, "-", 1.0),
        "CODAC-omega-10": (7, "-", 1.0),
        "CODAC-omega-100": (3, "-", 1.0),
        "CODAC-b10": (0, "-", 1.0),
        "CODAC-b50": (1, "-", 1.0),
        "CODAC-b200": (2, "-", 1.0),
        "CODAC-b500": (3, "-", 1.0),
        "CODAC-random": (7, ":", 0.8),
        "COBRA": (0, "-", 0.8),
        "Deep-COBRA": (0, "--", 0.8),
        "COBRA-bound": (6, "-", 0.8),
        "Deep-COBRA-bound": (6, "--", 0.8),
        "COBRAS": (1, "-", 0.8),
        "Deep-COBRAS": (1, "--", 0.8),
        "ACDM": (2, "-", 0.8),
        "Deep-ACDM": (2, "--", 0.8),
        "FFQS": (3, "-", 0.8),
        "Deep-FFQS": (3, "--", 0.8),
        "A3S": (5, "-", 0.8),
        "Deep-A3S": (5, "--", 0.8),
    }
    
    if plot_type == "compare":
        methods_ordered = ["CODAC", "CODAC-random", "COBRA", "Deep-COBRA", "COBRAS", "Deep-COBRAS", "ACDM", "Deep-ACDM", "FFQS", "Deep-FFQS", "A3S", "Deep-A3S"]
        legend_names = ["CODAC", "Random", "COBRA", "Deep-COBRA", "COBRAS", "Deep-COBRAS", "ACDM", "Deep-ACDM", "FFQS", "Deep-FFQS", "A3S", "Deep-A3S"]
    elif plot_type == "abl-loss":
        methods_ordered = ["CODAC", "CODAC-nopush", "CODAC-nosoftmin", "CODAC-nossl"]
        legend_names = ["CODAC", r"No $\mathcal{L}_{cl}$ I (CL Push)", r"No $\mathcal{L}_{cl}$ II (SoftMin)", r"No $\mathcal{L}_{ssl}$"]
    elif plot_type == "abl-sampling":
        methods_ordered = ["CODAC", "CODAC-nodensity", "CODAC-uncertainty", "CODAC-diversity", "CODAC-random-sample"]
        legend_names = ["CODAC", "No Density", "Only Uncertainty", "Only Diversity", "Random"]
    elif plot_type == "abl-clusterer":
        methods_ordered = ["CODAC-DCN", "CODAC-DKM", "CODAC-IDEC", "CODAC-deepkmeans"]
        legend_names = ["DCN", "DKM", "IDEC", "k-Means"]
    elif plot_type == "abl-margins":
        methods_ordered = ["CODAC", "CODAC-nomust", "CODAC-betamax", "CODAC-beta1000"]
        legend_names = ["CODAC", r"$\alpha = 0$", r"$\beta_{max}$", r"$\beta = 1000$"]
    elif plot_type == "test":
        methods_ordered = ["CODAC", "CODAC-train", "CODAC-test"]
        legend_names = ["All", "Train", "Test"]
    elif plot_type == "guess-k":
        methods_ordered = ["CODAC", "CODAC-guess0.5", "CODAC-guess2.0"]
        legend_names = [r"$\hat{k} = k_{gt}$", r"$\hat{k} = k_{gt}/2$", r"$\hat{k} = 2k_{gt}$"]
    elif plot_type == "abl-eta":
        methods_ordered = ["CODAC-0.5", "CODAC-0.6", "CODAC-0.7", "CODAC-0.8"]
        legend_names = [r"$\eta=0.5$", r"$\eta=0.6$", r"$\eta=0.7$", r"$\eta=0.8$"]
    elif plot_type == "loss-weight":
        methods_ordered = ["CODAC-both0.01", "CODAC-rec0.1-clus0.01", "CODAC-rec0.01-clus0.001"]
    elif plot_type == "softmin-omega":
        methods_ordered = ["CODAC-omega-0.01", "CODAC-omega-0.1", "CODAC-omega-1", "CODAC-omega-10", "CODAC-omega-100"]
        legend_names = [r"$\omega=0.01$", r"$\omega=0.1$", r"$\omega=1$", r"$\omega=10$", r"$\omega=100$"]
    elif plot_type == "density-neighbors":
        methods_ordered = ["CODAC-n=2", "CODAC-n=10", "CODAC-n=20", "CODAC-n=100", "CODAC-n=1000"]
        legend_names = ["neighbors=2", "neighbors=10", "neighbors=20", "neighbors=100", "neighbors=1000"]
    elif plot_type == "init-budget":
        methods_ordered = ["CODAC-b10", "CODAC-b50", "CODAC", "CODAC-b200", "CODAC-b500"]
        legend_names = [r"$b_1=10$", r"$b_1=50$", r"$b_1=100$", r"$b_1=200$", r"$b_1=500$"]
    else:
        raise TypeError("Unknown plot type")

    query_batch_size = 100
    
    if len(methods_ordered) > 5:
        legend_cols = 2
    else:
        legend_cols = 1

    for dataset in df["dataset"].unique():
        fig, ax = plt.subplots()

        print("\n----------- Plotting {} -----------".format(dataset))

        # horizontal offset to make the errorbars not overlap
        df_data = df[df["dataset"] == dataset]
        max_queries = df_data[df_data["model"]=="CODAC"]["queries"].max()

        x_range = max_queries - df_data["queries"].min()
        jitter_scale = x_range * 0.001
        #for i, model in enumerate(df["model"].unique()):
        for i, model in enumerate(methods_ordered):
            plot_name = legend_names[i] #r"{}".format(legend_names[i])
            df_subset = df_data[df_data["model"] == model]
            # verify that all runs completed
            n_completed_runs = len(df_subset["run-index"].unique())
            if n_completed_runs != 10:
                print("WARNING: {}:{} completed runs {}/10".format(model, dataset, n_completed_runs))

            if n_completed_runs == 0:
                continue
            # Aggregate the results from repeated clusterings
            if model == "ACDM" or model == "Deep-ACDM" or model == "A3S" or model == "Deep-A3S":
                #df_subset = df_subset[['dataset', 'model', 'queries', 'run-index', 'ari', 'accuracy', 'fnmi']]
                df_subset = interpolate_results(df_subset, query_batch_size, max_queries)
                #df_agg = process_results(df_subset)
            #else:
                #df_agg = process_results(df_subset)
            if dataset == "pendigits" or dataset == "reuters" or dataset == "waveform" or dataset == "webkb":
                df_subset = df_subset[df_subset["queries"]<= 800]
            df_agg = process_results(df_subset)
            #if dataset == "mnist":
            #    print(df_agg)
            if np.all(df_agg["ari_q_high"] <= 0.01):
                print("THE LINE {} IS CONSTANT ZERO!".format(model))
                # print(df_agg)

            # Get color index from map, fallback to i if not found
            color_idx, linestyle, linewidth = linestyle_map.get(model, (i, "-"))
            color = colors[color_idx]#colors(color_idx)

            # get random uniform jitter
            h_offset = 0.0 #np.random.uniform(low=-jitter_scale, high=jitter_scale)

            # Shaded IQR/CI band
            y_lo = df_agg["ari_q_low"].to_numpy(float)
            y_hi = df_agg["ari_q_high"].to_numpy(float)
            ax.fill_between(df_agg["queries"]+i*h_offset, y_lo, y_hi, color=color, alpha=0.15, linewidth=0, zorder=1)
            ax.plot(df_agg["queries"]+i*h_offset, df_agg["ari_median"], color=color, label=plot_name, linestyle=linestyle, linewidth=linewidth, zorder=3)
            #ax.errorbar(x=df_agg["queries"]+i*h_offset, y=df_agg["ari_median"], yerr=[np.abs(df_agg["ari_q_low"]-df_agg["ari_median"]),df_agg["ari_q_high"]-df_agg["ari_median"]], fmt="o", color=color)
            #ax.plot(df_agg["queries"]+i*h_offset, df_agg["ari_median"], color=color, label=model, linewidth=1, linestyle=linestyle)
            #ax.errorbar(x=df_agg["queries"]+i*h_offset, y=df_agg["ari_median"], yerr=[np.abs(df_agg["ari_q_low"]-df_agg["ari_median"]),df_agg["ari_q_high"]-df_agg["ari_median"]], fmt=".", color=color, linewidth=1)
            #ax.plot(df_agg["queries"]+i*h_offset, df_agg["ari_mean"], color=colors(i), label=model, linewidth=1)
            #ax.errorbar(x=df_agg["queries"]+i*h_offset, y=df_agg["ari_mean"], yerr=df_agg["ari_std"], fmt=".", color=colors(i), linewidth=1)
        ax.set_xlabel("Queries")
        ax.set_ylabel("ARI")

        # set the title (dataset name)
        #ax.set_title(dataset)
        ax.text(
            0.02, 0.98, dataset,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.0),
            zorder=5
        )


        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='-', alpha=0.4, linewidth=0.5)

        # ax.set_ylim(0.0, 1.0)

        # Comment this block if you prefer only a legend.
        # for line in ax.get_lines():
        #     xdata, ydata = line.get_xdata(), line.get_ydata()
        #     if len(xdata) == 0:
        #         continue
        #     x_last, y_last = xdata[-1], ydata[-1]
        #     label = line.get_label()
        #     # Slight right shift in data coords
        #     ax.text(
        #         x_last + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
        #         y_last,
        #         label,
        #         color=line.get_color(),
        #         va="center", ha="left", fontsize=8.5,
        #         bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.5),
        #         zorder=4,
        #     )

        if add_legend == "all" or add_legend == dataset:
            leg = ax.legend(loc="lower right", fontsize=8, ncol=legend_cols, framealpha=0.5, handlelength=1.2, handletextpad=0.5, borderpad=0.2, columnspacing=0.8)
            leg.get_frame().set_linewidth(0.0)  # remove border
            #plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        file_name = "{}.pdf".format(dataset)
        save_path = save_dir / file_name
        print("Saving figure to path {}".format(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def read_results_from_dir(path):

    path = Path(path)
    search_path = "{}/*.csv".format(str(path))
    results = [pd.read_csv(path) for path in glob.glob(search_path)]
    df = pd.concat(results)
    return df

def main():
    np.random.seed(42)
    args = parse_args()

    save_path = Path(args.output)
    save_path.mkdir(parents=True, exist_ok=True)

    all_results = [read_results_from_dir(path) for path in args.files]
    df = pd.concat(all_results)
    plot_results(df, save_path, args.legend, args.type)

if __name__ == "__main__":
    main()
