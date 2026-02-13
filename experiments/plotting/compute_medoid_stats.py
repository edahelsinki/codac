import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

DATA_CLUSTERS = {
    "optdigits": 10,
    "pendigits": 10,
    "har": 6,
    "mnist": 10,
    "usps": 10,
    "fashion": 10,
    "waveform": 3,
    "handwritten": 10,
    "bloodmnist": 8,
    "reuters": 5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the CSV output file.",
    )
    args = parser.parse_args()
    return args

def compute_stats(df):
    models = ["CODAC-guess0.5", "CODAC", "CODAC-guess2.0"]

    results = []
    for dataset_name in df["dataset"].unique():
        print("--- {} ---".format(dataset_name))
        result = {"dataset": dataset_name}
        for model in models:
            mask = (df["dataset"] == dataset_name) & (df["model"] == model) & (df["queries"] == 0)
            df_subset = df[mask]
            #df_subset = df_subset[df_subset["queries"] == df_subset["queries"].max()]
            if len(df_subset) != 10:
                print(f"Warning: the model {model} had {len(df_subset)} entries for {dataset_name}")
            nan_mask = np.isnan(df_subset["queries_for_verifying"])
            if np.any(nan_mask):
                print("{}: {}: NaNs {}".format(dataset_name, model, np.sum(nan_mask)))

            mean_var = "{}_mean".format(model)
            std_var = "{}_std".format(model)
            result[mean_var] = np.round(df_subset["queries_for_verifying"].mean(), 1)
            result[std_var] = np.round(df_subset["queries_for_verifying"].std(), 1)
        results.append(result)
    return results

def main():
    args = parse_args()

    # load the results
    results = []
    if ".csv" in args.file:
        res = compute_stats(args.file)
        if res:
            results.append(res)
    else:
        # input is a directory, load and combine all .csv files
        path = Path(args.file)
        search_path = f"{str(path)}/*.csv"
        paths = [p for p in glob.glob(search_path)]
        df = pd.concat([pd.read_csv(p) for p in paths])
        results = compute_stats(df)
    if results:
        df_results = pd.DataFrame(results)
        df_results.set_index("dataset", inplace=True)
       
      

        # min_verify_queries = []
        # for dataset in df_results.index:
        #     k = DATA_CLUSTERS.get(dataset, 0)
        #     min_verify_queries.append((k * (k-1))/2)
        # df_results["min_queries"] = min_verify_queries

        processed_vars = ["CODAC-guess0.5", "CODAC", "CODAC-guess2.0"]
        print("\nSummary DataFrame:")
        print(df_results)
        print("\nLaTeX Table:")
        # Combine mean and std into a single column for each variable for LaTeX output
        df_latex = df_results.copy()
        for var in processed_vars:
            mean_col = f"{var}_mean"
            std_col = f"{var}_std"
            df_latex[var] = df_latex.apply(lambda row: f"{row[mean_col]:.1f} ({row[std_col]:.1f})", axis=1)
    
    df_latex = df_latex[processed_vars]
    print(df_latex.to_latex())


if __name__ == "__main__":
    main()