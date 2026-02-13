import argparse
import datetime
import glob
import io
import math
import os
import time
import traceback
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from active_clustering_config import DATASETS, MODEL_LOADERS, _data_specs
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.metrics import (
    fair_normalized_mutual_information,
    unsupervised_clustering_accuracy,
)
from sklearn.metrics.cluster import adjusted_rand_score

plt.rcParams["figure.figsize"] = (6, 4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the CSV output file.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use train/test split for CODAC",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output directory in results.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize data",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default=None,
        help="Dataset to be used.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to the experiment configuration YAML.",
    )
    parser.add_argument(
        "-q",
        "--queries",
        type=int,
        default=None,
        help="Maximum number of queries (overrides the config).",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Index for repeated runs.",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use wandb to log the results.",
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


def create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total):
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
        "violated_frac": np.nan,
        "runtime_total": runtime_total,
        "runtime_querying": np.nan,
        "runtime_optimizing": np.nan,
        "model_params": params_str
        }
    return res


def cobra_init_clusters_bound(Q, K):
    # Q max number of queries, K number of true clusters
    return math.ceil(Q / K + (K + 1) / 2)


def train_eval_cobra(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    max_queries,
    query_batch_size,
    save_path,
    ae_model,
    run_index,
    pretrain_time,
    use_bound_eval=False,
    random_seed=None
):  
    model_params.pop("query_sample_size") # not used in cobra

    # add random seed
    model_params["random_state"] = random_seed + run_index * 53

    # cleaned parameters for saving in csv
    to_text_params = model_params.copy()

    if "Deep" in model_name:
        model_params["neural_network"] = ae_model

    # fit COBRA
    if use_bound_eval:
        # we compute the initial number of clusters from a bound given the number of true clusters and the queries
        # COBRA will be run on this number of initial clusters until it converges

        # initialize the number of queries
        n_queries = 0
        for _ in range(math.ceil(max_queries/query_batch_size)):
            # increment the queries
            n_queries += query_batch_size
            # fit the model with increasing number of queries (each separate .fit() call has the same query budget)
            if n_queries > max_queries:
                # fix the last iteration query number if it went over the limit
                n_queries = max_queries

            model_params["max_queries"] = n_queries
            model = model_class(**model_params)
            time_start = time.time()
            model.fit(X, y)
            runtime_total = time.time() - time_start + pretrain_time

            preds = model.labels_

            # extract the number of used queries
            n_queries_used = model.n_queries_used_

            to_text_params["n_init_clusters"] = model.n_clusters_init
            to_text_params["max_queries"] = n_queries
            to_text_params["queries_used"] = n_queries_used
            params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

            # append the results to a csv file
            ari, acc, fnmi = compute_metrics(y, preds)
            res = create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total)
            write_row_to_csv(res, save_path)
    else:
        # use regular evaluation for COBRA
        query_interval = []
        q = 0
        while q <= max_queries:
            query_interval.append(q)
            q += query_batch_size

        params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())
        model_params["eval_interval"] = query_interval
        model = model_class(**model_params)
        time_start = time.time()
        model.fit(X, y)
        runtime_total = time.time() - time_start + pretrain_time

        for queries, preds in zip(model.queries_record, model.labels_record):
            ari, acc, fnmi = compute_metrics(y, preds)
            res = create_csv_row(dataset_name, model_name, queries, run_index, ari, acc, fnmi, params_str, runtime_total)
            write_row_to_csv(res, save_path)


def train_eval_cobras(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    max_queries,
    query_batch_size,
    save_path,
    run_index,
    ae_model,
    pretrain_time,
):
    # remove sample size param (not used in COBRAS)
    model_params.pop("query_sample_size")

    to_text_params = model_params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

    if "Deep" in model_name:
        model_params["neural_network"] = ae_model

    print("Eval: {}, \n{}".format(model_name, params_str))
    # define and fit the model
    model = model_class(**model_params)
    time_start = time.time()
    model.fit(X, y)
    runtime_total = time.time() - time_start + pretrain_time

    labels = []
    queries = []

    # add initial state (without queries)
    queries.append(0)
    labels.append(np.array(model.intermediate_clusterings[0]))
    # get the state at 100, 200, ..., max_queries
    q = query_batch_size
    while q <= max_queries:
        queries.append(q)
        labels.append(np.array(model.intermediate_clusterings[q-1]))
        q = q + query_batch_size

    # write results (predictions are saved during fitting)
    for preds, n_queries in zip(labels, queries):
        # write results
        ari, acc, fnmi = compute_metrics(y, preds)
        res = create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total)
        write_row_to_csv(res, save_path)


def train_eval_codac(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    save_path,
    ae_model,
    run_index,
    experiment_name,
    use_wandb,
    random_seed,
    pretrain_time,
    X_test=None,
    y_test=None,
):  
    to_text_params = model_params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

    # add pre-trained ae to params
    model_params["neural_network"] = ae_model
    
    # random seed
    model_params["random_state"] = random_seed + run_index * 53

    if use_wandb:
        # wandb configuration
        run_name = "{}_R{}".format(model_name, run_index)
        group_name = "{}_{}".format(experiment_name, dataset_name)
        try:
            run = wandb.init(group=group_name, name=run_name, config=to_text_params)
        except Exception:
            print("Exception occurred while initializing wandb, disabling wandb logging.")
            print(traceback.format_exc())
            run = wandb.init(mode="disabled")

        model_params["wandb_run"] = run
    else:
        run = wandb.init(mode="disabled")

    print("Eval: {}, \n{}".format(model_name, params_str))
    # define and fit the model
    model = model_class(**model_params)
    time_start = time.time()
    model.fit(X, y, X_test, y_test)
    runtime_total = time.time() - time_start + pretrain_time

    n_queries_for_verifying = model.metadata["queries_used_for_verifying"]
    different_medoids_init = model.metadata["different_medoids_at_init"]
    violated_frac = model.metadata["violated_frac"]
    runtime_query = model.runtime["querying"]
    runtime_optimize = model.runtime["optimizing"]

    # write results (labels are saved during fitting)
    if X_test is None:
        model_name_str = model_name
    else:
        model_name_str = "{}-train".format(model_name)

    for preds, n_queries in zip(model.hist["labels"], model.hist["queries"]):
        # metrics
        ari = adjusted_rand_score(y, preds)
        acc = unsupervised_clustering_accuracy(y, preds)
        fnmi = fair_normalized_mutual_information(y, preds)
        # write results
        # append the results to a csv file
        res = {
            "dataset": dataset_name,    
            "model": model_name_str,
            "queries": n_queries,
            "run-index": run_index,
            "ari": ari,
            "accuracy": acc,
            "fnmi": fnmi,
            "queries_for_verifying": n_queries_for_verifying,
            "different_medoids_at_init": different_medoids_init,
            "violated_frac": violated_frac,
            "runtime_total": runtime_total,
            "runtime_querying": runtime_query,
            "runtime_optimizing": runtime_optimize,
            "model_params": params_str
        }
        write_row_to_csv(res, save_path)

    if X_test is not None:
        # write test results
        model_name_str = "{}-test".format(model_name)
        for preds, n_queries in zip(model.hist["test_labels"], model.hist["queries"]):
            ari = adjusted_rand_score(y_test, preds)
            acc = unsupervised_clustering_accuracy(y_test, preds)
            fnmi = fair_normalized_mutual_information(y_test, preds)
            res = {
                "dataset": dataset_name,    
                "model": model_name_str,
                "queries": n_queries,
                "run-index": run_index,
                "ari": ari,
                "accuracy": acc,
                "fnmi": fnmi,
                "queries_for_verifying": n_queries_for_verifying,
                "different_medoids_at_init": different_medoids_init,
                "violated_frac": violated_frac,
                "runtime_total": runtime_total,
                "runtime_querying": runtime_query,
                "runtime_optimizing": runtime_optimize,
                "model_params": params_str
            }
            write_row_to_csv(res, save_path)


def train_eval_acdm(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    max_queries,
    query_batch_size,
    save_path,
    run_index,
    ae_model,
    pretrain_time,
):  
    # remove sample size param (not used in ACDM)
    model_params.pop("query_sample_size")

    to_text_params = model_params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

    if "Deep" in model_name:
        model_params["neural_network"] = ae_model

    print("Eval: {}, \n{}".format(model_name, params_str))
    # define and fit the model
    model = model_class(**model_params)

    time_start = time.time()
    model.fit(X, y)
    runtime_total = time.time() - time_start + pretrain_time

    for item in model.records_:
        n_queries = item["queries"]
        pred = np.array(item["labels"])
        ari = adjusted_rand_score(y, pred)
        acc = unsupervised_clustering_accuracy(y, pred)
        fnmi = fair_normalized_mutual_information(y, pred)
        # append the results to a csv file
        res = create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total)
        write_row_to_csv(res, save_path)


def train_eval_a3s(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    max_queries,
    query_batch_size,
    save_path,
    run_index,
    ae_model,
    pretrain_time,
):  
    # remove sample size param
    model_params.pop("query_sample_size")

    to_text_params = model_params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

    if "Deep" in model_name:
        model_params["neural_network"] = ae_model

    print("Eval: {}, \n{}".format(model_name, params_str))
    # define and fit the model
    model = model_class(**model_params)
    time_start = time.time()
    model.fit(X, y)
    runtime_total = time.time() - time_start + pretrain_time

    for n_queries, pred in zip(model.history["queries"], model.history["labels"]):
        pred = np.array(pred)
        ari = adjusted_rand_score(y, pred)
        acc = unsupervised_clustering_accuracy(y, pred)
        fnmi = fair_normalized_mutual_information(y, pred)
        # append the results to a csv file
        res = create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total)
        write_row_to_csv(res, save_path)


def train_eval_ffqs(
    X,
    y,
    dataset_name,
    model_class,
    model_params,
    model_name,
    save_path,
    run_index,
    ae_model,
    pretrain_time,
): 
    to_text_params = model_params.copy()
    params_str = ", ".join(f"{k}={v}" for k, v in to_text_params.items())

    if "Deep" in model_name:
        model_params["neural_network"] = ae_model

    print("Eval: {}, \n{}".format(model_name, params_str))
    # define and fit the model
    model = model_class(**model_params)
    time_start = time.time()
    model.fit(X, y)
    runtime_total = time.time() - time_start + pretrain_time

    # write results (predictions are saved during fitting)
    for preds, n_queries in zip(model.label_record, model.query_record):
        # write results
        ari, acc, fnmi = compute_metrics(y, preds)
        res = create_csv_row(dataset_name, model_name, n_queries, run_index, ari, acc, fnmi, params_str, runtime_total)
        write_row_to_csv(res, save_path)


def process_results(
    df, dataset=None, model=None, q_low=0.1, q_high=0.9
):
    """
    Process raw results dataframe for plotting

    - select results for a given dataset and model
    - compute medians and quantiles over repeats
    """
    #features_methods = (
    #    df["features_method"].unique() if features_methods is None else features_methods
    #)
    df0 = (
        df.query(
            "dataset == @dataset and "
            "model in @model"
        )
        .groupby(["dataset", "model", "queries"])
        .agg(
            ari_median=pd.NamedAgg(column="ari", aggfunc="median"),
            ari_mean=pd.NamedAgg(column="ari", aggfunc="mean"),
            ari_std=pd.NamedAgg(column="ari", aggfunc="std"),
            ari_q_low=pd.NamedAgg(
                column="ari", aggfunc=lambda x: np.quantile(x, q=q_low)
            ),
            ari_q_high=pd.NamedAgg(
                column="ari", aggfunc=lambda x: np.quantile(x, q=q_high)
            )
            # ari_mean=pd.NamedAgg(column="ari", aggfunc="mean"),
            # ari_std=pd.NamedAgg(
            #     column="ari", aggfunc=lambda x: np.std(x)
            # ),
        )
        .reset_index()
    )
    # # Sort data frame by the given order of feature_methods. Affects legend order.
    # df0["features_method"] = pd.Categorical(
    #     df0["features_method"], categories=features_methods, ordered=True
    # )
    # df0 = df0.sort_values(by=["features_method"])
    return df0


def plot_results(df, save_dir):
    colors = matplotlib.colormaps.get_cmap('tab10')

    # create a figures dir
    #save_dir = Path(__file__).parent / "figures"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for dataset in df["dataset"].unique():
        fig, ax = plt.subplots()
        # horizontal offset to make the errorbars not overlap
        df_ = df[df["dataset"] == dataset]
        x_range = df_["queries"].max() - df_["queries"].min()
        h_offset = x_range * 0.002
        for i, model in enumerate(df["model"].unique()):
            # Aggregate the results from repeated clusterings
            df_agg = process_results(df, dataset, model)
            ax.plot(df_agg["queries"]+i*h_offset, df_agg["ari_median"], color=colors(i), label=model, linewidth=1)
            ax.errorbar(x=df_agg["queries"]+i*h_offset, y=df_agg["ari_median"], yerr=[np.abs(df_agg["ari_q_low"]-df_agg["ari_median"]),df_agg["ari_q_high"]-df_agg["ari_median"]], fmt=".", color=colors(i), linewidth=1)
            #ax.plot(df_agg["queries"]+i*h_offset, df_agg["ari_mean"], color=colors(i), label=model, linewidth=1)
            #ax.errorbar(x=df_agg["queries"]+i*h_offset, y=df_agg["ari_mean"], yerr=df_agg["ari_std"], fmt=".", color=colors(i), linewidth=1)
        ax.set_xlabel("Queries")
        ax.set_ylabel("ARI")
        ax.set_title(dataset)
        plt.legend(loc="best")
        file_name = "active_clustering_{}.pdf".format(dataset)
        save_path = save_dir / file_name
        print("Saving figure to path {}".format(save_path))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def main():
    X_test = None
    y_test = None

    args = parse_args()

    all_datasets = ["optdigits", "pendigits", "har", "three-mnist", "letter-recog"]
    
    # get a random seed from date
    date = datetime.datetime.now()
    random_seed_base = date.year + date.month + date.day

    if args.data is not None:
        all_datasets = [args.data]

    # create a results dir
    results_dir = Path(__file__).parent / "results"
    results_dir = results_dir / Path(args.output)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # read the experiment config
    with open(args.config) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    MODELS = CONFIG["models"]
    general_config = CONFIG["general"]
    # write the general config to results
    config_dump_dir = results_dir / "ex_configs"
    Path(config_dump_dir).mkdir(parents=True, exist_ok=True)
    with io.open(config_dump_dir / "general_config.yaml", "w", encoding="utf8") as outfile:
        yaml.dump(general_config, outfile, default_flow_style=False, allow_unicode=True)

    # for logging the used model configs
    configs_dump = {}

    n_datasets = len(all_datasets)
    n_models = len(MODELS.keys())

    n_total_experiments = n_datasets * n_models

    i = 1
    for dataset_name in all_datasets:
        dataset = DATASETS[dataset_name]
        print("------------------")
        print(dataset_name)
        print("------------------")

        # load dataset
        loaded_data = dataset["loader"](normalize=args.normalize, test_train_split=args.test)
        if args.test:
            X, X_test, y, y_test = loaded_data
        else:
            X, y = loaded_data

        max_queries = dataset["max_queries"]

        # form a path for the results
        if args.data is None:
            res_path = results_dir / "active_clustering.csv"
        else:
            res_path = results_dir / "{}_{}.csv".format(dataset_name, args.job_index)
            
        # pre-train an autoencoder
        time_start = time.time()
        ae_model = FeedforwardAutoencoder(layers=_data_specs[dataset_name]["ae_layers"]).fit(n_epochs=general_config["pretrain_epochs"], batch_size=256, data=X)
        pretrain_time = time.time() - time_start

        for model_name in MODELS.keys():
            model_config = MODELS[model_name]
            model_class = MODEL_LOADERS[model_config["model_loader_key"]]
            model_params = model_config["params"].copy()
            # dataset specific configs
            model_data_config = dataset["model_config"][model_config["data_config_key"]]
            # add the data specific parameters to general parameters
            for param in model_data_config.keys():
                model_params[param] = model_data_config[param]

            configs_dump = {model_name: model_params.copy()}
            config_filename = "configs_{}_{}.yaml".format(dataset_name, model_name)
            # Write the experiment config to the result dir (before every experiment)
            with io.open(config_dump_dir / config_filename, "w", encoding="utf8") as outfile:
                yaml.dump(configs_dump, outfile, default_flow_style=False, allow_unicode=True)

            # override the max_queries from the config
            if args.queries is not None:
                max_queries = args.queries
                model_params["max_queries"] = args.queries
            print("\n######################################")
            print(model_name)
            print("######################################\n")
            query_sample_size = model_params["query_sample_size"]
            # evaluate and write results to a file
            print("[{}/{}] dataset: {}, model: {}".format(i, n_total_experiments, dataset_name, model_name))
            
            start = time.time()
            # fit and evaluate a single run
            try:
                if "COBRAS" in model_name:
                    train_eval_cobras(
                    X,
                    y,
                    dataset_name,
                    model_class,
                    model_params,
                    model_name,
                    max_queries,
                    query_sample_size,
                    res_path,
                    args.job_index,
                    ae_model,
                    pretrain_time,
                )
                elif "COBRA" in model_name:
                    # determine if we use bound evaluation or "regular" evaluation
                    if "bound" in model_name:
                        use_bound_eval = True
                    else:
                        use_bound_eval = False
                    train_eval_cobra(
                    X,
                    y,
                    dataset_name,
                    model_class,
                    model_params,
                    model_name,
                    max_queries,
                    query_sample_size,
                    res_path,
                    ae_model,
                    args.job_index,
                    pretrain_time,
                    use_bound_eval,
                    random_seed_base,
                )
                elif "AAEC" in model_name or "CODAC" in model_name:
                    train_eval_codac(
                        X,
                        y,
                        dataset_name,
                        model_class,
                        model_params,
                        model_name,
                        res_path,
                        ae_model,
                        args.job_index,
                        args.output,
                        args.wandb,
                        random_seed_base,
                        pretrain_time,
                        X_test,
                        y_test,
                    )
                elif "ACDM" in model_name:
                    train_eval_acdm(X, y, dataset_name, model_class, model_params, model_name, max_queries, query_sample_size, res_path, args.job_index, ae_model, pretrain_time)
                elif "A3S" in model_name:
                    train_eval_a3s(X, y, dataset_name, model_class, model_params, model_name, max_queries, query_sample_size, res_path, args.job_index, ae_model, pretrain_time)
                elif "FFQS" in  model_name:
                    train_eval_ffqs(X, y, dataset_name, model_class, model_params, model_name, res_path, args.job_index, ae_model, pretrain_time)
                else:
                    print("Warning, unknown method. Skipping.")
            except Exception:
                print("Ran into an error while evaluating the method")
                traceback.print_exc()

            i +=1
            duration = time.time() - start
            print("Evaluation completed in {} seconds".format(duration))

if __name__ == "__main__":
    main()
