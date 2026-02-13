import argparse
import datetime
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from clustering.codac import CODAC
from clustering.dcc import (
    IDEC,
    MNIST,
    FashionMNIST,
    generate_random_pair,
    transitive_closure,
)
from clustpy.deep import DCN
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from clustpy.metrics import (
    fair_normalized_mutual_information,
    unsupervised_clustering_accuracy,
)
from sklearn.metrics.cluster import adjusted_rand_score

from active_clustering_config import DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
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
        default="mnist",
        help="Dataset to be used.",
    )
    parser.add_argument(
        "-q",
        "--queries",
        type=int,
        default=1000,
        help="Maximum number of queries.",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=0,
        help="Index for repeated runs.",
    )
    parser.add_argument(
        "--load",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load a pre-trained model for CODAC",
    )
    parser.add_argument(
        "--ver2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ver 2 eval (DCC uses CODAC queries).",
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


def dcc_constraints(y, constraint_pairs):
    """
    Create constraint arrays for DCC from already sampled constraint pairs.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = y.to(torch.device("cpu"))
    y = y.numpy()
    for pair in constraint_pairs:
        tmp1 = pair[0]
        tmp2 = pair[1]
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)

    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def run_codac(
    n_queries,
    dataset_name,
    load_model,
    model_dir,
    model_name,
    labels_path,
    save_path,
    run_index,
    random_seed,
    use_active_queries=False,
    normalize_data=True
):  
    if dataset_name == "mnist" or dataset_name == "fashion":
        ae_layers = [784, 500, 500, 2000, 10]
    elif dataset_name == "bloodmnist":
        ae_layers = [2352, 500, 500, 2000, 10]
    else:
        raise TypeError("Unknown dataset.")

    # load dataset
    dataset = DATASETS[dataset_name]
    loaded_data = dataset["loader"](normalize=normalize_data, test_train_split=False)
    X, y = loaded_data

    if load_model:
        ae_model = FeedforwardAutoencoder(layers=ae_layers)
        ae_model.load_state_dict(torch.load(model_dir / model_name, weights_only=True))
        labels = np.load(model_dir / labels_path)
        print("pre-train ARI", adjusted_rand_score(y, labels))

    else:
        # pre-train an autoencoder (for codac)
        pre_train_model = DCN(n_clusters=10, pretrain_epochs=50, clustering_epochs=150, embedding_size=10)
        pre_train_model.fit(X)
        ae_model = pre_train_model.neural_network_trained_
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        labels = pre_train_model.predict(X)
        print("pre-train ARI", adjusted_rand_score(y, labels))
        torch.save(ae_model.state_dict(), model_dir / model_name)
        np.save(model_dir / labels_path, labels)

    # random seed
    seed = random_seed + run_index * 53

    # define and fit the model
    if use_active_queries:
        model = CODAC(n_clusters=10, embedding_size=10, max_queries=n_queries, neural_network=ae_model, random_state=seed)
        model_name = "CODAC-active"
    else:
        model = CODAC(n_clusters=10, embedding_size=10, max_queries=n_queries, sample_mode="random", neural_network=ae_model, random_state=seed)
        model_name = "CODAC"
    model.fit(X, y, init_labels=labels)

    n_queries_for_verifying = model.metadata["queries_used_for_verifying"]
    different_medoids_init = model.metadata["different_medoids_at_init"]
    violated_frac = model.metadata["violated_frac"]

    queried_points = model.queried_points

    preds = model.hist["labels"][-1]
    n_queries = model.hist["queries"][-1]

    #preds = model.predict(X)
    ari = adjusted_rand_score(y, preds)
    acc = unsupervised_clustering_accuracy(y, preds)
    fnmi = fair_normalized_mutual_information(y, preds)
    # write results
    # append the results to a csv file
    res = {
        "dataset": dataset_name,    
        "model": model_name,
        "queries": n_queries,
        "run-index": run_index,
        "ari": ari,
        "accuracy": acc,
        "fnmi": fnmi,
        "queries_for_verifying": n_queries_for_verifying,
        "different_medoids_at_init": different_medoids_init,
        "violated_frac": violated_frac,
    }
    write_row_to_csv(res, save_path)
    return queried_points


def run_DCC(dataset, num_constraints, load_model, save_path, run_index, queried_points, ver2=False):

    update_interval = 1
    epochs = 500
    batch_size = 256
    learning_rate = 0.001
    without_kmeans = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    if dataset == "mnist":
        mnist_train = MNIST('./dataset/mnist', train=True, download=True)
        mnist_test = MNIST('./dataset/mnist', train=False)
        X = mnist_train.train_data
        y = mnist_train.train_labels
        test_X = mnist_test.test_data
        test_y = mnist_test.test_labels
        X = torch.concat([X, test_X])
        y = torch.concat([y, test_y])
        pretrain_path = "../model/mnist_sdae_weights.pt"
        input_dim = 784
        n_clusters = 10
        
    if dataset == "fashion":
        fashionmnist_train = FashionMNIST('./dataset/fashion_mnist', train=True, download=True)
        fashionmnist_test = FashionMNIST('./dataset/fashion_mnist', train=False)
        X = fashionmnist_train.train_data
        y = fashionmnist_train.train_labels
        test_X = fashionmnist_test.test_data
        test_y = fashionmnist_test.test_labels
        X = torch.concat([X, test_X])
        y = torch.concat([y, test_y])
        pretrain_path = "../model/fashion_sdae_weights.pt"
        ml_penalty = 1
        input_dim = 784
        n_clusters = 10
    
    if dataset == "bloodmnist":
        loaded_data = DATASETS["bloodmnist"]["loader"](normalize=True, test_train_split=False)
        X, y = loaded_data
        X = torch.from_numpy(X).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        load_model = False
        input_dim = 2352
        n_clusters = 8
    
    # Set parameters
    ml_penalty, cl_penalty = 0.1, 1
    idec = IDEC(input_dim=input_dim, z_dim=10, n_clusters=n_clusters,
                encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", dropout=0)

    if load_model:
        print("loading a pretrained model")
        idec.load_model(pretrain_path)

    # Construct Constraints
    if ver2:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = dcc_constraints(y, queried_points)
    else:
        ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(y, num_constraints)
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, X.shape[0])

    # ml_ind1 = ml_ind1[:num_constraints]
    # ml_ind2 = ml_ind2[:num_constraints]
    # cl_ind1 = cl_ind1[:num_constraints]
    # cl_ind2 = cl_ind2[:num_constraints]

    anchor, positive, negative = np.array([]), np.array([]), np.array([])
    #instance_guidance = torch.zeros(X.shape[0]).cuda()
    instance_guidance = torch.zeros(X.shape[0]).to(device)
    #instance_guidance = torch.zeros(X.shape[0])
    use_global = False
    
    # Train Neural Network
    train_acc, train_nmi, epo = idec.fit(anchor, positive, negative, ml_ind1, ml_ind2, cl_ind1, cl_ind2, instance_guidance, use_global,  ml_penalty, cl_penalty, X, y,
                             lr=learning_rate, batch_size=batch_size, num_epochs=epochs,
                             update_interval=update_interval,tol=1*1e-3,use_kmeans=without_kmeans,plotting="")
    
    # Make Predictions
    preds = idec.predict(X)

    y_numpy = y.cpu().numpy()
    ari = adjusted_rand_score(y_numpy, preds)
    acc = unsupervised_clustering_accuracy(y_numpy, preds)
    fnmi = fair_normalized_mutual_information(y_numpy, preds)
    # write results
    # append the results to a csv file
    res = {
        "dataset": dataset,    
        "model": "DCC",
        "queries": num_constraints,
        "run-index": run_index,
        "ari": ari,
        "accuracy": acc,
        "fnmi": fnmi,
        "queries_for_verifying": np.nan,
        "different_medoids_at_init": np.nan,
        "violated_frac": np.nan,
    }
    write_row_to_csv(res, save_path)


def main():
    args = parse_args()
    
    # get a random seed from date
    date = datetime.datetime.now()
    random_seed_base = date.year + date.month + date.day

    # create a results dir
    results_dir = Path(__file__).parent / "results"
    results_dir = results_dir / Path(args.output)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # form a path for the results
    if args.ver2:
        res_path = results_dir / "constrained_clustering_{}_ver2.csv".format(args.data)
    else:
        res_path = results_dir / "constrained_clustering_{}.csv".format(args.data)

    model_dir = Path(__file__).parent / "models"
    model_name = "pretrained_DCN_{}.pt".format(args.data)
    labels_path = "pretrain_labels_{}.npy".format(args.data)
    
    queried_points = None

    print("Evaluating CODAC")
    start = time.time()
    queried_points = run_codac(args.queries, args.data, args.load, model_dir, model_name, labels_path, res_path, args.job_index, random_seed_base, normalize_data=args.normalize)
    duration = time.time() - start
    print("Evaluation completed in {} seconds".format(duration))

    print("Evaluating DCC")
    start = time.time()
    run_DCC(args.data, args.queries, args.load, res_path, args.job_index, queried_points, args.ver2)
    duration = time.time() - start
    print("Evaluation completed in {} seconds".format(duration))

    print("Evaluating CODAC active")
    start = time.time()
    run_codac(args.queries, args.data, args.load, model_dir, model_name, labels_path, res_path, args.job_index, random_seed_base, use_active_queries=True, normalize_data=args.normalize)
    duration = time.time() - start
    print("Evaluation completed in {} seconds".format(duration))

if __name__ == "__main__":
    main()
