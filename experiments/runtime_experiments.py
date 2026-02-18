from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.metrics import unsupervised_clustering_accuracy as acc, fair_normalized_mutual_information as fnmi
import time
import pandas as pd
import numpy as np
import os

from active2025.clustering.codac import CODAC
#from active2025.clustering.cobra import COBRA
from active2025.clustering.cobras import COBRAS
from active2025.clustering.acdm import ACDM
from active2025.clustering.ffqs import FFQS
from active2025.clustering.a3s import A3S


import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
#from sklearn_extra.cluster import KMedoids

import matplotlib.pyplot as plt
import matplotlib as mpl

MAX_TIME = 3 * 3600

def evaluate(L, clusterer):
    if hasattr(clusterer, "labels_"):
        labels = clusterer.labels_
    elif hasattr(clusterer, "intermediate_clusterings"):
        labels = np.array(clusterer.intermediate_clusterings[-1])
    elif hasattr(clusterer, "label_record"):
        labels = np.array(clusterer.label_record[-1])
    elif hasattr(clusterer, "records_"):
        labels = np.array(clusterer.records_[-1]["labels"])
    elif hasattr(clusterer, "history"):
        labels = np.array(clusterer.history["labels"][-1])
    else:
        raise Exception("Labels not available")
    clust_acc = acc(L, labels)
    clust_ari = ari(L, labels)
    clust_nmi = nmi(L, labels)
    clust_fnmi = fnmi(L, labels)
    print("ACC", clust_acc, "/ ARI", clust_ari, " / NMI", clust_nmi, " / FNMI", clust_fnmi)
    return clust_acc, clust_ari, clust_nmi, clust_fnmi

"""
COBRA
"""

def get_final_labels(N, medoid_labels, init_labels):
    # get final labels
    medoid_labels_orig = medoid_labels.copy()
    medoid_labels_new = medoid_labels.copy()
    unique_labels = np.unique(medoid_labels)
    final_labels = np.zeros(N, dtype=int)
    for i, l in enumerate(unique_labels):
        which_medoids = np.where(medoid_labels_orig == l)[0]
        medoid_labels_new[which_medoids] = i
        final_labels[np.isin(init_labels, which_medoids)] = i
    return final_labels, medoid_labels_new

class COBRA():    
    """COBRA algorithm.

    Craenendonck, T. V., Dumancic, S., & Blockeel, H. (2018).
    COBRA: A Fast and Simple Method for Active Clustering with Pairwise Constraints (No. arXiv:1801.09955).
    arXiv. https://doi.org/10.48550/arXiv.1801.09955
    """

    def __init__(self, n_clusters_init: int = None, max_queries: int = None, gt_n_clusters: int = None, init_clustering_method: str = "kmeans", eval_interval = None, random_state: int | np.random.RandomState = None):
        init_clustering_method = init_clustering_method.lower()
        assert init_clustering_method in ["kmedoids", "kmeans"], "methods must be 'kmedoids' or 'kmeans'"
        assert n_clusters_init is not None or max_queries is not None, "if n_init_clusters is None, max_queries must be specified"
        if n_clusters_init is None:
            if gt_n_clusters is not None:
                self.n_clusters_init = int(max_queries / gt_n_clusters + (gt_n_clusters + 1) / 2)
            else:
                self.n_clusters_init = int(0.5 + np.sqrt(0.25 + 2*max_queries))
            print("Setting n_clusters_init automatically. n_clusters_init =", self.n_clusters_init)
        else:
            self.n_clusters_init = n_clusters_init
        self.max_queries = max_queries
        self.gt_n_clusters = gt_n_clusters
        self.init_clustering_method = init_clustering_method
        self.eval_interval = eval_interval
        self.random_state = check_random_state(random_state)

    def fit(self, X:np.ndarray, y:np.ndarray):
        N = X.shape[0]
        assert N >= self.n_clusters_init, "n_clusters_init can not be larger then the number of samples in the dataset"
        if self.eval_interval is not None:
            eval_interval = self.eval_interval.copy()
        else:
            eval_interval = []
        # for recording the labels over the run
        labels_record = []
        queries_record = []
        # Initial over-clustering
        init_clustering, medoid_ids = run_init_clustering(X, self.n_clusters_init, self.init_clustering_method, self.random_state)
        # Get distances between medoids
        medoids = X[medoid_ids]
        medoids_distances = pdist(medoids)
        square_medoids_distances = squareform(medoids_distances)
        # Remove double entries by setting values in lower triangle matrix to inf
        square_medoids_distances += np.tril(np.ones(square_medoids_distances.shape) + np.inf)
        argsorted_medoid_distances = np.argsort(square_medoids_distances, axis=None)
        # Begin queries
        n_queries_used = 0
        medoid_labels = np.arange(self.n_clusters_init)
        ground_truth_medoid_labels = y[medoid_ids]
        cannot_links = np.zeros((self.n_clusters_init, self.n_clusters_init), dtype=bool)
        # Iterate over all (useful) medoid distances
        for entry in argsorted_medoid_distances[:len(medoids_distances)]:
            if n_queries_used in eval_interval:
                # record the labels in the query interval (for evaluation)
                eval_interval.remove(n_queries_used)
                final_labels, _ = get_final_labels(N, medoid_labels, init_clustering.labels_)
                queries_record.append(n_queries_used)
                labels_record.append(final_labels)

            s1, s2 = np.unravel_index(entry, square_medoids_distances.shape)
            # Check if already merged or cannot-link
            if medoid_labels[s1] != medoid_labels[s2] and cannot_links[medoid_labels[s1], medoid_labels[s2]] == 0:
                n_queries_used += 1
                # Query ground truth labels
                if ground_truth_medoid_labels[s1] == ground_truth_medoid_labels[s2]:
                    # must_link
                    cannot_links[medoid_labels == medoid_labels[s1]] += cannot_links[medoid_labels[s2]]
                    medoid_labels[(medoid_labels == medoid_labels[s1]) | (medoid_labels == medoid_labels[s2])] = medoid_labels[s1]
                    cannot_links[:, medoid_labels == medoid_labels[s1]] += cannot_links[medoid_labels[s2]].reshape(-1, 1)
                else:
                    # cannot-link
                    cannot_links[medoid_labels[s1], medoid_labels[s2]] = 1
                    cannot_links[medoid_labels[s2], medoid_labels[s1]] = 1
            if n_queries_used == self.max_queries:
                break
        # get final labels
        final_labels, new_medoid_labels = get_final_labels(N, medoid_labels, init_clustering.labels_)

        # populate the query/label record (if the algorithm terminated before covering all interval points)
        for q in eval_interval:
            queries_record.append(q)
            labels_record.append(final_labels)

        # save parameters
        self.labels_= final_labels
        self.cluster_medoids_ = medoids
        self.cluster_medoid_ids_ = medoid_ids
        self.medoid_labels_ = new_medoid_labels
        self.n_queries_used_ = n_queries_used
        self.queries_record = queries_record
        self.labels_record = labels_record
        return self

    def predict(self, X: np.ndarray):
        distances = cdist(X, self.cluster_medoids_)
        labels_tmp = np.argmin(distances, axis=1)
        labels = self.medoid_labels_[labels_tmp]
        return labels
    
    def fit_predict(self, X: np.ndarray, y:np.ndarray):
        self.fit(X, y)
        return self.labels_


def run_init_clustering(X:np.ndarray, n_clusters:int, init_clustering_method:str="kmedoids", random_state: int | np.random.RandomState=None):
    """Run k-medoids or k-means and return the clustering model (containing cluster labels) and the medoid ids.

    Returns:
        tuple of initial clustering model and array of mdedoid ids
    """
    if init_clustering_method == "kmedoids":
        raise Exception("kmedoids not supported at the moment")
        #init_clustering = KMedoids(n_clusters=n_clusters, init="k-medoids++", random_state=random_state)
        #init_clustering.fit(X)
        #medoid_ids = init_clustering.medoid_indices_
    else:
        init_clustering = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state)
        init_clustering.fit(X)
        # get medoids
        distances_to_clusters = cdist(init_clustering.cluster_centers_, X)
        medoid_ids = distances_to_clusters.argmin(1)
    return init_clustering, medoid_ids

"""
-------
"""

def run_codac(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        codac = CODAC(pretrain_epochs = 50, deep_cluster_epochs = 150, clustering_epochs = 100, ssl_loss_weight = 0.01, clustering_loss_weight = 0.001,
                      constraint_loss_weight = 1.0, mode = "cannot", sample_mode = "hybrid-sum-knn", uncertainty_weight = 0.6, deep_clusterer = "DCN",
                     max_queries=mq, embedding_size=n_clusters, n_clusters=n_clusters, random_state=random_state)
        codac.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, codac)
    except Exception as e:
        print("[ERROR]", e) 
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def run_cobra(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        cobra = COBRA(n_clusters_init=None, max_queries=mq, gt_n_clusters=n_clusters, random_state=random_state)
        cobra.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, cobra)
    except Exception as e:
        print("[ERROR]", e) 
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def run_cobras(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        cobras = COBRAS(max_queries=mq)
        cobras.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, cobras)
    except Exception as e:
        print("[ERROR]", e)
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def run_acdm(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        acdm = ACDM(max_queries=mq, random_state=random_state)
        acdm.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, acdm)
    except Exception as e:
        print("[ERROR]", e) 
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def run_ffqs(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        ffqs = FFQS(n_clusters=n_clusters, query_sample_size=100, max_queries=mq)
        ffqs.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, ffqs)
    except Exception as e:
        print("[ERROR]", e) 
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def run_a3s(n_clusters, X, L, mq, random_state):
    np.random.seed(random_state)
    b = time.time()
    try:
        a3s = A3S(max_queries=mq, T=1000, n_neighbors=50, tau=0.5, random_state=random_state)
        a3s.fit(X, L)
        e = time.time()
        runtime = e-b
        print("runtime", runtime)
        clust_acc, clust_ari, clust_nmi, clust_fnmi = evaluate(L, a3s)
    except Exception as e:
        print("[ERROR]", e) 
        runtime = -1
        clust_acc = 0
        clust_ari = 0
        clust_nmi = 0
        clust_fnmi = 0
    return clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime

def create_data(N, d, k, random_state):
    if N == 0:
        N = 1000
    if d == 0:
        d = 10
    if k == 0:
        k = 2
    X, L = make_blobs(n_samples=N, n_features=d, centers=k, random_state=random_state)
    return X, L, k

def run_runtime_experiment(exp, algo, iteration_init = 0, iteration_end = 11, n_runs = 10, save_dir = None):
    assert iteration_init >= 0
    assert iteration_end > iteration_init
    assert exp in ["N", "d", "k", "q"]
    assert algo in ["CODAC", "COBRA", "COBRAS", "ACDM", "FFQS", "A3S"]
    all_random_states = [1, 5, 15, 30, 50, 100, 200, 500, 700, 1000, 1500]

    d_base = 100
    k_base = 5
    N_base = 10000
    q_base = 500

    if save_dir is None:
        file_local = "runtime_exp_{0}_{1}.csv".format(algo, exp)
    else:
        file_local = save_dir + "runtime_exp_{0}_{1}.csv".format(algo, exp)
    if os.path.isfile(file_local):
        df = pd.read_csv(file_local, index_col = 0)
        print("read old csv for "+ exp)
    else:
        df = None
        
    for i in range(iteration_init, iteration_end):
        row_list = []
        for run in range(n_runs):
            random_state = all_random_states[run]
            if exp == "N":
                X, L, k = create_data(i*N_base, d_base, k_base, random_state)
            elif exp == "d":
                X, L, k = create_data(N_base, i * d_base, k_base, random_state)
            elif exp == "k":
                X, L, k = create_data(N_base, d_base, i * k_base, random_state)
            if exp == "q":
                X, L, k = create_data(N_base, d_base, k_base, random_state)
                mq = q_base * (i + 1)
            else:
                mq = q_base
            print("{0} - iteartion {1} - run {2} - X shape {3} - k {4} - q {5}".format(exp, i, run, X.shape, k, mq))
            if algo == "CODAC":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_codac(k, X, L, mq, random_state)
            elif algo == "COBRA":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_cobra(k, X, L, mq, random_state)
            elif algo == "COBRAS":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_cobras(k, X, L, mq, random_state)
            elif algo == "ACDM":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_acdm(k, X, L, mq, random_state)
            elif algo == "FFQS":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_ffqs(k, X, L, mq, random_state)
            elif algo == "A3S":
                clust_acc, clust_ari, clust_nmi, clust_fnmi, runtime = run_a3s(k, X, L, mq, random_state)
                
            row_list.append({"Experiment": "Increase_" + exp, "Algo": algo, "Run": run, "N": X.shape[0], "d": X.shape[1], "k": k, "q": mq, "ACC": clust_acc, "ARI": clust_ari, "NMI": clust_nmi, "FNMI": clust_fnmi, "runtime": runtime})
        df_new = pd.DataFrame(row_list, columns=["Experiment", "Algo", "Run", "N", "d", "k", "q", "ACC", "ARI", "NMI", "FNMI", "runtime"])
        if df is None:
            df = df_new
        else:
            df = pd.concat([df, df_new], ignore_index=True, sort=False)
        df.to_csv(file_local)
        avg_runtime = np.mean([entry["runtime"] for entry in row_list])
        if avg_runtime > MAX_TIME:
            print(f"[WARNING] Average runtime in last iteration was {avg_runtime} => ABORT")
            break
    return df


def create_plot(df, relevant_column, q_low=0.1, q_high=0.9, save_path=""):
    mpl.rcParams.update({
        # Figure & DPI (PDF is vector; DPI applies to raster outputs)
        "figure.figsize": (4.5, 3.2),          # per-subplot size; adjust per journal
        "figure.dpi": 150,
    
        # Fonts
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",          # or "Times New Roman" if journal requires serif
    
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
        "pdf.fonttype": 42,  # embed fonts as TrueType (editable in Illustrator)
        "ps.fonttype": 42,
    })
    colors = [
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
    # Define specific color mapping
    linestyle_map = {
        "CODAC": (7, "-", 1.0),
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

    fig, ax = plt.subplots()
    
    exp = "Increase_" + relevant_column
    df_data = df[df["Experiment"] == exp]
    max_x = df_data[relevant_column].max()
    min_x = df_data[relevant_column].min()
    x_range = max_x - min_x
    jitter_scale = x_range * 0.001
    for i, model in enumerate(df["Algo"].unique()):
        plot_name = model
        df_subset = df_data[df_data["Algo"] == model]
        df_subset = df_subset[df_subset["runtime"] != -1]

        df_agg = df_subset.groupby(["N", "d", "k", "q"]).agg(
                ari_median=pd.NamedAgg(column="ARI", aggfunc="median"),
                ari_mean=pd.NamedAgg(column="ARI", aggfunc="mean"),
                ari_std=pd.NamedAgg(column="ARI", aggfunc="std"),
                ari_q_low=pd.NamedAgg(
                    column="ARI", aggfunc=lambda x: np.quantile(x, q=q_low)
                ),
                ari_q_high=pd.NamedAgg(
                    column="ARI", aggfunc=lambda x: np.quantile(x, q=q_high)
                ),
                runtime_median=pd.NamedAgg(column="runtime", aggfunc="median"),
                runtime_mean=pd.NamedAgg(column="runtime", aggfunc="mean"),
                runtime_std=pd.NamedAgg(column="runtime", aggfunc="std"),
                runtime_q_low=pd.NamedAgg(
                    column="runtime", aggfunc=lambda x: np.quantile(x, q=q_low)
                ),
                runtime_q_high=pd.NamedAgg(
                    column="runtime", aggfunc=lambda x: np.quantile(x, q=q_high)
                )
            ).reset_index()
        print(df_agg["runtime_q_low"], df_agg["runtime_median"], df_agg["runtime_q_high"])
            
        # Get color index from map, fallback to i if not found
        color_idx, linestyle, linewidth = linestyle_map.get(model, (i, "-"))
        color = colors[color_idx]
    
        # Shaded IQR/CI band
        y_lo = df_agg["runtime_q_low"].to_numpy(float)
        y_hi = df_agg["runtime_q_high"].to_numpy(float)
        ax.fill_between(df_agg[relevant_column], y_lo, y_hi, color=color, alpha=0.15, linewidth=0, zorder=1)
        ax.plot(df_agg[relevant_column], df_agg["runtime_median"], color=color, label=plot_name, linestyle=linestyle, linewidth=linewidth, zorder=3)
    if relevant_column == "N":
        ax.set_xlabel("Number of data points ($n$)")
    elif relevant_column == "d":
        ax.set_xlabel("Number of dimensions ($d$)")
    elif relevant_column == "k":
        ax.set_xlabel("Number of ground truth clusters ($k_{gt}$)")
    elif relevant_column == "q":
        ax.set_xlabel("Total budget ($b_{total}$)")
    ax.set_ylabel("Runtime (s)")
    
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='-', alpha=0.4, linewidth=0.5)


    if True:
        leg = ax.legend(loc="lower right", fontsize=8, ncol=1, framealpha=0.5, handlelength=1.2, handletextpad=0.5, borderpad=0.2, columnspacing=0.8)
        leg.get_frame().set_linewidth(0.0)  # remove border
    ax.set_yscale('log')
    ax.set_ylim(0, MAX_TIME)

    #save_path = save_dir / file_name
    plt.savefig(save_path + "runtime_{0}.pdf".format(relevant_column), bbox_inches='tight')
    #plt.close()
    plt.show()
