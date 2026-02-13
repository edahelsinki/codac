import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state
from sklearn_extra.cluster import KMedoids


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
        init_clustering = KMedoids(n_clusters=n_clusters, init="k-medoids++", random_state=random_state)
        init_clustering.fit(X)
        medoid_ids = init_clustering.medoid_indices_
    else:
        init_clustering = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state)
        init_clustering.fit(X)
        # get medoids
        distances_to_clusters = cdist(init_clustering.cluster_centers_, X)
        medoid_ids = distances_to_clusters.argmin(1)
    return init_clustering, medoid_ids


if __name__ == "__main__":
    import time

    from clustpy.data import (
        load_banknotes,
        load_iris,
        load_mnist,
        load_optdigits,
        load_wine,
    )
    from sklearn.metrics import adjusted_rand_score
    X, y = load_optdigits(return_X_y=True)

    n_queries = 1000
    n_init_clusters = None
    random_state = 42
    n_ground_truth_clustres = 10
    
    # b = time.time()
    # model = COBRA(n_clusters_init=n_init_clusters, max_queries=n_queries, init_clustering_method="kmedoids", random_state=random_state, gt_n_clusters=n_ground_truth_clustres)
    # model.fit(X, y)
    # e = time.time()
    # labels = model.labels_
    # print("ARI:", adjusted_rand_score(y, labels))
    # print("Queries used", model.n_queries_used_)
    # print("Time", e-b)
    
    b = time.time()
    model = COBRA(n_clusters_init=n_init_clusters, max_queries=n_queries, init_clustering_method="kmeans", random_state=random_state, gt_n_clusters=n_ground_truth_clustres, eval_interval=[10, 20, 50, 100, 150])
    model.fit(X, y)
    e = time.time()
    labels = model.labels_
    print("ARI (labels):", adjusted_rand_score(y, labels))
    print("ARI (predict):", adjusted_rand_score(y, model.predict(X)))
    print("Queries used", model.n_queries_used_)
    print("Time", e-b)