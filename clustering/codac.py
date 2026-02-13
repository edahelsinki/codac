import math
import time
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from clustpy.deep import DCN, DKM, IDEC
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import (
    detect_device,
    encode_batchwise,
    mean_squared_error,
    squared_euclidean_distance,
)
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from clustpy.utils import plot_2d_data
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.neighbors import NearestNeighbors

from clustering.deep_kmeans import DeepKmeans
from clustering.utils import (
    ConstraintManager,
    extract_constraint_pairs,
    purity_score,
)


def evaluate(labels: np.ndarray, ground_truth: np.ndarray, labels_test: np.ndarray = None, gt_test: np.ndarray = None):
    my_acc = acc(ground_truth, labels)
    my_ari = ari(ground_truth, labels)
    my_nmi = nmi(ground_truth, labels)
    print(f"ACC: {my_acc}, ARI: {my_ari}, NMI: {my_nmi}")
    assert (labels_test is None and gt_test is None) or (labels_test is not None and gt_test is not None)
    if gt_test is not None:
        my_acc_test = acc(gt_test, labels_test)
        my_ari_test = ari(gt_test, labels_test)
        my_nmi_test = nmi(gt_test, labels_test)
        print(f"TEST - ACC: {my_acc}, ARI: {my_ari}, NMI: {my_nmi}")


def _plot_embedding(embedded_data, labels, medoid_ids, ground_truth, constraints):
    plot_2d_data(
        embedded_data,
        labels,
        embedded_data[medoid_ids],
        ground_truth,
        show_plot=False,
        scattersize=20,
        equal_axis=True,
        centers_scattersize=5,
    )
    n_clusters = len(medoid_ids)
    for c in range(n_clusters):
        with_ml = constraints[:, c] == 1
        # cannot linked points
        cl_points = constraints[:, c] == -1
        plt.scatter(
            embedded_data[cl_points, 0],
            embedded_data[cl_points, 1],
            s=10,
            c="gray",
            marker="x",
        )
        # plot must linked points
        in_cluster = labels[with_ml] == c
        plt.scatter(
            embedded_data[with_ml, 0][in_cluster],
            embedded_data[with_ml, 1][in_cluster],
            s=5,
            c="g",
        )
        plt.scatter(
            embedded_data[with_ml, 0][~in_cluster],
            embedded_data[with_ml, 1][~in_cluster],
            s=5,
            c="r",
        )

        # plot must lines
        with_ml_idx = np.where(constraints[:, c] == 1)[0]
        for ml in with_ml_idx:
            plt.plot(
                [embedded_data[ml, 0], embedded_data[medoid_ids[c], 0]],
                [embedded_data[ml, 1], embedded_data[medoid_ids[c], 1]],
                c="g",
                alpha=0.3,
            )

        # plot cannot lines
        with_cl = np.where(constraints[:, c] == -1)[0]
        for cl in with_cl:
            # only plot cannot-links that are violated
            if c == labels[cl]:
                plt.plot(
                    [embedded_data[cl, 0], embedded_data[medoid_ids[c], 0]],
                    [embedded_data[cl, 1], embedded_data[medoid_ids[c], 1]],
                    c="r",
                    alpha=0.5,
                )

    plt.show()


def _get_amount_fulfilled_constraints(labels, medoid_ids, constraints):
    # Must links
    must_links_fulfilled = [0, 0]
    # Cannot links
    cannot_links_fulfilled = [0, 0]
    for c in range(medoid_ids.shape[0]):
        must_links = constraints[:, c] == 1
        must_links_fulfilled[0] += (labels[must_links] == c).sum() - 1 # (medoid to itself should be ignored)
        must_links_fulfilled[1] += must_links.sum() - 1
        cannot_links = constraints[:, c]==-1
        cannot_links_fulfilled[0] += (labels[cannot_links] != c).sum()
        cannot_links_fulfilled[1] += cannot_links.sum()
    return must_links_fulfilled, cannot_links_fulfilled


def _update_labels_and_medoids(medoid_ids, embedded_data, constraints):
    #medoid_ids = np.unique(medoid_ids)
    labels, _ = pairwise_distances_argmin_min(X=embedded_data, Y=embedded_data[medoid_ids],
                                              metric='euclidean',
                                              metric_kwargs={'squared': True})

    # Update medoids
    for c in range(medoid_ids.shape[0]):
        mask = labels == c
        if np.any(mask):
            optimal_embedded_centers = [np.mean(embedded_data[labels == c], axis=0)]
            connected_ids = np.where(constraints[:, c] == 1)[0]
            new_medoid_ids = _get_nearest_object(optimal_embedded_centers, embedded_data[connected_ids])[0]
            medoid_ids[c] = connected_ids[new_medoid_ids]
    return labels, medoid_ids

def min_max_scale(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def _initialize_empty_cluster(ids_relevant, X_embedded, constraints, subsample_coverage=True):
    # pick a new candidate medoid by maximizing coverage
    X_relevant = X_embedded[ids_relevant]
    
    if subsample_coverage and len(ids_relevant) > 10_000:
        # subsample to avoid waisting memory and compute resources on the pairwise distance computations
        ids_relevant = np.random.choice(ids_relevant, size=10_000, replace=False)
        X_relevant = X_embedded[ids_relevant]

    sampled_mask = np.any(constraints != 0, axis=1)
    dists = cdist(X_relevant, X_embedded[sampled_mask])
    dist_nearest_medoid = np.min(dists, axis=1)
    D = cdist(X_relevant, X_relevant)
    n_relevant = len(X_relevant)
    scores = np.zeros(n_relevant)
    for i in range(n_relevant):
        # compute gain: how much addition of the current point i as the medoid would improve the coverage
        scores[i] = np.sum(np.maximum(0, dist_nearest_medoid - D[i]))

    # vectorized marginal gain computation 
    # NOTE: this uses a lot of memory: consider subsampling X_relevant or using float32
    # scores = np.maximum(0, dist_nearest_medoid[:, None] - D).sum(axis=0)
    best_new_medoid = ids_relevant[np.argmax(scores)]

    return best_new_medoid


def _query_init_medoids(X_embedded: np.ndarray, labels: np.ndarray, ground_truth: np.ndarray, medoid_ids: np.ndarray, n_max_queries: int, constraints: np.ndarray):

    n_constraints_used = 0
    X_medoids = X_embedded[medoid_ids]
    medoid_dists = squareform(pdist(X_medoids))

    deleted_idx = []
    # pairs to query
    pairs = []
    for c1 in range(medoid_ids.shape[0] - 1):
        for c2 in range(c1 + 1, medoid_ids.shape[0]):
            pairs.append([c1, c2])
    pairs = np.array(pairs)

    queried_points = []

    # sort pairs by distance
    sorted_pair_idx = np.argsort(medoid_dists[pairs[:, 0], pairs[:, 1]])
    for pair in pairs[sorted_pair_idx]:
        c1 = pair[0].item()
        c2 = pair[1].item()

        if n_constraints_used == n_max_queries:
            break
        
        if c1 in deleted_idx or c2 in deleted_idx:
            # one of the medoids were deleted already, skip
            continue

        # Check that no constraint already exists
        if constraints[medoid_ids[c1], labels[medoid_ids[c2]]] != 0:
            continue

        # the medoid pair has not been queried yet
        n_constraints_used += 1
        queried_points.append([medoid_ids[c1], medoid_ids[c2]])
        if ground_truth[medoid_ids[c1]] != ground_truth[medoid_ids[c2]]:
            # print(f"clusters {c1} and {c2} are different")
            # propagate a cannot-link for each must linked points
            constraints[constraints[:, c1] == 1, c2] = -1
            constraints[constraints[:, c2] == 1, c1] = -1
        else:
            # the medoids are supposed to be in the same cluster, merge c1 and c2
            size_c1 = (constraints[:, c1] == 1).sum()
            size_c2 = (constraints[:, c2] == 1).sum()
            # merge the cluster with fewer must links
            cluster_to_substitute = c1 if size_c1 < size_c2 else c2
            cluster_to_keep = c2 if size_c1 < size_c2 else c1
            # print(f"clusters {c1} and {c2} are the same, delete {cluster_to_substitute}")
            # Move all constraints to cluster_to_keep and delete everything in cluster_to_substitute
            has_constraint_with_substitute = constraints[:, cluster_to_substitute] != 0
            constraints[has_constraint_with_substitute, cluster_to_keep] = constraints[has_constraint_with_substitute, cluster_to_substitute]
            constraints[:, cluster_to_substitute] = 0
            # add the deleted medoid to a list (for skipping the queries concerning it)
            deleted_idx.append(cluster_to_substitute)

    # keep only the verified medoids
    if len(deleted_idx)>0:
        deleted_medoids = medoid_ids[np.array(deleted_idx)]
        verified_medoids_idx = np.where(~np.isin(medoid_ids, deleted_medoids))[0]
        medoid_ids = medoid_ids[verified_medoids_idx]
        constraints = constraints[:, verified_medoids_idx]
    return constraints, n_constraints_used, medoid_ids, queried_points


def add_new_medoid(X_embedded, labels, constraints, medoid_ids, new_medoid_id):
    # append a new column to constraint matrix
    constraints = np.concatenate((constraints, np.zeros((X_embedded.shape[0], 1), dtype=int)), axis=1)
    # other medoids should be cannot-linked with the new medoid (by definition)
    constraints[medoid_ids, -1] = -1
    # make the point a medoid
    medoid_ids = np.concat([medoid_ids, [new_medoid_id]])
    # medoid is must linked to itself and cannot linked to other medoids
    constraints[new_medoid_id, :-1] = -1
    constraints[new_medoid_id, -1] = 1
    labels[new_medoid_id] = constraints.shape[1] - 1 # update the new medoid label
    # update clustering with new medoids (re-assign cluster labels)
    labels, medoid_ids = _update_labels_and_medoids(medoid_ids, X_embedded, constraints)
    return labels, constraints, medoid_ids


def verify_embedding(X_embedded: np.ndarray, labels: np.ndarray, ground_truth: np.ndarray, medoid_ids: np.ndarray, max_queries: int, n_constraints_used, constraints: np.ndarray, n_true_clusters, n_queries_true_k, queried_points):
    """Sample a new point from regions with no queries and match the point with medoids until a must link is found.
    Continue until the budget is spent.
    """
    n_clusters = constraints.shape[1]
    while n_constraints_used < max_queries:
        # points that have not been sampled yet
        ids_relevant = np.where(~np.any(constraints != 0, axis=1))[0]

        # sample a new point
        best_new_point = _initialize_empty_cluster(ids_relevant, X_embedded, constraints)
        found_cluster = False
        dists = cdist(X_embedded[best_new_point].reshape(1, -1), X_embedded[medoid_ids]).flatten()
        idx_sorted = np.argsort(dists)
        for i in range(n_clusters):
            if n_constraints_used == max_queries:
                found_cluster = True # set the flag to true to avoid adding a new medoid when stopping verifying
                break

            c = labels[medoid_ids[idx_sorted[i]]]
            queried_points.append([best_new_point, medoid_ids[c]])
            if ground_truth[best_new_point] == ground_truth[medoid_ids[c]]:
                constraints[best_new_point, c] = 1
                n_constraints_used +=1
                found_cluster = True
                break

            else:
                constraints[best_new_point, c] = -1
                n_constraints_used +=1

        if not found_cluster:
            # print("Found a point that does not belong to any current cluster, adding a new medoid.")
            labels, constraints, medoid_ids = add_new_medoid(X_embedded, labels, constraints, medoid_ids, best_new_point)
            n_clusters = len(medoid_ids)
            if n_clusters == n_true_clusters:
                n_queries_true_k = n_constraints_used

    # handle cannot transitivity
    constraints = cannot_transitivity(constraints)
    return constraints, n_constraints_used, medoid_ids, labels, n_queries_true_k, queried_points


def cannot_transitivity(constraints):
    """Infere additional cannot-links through transitivity"""
    for c in range(constraints.shape[1]):
        in_same_cluster = constraints[:, c] == 1
        any_cannots = np.where(np.any(constraints[in_same_cluster] == -1, axis=0))[0]
        for cannot_id in any_cannots:
            constraints[in_same_cluster, cannot_id] = -1
    return constraints


def _add_constraints(X_embedded: np.ndarray, labels: np.ndarray, ground_truth: np.ndarray,
                     medoid_ids: np.ndarray, n_new_constraints: int,
                     constraints: np.ndarray, sample_mode="uncertainty", uncertainty_weight=1.0, n_nearest_neighbors=20, queried_points=None):
    time_start = time.time()
    n_added = 0
    last_added = -1
    cluster_medoids = X_embedded[medoid_ids]
    # Get points without must_link
    has_must_link = np.any(constraints == 1, axis=1)
    ids_points_wo_must_link = np.where(~has_must_link)[0]
    n_points_wo_must_link = ids_points_wo_must_link.shape[0]
    
    # Check if number of new constraints is possible
    n_new_constraints_adj = min(n_new_constraints, n_points_wo_must_link)

    if sample_mode == "random":
        # handle random querying
        n_data_points, n_clusters = constraints.shape

        while n_added < n_new_constraints_adj:
            data_idx = np.random.choice(n_data_points)
            cluster_idx = np.random.choice(n_clusters)

            if constraints[data_idx, cluster_idx] != 0:
                # constraint already sampled, continue
                continue
            
            # add must/cannot link
            n_added +=1
            queried_points.append([data_idx, medoid_ids[cluster_idx]])
            if ground_truth[data_idx] == ground_truth[medoid_ids[cluster_idx]]:
                constraints[data_idx, cluster_idx] = 1
            else:
                constraints[data_idx, cluster_idx] = -1

        # adjust cannot-links (transitivity)
        constraints = cannot_transitivity(constraints)
        runtime = time.time() - time_start
        return labels, medoid_ids, constraints, n_added, runtime, queried_points

    # non-random querying

    # Get the shortest distance to other cluster (uncertainty score)
    distances_to_clusters = cdist(X_embedded[ids_points_wo_must_link], cluster_medoids)
    distances_to_clusters = np.sort(distances_to_clusters, axis=1)
    distances_to_other_cluster = (distances_to_clusters[:, 0] / distances_to_clusters[:, 1])

    if sample_mode == "hybrid-sum-knn" or sample_mode == "diversity" or sample_mode == "linear":
        knn = NearestNeighbors(n_neighbors=n_nearest_neighbors)
        knn.fit(X_embedded[ids_points_wo_must_link])
        d, _ = knn.kneighbors(X_embedded[ids_points_wo_must_link])
        avg_knn_dist = d.mean(axis=1)
        avg_knn_dist = np.maximum(avg_knn_dist, 1e-8) # to avoid division by zero

    # compute exploration score
    for j in range(n_points_wo_must_link):
        if n_added >= n_new_constraints_adj:
            break
        # Get Object farthest away from points that are already used for a constraint
        if j == 0:
            distances_to_constraint = np.min(cdist(X_embedded[ids_points_wo_must_link], X_embedded[has_must_link]), axis=1)
            distances_to_constraint = distances_to_constraint

        else:
            distances_to_constraint = np.minimum(distances_to_constraint,
                                                 cdist(X_embedded[ids_points_wo_must_link], [X_embedded[last_added]])[:, 0])
        
        if sample_mode == "diversity":
            max_id = np.argmax(distances_to_constraint/avg_knn_dist)
            id_new_constraint = ids_points_wo_must_link[max_id]
        elif sample_mode == "hybrid":
            max_id = np.argmax(distances_to_constraint * distances_to_other_cluster**uncertainty_weight)
            id_new_constraint = ids_points_wo_must_link[max_id]
        elif sample_mode == "hybrid-sum":
            # minmax scale the diversity
            min_dist = distances_to_constraint.min()
            max_dist = distances_to_constraint.max()
            diversity = (distances_to_constraint - min_dist) / (max_dist - min_dist)
            score = uncertainty_weight*distances_to_other_cluster + (1-uncertainty_weight)*diversity
            max_id = np.argmax(score)
            id_new_constraint = ids_points_wo_must_link[max_id]
            # uncert to -inf to avoid selecting the point again 
            distances_to_other_cluster[max_id] = -np.inf
        elif sample_mode == "hybrid-sum-knn":
            # minmax scale the diversity
            diversity = distances_to_constraint / avg_knn_dist
            min_dist = diversity.min()
            max_dist = diversity.max()
            diversity = (diversity - min_dist) / (max_dist - min_dist)
            score = uncertainty_weight*distances_to_other_cluster + (1-uncertainty_weight)*diversity
            max_id = np.argmax(score)
            id_new_constraint = ids_points_wo_must_link[max_id]
            # uncert to -inf to avoid selecting the point again 
            distances_to_other_cluster[max_id] = -np.inf
        elif sample_mode == "uncertainty":
            idx = np.argmax(distances_to_other_cluster)
            id_new_constraint = ids_points_wo_must_link[idx]
            # set the selected point's uncertainty score to -inf to avoid selecting it again
            distances_to_other_cluster[idx] = -np.inf
        cluster_new_constraint = labels[id_new_constraint]
        # Save id of that point for next iteration
        last_added = id_new_constraint
        # Add must-link or cannot-link constraint
        if constraints[id_new_constraint, cluster_new_constraint] != 0:
            # constraint already sampled, continue
            # we can hit an already sampled pair if all violations are not resolved (e.g., in ablations)
            continue
        if ground_truth[medoid_ids[cluster_new_constraint]] == ground_truth[id_new_constraint]:
            # Add must-link
            constraints[id_new_constraint, cluster_new_constraint] = 1
            n_added +=1
            # other_cluster = np.ones(medoid_ids.shape[0], dtype=bool)
            # other_cluster[cluster_new_constraint] = 0
            # constraints[id_new_constraint, other_cluster] = -1
        else:
            # Add cannot-link
            constraints[id_new_constraint, cluster_new_constraint] = -1
            n_added +=1

    # adjust cannot-links (transitivity)
    constraints = cannot_transitivity(constraints)

    # check if any point has cannot-link to all current medoids
    cl_mask = constraints == -1
    all_cl_mask = np.sum(cl_mask, axis=1) == len(medoid_ids)
    if np.any(all_cl_mask):
        # print("Some points have cannot-link to all medoids, adding a new medoid")
        # pick the first id to be the new medoid
        new_medoid_id = np.where(all_cl_mask)[0][0]
        labels, constraints, medoid_ids = add_new_medoid(X_embedded, labels, constraints, medoid_ids, new_medoid_id)
    
    runtime = time.time() - time_start
    return labels, medoid_ids, constraints, n_added, runtime, queried_points


def _get_nearest_object(optimal_centers: np.ndarray, embedded_data: np.ndarray):
    best_center_points = np.argmin(cdist(optimal_centers, embedded_data), axis=1)
    return best_center_points


def _update_labels_and_centers(centers, embedded_data):
    labels, _ = pairwise_distances_argmin_min(X=embedded_data, Y=centers,
                                              metric='euclidean',
                                              metric_kwargs={'squared': True})
    # Update centers
    updated_centers = centers.copy()
    for c in range(centers.shape[0]):
        mask = labels == c
        if mask.sum() > 0:
            updated_centers[c] = np.mean(embedded_data[labels == c], axis=0)
    return labels, updated_centers


def _plot_embedding_random(embedded_data, labels, centers, ground_truth, constraint_pairs, constraint_labels, fig_path=None):
    plot_2d_data(
        embedded_data,
        labels,
        centers,
        ground_truth,
        show_plot=False,
        scattersize=20,
        equal_axis=True,
        centers_scattersize=5,
    )
    for pair in constraint_pairs[constraint_labels==1]:
        l,r = pair
        plt.plot(
                [embedded_data[l, 0], embedded_data[r, 0]],
                [embedded_data[l, 1], embedded_data[r, 1]],
                c="g",
                alpha=0.3,
            )
    for pair in constraint_pairs[constraint_labels==-1]:
        l,r = pair
        if labels[l] != labels[r]:
            continue # skip if the points are in different clusters
        plt.plot(
                [embedded_data[l, 0], embedded_data[r, 0]],
                [embedded_data[l, 1], embedded_data[r, 1]],
                c="r",
                alpha=0.3,
            )
    plt.show()

def _method_random(
    X_torch: torch.tensor,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    centers: np.ndarray,
    max_queries: int,
    n_constraints_per_query: int,
    n_necessary_successes: int,
    threshold: int,
    clustering_epochs: int,
    trainloader,
    testloader,
    autoencoder,
    clustering_loss_weight: float,
    ssl_loss_weight: float,
    ssl_loss_fn: torch.nn.modules.loss._Loss,
    optimizer,
    device,
    random_state: np.random.RandomState,
    wandb_run=None,
    enable_plots=False,
    constraint_loss_weight=1.0,
    constraint_batch_size=256,
):  
    use_must_margin = True

    plot_interval = 1
    figure_path = None

    if wandb_run is None:
        wandb_run = wandb.init(mode="disabled")

    # for storing constraints
    cm = ConstraintManager()

    n_given_constraints = 0

    # cluster labels/number of queries used record
    hist = {"labels": [], "queries": []}

    # save the initial labels/constraints
    hist["labels"].append(labels.copy())
    hist["queries"].append(n_given_constraints)

    avg_medoid_dist = torch.tensor(pdist(centers).mean())
    
    if use_must_margin:
        min_medoid_dist = torch.tensor(pdist(centers).min()) / 3
        must_margin = min_medoid_dist**2
        # must_margin = (avg_medoid_dist/3)**2

    else:
        must_margin = torch.tensor(0)

    # average medoid distance
    cannot_margin = avg_medoid_dist**2

    n_data_points = X_torch.shape[0]
    X_dims = X_torch.shape[1]
    embed_dim = centers.shape[1]
    
    # run the main query-optimization loop
    iteration = 0
    while n_given_constraints < max_queries:
        iteration += 1
        
        # querying constraints
        next_query_size = min(n_constraints_per_query, max_queries - n_given_constraints)
        n_constraints_added_batch = 0 # queries spent for this batch
        while n_constraints_added_batch < next_query_size:
            data_ids = np.random.choice(n_data_points, size=2)
            point_a = data_ids[0].item()
            point_b = data_ids[1].item()
            if point_a == point_b:
                continue
            # skip if the point has been sampled before
            if  point_b in cm.get_must_link_group(point_a):
                continue
            if point_b in cm.get_cannot_link_group(point_a):
                continue
            # add constraint
            n_constraints_added_batch +=1
            if ground_truth[data_ids[0]] == ground_truth[data_ids[1]]:
                cm.add_must_link(point_a, point_b)
            else:
                cm.add_cannot_link(point_a, point_b)

        n_given_constraints += n_constraints_added_batch
        # print("Used queries overall:", n_given_constraints)

        # constraint pairs with labels (1 must, -1 cannot)
        const_pairs, const_labels = extract_constraint_pairs(cm)
        unique_const = const_pairs[:, 0] < const_pairs[:, 1] # drop mirrored pairs
        const_pairs = const_pairs[unique_const]
        const_labels = const_labels[unique_const]

        # PLOTTING after querying
        if enable_plots:
            evaluate(labels, ground_truth) # print metrics when plot is shown
            embedded_data = encode_batchwise(testloader, autoencoder)
            _plot_embedding_random(
                embedded_data, labels, centers, ground_truth, const_pairs, const_labels, figure_path
            )

        n_const_batch = min(constraint_batch_size, len(const_pairs))
        # create dataloader for constraints
        constraint_loader = get_dataloader(const_pairs, n_const_batch, shuffle=True, drop_last=True)
        constraint_loader_iter = iter(constraint_loader)

        # Init variables for next run with new constraints
        n_successes = 0
        centers_torch = torch.tensor(centers).to(device)
        autoencoder.train()
        for epoch in range(clustering_epochs):
            total_loss = torch.zeros(5)
            labels_torch = torch.from_numpy(labels).int().to(device)
            # Update embedding
            for batch in trainloader:
                # Get data batch
                ids = batch[0].detach().to(device)
                batch_data = batch[1].to(device)
                # encode/decode X
                embedded = autoencoder.encode(batch_data)
                reconstruction = autoencoder.decode(embedded)
                # Reconstruction Loss
                ae_loss = ssl_loss_fn(batch_data, reconstruction) / X_dims  # divide by number of X features
                # get constraint batch
                try:
                    constraint_batch = next(constraint_loader_iter)
                except:
                    constraint_loader_iter = iter(constraint_loader)
                    constraint_batch = next(constraint_loader_iter)
                constraint_batch_pairs = constraint_batch[1]
                constraint_labels = const_labels[constraint_batch[0]]

                # extract X features from the main dataloader
                uniques, inverse_idx = torch.unique(constraint_batch_pairs, return_inverse=True)
                # take only the unique X's
                X_uniques = X_torch[uniques]
                X_uniques = X_uniques.to(device)
                # push only the unique datapoints through the network (to avoid computing the embedding multiple times for the same X's)
                Z_uniques = autoencoder.encode(X_uniques)
                # get the left and right pairs in the original order
                z_left = Z_uniques[inverse_idx[:,0]]
                z_right = Z_uniques[inverse_idx[:,1]]

                # must-link loss (pull)
                constraint_is_ml = (constraint_labels == 1)
                if torch.any(constraint_labels == 1):
                    must_link_loss_tmp = torch.sum((z_left[constraint_is_ml] - z_right[constraint_is_ml])**2, axis=1)
                    if use_must_margin:
                        must_link_loss = torch.nn.functional.relu(must_link_loss_tmp - must_margin).mean() / embed_dim
                    else:
                        must_link_loss = must_link_loss_tmp.mean() / embed_dim

                # cannot-link losses
                constraint_is_cl = (constraint_labels == -1)
                cannot_push_loss = torch.tensor(0) # cannot push loss
                if torch.any(constraint_is_cl):
                    # cannot-link loss (push)
                    all_sq_dist = torch.sum((z_left[constraint_is_cl] - z_right[constraint_is_cl])**2, axis=1)
                    cannot_push_loss = torch.mean(torch.nn.functional.relu(cannot_margin - all_sq_dist)) / embed_dim

                # Cluster loss
                cluster_loss = torch.tensor(0)
                # compute clustering loss wrt. all points in batch
                cluster_loss_tmp = embedded - centers_torch[labels_torch[ids]]
                cluster_loss_tmp = cluster_loss_tmp.pow(2).sum(1)
                cluster_loss = cluster_loss_tmp.mean() / X_dims  # divide by number of X features

                # scalarized loss function
                loss = ssl_loss_weight * ae_loss + clustering_loss_weight * cluster_loss + constraint_loss_weight * (cannot_push_loss + must_link_loss)
                
                # for logging
                total_loss[0] += loss.item()
                total_loss[1] += ae_loss.item()
                total_loss[2] += must_link_loss.item()
                total_loss[3] += cannot_push_loss.item()
                total_loss[4] += cluster_loss.item()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update labels and medoids
            embedded_data = encode_batchwise(testloader, autoencoder)
            labels, centers = _update_labels_and_centers(centers, embedded_data)
            centers_torch = torch.tensor(centers).to(device)

            # number of different constraints, violated and satisfied
            n_constraints = len(const_pairs)
            n_must_links = torch.sum(const_pairs==1).item()
            n_cannot_links = n_constraints-n_must_links

            n_violated_must = 0
            n_violated_cannot = 0
            for i in range(len(const_pairs)):
                pair = const_pairs[i]
                pair_label = const_labels[i]
                if pair_label == 1:
                    # must link
                    if labels[pair[0]] != labels[pair[1]]:
                        n_violated_must +=1
                else:
                    # cannot link
                    if labels[pair[0]] == labels[pair[1]]:
                        n_violated_cannot +=1

            n_satisfied_constraints = n_constraints - n_violated_must - n_violated_cannot
            prop_satisfied_constraints = n_satisfied_constraints / n_constraints

            # Log results to wandb
            wandb_run.log(
                {
                    "total_loss": total_loss[0],
                    "reconstruction_loss": total_loss[1],
                    "clustering_loss": total_loss[4],
                    "must_link_loss": total_loss[2],
                    "cannot_link_loss": total_loss[3],
                    "ari": ari(ground_truth, labels),
                    "purity": purity_score(ground_truth, labels),
                    "queries": n_given_constraints,
                    "must_violations": n_violated_must,
                    "cannot_violations": n_violated_cannot,
                    "must_links": n_must_links,
                    "cannot_links": n_cannot_links,
                    "avg_meadoid_dist": pdist(centers).mean(),
                    "min_meadoid_dist": pdist(centers).min(),
                },
            )

            if prop_satisfied_constraints >= threshold:
                n_successes += 1
                if n_successes == n_necessary_successes:
                    # if n_successes == n_necessary_successes and iteration != last_iteration:
                    # it is not the last iteration (all queries used) and sufficiently many constraints are fulfilled
                    # we can query again and start another updating iteration
                    # print("ALL CONSTRAINTS FULFILLED!")
                    break
            else:
                n_successes = 0

            if epoch % plot_interval == 0 and enable_plots:
                evaluate(labels, ground_truth) # print metrics when plot is shown
                _plot_embedding_random(
                    embedded_data, labels, centers, ground_truth, const_pairs, const_labels, figure_path
                )

        # print("number of queries used", n_given_constraints)

        # end of iteration
        # update history (used for evaluating the algorithm)
        hist["labels"].append(labels.copy())
        hist["queries"].append(n_given_constraints)

    # finish logging
    wandb_run.finish()

    metadata = {"queries_used_for_verifying": 0, "different_medoids_at_init": np.nan, "violated_frac": np.nan}
    runtime = {"querying": np.nan, "optimizing": np.nan}

    return labels, centers, autoencoder, hist, metadata, runtime


def verify_medoids(embedded_data, labels, ground_truth, medoid_ids, query_budget, constraints):
    n_true_clusters = len(np.unique(ground_truth))
    n_queries_true_k = np.nan
    # verify the initial medoids
    constraints, n_queries_used, medoid_ids, queried_points = _query_init_medoids(embedded_data, labels, ground_truth, medoid_ids, query_budget, constraints)
    # update clustering with new medoids (re-assign cluster labels)
    labels, medoid_ids = _update_labels_and_medoids(medoid_ids, embedded_data, constraints)
    # print("Used {} queries for querying the initial medoids".format(n_queries_used))
    # verify_budget = max(0, query_budget - n_queries_used)
    if len(medoid_ids) == n_true_clusters:
        n_queries_true_k = n_queries_used
    # print("Using the remaining budget ({}) for verifying".format(verify_budget))
    constraints, n_queries_used, medoid_ids, labels, n_queries_true_k, queried_points = verify_embedding(embedded_data, labels, ground_truth, medoid_ids, query_budget, n_queries_used, constraints, n_true_clusters, n_queries_true_k, queried_points)

    # handle cannot transitivity
    constraints = cannot_transitivity(constraints)
    return constraints, n_queries_used, medoid_ids, labels, n_queries_true_k, queried_points


def _predict(data_loader, center_embed, autoencoder):
    autoencoder.eval()
    Z = encode_batchwise(data_loader, autoencoder)
    labels = np.argmin(cdist(Z, center_embed), axis=1)
    return labels


def _method(
    X_torch: torch.tensor,
    labels: np.ndarray,
    ground_truth: np.ndarray,
    embedded_data,
    medoid_ids: np.ndarray,
    n_max_queries: int,
    n_constraints_per_query: int,
    init_budget,
    n_necessary_successes: int,
    threshold: int,
    clustering_epochs: int,
    trainloader,
    testloader,
    testloader_test,
    autoencoder,
    clustering_loss_weight: float,
    ssl_loss_weight: float,
    ssl_loss_fn: torch.nn.modules.loss._Loss,
    optimizer,
    device,
    random_state: np.random.RandomState,
    wandb_run=None,
    enable_plots=False,
    constraint_loss_weight=1.0,
    constraint_batch_size=256,
    mode="cannot",
    use_must_margin=True,
    softmin_weight=1.1,
    sample_mode="hybrid",
    uncertainty_weight=1.0,
    softmin_omega=-10,
    density_nearest_neighbors=20,
    cannot_margin_style = "avg",
):  
    push_loss = False
    softmin_loss = False

    if mode == "nosoftmin":
        push_loss = True
    elif mode == "nopush":
        softmin_loss = True
    elif mode == "cannot":
        push_loss = True
        softmin_loss = True
    else:
        raise TypeError("Unknown mode!")

    if wandb_run is None:
        wandb_run = wandb.init(mode="disabled")

    plot_interval = 1

    n_queries_true_k = np.nan
    all_medoids_verified = False
    n_clusters = medoid_ids.shape[0]
    n_given_constraints = 0

    # cluster labels/number of queries used record
    hist = {"labels": [], "queries": [], "test_labels": [], "embeddings": [], "constraints": [], "medoids": []}

    runtime = {"querying": 0, "optimizing": 0}

    queried_points = None

    # save the initial labels/constraints
    hist["labels"].append(labels.copy())
    hist["queries"].append(n_given_constraints)
    if testloader_test is not None:
        hist["test_labels"].append(_predict(testloader_test, embedded_data[medoid_ids], autoencoder))

    # initialize constraint weights
    # constraint_weight = torch.ones((X_torch.shape[0], n_clusters)).to(device)

    # different medoids at the initialization
    n_unique_medoids_init = len(np.unique(ground_truth[medoid_ids]))

    # print("Different medoids at init {}/{}".format(n_unique_medoids_init, n_clusters))

    X_dims = X_torch.shape[1]
    embed_dim = embedded_data.shape[1]

    # initialize constraints matrix (unverified)
    constraints = np.zeros((X_torch.shape[0], n_clusters), dtype=int)
    # medoids should have must link with their own cluster
    for c in range(n_clusters):
        constraints[medoid_ids[c], c] = 1

    # if enable_plots:
    #     evaluate(labels, ground_truth) # print metrics when plot is shown
    #     _plot_embedding(
    #         embedded_data, labels, medoid_ids, ground_truth, constraints
    #     )

    # run the main query-optimization loop
    iteration = 0
    while n_given_constraints < n_max_queries:
        iteration +=1
        # print("#### Iteration {} ####".format(iteration))

        if not all_medoids_verified:
            # print("different", len(np.unique(ground_truth[medoid_ids])))

            # verify medoids by pairwise querying
            print("verify medoids (run init)")
            constraints, n_queries_verifying, medoid_ids, labels, n_queries_true_k, queried_points = verify_medoids(
                embedded_data,
                labels,
                ground_truth,
                medoid_ids,
                init_budget, #n_constraints_per_query,
                constraints,
            )

            # print("Number of queries used to find true number of clusters", n_queries_true_k)
            
            # update queries used overall
            n_given_constraints = n_queries_verifying

            all_medoids_verified = True

            # cannot/must margins
            avg_medoid_dist = torch.tensor(pdist(embedded_data[medoid_ids]).mean())
            max_medoid_dist = torch.tensor(pdist(embedded_data[medoid_ids]).max())
            if use_must_margin:
                min_medoid_dist = torch.tensor(pdist(embedded_data[medoid_ids]).min()) / 3
                must_margin = min_medoid_dist**2
            else:
                must_margin = torch.tensor(0)

            if cannot_margin_style == "avg":
                cannot_margin = avg_medoid_dist**2
            elif cannot_margin_style == "max":
                cannot_margin = max_medoid_dist**2
            elif cannot_margin_style == "const":
                cannot_margin = 1000
            else:
                raise TypeError("Unknown cannot margin.")


        else:
            # querying constraints
            if n_constraints_per_query < n_max_queries - n_given_constraints:
                n_next_query = n_constraints_per_query
            else:
                n_next_query = max(0, n_max_queries - n_given_constraints)
            labels, medoid_ids, constraints, n_constraints_used, time_spent, queried_points = _add_constraints(
                embedded_data,
                labels,
                ground_truth,
                medoid_ids,
                n_next_query,
                constraints,
                sample_mode,
                uncertainty_weight,
                density_nearest_neighbors,
                queried_points
            )
            print("querying added const", n_constraints_used)
            n_clusters = len(medoid_ids)
            n_given_constraints += n_constraints_used
            runtime["querying"] += time_spent

        print("Used queries overall:", n_given_constraints)

        optimize_start = time.time()

        # check the number of different medoids
        n_unique_medoid = len(np.unique(ground_truth[medoid_ids]))

        # Update clusters with new constraint information
        labels, medoid_ids = _update_labels_and_medoids(medoid_ids, embedded_data, constraints)


        # save embedding after sampling
        # hist["embeddings"].append(embedded_data.copy())
        # hist["constraints"].append(constraints.copy())
        # hist["medoids"].append(medoid_ids.copy())


        #PLOTTING after querying
        # if enable_plots:
        #     evaluate(labels, ground_truth) # print metrics when plot is shown
        #     _plot_embedding(
        #         embedded_data, labels, medoid_ids, ground_truth, constraints
        #     )

        # create loader for constraints
        obj_with_constraint = np.any(constraints != 0, axis=1)
        if constraint_batch_size is not None:
            n_const_rows = min(constraint_batch_size, np.sum(obj_with_constraint).item()) # number of non zero constraint rows
        else:
            n_const_rows = np.sum(obj_with_constraint).item()
        # create dataloader for constraints
        constraint_loader = get_dataloader(X_torch[obj_with_constraint], n_const_rows, shuffle=True,
                                                 drop_last=True,
                                                 additional_inputs=[constraints[obj_with_constraint], 
                                                                               torch.arange(X_torch.shape[0])[obj_with_constraint]])
        constraint_loader_iter = iter(constraint_loader)

        # Init variables for next run with new constraints
        n_successes = 0
        autoencoder.train()
        for epoch in range(clustering_epochs):
            labels_torch = torch.from_numpy(labels).int().to(device)
            total_loss = torch.zeros(6)
            # Update embedding
            for batch in trainloader:
                # Get data batch
                ids = batch[0].detach().to(device)
                batch_data = batch[1].to(device)
                # encode/decode X
                embedded = autoencoder.encode(batch_data)
                reconstruction = autoencoder.decode(embedded)
                # embed medoids
                embedded_medoids = autoencoder.encode(X_torch[medoid_ids])
                # get constraint batch
                try:
                    constraint_batch = next(constraint_loader_iter)
                except:
                    constraint_loader_iter = iter(constraint_loader)
                    constraint_batch = next(constraint_loader_iter)
                batch_constraint_data = constraint_batch[1].to(device)
                batch_constraint_constraints = constraint_batch[2].int().to(device)

                # batch_constraint_ids = constraint_batch[3].int().to(device)
                batch_constraint_embedded = autoencoder.encode(batch_constraint_data)

                # Reconstruction Loss
                ae_loss = torch.tensor(0)
                ae_loss = ssl_loss_fn(batch_data, reconstruction) / X_dims  # divide by number of X features

                # squared distance to all medoids
                center_dist_all = squared_euclidean_distance(batch_constraint_embedded, embedded_medoids)

                # must-link loss (pull)
                constraint_is_ml = (batch_constraint_constraints == 1)
                if torch.any(constraint_is_ml):
                    must_link_loss_tmp = center_dist_all
                    if use_must_margin:
                        must_link_loss = torch.nn.functional.relu(must_link_loss_tmp[constraint_is_ml] - must_margin).mean() / embed_dim
                    else:
                        must_link_loss = must_link_loss_tmp[constraint_is_ml].mean() / embed_dim # divide by number of embedding dimensions

                # cannot-link losses
                constraint_is_cl = (batch_constraint_constraints == -1)
                n_cl_batch = constraint_is_cl.sum()
                cannot_link_loss = torch.tensor(0) # softmin loss
                cannot_push_loss = torch.tensor(0) # cannot push loss
                if torch.any(constraint_is_cl):
                    if softmin_loss:
                        # compute soft minimum
                        # https://mathoverflow.net/questions/35191/a-differentiable-approximation-to-the-minimum-function
                        # distances to all other clusters (exept the cannot linked clusters)
                        center_dist_wo_cl = center_dist_all.clone()
                        center_dist_wo_cl[constraint_is_cl] = torch.inf
                        # stable version
                        # Use the log-sum-exp trick for numerical stability
                        min_val, _ = torch.min(center_dist_wo_cl, axis=1, keepdims=True)
                        stable_distances = center_dist_wo_cl - min_val
                        sm = torch.exp(softmin_omega*stable_distances).sum(1)
                        sm = sm / (~constraint_is_cl).sum(1)
                        sm = torch.log(sm) / softmin_omega + min_val.flatten()
                        # compute softmin loss
                        soft_min_wo_cl = sm.reshape((-1, 1))
                        cannot_link_loss_tmp = torch.nn.functional.relu(softmin_weight * soft_min_wo_cl - center_dist_all)
                        cannot_link_loss = (cannot_link_loss_tmp * constraint_is_cl).sum() / (n_cl_batch*embed_dim)
                    # cannot-link loss (push)
                    if push_loss:
                        # all cannot links push (medoids)
                        cl_distances = center_dist_all[constraint_is_cl]
                        cannot_push_losses = torch.nn.functional.relu(cannot_margin - cl_distances)
                        cannot_push_loss = torch.sum(cannot_push_losses) / (n_cl_batch*embed_dim)


                # Cluster loss
                cluster_loss = torch.tensor(0)
                cluster_loss_tmp = embedded - embedded_medoids[labels_torch[ids]]# .detach()
                cluster_loss_tmp = cluster_loss_tmp.pow(2).sum(1)
                cluster_loss = cluster_loss_tmp.mean() / X_dims  # divide by number of X features

                # scalarized loss function
                loss = ssl_loss_weight * ae_loss + clustering_loss_weight * cluster_loss + constraint_loss_weight * (cannot_push_loss + must_link_loss) + cannot_link_loss

                # for logging
                total_loss[0] += loss.item()
                total_loss[1] += ae_loss.item()
                total_loss[2] += must_link_loss.item()
                total_loss[3] += cannot_link_loss.item()
                total_loss[4] += cluster_loss.item()
                total_loss[5] += cannot_push_loss.item()
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                optimizer.step()

            # Update labels and medoids
            embedded_data = encode_batchwise(testloader, autoencoder)
            labels, medoid_ids = _update_labels_and_medoids(medoid_ids, embedded_data, constraints)
            embedded_medoids = embedded_data[medoid_ids]

            # number of satisfied constraints
            must_links_fulfilled, cannot_links_fulfilled = _get_amount_fulfilled_constraints(labels, medoid_ids, constraints)
            constraints_fulfilled_total = must_links_fulfilled[0] + cannot_links_fulfilled[0]
            real_n_constraints_total = must_links_fulfilled[1] + cannot_links_fulfilled[1]
            amount_constraints_fulfilled_total = constraints_fulfilled_total / real_n_constraints_total

            # compute how many points cl points are in the wrong cluster
            n_wrong_cluster = 0
            # only_cl_mask = np.any(constraints == -1, axis=1) & ~np.any(constraints == 1, axis=1)
            # how many cl points are in wrong cluster
            # for idx in np.where(only_cl_mask)[0]:
            #     c = labels[idx]
            #     gt = ground_truth[idx]
            #     current_medoid = medoid_ids[c]
            #     medoid_gt = ground_truth[current_medoid]
            #     if gt != medoid_gt:
            #         n_wrong_cluster += 1

            # if enable_plots:
            #     print("=====[EPOCH {} / ITER {}]=====".format(epoch, iteration))
            #     print(
            #         "total loss: {0}\nrec loss:\t{1}\ncluster loss:\t{4}\nmust loss:\t{2}\ncannot (min):\t{3}\ncannot (push):\t{5}".format(
            #             total_loss[0], total_loss[1], total_loss[2], total_loss[3], total_loss[4], total_loss[5]))

            #     print("=========================================")
            #     print("Number of only CL points in wrong clusters", n_wrong_cluster)
            #     print("=========================================")

            # Log results to wandb
            wandb_run.log(
                {
                    "total_loss": total_loss[0],
                    "reconstruction_loss": total_loss[1],
                    "clustering_loss": total_loss[4],
                    "must_link_loss": total_loss[2],
                    "cannot_link_loss": total_loss[3] + total_loss[5],
                    "ari": ari(ground_truth, labels),
                    "purity": purity_score(ground_truth, labels),
                    "queries": n_given_constraints,
                    "different_medoids": n_unique_medoid,
                    "queries_for_verifying_medoids": n_queries_true_k,
                    "must_violations": must_links_fulfilled[1] - must_links_fulfilled[0],
                    "cannot_violations": cannot_links_fulfilled[1] - cannot_links_fulfilled[0],
                    "cannot_in_wrong_cluster": n_wrong_cluster,
                    "must_links": must_links_fulfilled[1],
                    "cannot_links": cannot_links_fulfilled[1],
                    "avg_meadoid_dist": pdist(embedded_medoids).mean(),
                    "min_meadoid_dist": pdist(embedded_medoids).min(),
                },
            )

            if amount_constraints_fulfilled_total >= threshold:
                n_successes += 1
                if n_successes == n_necessary_successes:
                    # if n_successes == n_necessary_successes and iteration != last_iteration:
                    # it is not the last iteration (all queries used) and sufficiently many constraints are fulfilled
                    # we can query again and start another updating iteration
                    # print("ALL CONSTRAINTS FULFILLED AT EPOCH {}!".format(epoch))
                    break
            else:
                n_successes = 0

            # if epoch % plot_interval == 0 and enable_plots:
            #     evaluate(labels, ground_truth) # print metrics when plot is shown
            #     _plot_embedding(
            #         embedded_data, labels, medoid_ids, ground_truth, constraints
            #     )

        # end of iteration
        optim_time = time.time() - optimize_start
        # print("end of optimization, time", optim_time)
        runtime["optimizing"] += optim_time
        # update history (used for evaluating the algorithm)
        hist["labels"].append(labels.copy())
        hist["queries"].append(n_given_constraints)
        if testloader_test is not None:
            hist["test_labels"].append(_predict(testloader_test, embedded_data[medoid_ids], autoencoder))

    # finish logging
    wandb_run.finish()

    # compute the fraction of violated constraints
    n_violated = must_links_fulfilled[1] - must_links_fulfilled[0] + cannot_links_fulfilled[1] - cannot_links_fulfilled[0]
    n_const = must_links_fulfilled[1] + cannot_links_fulfilled[1]
    violated_frac = n_violated / n_const

    metadata = {"queries_used_for_verifying": n_queries_true_k, "different_medoids_at_init": n_unique_medoids_init, "violated_frac": violated_frac}

    return labels, medoid_ids, autoencoder, hist, metadata, runtime, queried_points


class CODAC(_AbstractDeepClusteringAlgo):
    def __init__(
        self,
        n_clusters: int = 35,
        max_queries: int = 100,
        query_sample_size: int = 100,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params: dict = None,
        pretrain_epochs: int = 100,
        deep_cluster_epochs: int = 150,
        clustering_epochs: int = 100,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
        clustering_loss_weight: float = 0.0,
        ssl_loss_weight: float = 0.01,
        neural_network: torch.nn.Module | tuple = None,
        neural_network_weights: str = None,
        embedding_size: int = 10,
        custom_dataloaders: tuple = None,
        augmentation_invariance: bool = False,
        initial_clustering_class: ClusterMixin = KMeans,
        initial_clustering_params: dict = None,
        device: torch.device = None,
        random_state: np.random.RandomState | int = None,
        debug: bool = False,
        wandb_run=None,
        enable_plots=False,
        constraint_loss_weight: float = 1.0,
        mode="cannot",
        early_stopping_tol: int = 30,
        random_mode: bool = False,
        must_margin = True,
        cannot_margin="avg",
        softmin_weight = 1.0,
        sample_mode="hybrid-sum-knn",
        uncertainty_weight=0.6,
        constraint_batch_size=256,
        deep_clusterer="DCN",
        guess_k_multiplier=None,
        softmin_omega=-10,
        density_nearest_neighbors=20,
        init_budget=100,
    ):
        super().__init__(
            batch_size,
            neural_network,
            neural_network_weights,
            embedding_size,
            device,
            random_state,
        )
        self.n_clusters = n_clusters
        self.max_queries = max_queries
        self.query_sample_size = query_sample_size
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.deep_cluster_epochs = deep_cluster_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params
        self.debug = debug
        self.enable_plots = enable_plots
        self.constraint_loss_weight = constraint_loss_weight
        self.mode = mode
        self.early_stopping_tol = early_stopping_tol
        self.random_mode = random_mode
        self.must_margin = must_margin
        self.cannot_margin = cannot_margin
        self.softmin_weight = softmin_weight
        self.sample_mode = sample_mode
        self.uncertainty_weight = uncertainty_weight
        self.constraint_batch_size = constraint_batch_size
        self.deep_clusterer = deep_clusterer
        self.guess_k_multiplier = guess_k_multiplier
        self.softmin_omega = softmin_omega
        self.density_nearest_neighbors = density_nearest_neighbors
        self.init_budget = init_budget
        if self.clustering_optimizer_params is not None:
            # make sure that learning rate is float and not string
            try:
                self.clustering_optimizer_params["lr"] = float(self.clustering_optimizer_params["lr"])
            except KeyError:
                pass
            
        # simulate not exact estimation of the number of clusters
        if self.guess_k_multiplier is not None:
            self.n_clusters = max(2, int(self.n_clusters * self.guess_k_multiplier))
            # print("Setting the number of clusters to", self.n_clusters)

        # how many query batches should be collected
        self.n_query_batches = math.ceil(self.max_queries / self.query_sample_size)

        if wandb_run is None:
            self.wandb_run = wandb.init(mode="disabled")
        else:
            self.wandb_run = wandb_run

        self.hist = None
        self.metadata = None
        self.runtime = None
        self.queried_points = None

    def fit(self, X: np.ndarray, y: np.ndarray, X_test: np.ndarray=None, y_test: np.ndarray=None, init_labels: np.ndarray=None) -> 'CODAC':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : CODAC
            this instance of the CODAC algorithm
        """
        X, _, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params = self._check_parameters(X, y=y)

        device = detect_device()
        trainloader = get_dataloader(X, self.batch_size, True, True)
        testloader = get_dataloader(X, self.batch_size, False, False)

        if X_test is None:
            testloader_test = None
        else:
            testloader_test = get_dataloader(X_test, self.batch_size, False, False)

        if init_labels is not None:
            autoencoder = self.neural_network
            labels = init_labels
        else:
            # perform deep clustering for an init
            if self.deep_clusterer == "DCN":
                deep_clusterer = DCN(
                    self.n_clusters,
                    self.batch_size,
                    pretrain_optimizer_params,
                    clustering_optimizer_params,
                    self.pretrain_epochs,
                    self.deep_cluster_epochs,
                    self.optimizer_class,
                    self.ssl_loss_fn,
                    0.1, #self.clustering_loss_weight,
                    1.0, #self.ssl_loss_weight,
                    self.neural_network,
                    self.neural_network_weights,
                    self.embedding_size,
                    self.custom_dataloaders,
                    self.augmentation_invariance,
                    self.initial_clustering_class,
                    self.initial_clustering_params,
                    self.device,
                    self.random_state,
                )
            elif self.deep_clusterer == "DKM":
                deep_clusterer = DKM(
                    n_clusters=self.n_clusters,
                    batch_size=self.batch_size,
                    pretrain_optimizer_params=self.pretrain_optimizer_params,
                    pretrain_epochs=self.pretrain_epochs,
                    clustering_epochs=self.deep_cluster_epochs,
                    optimizer_class=self.optimizer_class,
                    ssl_loss_fn=self.ssl_loss_fn,
                    neural_network=self.neural_network,
                    neural_network_weights=self.neural_network_weights,
                    embedding_size=self.embedding_size,
                    clustering_loss_weight=0.1,
                    ssl_loss_weight=1.0,
                    custom_dataloaders=self.custom_dataloaders,
                    augmentation_invariance=self.augmentation_invariance,
                    initial_clustering_class=self.initial_clustering_class,
                    initial_clustering_params=self.initial_clustering_params,
                    device=self.device,
                    random_state=self.random_state,
                )
            elif self.deep_clusterer == "IDEC":
                deep_clusterer = IDEC(
                    self.n_clusters,
                    1,
                    self.batch_size,
                    pretrain_optimizer_params,
                    clustering_optimizer_params,
                    self.pretrain_epochs,
                    self.deep_cluster_epochs,
                    self.optimizer_class,
                    self.ssl_loss_fn,
                    self.neural_network,
                    self.neural_network_weights,
                    self.embedding_size,
                    0.1, #self.clustering_loss_weight,
                    1.0, #self.ssl_loss_weight,
                    self.custom_dataloaders,
                    self.augmentation_invariance,
                    self.initial_clustering_class,
                    self.initial_clustering_params,
                    self.device,
                    self.random_state,
                )
            elif self.deep_clusterer == "deepkmeans":
                deep_clusterer = DeepKmeans(self.n_clusters, self.neural_network, self.embedding_size)

            deep_clusterer.fit(X)
            autoencoder = deep_clusterer.neural_network_trained_
            labels = deep_clusterer.labels_

        # print("transforming data ...")
        # Get nearest points to optimal centers
        X_torch = torch.from_numpy(X).float().to(device)
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        embedded_data = encode_batchwise(testloader, autoencoder)


        # get cluster center given by deep clustering
        optimal_embedded_centers = []
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                optimal_embedded_centers.append(np.mean(embedded_data[mask], axis=0))
        optimal_embedded_centers = np.array(optimal_embedded_centers)
        medoid_ids = _get_nearest_object(optimal_embedded_centers, embedded_data)

        # print("Init score: ACC: {0} / NMI: {1} / ARI: {2}".format(acc(y, labels), nmi(y, labels), ari(y, labels)))

        optimizer = self.optimizer_class(autoencoder.parameters(), **clustering_optimizer_params)

        threshold = 1.
        if self.random_mode:
            labels, centers, autoencoder, history, metadata, runtime = _method_random(
            X_torch,
            labels,
            y,
            optimal_embedded_centers,
            self.max_queries,
            self.query_sample_size,
            self.early_stopping_tol,
            threshold,
            self.clustering_epochs,
            trainloader,
            testloader,
            autoencoder,
            self.clustering_loss_weight,
            self.ssl_loss_weight,
            self.ssl_loss_fn,
            optimizer,
            device,
            random_state,
            self.wandb_run,
            self.enable_plots,
            self.constraint_loss_weight,
            self.constraint_batch_size,
        )
            self.cluster_centers_ = centers
        else:
            labels, medoid_ids, autoencoder, history, metadata, runtime, queried_points = _method(
                X_torch,
                labels,
                y,
                embedded_data,
                medoid_ids,
                self.max_queries,
                self.query_sample_size,
                self.init_budget,
                self.early_stopping_tol,
                threshold,
                self.clustering_epochs,
                trainloader,
                testloader,
                testloader_test,
                autoencoder,
                self.clustering_loss_weight,
                self.ssl_loss_weight,
                self.ssl_loss_fn,
                optimizer,
                device,
                random_state,
                self.wandb_run,
                self.enable_plots,
                self.constraint_loss_weight,
                self.constraint_batch_size,
                self.mode,
                self.must_margin,
                self.softmin_weight,
                self.sample_mode,
                self.uncertainty_weight,
                self.softmin_omega,
                self.density_nearest_neighbors,
                self.cannot_margin,
            )
            self.cluster_centers_ = X[medoid_ids]
            self.queried_points = queried_points

        self.hist = history
        self.metadata = metadata
        self.labels_ = labels
        # self.n_clusters_ = medoid_ids.shape[0]
        self.autoencoder = autoencoder
        self.runtime = runtime
        return self

    def predict(self, X: np.ndarray):
        self.autoencoder.eval()
        Z = self.autoencoder.encode(torch.tensor(X).float()).detach().numpy()
        if self.random_mode:
            center_embed = self.cluster_centers_
        else:
            center_embed = self.autoencoder.encode(torch.tensor(self.cluster_centers_).float()).detach().numpy()
        assignments = np.argmin(cdist(Z, center_embed), axis=1)
        return assignments


if __name__ == "__main__":
    from clustpy.data import load_optdigits
    from clustpy.deep.neural_networks.feedforward_autoencoder import (
        FeedforwardAutoencoder,
    )
    X, L = load_optdigits(subset="all", return_X_y=True)

    mask = (L==1) | (L == 8) | (L == 7) | (L == 9) | (L == 0)
    L = L[mask]
    X = X[mask]
    # convert labels to start from 0
    L[L==1] = 1
    L[L==8] = 2
    L[L==7] = 3
    L[L==9] = 4
    X = X / 16.0

    n_clusters = 10#np.unique(L).shape[0]
    embed_dim = 2 #n_clusters
    X_dims = X.shape[1]

    # model = DCN(n_clusters=10, embedding_size=2, pretrain_epochs=2, clustering_epochs=2)
    # model.fit(X)
    # torch.save(model.neural_network_trained_.state_dict(), "test_ae.pt")
    # pred = model.predict(X)
    # np.save("preds.npy", pred)
    # print("pretrain TRAIN ARI", ari(L, pred))
    # ae = model.neural_network_trained_

    ae = FeedforwardAutoencoder(layers=[64, 500, 500, 2000, 2])
    ae.load_state_dict(torch.load("test_ae.pt", weights_only=True))
    pred = np.load("preds.npy")
    print("pretrain TRAIN ARI", ari(L, pred))

    codac = CODAC(
        n_clusters=n_clusters,
        max_queries=300,
        query_sample_size=50,
        pretrain_epochs=50,
        deep_cluster_epochs=10,
        batch_size=256,
        embedding_size=embed_dim,
        random_state=10,
        clustering_epochs=200,
        ssl_loss_weight=0.1,
        clustering_loss_weight=0.1,
        constraint_loss_weight=1.0,
        neural_network=ae,
        enable_plots=True,
        mode="cannot",
        early_stopping_tol=10,
        must_margin=True,
        softmin_weight=1.0,
        sample_mode="hybrid-sum-knn",
        uncertainty_weight=0.5,
        random_mode=True,
        constraint_batch_size=256,
        deep_clusterer="DCN",
        guess_k_multiplier = None,
        cannot_margin="avg",
        init_budget=1,
    )
    #codac.fit(X, L)
    codac.fit(X, L, init_labels=pred)
    pred = codac.predict(X)
    print("FINAL TRAIN ARI", ari(L, pred))

    print(codac.metadata)
