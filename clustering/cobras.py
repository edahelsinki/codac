import abc
import collections
import copy
import itertools
import time

import numpy as np
from sklearn.cluster import KMeans


class SuperInstance:
    """
        A class representing a super-instance used in the COBRAS algorithm
    """

    def __init__(self, data, indices, train_indices, parent=None):
        """

        :param data: the full dataset
        :param indices: the indices of the instances that are in this super-instance
        :param train_indices: all training indices (i.e. training indices of the full dataset)
        :param parent: the parent super-instance (if any)
        """
        if not isinstance(indices, list):
            raise ValueError('Should give a list of indices as input to SuperInstance...')

        self.data = data
        #: The indices of the instances in this super-instance
        self.indices = indices
        #: The indices of the training instances in this super-instance
        self.train_indices = [x for x in indices if x in train_indices]
        #: Whether or not we have tried splitting this super-instance in the past and failed to do so
        self.tried_splitting = False
        #: The index of the super-instance representative instance
        self.representative_idx = None

        self.children = None
        self.parent = parent


    def get_representative_idx(self):
        """
        :return: the index of the super-instance representative
        """
        try:
            return self.representative_idx
        except:
            raise ValueError('Super instances without training instances')

    @abc.abstractmethod
    def distance_to(self, other_superinstance):
        """
            Calculates the distance to the given super-instance
            This is COBRAS variant specific

        :param other_superinstance: the super-instance to calculate the distance to
        :return: the distance between this super-instance and the given other_superinstance
        """
        return

    def get_leaves(self):
        if self.children is None:
            return [self]
        else:
            d = []
            for s in self.children:
                d.extend(s.get_leaves())
            return d

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()

class SuperInstance_kmeans(SuperInstance):

    def __init__(self, data, indices, train_indices, parent=None):
        """
            Chooses the super-instance representative as the instance closest to the super-instance centroid
        """
        super(SuperInstance_kmeans, self).__init__(data, indices, train_indices, parent)

        self.centroid = np.mean(data[indices, :], axis=0)
        self.si_train_indices =  [x for x in indices if x in train_indices]

        try:
            self.representative_idx = min(self.si_train_indices, key=lambda x: np.linalg.norm(self.data[x, :] - self.centroid))
        except:
            raise ValueError('Super instance without training instances')



    def distance_to(self, other_superinstance):
        """
            The distance between two super-instances is equal to the distance between there centroids  
        """
        return np.linalg.norm(self.centroid - other_superinstance.centroid)


class Clustering:
    """
        An instance of Clustering represents a full clustering of the dataset
        Note that an instance of clustering is not always a valid clustering (e.g. we are in the middle of the merging phase)

    """

    def __init__(self,clusters):
        #: The clusters in this clustering
        self.clusters = clusters

    def construct_cluster_labeling(self):
        """
        :return: a list of labels indicating the clustering assignment of this clustering
        """

        pts_per_cluster = [cluster.get_all_points() for cluster in self.clusters]

        pred = [-1] * sum([len(x) for x in pts_per_cluster])

        for i, pts in enumerate(pts_per_cluster):
            for pt in pts:
                pred[pt] = i

        return pred

    def create_generalized_super_instances(self, si):
        """
        Collects a list of 'generalized super-instances'.
        COBRAS always splits a super-instances in at least two new super-instances.
        If there is a must-link between these super-instances, and similarly a must-link between all the future splits
        of these super-intances, there is no need to consider the points as belonging to conceptually different super-
        instances (i.e. super-instances corresponding to different behaviour).
        This procedure constructs generalized super-instances: super-instances (i.e. leaves) that are part of a
        subtree with only must-links amongst eachoter are collected into a list.
        :param si: the root super-instance
        :return: a list of lists of super-instances, each entry in the list corresponds to one generalized super-instance,
        that may contain several super-instances
        """
        leaves = si.get_leaves()

        all_in_same_cluster = True
        cur_cluster = None
        for c in self.clusters:
            if leaves[0] in c.super_instances:
                cur_cluster = c
                break

        for l in leaves:
            if l not in cur_cluster.super_instances:
                all_in_same_cluster = False
                break

        if all_in_same_cluster:
            return [leaves]
        else:
            generalized_leaves = []
            for l in si.children:
                generalized_leaves.extend(self.create_generalized_super_instances(l))
            return generalized_leaves

    def get_cluster_to_generalized_super_instance_map(self):
        # first get the generalized super-instances
        generalized_super_instance_sets = self.create_generalized_super_instances(self.clusters[0].super_instances[0].get_root())

        # now map each cluster to its leaves
        cluster_to_si = collections.defaultdict(list)
        for cluster in self.clusters:
            to_delete = []
            for l in generalized_super_instance_sets:
                if l[0] in cluster.super_instances:
                    cluster_to_si[cluster].append(l)
                    to_delete.append(l)

            for l in to_delete:
                generalized_super_instance_sets.remove(l)

        all_instances_ct = 0
        for k in cluster_to_si:
            for l in cluster_to_si[k]:
                for x in l:
                    all_instances_ct += len(x.indices)

        return cluster_to_si


class Cluster:
    """
        An instance of the class Cluster represents a single cluster.
        In the case of COBRAS a cluster contains several super-instances.

    """

    def __init__(self, super_instances):
        #: The super-instances in this cluster
        self.super_instances = super_instances
        #: Whether or not this cluster is pure (as indicated by the querier)
        self.is_pure = False # in the visual querier, the user can indicate that the entire cluster is pure
        self.is_finished = False

    def distance_to(self, other_cluster):
        """
            Calculates the distance between this cluster and the other cluster
            This is equal to the distance between the 2 closest super-instances in this cluster

        :param other_cluster: the cluster to which to calculate the distance
        :return: The distance between this cluster and the given other_cluster
        """
        super_instance_pairs = itertools.product(self.super_instances, other_cluster.super_instances)
        return min([x[0].distance_to(x[1]) for x in super_instance_pairs])

    def get_comparison_points(self, other_cluster):
        # any super-instance should do, no need to find closest ones!
        return self.super_instances[0], other_cluster.super_instances[0]

    def get_all_points(self):
        """
        :return: a list of all the indices of instances in this cluster
        """
        all_pts = []
        for super_instance in self.super_instances:
            all_pts.extend(super_instance.indices)
        return all_pts

    def cannot_link_to_other_cluster(self, c, cl):
        """
            Return whether or not there is a cannot-link between this cluster and cluster c in the given set of cannot-link constraints cl

        :param c: the other cluster c
        :param cl: a set of tuples each representing a cannot-link constraint
        :return: whether or not there is a cannot-link between this cluster and the cluster c
        """

        medoids_c1 = [si.representative_idx for si in self.super_instances]
        medoids_c2 = [si.representative_idx for si in c.super_instances]

        for x, y in itertools.product(medoids_c1, medoids_c2):
            if (x, y) in cl or (y, x) in cl:
                return True
        return False


class COBRAS(abc.ABC):
    def __init__(self, data, querier, max_questions, train_indices=None, store_intermediate_results=True):
        """
        COBRAS clustering

        :param data: Data set numpy array of size (nb_instances,nb_features)
        :type data: ndarray
        :param querier: Querier object that answers whether instances are linked through a must-link or cannot-link.
        :type querier: Querier
        :param max_questions: Maximum number of questions that are asked. Clustering stops after this.
        :type max_questions: int
        :param train_indices: the indices of the training data
        :type train_indices: List[int]
        """
        self.data = data
        self.querier = querier
        self.max_questions = max_questions
        self.store_intermediate_results = store_intermediate_results

        if train_indices is None:
            self.train_indices = range(self.data.shape[0])
        else:
            self.train_indices = train_indices

        self.clustering = None
        self.split_cache = dict()
        self.start_time = None
        self.intermediate_results = []
        self.ml = None
        self.cl = None

    def cluster(self):
        """Perform clustering

        :return: if cobras.store_intermediate_results is set to False, this method returns a single Clustering object
                 if cobras.store_intermediate_results is set to True, this method returns a tuple containing the following items:

                     - a :class:`~clustering.Clustering` object representing the resulting clustering
                     - a list of intermediate clustering labellings for each query (each item is a list of clustering labels)
                     - a list of timestamps for each query
                     - the list of must-link constraints that was queried
                     - the list of cannot-link constraints that was queried
        """
        self.start_time = time.time()

        # initially, there is only one super-instance that contains all data indices
        # (i.e. list(range(self.data.shape[0])))
        initial_superinstance = self.create_superinstance(list(range(self.data.shape[0])))

        self.ml = []
        self.cl = []

        self.clustering = Clustering([Cluster([initial_superinstance])])

        # the split level for this initial super-instance is determined,
        # the super-instance is split, and a new cluster is created for each of the newly created super-instances
        initial_k = self.determine_split_level(initial_superinstance,
                                               copy.deepcopy(self.clustering.construct_cluster_labeling()))

        # split the super-instance and place each new super-instance in its own cluster
        superinstances = self.split_superinstance(initial_superinstance, initial_k)
        self.clustering.clusters = []
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        # the first bottom up merging step
        # the resulting cluster is the best clustering we have so use this as first valid clustering
        self.merge_containing_clusters(copy.deepcopy(self.clustering.construct_cluster_labeling()))
        last_valid_clustering = copy.deepcopy(self.clustering)

        # while we have not reached the max number of questions
        while len(self.ml) + len(self.cl) < self.max_questions:
            # notify the querier that there is a new clustering
            # such that this new clustering can be displayed to the user
            self.querier.update_clustering(self.clustering)

            # after inspecting the clustering the user might be satisfied
            # let the querier check whether or not the clustering procedure should continue
            # note: at this time only used in the notebook queriers
            if not self.querier.continue_cluster_process():
                break

            # choose the next super-instance to split
            to_split, originating_cluster = self.identify_superinstance_to_split()
            if to_split is None:
                break

            # clustering to store keeps the last valid clustering
            clustering_to_store = None
            if self.intermediate_results:
                clustering_to_store = self.clustering.construct_cluster_labeling()

            # remove the super-instance to split from the cluster that contains it
            originating_cluster.super_instances.remove(to_split)
            if len(originating_cluster.super_instances) == 0:
                self.clustering.clusters.remove(originating_cluster)

            # - splitting phase -
            # determine the splitlevel
            split_level = self.determine_split_level(to_split, clustering_to_store)

            # split the chosen super-instance
            new_super_instances = self.split_superinstance(to_split, split_level)

            # add the new super-instances to the clustering (each in their own cluster)
            new_clusters = self.add_new_clusters_from_split(new_super_instances)
            if not new_clusters:
                # it is possible that splitting a super-instance does not lead to a new cluster:
                # e.g. a super-instance constains 2 points, of which one is in the test set
                # in this case, the super-instance can be split into two new ones, but these will be joined
                # again immediately, as we cannot have super-instances containing only test points (these cannot be
                # queried)
                # this case handles this, we simply add the super-instance back to its originating cluster,
                # and set the already_tried flag to make sure we do not keep trying to split this superinstance
                originating_cluster.super_instances.append(to_split)
                to_split.tried_splitting = True
                to_split.children = None

                if originating_cluster not in self.clustering.clusters:
                    self.clustering.clusters.append(originating_cluster)

                continue
            else:
                self.clustering.clusters.extend(new_clusters)

            # perform the merging phase
            fully_merged = self.merge_containing_clusters(clustering_to_store)
            # if the merging phase was able to complete before the query limit was reached
            # the current clustering is a valid clustering
            if fully_merged:
                last_valid_clustering = copy.deepcopy(self.clustering)

        # clustering procedure is finished
        # change the clustering result to the last valid clustering
        self.clustering = last_valid_clustering

        # return the correct result based on what self.store_intermediate_results contains
        if self.store_intermediate_results:
            return self.clustering, [clust for clust, _, _ in self.intermediate_results], [runtime for _, runtime, _ in
                                                                                           self.intermediate_results], self.ml, self.cl
        else:
            return self.clustering

    @abc.abstractmethod
    def split_superinstance(self, si, k):
        """Splits the super-instance si into k new super-instances

        :param si: the super-instance to split
        :type si: SuperInstance
        :param k: the desired number of new super-instances
        :type k: int
        :return: a list of the new super-instances
        :rtype: List[SuperInstance]
        """
        return

    @abc.abstractmethod
    def create_superinstance(self, indices, parent=None):
        """ Creates a new super-instance containing the given instances and with the given parent

        :param indices: the indices of the instances that should be in the new super-instance
        :type indices: List[int]
        :param parent: the parent of this super-instance
        :type parent: SuperInstance
        :return: the new super-instance
        """
        return

    def determine_split_level(self, superinstance, clustering_to_store):
        """ Determine the splitting level for the given super-instance using a small amount of queries

        For each query that is posed during the execution of this method the given clustering_to_store is stored as an intermediate result.
        The provided clustering_to_store should be the last valid clustering that is available

        :return: the splitting level k
        :rtype: int
        """
        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        si = self.create_superinstance(superinstance.indices)

        must_link_found = False
        # the maximum splitting level is the number of instances in the superinstance
        max_split = len(si.indices)
        split_level = 0
        while not must_link_found and len(self.ml) + len(self.cl) < self.max_questions:

            if len(si.indices) == 2:
                # if the superinstance that is being splitted just contains 2 elements split it in 2 superinstances with just 1 instance
                new_si = [self.create_superinstance([si.indices[0]]), self.create_superinstance([si.indices[1]])]
            else:
                # otherwise use k-means to split it
                new_si = self.split_superinstance(si, 2)

            if len(new_si) == 1:
                # we cannot split any further along this branch, we reached the splitting level
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            s1 = new_si[0]
            s2 = new_si[1]
            pt1 = min([s1.representative_idx, s2.representative_idx])
            pt2 = max([s1.representative_idx, s2.representative_idx])

            if self.querier.query_points(pt1, pt2):
                self.ml.append((pt1, pt2))
                must_link_found = True
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                continue
            else:
                self.cl.append((pt1, pt2))
                split_level += 1
                if self.store_intermediate_results:
                    self.intermediate_results.append(
                        (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))

            si_to_choose = []
            if len(s1.train_indices) >= 2:
                si_to_choose.append(s1)
            if len(s2.train_indices) >= 2:
                si_to_choose.append(s2)

            if len(si_to_choose) == 0:
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(max_split, split_n)

            si = min(si_to_choose, key=lambda x: len(x.indices))

        split_level = max([split_level, 1])
        split_n = 2 ** int(split_level)
        return min(max_split, split_n)

    def add_new_clusters_from_split(self, sis):
        """
            Given a list of super-instances creates a list of clusters such that each super-instance has its own cluster

        :param sis: a list of super-instances
        :return: a list of clusters ( :class:'~cluster.Cluster' )
        """
        new_clusters = []
        for x in sis:
            new_clusters.append(Cluster([x]))

        if len(new_clusters) == 1:
            return None
        else:
            return new_clusters

    def merge_containing_clusters(self, clustering_to_store):
        """
            Execute the merging phase on the current clustering


        :param clustering_to_store: the last valid clustering, this clustering is stored as an intermediate result for each query that is posed during the merging phase
        :return: a boolean indicating whether the merging phase was able to complete before the query limit is reached
        """
        query_limit_reached = False
        merged = True
        while merged and len(self.ml) + len(self.cl) < self.max_questions:

            clusters_to_consider = [cluster for cluster in self.clustering.clusters if not cluster.is_finished]

            cluster_pairs = itertools.combinations(clusters_to_consider, 2)
            cluster_pairs = [x for x in cluster_pairs if
                             not x[0].cannot_link_to_other_cluster(x[1], self.cl)]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))

            merged = False
            for x, y in cluster_pairs:

                if x.cannot_link_to_other_cluster(y, self.cl):
                    continue

                bc1, bc2 = x.get_comparison_points(y)
                pt1 = min([bc1.representative_idx, bc2.representative_idx])
                pt2 = max([bc1.representative_idx, bc2.representative_idx])

                if (pt1, pt2) in self.ml:
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    merged = True
                    break

                if len(self.ml) + len(self.cl) == self.max_questions:
                    query_limit_reached = True
                    break

                if self.querier.query_points(pt1, pt2):
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    self.ml.append((pt1, pt2))
                    merged = True

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))
                    break
                else:
                    self.cl.append((pt1, pt2))

                    if self.store_intermediate_results:
                        self.intermediate_results.append(
                            (clustering_to_store, time.time() - self.start_time, len(self.ml) + len(self.cl)))

        fully_merged = not query_limit_reached and not merged

        # if self.store_intermediate_results and not starting_level:
        if fully_merged and self.store_intermediate_results:
            self.intermediate_results[-1] = (self.clustering.construct_cluster_labeling(), time.time() - self.start_time,
                                             len(self.ml) + len(self.cl))
        return fully_merged

    def identify_superinstance_to_split(self):
        """
            From the current clustering (self.clustering) select the super-instance that has to be split next.
            The super-instance that contains the most instances is selected.

            The following types of super-instances are excluded from selection:

                - super-instances from clusters that are marked as pure or finished by the Querier
                - super-instances that failed to be split are excluded
                - super-instances with less than 2 training instances

            :return: the selected super-instance for splitting and the cluster from which it originates
            :rtype: Tuple[SuperInstance, Cluster]
        """

        if len(self.clustering.clusters) == 1 and len(self.clustering.clusters[0].super_instances) == 1:
            return self.clustering.clusters[0].super_instances[0], self.clustering.clusters[0]

        superinstance_to_split = None
        max_heur = -np.inf
        originating_cluster = None

        for cluster in self.clustering.clusters:

            if cluster.is_pure:
                continue

            if cluster.is_finished:
                continue

            for superinstance in cluster.super_instances:
                if superinstance.tried_splitting:
                    continue

                if len(superinstance.indices) == 1:
                    continue

                if len(superinstance.train_indices) < 2:
                    continue

                if len(superinstance.indices) > max_heur:
                    superinstance_to_split = superinstance
                    max_heur = len(superinstance.indices)
                    originating_cluster = cluster

        if superinstance_to_split is None:
            return None, None

        return superinstance_to_split, originating_cluster


class Querier(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def query_points(self, idx1, idx2):
        return

    def continue_cluster_process(self):
        """Returns whether or not the clustering process should continue"""
        return True

    def update_clustering(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        pass  # do nothing

    def update_clustering_detailed(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        pass  # do nothing


class LabelQuerier(Querier):

    def __init__(self, labels):
        super(LabelQuerier, self).__init__()
        self.labels = labels

    def query_points(self, idx1, idx2):
        return self.labels[idx1] == self.labels[idx2]


class COBRAS_kmeans(COBRAS):

    def split_superinstance(self, si, k):
        """
            Splits the given super-instance using k-means
        """

        data_to_cluster = self.data[si.indices, :]
        km = KMeans(k, n_init=10)
        km.fit(data_to_cluster)

        split_labels = km.labels_.astype(int)

        training = []
        no_training = []

        for new_si_idx in set(split_labels):
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance_kmeans(self.data, cur_indices, self.train_indices, si))
            else:
                no_training.append((cur_indices, np.mean(self.data[cur_indices,:],axis=0)))

        for indices, centroid in no_training:
            closest_train = min(training, key=lambda x: np.linalg.norm(self.data[x.representative_idx,:] - centroid))
            closest_train.indices.extend(indices)

        si.children = training

        return training

    def create_superinstance(self, indices, parent=None):
        """
            Creates a super-instance of type SuperInstance_kmeans
        """

        return SuperInstance_kmeans(self.data, indices, self.train_indices, parent)
    

class COBRAS():
    """Wrapper for COBRAS kmeans
    
    https://github.com/ML-KULeuven/cobras/tree/master
    """
    def __init__(self, max_queries=100):
        self.max_queries = max_queries
        self.trained = False
        self.clusterer = None
        self.clusterin = None
        self.intermediate_clusterings = None
        self.runtimes = None
        self.ml = None
        self.cl = None
    
    def fit(self, X, y):
        self.clusterer = COBRAS_kmeans(X, LabelQuerier(y), self.max_queries)
        self.clustering, self.intermediate_clusterings, self.runtimes, self.ml, self.cl = self.clusterer.cluster()
        self.trained = True


    # def predict(self, X):
    #     if not self.trained:
    #         raise TypeError("COBRAS not trained yet!")
    #     return self.clustering.construct_cluster_labeling()
    
if __name__ == "__main__":
    from clustpy.data import load_optdigits
    from sklearn.metrics import adjusted_rand_score
    
    X, y = load_optdigits(return_X_y=True)
    model = COBRAS(max_queries=100)
    model.fit(X, y)

    print(len(model.intermediate_clusterings))
    for i, clabels in enumerate(model.intermediate_clusterings):
        ari = adjusted_rand_score(y, clabels)
        print("iter {} ARI: {}".format(i, ari))