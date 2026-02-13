from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

from clustering.cobra import COBRA


class DeepCOBRA:
    """A Wrapper to combine an autoencoder and COBRA"""
    def __init__(self, n_clusters_init: int = None, max_queries: int = None, gt_n_clusters: int = None, init_clustering_method: str = "kmeans", neural_network=None, ae_layers=None, ae_epochs=100, eval_interval=None, random_state=None):
        self.n_clusters_init = n_clusters_init
        self.max_queries = max_queries
        self.gt_n_clusters = gt_n_clusters
        self.init_clustering_method = init_clustering_method
        self.epochs = ae_epochs
        self.centroids = None
        self.cluster_labels = None
        self.fitted = False
        self.ae_layers = ae_layers
        self.random_state = random_state
        self.eval_interval = eval_interval
        self.labels_ = None
        self.queries_record = None
        self.labels_record = None

        # init cobra
        self.cobra = COBRA(self.n_clusters_init, self.max_queries, self.gt_n_clusters, self.init_clustering_method, self.eval_interval, self.random_state)
        self.n_clusters_init = self.cobra.n_clusters_init

        if neural_network is not None:
            self.neural_network = neural_network
            self.fitted = True
        elif ae_layers is not None:
            self.neural_network = FeedforwardAutoencoder(layers=ae_layers, random_state=self.random_state)

    def fit(self, X, y):
        if not self.fitted:
            self.neural_network.fit(data=X, n_epochs=self.epochs)
        Z = self.transform(X)
        self.cobra.fit(Z, y)
        self.fitted = True
        self.n_queries_used_ = self.cobra.n_queries_used_
        self.labels_ = self.cobra.labels_
        self.queries_record = self.cobra.queries_record
        self.labels_record = self.cobra.labels_record

    def transform(self, X):
        Z = self.neural_network.transform(X)
        return Z

    def fit_predict(self, X, y):
        self.fit(X, y)
        pred = self.predict(X)
        return pred

    def predict(self, X):
        Z = self.transform(X)
        pred = self.cobra.predict(Z)
        return pred
