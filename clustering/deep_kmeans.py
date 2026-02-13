import copy

from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.cluster import KMeans


class DeepKmeans():
    """K-means clustering over an learned embedding"""
    def __init__(self, n_clusters, neural_network, embedding_size=10):
        self.n_clusters = n_clusters
        self.neural_network = neural_network
        self.embedding_size = embedding_size
        self.labels_ = None
        self.cluster_centers_ = None
        self.neural_network_trained = None

    def fit(self, X):
        if self.neural_network is None:
            self.neural_network = FeedforwardAutoencoder(layers=[X.shape[1], 500, 500, 2000, self.embedding_size]).fit(n_epochs=50, batch_size=256, data=X)
        else:
            # a pretrained network passed, copy to avoid changing the original
            self.neural_network = copy.deepcopy(self.neural_network)
        X_embedded = self.neural_network.transform(X)
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X_embedded)
        self.labels_ = kmeans.labels_
        self.cluster_centers_ = kmeans.cluster_centers_
        self.neural_network_trained_ = self.neural_network


if __name__ == "__main__":
    from clustpy.data import (
        load_optdigits,
    )
    from sklearn.metrics import adjusted_rand_score
    X, y = load_optdigits(return_X_y=True)

    ae_model = FeedforwardAutoencoder(layers=[X.shape[1], 256, 128, 10], random_state=10).fit(n_epochs=50, batch_size=256, data=X)
    
    model = DeepKmeans(n_clusters=10, neural_network=ae_model, embedding_size=10)
    model.fit(X)
    labels = model.labels_
    print("ARI:", adjusted_rand_score(y, labels))