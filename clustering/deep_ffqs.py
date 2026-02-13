from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.utils import check_random_state

from clustering.ffqs import FFQS


class DeepFFQS:
    """A Wrapper to combine an autoencoder and FFQS"""
    def __init__(self, n_clusters=10, query_sample_size=10, max_queries=100, neural_network=None, embedding_size=10, ae_epochs=100, random_state=None):
        self.n_clusters = n_clusters
        self.query_sample_size = query_sample_size
        self.max_queries = max_queries
        self.embedding_size = embedding_size
        self.ae_epochs = ae_epochs
        self.random_state = check_random_state(random_state)

        self.label_record = None
        self.query_record = None


        # init FFQS
        self.ffqs = FFQS(self.n_clusters, self.query_sample_size, self.max_queries)

        if neural_network is not None:
            self.neural_network = neural_network
            self.fitted = True
        else:
            self.fitted = False

    def fit(self, X, y):
        if not self.fitted:
            self.neural_network = FeedforwardAutoencoder(layers=[X.shape[1], 512, 256, self.embedding_size], random_state=self.random_state)
            self.neural_network.fit(data=X, n_epochs=self.ae_epochs)
            self.fitted = True
        Z = self.neural_network.transform(X)
        self.ffqs.fit(Z, y)
        self.label_record = self.ffqs.label_record
        self.query_record = self.ffqs.query_record


if __name__ == "__main__":
    from clustpy.data import load_optdigits
    from sklearn.metrics import adjusted_rand_score
    X, y = load_optdigits(return_X_y=True)

    model = DeepFFQS(n_clusters=10, query_sample_size=100, max_queries=500)
    model.fit(X, y)
    pred = model.label_record[-1]
    print("ARI", adjusted_rand_score(y, pred))
