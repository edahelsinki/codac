from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

from clustering.cobras import COBRAS


class DeepCOBRAS:
    """A Wrapper to combine an autoencoder and COBRAS"""
    def __init__(self, max_queries=100, neural_network=None, embedding_size=10, ae_epochs=100, random_state=None):
        self.max_queries = max_queries
        self.trained = False
        self.clusterer = None
        self.clusterin = None
        self.intermediate_clusterings = None
        self.runtimes = None
        self.ml = None
        self.cl = None

        self.random_state = random_state
        self.embedding_size = embedding_size
        self.ae_epochs = ae_epochs

        # init COBRAS
        self.cobras = COBRAS(self.max_queries)

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
        self.cobras.fit(Z, y)
        self.clustering = self.cobras.clustering
        self.intermediate_clusterings = self.cobras.intermediate_clusterings
        self.runtimes = self.cobras.runtimes
        self.ml = self.cobras.ml
        self.cl = self.cobras.cl


if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score

    from experiments.data import get_optdigits

    X, y = get_optdigits()
    model = DeepCOBRAS(max_queries=100, ae_epochs=20)
    model.fit(X, y)
    
    print(len(model.intermediate_clusterings))
    for i, clabels in enumerate(model.intermediate_clusterings):
        ari = adjusted_rand_score(y, clabels)
        print("iter {} ARI: {}".format(i, ari))
