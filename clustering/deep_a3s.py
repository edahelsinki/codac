from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder

from clustering.a3s import A3S


class DeepA3S:
    """A Wrapper to combine an autoencoder and A3S"""
    def __init__(self, max_queries=100, n_neighbors=50, T=1000, clus_method="fpc", tau=0.5, subsample=None, random_state=None, neural_network=None, embedding_size=10, ae_epochs=100):
        self.max_queries = max_queries
        self.n_neighbors = n_neighbors
        self.T = T
        self.clus_method = clus_method
        self.tau = tau
        self.subsample = subsample
        self.random_state = random_state
        self.embedding_size = embedding_size
        self.ae_epochs = ae_epochs

        self.history = {"queries": [], "labels": []}

        # init A3S
        self.a3s = A3S(self.max_queries, self.n_neighbors, self.T, self.clus_method, self.tau, self.subsample, self.random_state)

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
        self.a3s.fit(Z, y)
        self.history = self.a3s.history


if __name__ == "__main__":
    # Example usage
    from experiments.data import get_optdigits
    
    X, y = get_optdigits()
    model = DeepA3S(random_state=42, T=5, subsample=None)
    model.fit(X, y)
    print(model.history)