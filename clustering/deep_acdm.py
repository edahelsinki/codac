import numpy as np
from clustpy.deep.neural_networks.feedforward_autoencoder import FeedforwardAutoencoder
from sklearn.utils import check_random_state

from clustering.acdm import ACDM


class DeepACDM:
    """A Wrapper to combine an autoencoder and ACDM"""
    def __init__(self, knn: int = 14, omega: float = 100, beta: float = 100, max_queries: int = None, random_state: int | np.random.RandomState = None, neural_network=None, embedding_size=10, ae_epochs=100):
        self.knn = knn
        self.omega = omega
        self.beta = beta
        self.random_state = check_random_state(random_state)
        self.embedding_size = embedding_size
        self.ae_epochs = ae_epochs
        if max_queries is None:
            self.max_queries = np.inf
        else:
            self.max_queries = max_queries

        self.records_ = None

        # init ACDM
        self.acdm = ACDM(self.knn, self.omega, self.beta, self.max_queries, self.random_state)

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
        self.acdm.fit(Z, y)
        self.records_ = self.acdm.records_


if __name__ == "__main__":
    from experiments.data import get_optdigits

    X, y = get_optdigits()
    model = DeepACDM(max_queries=100, ae_epochs=20)
    model.fit(X, y)
    print(model.records_)
