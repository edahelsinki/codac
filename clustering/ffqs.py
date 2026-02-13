import time

import numpy as np
from active_semi_clustering.active.pairwise_constraints import (
    NPU,
    ExampleOracle,
    ExploreConsolidate,
)
from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans


class FFQS():

    """FFQS active clustering
    
    Basu, S., Banerjee, A., & Mooney, R. J. (2004).
    Active Semi-Supervision for Pairwise Constrained Clustering.
    In Proceedings of the 2004 SIAM International Conference on Data Mining (SDM) (pp. 333-344).
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611972740.31

    Code adapted from: https://github.com/xiangtanshi/A3S
    """

    def __init__(self, n_clusters=10, query_sample_size=10, max_queries=100):
        self.n_clusters = n_clusters
        self.query_sample_size = query_sample_size
        self.max_queries = max_queries

        self.label_record = None
        self.query_record = None


    def fit(self, X, y):
        print("Fitting FFQS")

        q = 0
        q_list = [0]
        while q<self.max_queries:
            q += self.query_sample_size
            if q < self.max_queries:
                q_list.append(q)
            else:
                q_list.append(self.max_queries)

        self.label_record = _FFQS(X, y, self.n_clusters, q_list)
        self.query_record = q_list


    # def predict(self, X):
    #     return self.label_record[-1]


def query(gt_label, x, y):
    return gt_label[x] == gt_label[y]

def state_resolution(pair, state, cl, ml):
    # Fast transitivity inference: 
    # ensure that there are no pair (i,j) where S[i,j]=0, while S[i,j] could be induced by known chains such as <S[i,k1], S[k1,k2], S[k2,j]>
    for idx in pair:
        link_set = np.where(state[idx]==1)[0]
        sep_set = np.where(state[idx]==-1)[0]
        if len(link_set)>2:
            for p in link_set:
                for q in link_set:
                    if p != q:
                        state[p,q] = 1
                        if (p,q) not in ml:
                            ml.append((p,q))
                            ml.append((q,p))
        if len(sep_set) and len(link_set):
            for p in link_set:
                for q in sep_set:
                    state[p,q] = -1
                    state[q,p] = -1
                    if (p,q) not in cl:
                        cl.append((p,q))
                        cl.append((q,p))

def Random_P(X, y, N, state, k, query_threshold_list):
    labels_record = []
    start_time = time.time()
    upper = query_threshold_list[-1]
    absolue_index = np.random.choice([p for p in range(N*N)],size=2*upper,replace=True).tolist()
    pair_set = []
    for i in range(2*upper):
        item = absolue_index.pop(0)
        row = int(item/N)
        col = item % N
        if row != col and [row,col] not in pair_set:
            pair_set.append([row,col])
    ml, cl = [], []
    q = 0
    checkpoint = query_threshold_list.pop(0)
    while q < upper and len(pair_set):
        pair = pair_set.pop(0)
        if state[pair[0],pair[1]] == 0:
            q += 1
        if query(y,pair[0],pair[1]):
            state[pair[0],pair[1]] = 1
        else:
            state[pair[0],pair[1]] = -1
        state_resolution(pair)
        if q == checkpoint:
            clusterer = PCKMeans(n_clusters=k, max_iter=50)
            clusterer.fit(X,ml=ml,cl=cl)
            labels_record.append(np.copy(clusterer.labels_))
            if len(query_threshold_list):
                checkpoint = query_threshold_list.pop(0)
            else:
                break
    
    end_time = time.time()
    print('total running time of Random: {}seconds.'.format(end_time-start_time))
    return labels_record


def _FFQS(X, y, k, query_threshold_list):
    labels_record = []
    start_time = time.time()
    for threshold in query_threshold_list:
        print("Finding {} queries".format(threshold))
        if threshold == 0:
            print("Fitting PCKMeans")
            clusterer = PCKMeans(n_clusters=k, max_iter=100)
            clusterer.fit(X)
            labels_record.append(np.copy(clusterer.labels_))
        else:
            oracle = ExampleOracle(y,max_queries_cnt=threshold)
            active_learner = ExploreConsolidate(n_clusters=k)
            active_learner.fit(X,oracle=oracle)
            constraints = active_learner.pairwise_constraints_
            print("Fitting PCKMeans")
            clusterer = PCKMeans(n_clusters=k, max_iter=100)
            clusterer.fit(X,ml=constraints[0],cl=constraints[1])
            labels_record.append(np.copy(clusterer.labels_))
    end_time = time.time()
    print('total running time of FFQS: {}seconds.'.format(end_time-start_time))
    return labels_record

def NPU_(X, y, k, query_threshold_list):
    labels_record = []
    start_time = time.time()
    for threshold in query_threshold_list:
        oracle = ExampleOracle(y,max_queries_cnt=threshold)
        clusterer = PCKMeans(n_clusters=k, max_iter=100)
        active_learner = NPU(clusterer=clusterer)
        active_learner.fit(X,oracle=oracle)
        constraints = active_learner.pairwise_constraints_
        clusterer = PCKMeans(n_clusters=k, max_iter=100)
        clusterer.fit(X,ml=constraints[0],cl=constraints[1])
        labels_record.append(np.copy(clusterer.labels_))
    end_time = time.time()
    print('total running time of FFQS: {}seconds.'.format(end_time-start_time))
    return labels_record


# # load the data
# X, y = load_iris(return_X_y=True)
# N = X.shape[0]
# k = len(np.unique(y))
# state = np.eye(N)
# ml, cl = [],[]
# #labels_record = []

# #checkpoint_list = [1] + [int(4000/6*i) for i in range(1,7)]
# checkpoint_list = [5, 10, 15]
# print(checkpoint_list)
# checkpoint_list_ = deepcopy(checkpoint_list)
# if args.method == 'random':
#     result = Random_P(X, y, checkpoint_list)
# elif args.method == 'ffqs':
#     result = FFQS(X, y, checkpoint_list)
# elif args.method == 'npu':
#     result = NPU_(X, y, checkpoint_list)

if __name__ == "__main__":

    from clustpy.data import load_optdigits
    from sklearn.metrics import adjusted_rand_score
    X, y = load_optdigits(return_X_y=True)

    model = FFQS(n_clusters=10, query_sample_size=100, max_queries=300)
    model.fit(X, y)
    pred = model.label_record[-1]
    print("ARI", adjusted_rand_score(y, pred))

