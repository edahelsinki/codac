import copy
import heapq
import math

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.spatial.distance import euclidean
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

"""
Source: https://github.com/briceloskie/ACDM/
"""


def influence_model_propagation(skeleton,neighborhood):
    for i in range(len(skeleton.nodes)):
        skeleton.nodes[i]['state'] = 'inactive'
        skeleton.nodes[i]['label'] = 'unclear'
    nodes_in_neighborhood = []
    for i in range(len(neighborhood)):
        for node in neighborhood[i]:
            skeleton.nodes[node]['state'] = 'active'
            skeleton.nodes[node]['label'] = i
            nodes_in_neighborhood.append(node)
    for node in nodes_in_neighborhood:
        container = [node]
        label=skeleton.nodes[node]['label']
        while(True):
            neighbors = list(skeleton.neighbors(container[-1]))
            container.pop()
            for neighbor in neighbors:
                if (skeleton.nodes[neighbor]['ranking'] > skeleton.nodes[node]['ranking']) and (skeleton.nodes[neighbor]['state']=='inactive'):
                    container.append(neighbor)
                    skeleton.nodes[neighbor]['state'] = 'active'
                    skeleton.nodes[neighbor]['label'] = label
            if len(container)==0:
                break
    predicted_labels=[]
    for i in range(len(skeleton.nodes)):
        label=skeleton.nodes[i]['label']
        predicted_labels.append(label)
    return predicted_labels


def centrality_ranking(G, start_node):
    traversed_nodes = [start_node]
    candidate_edge = []
    for edge in G.edges(start_node, data=True):
        heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    while candidate_edge:
        max_edge = heapq.heappop(candidate_edge)
        weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
        traversed_nodes.append(new_node)
        for edge in G.edges(new_node, data=True):
            if edge[1] != current_node:
                heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    return traversed_nodes


def order_allocation(skeleton, representative):
    decision_list = centrality_ranking(skeleton, start_node=representative)
    for i in range(len(decision_list)):
        skeleton.nodes[decision_list[i]]['ranking'] = i
    return skeleton,decision_list


def connections_cal(data, node, neighborhood_r):
    connections = []
    for i in range(len(neighborhood_r)):
        distances = []
        for neighbor in neighborhood_r[i]:
            distances.append(euclidean(data[node], data[neighbor]))
        index = np.argmin(distances)
        connections.append([node, neighborhood_r[i][index], distances[index],i])
    connections = np.array(connections)
    sorted_indices = np.argsort(connections[:, 2])
    connections = connections[sorted_indices]
    return connections

def interaction_process(connections, real_labels, neighborhood, count, neighborhood_r, neighborhood_r_behind, skeleton, l):
    flag = False
    for i in range(len(connections)):
        node1 = int(connections[i][0])
        node2 = int(connections[i][1])
        neighborhood_index = int(connections[i][3])
        if real_labels[node1] == real_labels[node2]:
            count=count+1
            flag = True

            if len(neighborhood[neighborhood_index])<l:
                neighborhood[neighborhood_index].append(node1)
                neighborhood_r[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['ranking']>skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]]['ranking']:
                    neighborhood_r_behind[neighborhood_index]=[node1]
            if len(neighborhood[neighborhood_index])>=l:
                neighborhood[neighborhood_index].append(node1)
                if skeleton.nodes[node1]['ranking']<skeleton.nodes[neighborhood_r_behind[neighborhood_index][0]]['ranking']:
                    neighborhood_r[neighborhood_index].remove(neighborhood_r_behind[neighborhood_index][0])
                    neighborhood_r[neighborhood_index].append(node1)
                    a=[]
                    for j in neighborhood_r[neighborhood_index]:
                        a.append(skeleton.nodes[j]['ranking'])
                    c=neighborhood_r[neighborhood_index][np.argmax(a)]
                    neighborhood_r_behind[neighborhood_index]=[c]
            break
        if real_labels[node1] != real_labels[node2]:
            count = count + 1
    if flag == False:
        neighborhood.append([node1])
        neighborhood_r.append([node1])
        neighborhood_r_behind.append([node1])
    return neighborhood,neighborhood_r,neighborhood_r_behind,count


def neighborhood_initialization(data, decision_list, representative, real_labels, skeleton, m, l):
    count = 0
    neighborhood = []
    neighborhood.append([representative])
    neighborhood_r = []
    neighborhood_r.append([representative])
    neighborhood_r_behind=[]
    neighborhood_r_behind.append([representative])
    nodes = decision_list[1:m]
    decision_list.remove(representative)
    for node in nodes:
        decision_list.remove(node)
        connections = connections_cal(data, node, neighborhood_r)
        neighborhood,neighborhood_r,neighborhood_r_behind,count=interaction_process(connections, real_labels, neighborhood, count, neighborhood_r,neighborhood_r_behind, skeleton, l)
    return neighborhood,neighborhood_r,neighborhood_r_behind,count,decision_list


def k_nearest_neighbor_cal(data,k):

    neighbors = NearestNeighbors(n_neighbors=k).fit(data)
    k_nearest_neighbors = neighbors.kneighbors(data, return_distance=False)
    return k_nearest_neighbors


def uncertainty_oneNode(predict_labels, k_nearest_neighbor,k):
    dict={}
    for i in range(len(k_nearest_neighbor)):
        point=k_nearest_neighbor[i]
        if predict_labels[point] not in dict.keys():
            dict[predict_labels[point]]=[point]
        else:
            dict[predict_labels[point]].append(point)
    sum=0
    for m in dict.keys():
        proportion=len(dict[m])/k
        if proportion != 0:
            sum = sum + proportion * math.log2(proportion)
    sum = -sum
    if sum==-0.0:
        sum=0.0
    return sum

def uncertainty_cal(predict_labels,k_nearest_neighbors,candidates,k):
    uncertainty_dict=dict()
    for candidate in candidates:
        k_nearest_neighbor=k_nearest_neighbors[candidate]
        uncertainty=uncertainty_oneNode(predict_labels, k_nearest_neighbor,k)
        uncertainty_dict[candidate]=uncertainty
    return uncertainty_dict

def first_n_nodes_cal(my_dict,n):
    if n>len(my_dict):
        n=len(my_dict)
    sliced_list=[]

    heap = [(-value, key) for key, value in my_dict.items()]
    heapq.heapify(heap)
    count=0
    for _ in range(n):
        if heap:
            neg_value, key = heapq.heappop(heap)
            value = -neg_value
            if value==0.0:
                break
            sliced_list.append(key)
            del my_dict[key]
            count=count+1
        else:
            break
    remaining_count = n - count
    if remaining_count > 0:
        for i in range(remaining_count):
            first_key, first_value = next(iter(my_dict.items()))
            del my_dict[first_key]
            sliced_list.append(first_key)
            count = count + 1
    return sliced_list,my_dict



def neighborhood_learning(skeleton, data, predict_labels, neighborhood,neighborhood_r,neighborhood_r_behind, k_nearest_neighbors, count, order,real_labels, record, n, k, l, max_queries):
    candidates = dict()
    for i in range(len(order)):
        candidates[order[i]] = 0
    flag = False
    iter=2
    while (True):
        candidates = uncertainty_cal(predict_labels, k_nearest_neighbors, candidates,k)
        sliced_list,candidates=first_n_nodes_cal(candidates, n)
        for node in sliced_list:
            connections = connections_cal(data, node, neighborhood_r)
            neighborhood,neighborhood_r,neighborhood_r_behind,count=interaction_process(connections, real_labels, neighborhood, count, neighborhood_r,neighborhood_r_behind, skeleton,l)
        if candidates == dict():
            flag = True
        predict_labels=influence_model_propagation(skeleton, neighborhood)
        ari = adjusted_rand_score(real_labels, predict_labels)
        record.append({"iter": iter, "queries": count, "labels": predict_labels})
        print("iteration: %d, queries: %d, ari: %s" % (iter, count, ari))
        iter=iter+1
        if flag == True:
            break
        if ari == 1:
            break
        if count > max_queries:
            print("Max queries ({}) reached, stopping.".format(max_queries))
            break
    return record



def clusters_to_predict_vec(clusters):
    tranversal_dict = {}
    predict_vec = []
    for i in range(len(clusters)):
        for j in clusters[i]:
            tranversal_dict[j] = i
    for i in range(len(tranversal_dict)):
        predict_vec.append(tranversal_dict[i])
    return predict_vec


def initialization_cut(skeleton,m,start_node):
    G=copy.deepcopy(skeleton)
    traversed_nodes = [start_node]
    candidate_edge = []
    for edge in G.edges(start_node, data=True):
        heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    while candidate_edge:
        max_edge = heapq.heappop(candidate_edge)
        G.remove_edge(max_edge[1],max_edge[2])
        if len(traversed_nodes)==m:
            break
        weight, current_node, new_node = -max_edge[0], max_edge[1], max_edge[2]
        traversed_nodes.append(new_node)
        for edge in G.edges(new_node, data=True):
            if edge[1] != current_node:
                heapq.heappush(candidate_edge, (-edge[2]['weight'], edge[0], edge[1]))
    clusters = []
    S = [G.subgraph(c) for c in nx.connected_components(G)]
    for i in S:
        clusters.append(list(i.nodes))
    predict_labels = clusters_to_predict_vec(clusters)
    return predict_labels

def draw_graph(G):
    pos = graphviz_layout(G, prog="twopi")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos,alpha=0.5, node_color="blue", with_labels=True,font_size=20,node_size=30)
    plt.axis("equal")
    plt.show()

def nearest_neighbor_cal(feature_space):
    neighbors=NearestNeighbors(n_neighbors=2).fit(feature_space)
    distance,nearest_neighbors= neighbors.kneighbors(feature_space,return_distance=True)
    distance=distance[:,1]
    nearest_neighbors=nearest_neighbors.tolist()
    for i in range(len(nearest_neighbors)):
        nearest_neighbors[i].append(distance[i])
    return nearest_neighbors


def add_randomness(data, random_state):
    size=np.shape(data)
    random_matrix=random_state.rand(size[0],size[1]) * 0.000001
    data=data+random_matrix
    return data


def representative_cal(sub_S):
    degree_dict = dict(sub_S.degree())
    max_degree = max(degree_dict.values())

    nodes_with_max_degree = [node for node, degree in degree_dict.items() if degree == max_degree]

    min_weighted_degree_sum = float('inf')
    min_weighted_degree_node = None
    for node in nodes_with_max_degree:

        weighted_degree_sum = sum(weight for _, _, weight in sub_S.edges(data='weight', nbunch=node))

        if weighted_degree_sum < min_weighted_degree_sum:
            min_weighted_degree_sum = weighted_degree_sum
            min_weighted_degree_node = node
    representative=min_weighted_degree_node
    return representative

def clustering_loop(feature_space,dict_mapping,skeleton):
    Graph=nx.Graph()
    representatives = []
    edges=nearest_neighbor_cal(feature_space)
    Graph.add_weighted_edges_from(edges)
    S = [Graph.subgraph(c).copy() for c in nx.connected_components(Graph)]
    for sub_S in S:
        representative=representative_cal(sub_S)
        representatives.append(representative)
    for i in range(len(edges)):
        edges[i][0] = dict_mapping[edges[i][0]]
        edges[i][1] = dict_mapping[edges[i][1]]
    for i in range(len(representatives)):
        representatives[i]=dict_mapping[representatives[i]]
    skeleton.add_weighted_edges_from(edges)
    dict_mapping={}
    for i in range(len(representatives)):
        dict_mapping[i]=representatives[i]
    return representatives,skeleton,dict_mapping

def graph_initialization(data):
    feature_space = copy.deepcopy(data)
    dict_mapping = {}
    for i in range(len(feature_space)):
        dict_mapping[i] = i
    skeleton = nx.Graph()
    while (True):
        representatives, skeleton, dict_mapping = clustering_loop(feature_space, dict_mapping, skeleton)
        feature_space = data[representatives]
        if len(representatives) == 1:
            break
    representative=representatives[0]
    return skeleton,representative


class ACDM():

    def __init__(self, knn: int = 14, omega: float = 100, beta: float = 100, max_queries: int = None, random_state: int | np.random.RandomState = None):
        self.knn = knn
        self.omega = omega
        self.beta = beta
        self.random_state = check_random_state(random_state)
        if max_queries is None:
            self.max_queries = np.inf
        else:
            self.max_queries = max_queries

        self.records_ = None

    def fit(self, X, y):
        data = add_randomness(X, self.random_state)
        real_labels = y

        m = int(len(data) * (1 / self.beta))
        n = int(len(data) * (1 / self.beta))
        k_nearest_neighbors = k_nearest_neighbor_cal(data, self.knn)
        skeleton, representative = graph_initialization(data)
        record = [{"iter": 0, "queries": 0, "labels": np.zeros_like(y)}]
        skeleton, order = order_allocation(skeleton, representative)
        neighborhood,neighborhood_r,neighborhood_r_behind,count,order=neighborhood_initialization(data, order, representative, real_labels, skeleton, m, self.omega)
        predict_labels = influence_model_propagation(skeleton, neighborhood)
        record.append({"iter": 1, "queries": count, "labels": predict_labels})
        record = neighborhood_learning(skeleton, data, predict_labels, neighborhood, neighborhood_r, neighborhood_r_behind, k_nearest_neighbors, count, order,
                                    real_labels, record, n, self.knn, self.omega, self.max_queries)

        self.records_ = record

if __name__ == '__main__':
    from experiments.data import get_waveform
    # parameters
    k = 14
    beta = 100
    omega = 100
    data, labels = get_waveform()
    
    acdm = ACDM(k, omega, beta)
    acdm.fit(data, labels)
    for item in acdm.records_:
        print("iter {} queries {}".format(item["iter"], item["queries"]))
