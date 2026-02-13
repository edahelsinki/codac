import copy
import math
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics.cluster import contingency_matrix


def sparse_fisher_yates(n):
     """Efficient interger sampling without replacement"""
     A = dict()
     for i in reversed(range(n)):
         r = np.random.randint(i+1)
         yield A.get(r, r)
         A[r] = A.get(i, i)
         if i in A:
             del A[i]

def row_col_from_condensed_index(d, index):
	b = 1 - (2 * d)
	i = int((-b - math.sqrt(b ** 2 - 8 * index)) // 2)
	j = index + i * (b + i + 2) // 2 + 1
	return (i, j)


def extract_upper_tri_without_diagonal(A):
    """Get values from a matrix upper triangle excluding the diagonal"""
    # https://stackoverflow.com/questions/47314754/how-to-get-triangle-upper-matrix-without-the-diagonal-using-numpy
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

class ConstraintManager:
    """Disjoint set union datastructure for handling must/cannot link relations"""

    def __init__(self):
        # parent dict holds each entry's parent (cluster representative)
        # this is used to handle must-links
        self.parent = {}

        # size of a group
        self.size = defaultdict(lambda: 1)

        # group_rep -> members
        self.group_members = defaultdict(set)

        # cannot links are stored between each representative
        self.cannot_link = defaultdict(set)
        

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.group_members[x].add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def add_must_link(self, a, b):
        # find representatives
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return  # x, y belong to same group already

        # Check for CL conflict before merging
        if self.cannot_link[ra] & {rb} or self.cannot_link[rb] & {ra}:
            raise ValueError(f"Cannot must-link {a} and {b}: they are cannot-linked.")

        # always merge the smallest
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        # merge y to x
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

        # Merge group members
        self.group_members[ra].update(self.group_members[rb])
        del self.group_members[rb]

        # Merge cannot-link sets

        # cannot links of y should point to x
        for c in self.cannot_link[rb]:
            self.cannot_link[c].discard(rb)
            self.cannot_link[c].add(ra)

        # add cannot links of y to x
        self.cannot_link[ra].update(self.cannot_link[rb])
        if rb in self.cannot_link:
            del self.cannot_link[rb]

    def add_cannot_link(self, a, b):
        # finds the representatives of a,b
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            raise ValueError(f"Cannot-link between {a} and {b} would violate existing must-link group.")
        # the cannot link is symmetric (added to both representatives)
        self.cannot_link[ra].add(rb)
        self.cannot_link[rb].add(ra)

    def get_must_link_group(self, x):
        root = self.find(x)
        # get all members of a group
        result = self.group_members[root].copy()
        # remove itself from the must links
        result.remove(x)
        return result

    def get_cannot_link_group(self, x):
        root = self.find(x)
        result = set()
        for item in self.cannot_link[root]:
            result.update(self.group_members[item])
        return result


def query_constraints(index_pairs, labels, constraints):
    constraints_new = copy.deepcopy(constraints)
    for pair in index_pairs:
        if labels[pair[0]] == labels[pair[1]]:
            constraints_new["must-link"].append(pair)
        else:
            constraints_new["cannot-link"].append(pair)
    return constraints_new


def extract_constraint_pairs(cm):
    must_pairs = []
    cannot_pairs = []

    # extract must/cannot pairs for each data index in ConstraintManager
    for key in cm.parent.keys():
        ml = cm.get_must_link_group(key)
        for link in ml:
            must_pairs.append((key, link))

        cl = cm.get_cannot_link_group(key)
        for link in cl:
            cannot_pairs.append((key, link))

    # concatenate the must and cannot pairs, make sure the indices are integers
    pairs = torch.concatenate([torch.tensor(must_pairs), torch.tensor(cannot_pairs)]).int()

    # label 1: pair is a must link, label -1: pair is a cannot link
    labels = torch.concatenate([torch.ones(len(must_pairs), dtype=torch.int), -torch.ones(len(cannot_pairs), dtype=torch.int)])

    return pairs, labels


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 