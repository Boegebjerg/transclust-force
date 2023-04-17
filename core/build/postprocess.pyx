#!python
#cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np


cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i,
                                         np.npy_int64 j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)


cdef clean_clustering(int[:] clustering, int n):
    cdef int i,j,cluster
    cdef int counter = 0


    cdef int[:] seen = np.full(shape=clustering.shape[0], fill_value=-1, dtype='int32')

    for i in range(n):
        cluster = clustering[i]
        if seen[cluster] == -1:
            seen[cluster] = counter
            clustering[i] = counter
            counter += 1
        else:
            clustering[i] = seen[cluster]






def recluster(double[:] sims,
                 int[:] clustering,
                 int n):
    cdef int i,j,k, current_cluster
    cdef int clustering_n = -1
    cdef double cost = 0
    cdef double cost_to_leave = 0
    cdef double best_cost = 0
    cdef int best_cluster

    for i in range(n):
        j = clustering[i]
        if j > clustering_n:
            clustering_n = j
    for i in range(n):
        cost_to_leave = 0
        best_cost = np.inf
        current_cluster = clustering[i]

        for j in range(clustering_n+1):

            cost = 0
            for k in range(n):
                if i == k:
                    continue
                if clustering[k] == j:
                    cost -= sims[condensed_index(n,i,k)]
                else:
                    cost += sims[condensed_index(n,i,k)]

            if cost < best_cost:
                best_cost = cost
                best_cluster = j

        clustering[i] = best_cluster


    clean_clustering(clustering,n)
    return np.asarray(clustering)



def merge(double[:] sims,
                 int[:] clustering,
                 int n):
    cdef int i,j,k
    cdef int clustering_n = -1


    for i in range(n):
        j = clustering[i]
        if j > clustering_n:
            clustering_n = j

    cdef costs_to_stay = np.zeros(clustering_n+1)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if clustering[i] == clustering[j]:
                costs_to_stay[clustering[i]] -= sims[condensed_index(n,i,j)]
            else:
                costs_to_stay[clustering[i]] += sims[condensed_index(n,i,j)]
