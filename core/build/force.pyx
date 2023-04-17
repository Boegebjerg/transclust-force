#cython: language_level=3, profile=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow
import cython
import pandas as pd
import plotly.express as px
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libc.math cimport sqrt
from libc.string cimport memset


cdef inline np.npy_int64 condensed_index(np.npy_int64 n, np.npy_int64 i,
                                         np.npy_int64 j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) // 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) // 2) + (i - j - 1)



cdef double[:,::1] _initialize(int dims, int n):
    cdef double[:,::1] positions = np.zeros((n,dims))
    cdef int[:] nodes = np.arange(n)
    cdef int dim, node, start_index, end_index, i, size
    cdef double slice
    start_index = <int>np.floor(((0*n)/(dims)))
    for dim in range(dims):
        end_index = <int>np.ceil((((dim+1)*n)/(dims)))
        size = end_index-start_index
        slice = 2 * np.pi / size
        for i in range(start_index, end_index):
            wrap = (dim+1) % dims
            positions[i,dim] = np.sin(slice * (i-start_index))
            positions[i,wrap] = np.cos(slice * (i-start_index))
        start_index = end_index
    return positions

cdef double calc_norm(double[:,:] positions, int i, int dims):
    cdef double norm = 0
    cdef int dim
    for dim in range(dims):
        norm += positions[i, dim] * positions[i, dim]
    return np.sqrt(norm)


cdef inline double _calc_temperature(int i, int n, double temp):
    return pow((1.0 / (i+1)),2) * temp * n



cdef double euclidean_distance(double[:, ::1] X,
                               np.intp_t i1, np.intp_t i2):
    cdef double tmp, d
    cdef np.intp_t j

    d = 0

    for j in range(X.shape[1]):
        tmp = X[i1, j] - X[i2, j]
        d += tmp * tmp

    return sqrt(d)


cdef inline double[:,::1] _calc_displacements(double[:,::1] similarities,
                              double[:,::1] positions,
                              double attraction,
                              double repulsion,
                              int n,
                              int dims):
    cdef double distance, force, displacement
    cdef int i, j, dim
    cdef double[:,::1] displacements = np.zeros((n, dims))
    for i in range(n):
        for j in range(i+1,n):
            distance = euclidean_distance(positions, i, j)
            #print(f"Sim {i},{j}: {similarities[i,j]}")
            if similarities[i,j] > 0:
                force = (
                    np.log2(distance+1) * similarities[i,j] * attraction
                ) / distance
                #if i == 0:
                #    print(f"Attract: {force}")
                #print(f"{np.log2(distance+1)} {similarities[i,j]} {attraction} {distance}")

                for dim in range(dims):
                        displacement = (positions[j, dim]-positions[i, dim]) * force
                        displacements[i, dim] += displacement
                        displacements[j, dim] -= displacement
            else:
                force = (
                    (
                        similarities[i,j] * repulsion
                    ) / np.log2(distance+1)
                ) / distance
                #print(f"Repulse: {force}")

                #print(f"{np.log2(distance+1)} {similarities[i,j]} {attraction} {distance}")

                for dim in range(dims):
                    displacement = (positions[j, dim]-positions[i, dim]) * force
                    displacements[i, dim] += displacement
                    displacements[j, dim] -= displacement
    #print("End")
    return displacements



cdef double _calc_cluster_score(int[:] clusters, double[:,::1] sims):
    cdef double score = 0
    cdef double cost
    cdef int i,j
    cdef int length = len(clusters)
    for i in range(length):
        for j in range(length):
            cost = sims[i,j]
            if clusters[i] != clusters[j]:
                if cost > 0:
                    score += cost
            elif cost < 0:
                score -= cost
    return score


cdef double[:,::1] _force(double[:,::1] similarities, int n, int dims, double temp, int iterations):
    cdef double attraction = 100.0 / n, repulsion = 100.0 / n
    cdef double[:,::1] positions = _initialize(dims,n)
    cdef double[:,::1] displacements
    cdef int i, j, dim
    cdef double current_temp, norm
    for i in range(iterations):
        print(i)
        current_temp = _calc_temperature(i,n,temp)
        displacements = _calc_displacements(similarities,
                                            positions,
                                            attraction,
                                            repulsion,
                                            n,
                                            dims)

        for j in range(n):
            norm = calc_norm(displacements, j, dims)
            for dim in range(dims):
                if norm > current_temp:
                    displacements[j, dim] = (displacements[j, dim] / norm) * temp
                positions[j, dim] = positions[j, dim] + displacements[j, dim]
        res = np.asarray(positions)
        #with open(f'runs/force_{i}.txt','w') as f:
        #    f.write('\n'.join(['\t'.join([str(b) for b in a]) for a in res]))
        """
        print(res.shape)
        my_dict = {
            'x':res[:,0],
            'y':res[:,1],
            'z':res[:,2],
            'Cluster':[b for a in range(4) for b in [a]*1000]
        }
        fig = px.scatter_3d(pd.DataFrame(my_dict),
                            x='x',
                            y='y',
                            z='z',
                            color='Cluster')
        fig.write_image(f"images/iteration_{str(i).rjust(6,'0')}.png", height=1000, width=1000)"""



    return np.asarray(positions)


def calc_cluster_score(clusters, sims):
    return _calc_cluster_score(clusters, sims)



def force(similarities, dims, temp, iterations, n=None):
    if not n:
        n = similarities.shape[0]
    positions = _force(similarities,n, dims, temp, iterations)
    return np.asarray(positions)

def init(sims, dims):
    return np.asarray(_initialize(dims, sims.shape[0]))




def test():
    print("Hello")