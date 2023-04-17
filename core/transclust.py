import pyximport
import numpy as np


pyximport.install(setup_args={
    "include_dirs": np.get_include(),
    'options': {
        'build_ext': {
            'cython_directives': {
                'language_level': 3,
                'boundscheck':False,
                'wraparound':False,
                'initializedcheck':False,
                'nonecheck':False,
                'overflowcheck':False,
                'cdivision':True,
                'optimize.use_switch': True,
                'profile': True,
            }
        }
    }
}, reload_support=True)
from core.build.postprocess import recluster

from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist
from core.build.force import force, calc_cluster_score
from core.preprocessing.read import read, normalize_triangle


def cluster(similarity, threshold, iterations, use_pp):
    data, n = read(similarity)
    sims = normalize_triangle(data-threshold)
    res = force(similarities=sims,
            dims=3,
            temp=100,
            iterations=iterations,
            n=n)
    best_clustering = []
    best_score = np.finfo('d').max
    z = single(pdist(res))
    for t in z[::-1,2][range(0,n,n//20)]:
        print(t)
        clustering = fcluster(z,t,criterion="distance")
        score = calc_cluster_score(clustering, sims, n)
        if score < best_score:
            best_score = score
            best_clustering = clustering
    if use_pp:
        convergence = False
        while not convergence:
            print("Clustering..")
            old_clustering = best_clustering.copy()
            recluster(data,best_clustering,n)
            convergence = (best_clustering == old_clustering).all()
    return best_clustering












