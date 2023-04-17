import pyximport
import numpy as np
from scipy.cluster.hierarchy import single, fcluster
from scipy.spatial.distance import pdist

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
from core.build.force import force, init, calc_cluster_score
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from core.preprocessing.read import *
import timeit

#with open('4_cluster_revised.txt') as data_file:
#    data = data_file.read()
#arr = np.fromstring(data, sep='\t')
#data = output = np.zeros((4000,4000))
#data[np.triu_indices(4000,1)] = arr
#data2 = data.T
#np.fill_diagonal(data2,0)
#sims = data + data2
#sims = sims / max(np.max(sims),np.abs(np.min(sims)))
#sims = np.zeros((4000,4000))
#for i in range(data.shape[0]):
#    print(i)
#    sims[data.iloc[i,0]-1,data.iloc[i,1]-1] = data.iloc[i,2]
size = 300
dims = 3

#sims = np.zeros((size, size))

"""
print(timeit.repeat(lambda: force(similarities=sims,
                                  dims=3,
                                  temp=100,
                                  iterations=50),
                    number=10))
"""
data, n = read('4_cluster_revised.txt')
sims = normalize_triangle(data)



res = force(similarities=sims,
                       dims=3,
                       temp=100,
                       iterations=100,
                       n=n)

#res = init(sims, 3)
single(pdist(res))

scores = []
best_clustering = []
best_score = np.finfo('d').max
z = single(pdist(res))

for t in z[::-1,2][range(0,n,n//20)]:
    print(t)
    clustering = fcluster(z,t,criterion="distance")
    score = calc_cluster_score(clustering, sims, n)
    scores.append(score)
    if score < best_score:
        best_score = score
        best_clustering = clustering


fig = px.scatter_3d(pd.DataFrame({'x':res[:,0],'y':res[:,1],'z':res[:,2],'Cluster':[b for a in range(4) for b in [a]*1000]}),
                    x='x',
                    y='y',
                    z='z',
                    color='Cluster')

fig.write_html(f"100_iters.html")


exit()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(res[:,0],res[:,1],res[:,2])
#for i in range(0, size, size // dims):
#    ax.scatter(res[0, i:i + size // dims], res[1, i:i + size // dims], res[2, i:i + size // dims])

plt.show()

print(res)
