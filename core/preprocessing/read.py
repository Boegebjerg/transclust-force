import numpy as np


def read(file):

    with open(file) as data_file:
        data = data_file.read()
        n = len(data[:data.find('\n')].split('\t'))+1
    return np.fromstring(data, sep='\t'),n

def normalize(data_inc):
    data = np.zeros((4000,4000))
    data[np.triu_indices(4000,1)] = data_inc
    data2 = data.T
    np.fill_diagonal(data2,0)
    sims = data + data2
    sims = sims / max(np.max(sims),np.abs(np.min(sims)))
    return sims

def normalize_triangle(data):
    return data / max(np.max(data),np.abs(np.min(data)))


