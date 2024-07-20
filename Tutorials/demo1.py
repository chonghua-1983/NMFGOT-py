# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:30:48 2024

@author: PC
"""

# from NMFGOT.model import NMFGOT

import torch
import numpy as np
import scanpy as sc
import anndata
import scipy.io

import sys
sys.path.append('C:/Users/PC/Desktop/NMFGOT/NMFGOT-py/')
from Code import model
from Code import utils


# load preprocessed data
mat = scipy.io.loadmat('Tutorials/data/data1.mat')
X1 = mat['X1']
X2 = mat['X2']
label = mat['label'][:,0]
X1 = X1.T
X2= X2.T

# create AnnData object
mic = anndata.AnnData(X1.T,dtype = np.float64)
met = anndata.AnnData(X2.T,dtype = np.float64)
mic.obs['sampletype'] = label

# get the taxa and metabolites
# taxa = np.array([ val[0]  for val in mat['taxa'][:,0]])
# mets = np.array([ val[0]  for val in mat['mets'][:,0]])
test_model = model.NMFGOT(mic,met)
test_model.run()

# Cluster and Visualization
# use S  to do clustering with louvain
clu = test_model.cluster()
from sklearn import metrics
ARI = metrics.adjusted_rand_score(label, clu)
NMI = metrics.normalized_mutual_info_score(label, clu)
print('NMI=: {:.4f}, ARI=: {:.4f}'.format(NMI, ARI))

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
# silhouette_score(met.T, clu)
# sil = silhouette_samples(X_ms,y_pred)
# print('sil=: {:.4f}'.format(np.mean(sil)))


# Umap with clustering label
test_model.visualize(clu, min_dist = 0.2, n_neighbors=5)

# Umap with true label
test_model.visualize(label,min_dist = 0.2, n_neighbors=5)



# import snf
# affinity_networks = snf.make_affinity(digits.data, metric='euclidean', K=20, mu=0.5)

# fused_network = snf.snf(affinity_networks, K=20)
# best, second = snf.get_n_clusters(fused_network)

# from sklearn.cluster import spectral_clustering
# from sklearn.metrics import v_measure_score
# labels = spectral_clustering(fused_network, n_clusters=best)
# v_measure_score(labels, digits.labels)
