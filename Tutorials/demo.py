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
#X1 = X1*10000  #computing ot by using scaling
X2= np.log(X2+1).T


# create AnnData object
mic = anndata.AnnData(X1.T,dtype = np.float64)
met = anndata.AnnData(X2.T,dtype = np.float64)
mic.obs['sampletype'] = label

#import scipy.io as scio
# A1 = A1.cpu()
# A2 = A2.cpu()
# scio.savemat('opt_A.mat', {'opt_A1': A1, 'opt_A2': A2})

# get the taxa and metabolites
# taxa = np.array([ val[0]  for val in mat['taxa'][:,0]])
# mets = np.array([ val[0]  for val in mat['mets'][:,0]])

ot_sim = scipy.io.loadmat('opt_A.mat')
A1 = ot_sim['opt_A1']
A2 = ot_sim['opt_A2']

test_model = model.NMFGOT(mic, met, A1, A2)
test_model.run()

# Cluster and Visualization
# use S  to do clustering with louvain
clu = test_model.cluster()
from sklearn import metrics

AC = metrics.accuracy_score(label, clu, normalize=True)
ARI = metrics.adjusted_rand_score(label, clu)
NMI = metrics.normalized_mutual_info_score(label, clu)
print('AC=: {:.4f}, NMI=: {:.4f}, ARI=: {:.4f}'.format(AC, NMI, ARI))

# Umap with clustering label
test_model.visualize(clu, min_dist = 0.2, n_neighbors=5)

# Umap with true label
test_model.visualize(label,min_dist = 0.2, n_neighbors=5)


