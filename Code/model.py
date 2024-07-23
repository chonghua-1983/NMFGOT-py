from tkinter import W
import torch  
import numpy as np
import scanpy as sc
# import scvi
from anndata import AnnData
import time 
from scipy.sparse import isspmatrix
import bct
import umap,random
import matplotlib.pyplot as plt
import os
import pandas as pd

import sys
sys.path.append('C:/Users/PC/Desktop/NMFGOT/NMFGOT-py/code')
from utils import *
import ot

torch.set_default_dtype(torch.float64)

class NMFGOT:
    '''
    NMFGOT model. 

    Parameters
    ----------
    X1         : AnnData object that contains the data from microbial abundance profile data, true sample type should be
                 stored in RNA.obs['celltype']
                 row: taxa, column: sample
    X2         : AnnData object that contains the data from metabolome, or other modality
                 row: metabolite, column: sample
    A1,A2      : optimal transport similarity matrix for microbiome and metabolome data
    max_epochs : Number of max epochs to run, optinal, default is 200 epochs

   '''

    def __init__(self, micro: AnnData, metab: AnnData, A1, A2, max_epochs: int = 200):
        
        self.mic = micro
        self.met = metab
        self.label = micro.obs['sampletype'].values
        self.max_epochs = max_epochs
        self.num_c  = num_c = len(np.unique(self.label))
        self.embedding = None
        self.A1 = A1  #
        self.A2 = A2  #
        
        try:
            if  isspmatrix(micro.X):
                self.X1 = torch.from_numpy(micro.X.toarray()).T.cuda()
                self.X2 = torch.from_numpy(metab.X.toarray()).T.cuda()
                self.A1 = torch.from_numpy(A1.toarray()).T.cuda()  #
                self.A2 = torch.from_numpy(A2.toarray()).T.cuda()  #
            else:
                self.X1 = torch.from_numpy(micro.X).T.cuda()
                self.X2 = torch.from_numpy(metab.X).T.cuda()
                self.A1 = torch.from_numpy(A1).T.cuda()   #
                self.A2 = torch.from_numpy(A2).T.cuda()   #

        except:
            if isspmatrix(micro.X):
                self.X1 = torch.from_numpy(micro.X.toarray()).T
                self.X2 = torch.from_numpy(metab.X.toarray()).T
                self.A1 = torch.from_numpy(A1.toarray()).T  #
                self.A2 = torch.from_numpy(A2.toarray()).T  #
            else:
                self.X1 = torch.from_numpy(micro.X).T
                self.X2 = torch.from_numpy(metab.X).T
                self.A1 = torch.from_numpy(A1).T   #
                self.A2 = torch.from_numpy(A2).T   #


    def parameter_selection(self):
        '''
    Calculates the initilization values for the model. 

    Returns
        ----------
        alpha   : init hyperparameter alpha
        beta    : init hyperparameter beta
        Inits   : dict stroing the init values
        opt_dist: dict stroing the opt similarities

       '''
    
        Inits = {}
        opt_dist = {}
        num_c = self.num_c
        W1, H1 =  nndsvd(self.X1,num_c)
        print('nnsvd W1 done')
        W2, H2 = nndsvd(self.X2,num_c)
        print('nnsvd W2 done')
        
        Inits['W1'] = W1
        Inits['W2'] = W2
        Inits['H1'] = H1
        Inits['H2'] = H2
        
        D1 = dist2(self.X1.T,self.X1.T)
        print('D1 done')
        S1 = affinityMatrix(D1,20) # sample by sample
        print('S1 done')
        D2 = dist2(self.X2.T,self.X2.T)
        print('D2 done')
        S2 = affinityMatrix(D2,20) #  sample by sample
        print('S2 done')
        
        Inits['A1'] = S1
        Inits['D1'] = torch.diag(S1.sum(0))
        Inits['L1'] = Inits['D1'] - Inits['A1']
        
        Inits['A2'] = S2
        Inits['D2'] = torch.diag(S2.sum(0))
        Inits['L2'] = Inits['D2'] - Inits['A2']
        
        #opt_A1 = ot_mi(self.X1, 10)
        #opt_A1 = ot_mi2(self.X1)
        opt_A1 = self.A1
        print('opt1 done')
        opt_A1 = torch.Tensor(opt_A1)
        opt_D1 = torch.diag(opt_A1.sum(0))
        
        #opt_A2 = ot_mi(self.X2, 10)
        #opt_A2 = ot_mi2(self.X2)
        opt_A2 = self.A2
        print('opt2 done')
        opt_A2 = torch.Tensor(opt_A2)
        opt_D2 = torch.diag(opt_A2.sum(0))
        
        opt_dist['A1'] = opt_A1
        opt_dist['D1'] = opt_D1
        opt_dist['A2'] = opt_A2
        opt_dist['D2'] = opt_D2
        
        
        print('SNF starts')
        W = SNF(S1,S2,K = 20)
        print('SNF done')
        Inits['S'] = W
        
        H1tH1 = H1.T @ H1
        H2tH2 = H2.T @ H2
        n = W.shape[0]                                            
        allones = torch.ones(n,1).cuda()
        
        err1 = torch.square(torch.norm(self.X1 - W1@H1))
        err2 = torch.square(torch.norm(self.X2 - W2@H2))
        err3 = (torch.square(torch.norm(Inits['S'] - H1tH1)) +  torch.square(torch.norm(Inits['S'] - H2tH2)))
        err4 = (H1@Inits['L1']@H1.T + H2@Inits['L2']@H2.T).trace()
        err5 = torch.square(torch.norm(Inits['S']@allones - allones))
        
        alpha = (err1 + err2) / err3
        gamma = torch.abs((err1 + err2) / err4)
        phi = (err1 + err2) / err5
        
        alpha = alpha / 5
        gamma = gamma / 1000
        phi = phi/1000

        return alpha,gamma,phi,Inits,opt_dist


    def gcnmf(self):
        '''
        Main function to run the model. 

        Returns
        ----------
        W1,W2,H1,H2    : resulting values of init  after running the model
        S              : resulting complete graph

        use_epochs     : used epochs to converge

        objs		   : value of objective during each iteration

        '''
        Maxiter = self.max_epochs
        X1 = self.X1
        X2 = self.X2        
        
        alpha = self.alpha
        gamma = self.gamma
        phi = self.phi
        W1 = self.Inits['W1']
        W2 = self.Inits['W2']
        H1 = self.Inits['H1'].T
        H2 = self.Inits['H2'].T
        A1 = self.opt_dist['A1']
        D1 = self.opt_dist['D1']
        S = self.Inits['S']
        S = S / S.sum(dim =0,keepdim = True)
        A2 = self.opt_dist['A2']
        D2 = self.opt_dist['D2']
        n = S.size(0)

        try:
            beta = torch.tensor(0.1).cuda()
            gamma = torch.tensor(gamma).cuda()
            obj_old = torch.tensor(1.).cuda()
            objs = torch.zeros((Maxiter,1)).cuda()
            D1 = torch.tensor(D1).cuda()
            A1 = torch.tensor(A1).cuda()
            D2 = torch.tensor(D2).cuda()
            A2 = torch.tensor(A2).cuda()

            
        except:
            beta = torch.tensor(0.1)
            gamma = torch.tensor(gamma)
            obj_old = torch.tensor(1.)
            objs = torch.zeros((Maxiter,1))
            D1 = torch.tensor(D1)
            A1 = torch.tensor(A1)
            D2 = torch.tensor(D2)
            A2 = torch.tensor(A2)


        for i in range(Maxiter):
            # update W1 W2 using bpp algorithm
            # W1, _ = nnlsm_blockpivot(H1, X1.T, False, W1.T) 
            # W2, _ = nnlsm_blockpivot(H2, X2.T, False, W2.T) 
            # W1 = W1.T
            # W2 = W2.T
            
            # updating w1 w2 via multiplication rule
            H1tH1 = H1.T @ H1
            H2tH2 = H2.T @ H2
            tmp1 = W1@H1tH1
            tmp1[tmp1 < 1e-10] = 1e-10
            W1 = W1 * (X1@H1) /tmp1
            tmp2 = W2@H2tH2
            tmp2[tmp2 < 1e-10] = 1e-10
            W2 = W2 * (X2@H2) / tmp2
            
            
            # update H1, H2
            W1tW1 = W1.T @ W1
            W2tW2 = W2.T @ W2
            H1tH1 = H1.T @ H1
            H2tH2 = H2.T @ H2
            tmp_deno_1 = gamma * D1 @ H1 + 2*alpha * H1@H1tH1 + H1@W1tW1
            tmp_deno_1[tmp_deno_1 < 1e-10] = 1e-10
            tmp_nume_1 = 2*alpha * S.T@H1 + X1.T @ W1 + gamma * A1@H1 + beta* (H2@H2.T) @ H1
            H1 = H1 * (tmp_nume_1 / tmp_deno_1)
            
            tmp_deno_2 = gamma * D2 @ H2 + 2*alpha * H2@H2tH2 + H2@W2tW2
            tmp_deno_2[tmp_deno_2 < 1e-10] = 1e-10
            tmp_nume_2 = 2*alpha * S.T@H2 + X2.T @ W2 + gamma * A2@H2 + beta * (H1@H1.T) @ H2
            H2 = H2 * (tmp_nume_2 / tmp_deno_2)
            
            # update S
            H1tH1 = H1@H1.T 
            H2tH2 = H2@H2.T
            Q = H1tH1 + H2tH2
            tmp = 2*alpha*S + 2*phi * S.sum(dim = 0, keepdim =True)
            tmp[tmp<1e-10] = 1e-10
            try: 
                tmp_ones = torch.ones((n,n)).cuda()
            except:
                tmp_ones = torch.ones((n,n))
            S = S * ((alpha * Q + 2*phi * tmp_ones) / tmp)
                        
            
            #if stop_rule == 2:
            obj = compute_obj(X1,X2,S,W1,W2,H1,H2,D1,A1,D2,A2,alpha,beta,gamma,phi)
            objs[i,0] = obj
            error = torch.abs(obj_old - obj) / obj_old
            if  (error < 1e-6 and i > 0) or i == Maxiter - 1:
                print('number of epoch:', i+1)
                print('obj:',obj)
                print('converged!')
                break
            
            obj_old = obj
            
            print('number of epoch:', i+1)
            print('obj:',obj)

        S = (S + S.T)/2
        use_epochs = i+1
            
        return W1,W2,H1,H2,S,use_epochs,objs

    def run(self):
        '''
        Run the NMFGOT model. Init time and main function time are recorded.

        Returns
        ----------
        result  :  dict storing some information during the model running

        '''

        start = time.time()
        alpha, gamma, phi, Inits, opt_dist = self.parameter_selection()
        end = time.time()
        self.init_t = end - start
        self.alpha = alpha
        self.gamma = gamma
        self.phi = phi
        self.Inits = Inits
        self.opt_dist = opt_dist
        
        print('Init done')

        start = time.time()
        W1,W2,H1,H2,S,used_epoch,objs  = self.gcnmf()
        end = time.time()
        self.run_t = end - start
        self.used_epoch = used_epoch

        result = dict(W1 = W1, W2 = W2,
                    H1 = H1, H2 = H2,
                    S = S, used_epoch = used_epoch,
                    objs = objs,
                    init_t = self.init_t,
                    run_t  = self.run_t)
        self.result = result
       #return result

    def cluster(self, K = 20, step = 0.01, start = 2.3, upper = 4 ,seed = 3):
        '''
		Use louvain to cluster the cells based on the complete graph S, note that
		the function tries to find  a partition that has the same number of clusters
		as the true labels, the resolution parameter of louvain is found using binary search

		Parameters
		----------
		K      : (0, N) int, parameter of Wtrim
			     Number of neighbors to retain
		step   :  the step of binary search to find the partition, default 0.01
		start  :  start searching point of binary search
		upper  :  the upper bound of the reolution paramter to be searched
		seed   : seed parameter for louvain algorithm

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		Note that sometimes exact number of clusters as true labels may not be found, paramters
		need to be adjusted then, like step, seed and upper

        '''
        S = self.result['S']
        A = Wtrim(S, K = 20)
        A = A.cpu()
        A = A.numpy()
        
        num_c = self.num_c
        
        tmp_gamma = start
        #use louvain to cluster based on A
        clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma, seed = seed)
        tmp_c = len(np.unique(clusters))
        tmp_clu = clusters
        
        res_clu = None
        
        
        # use binary search to find the corret gamma parameter
        while True:
            if tmp_c == num_c:
                res_clu = tmp_clu
                break

            if tmp_c < num_c:
                tmp_gamma = tmp_gamma + step
            else:
                tmp_gamma = tmp_gamma - step

            if tmp_gamma < 0 or tmp_gamma > upper:
                break
            tmp_gamma = round(tmp_gamma,2)
            #print(tmp_res)
            clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma,seed = seed)
            tmp_c = len(np.unique(clusters))
            tmp_clu = clusters

        return res_clu

    def visualize(self, label, tag = False, **kwarg):
        
        '''
		Visualize based on the complete graph S using Umap

		Parameters
		----------
		label     : array, true or clustered (louvain result) labels for each cell
		tag		  : if recalculte umap embedding
		**kwarg   : kwarg for the umap 	    

		Returns
		-------
		res_clu : resulting cluster labels for each cell

        '''


        # transfer S to distance matrix first
        S = self.result['S']
        S = S.cpu()
        data = 1-S
        data = data-np.diag(data.diagonal())
        reducer = umap.UMAP(**kwarg)
        
        # avoid recompute umap embedding
        if self.embedding is None:
            #min_dist = 0.68, n_neighbors=12
            embedding = reducer.fit_transform(data)
            self.embedding = embedding

        # recaculate embedding if needed
        if tag is True:
            embedding = reducer.fit_transform(data)
            self.embedding = embedding


        # plt.figure(figsize=(3, 1.5), dpi=300)
        # visualize
        for i in range(1,label.max() + 1):
            ind = label == i
            rgb = (random.random(), random.random() /2, random.random()/2)
            plt.scatter(self.embedding[ind, 0],self.embedding[ind, 1], s = 1.5, label = i,color = rgb)
        plt.legend(ncol=2,bbox_to_anchor=(1, 1.2))
        plt.show()





