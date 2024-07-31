# NMFGOT-py
NMFGOT: A Multi-view Learning Framework for the Microbiome and Metabolome Integrative Analysis with Optimal Transport Plan。

This is the Python implementation of the NMFGOT algorithm. Note that this implementation can employ GPU to speed up the code. 

## 1. Enviroment and dependent packages
NMFGOT was developed in conda enviroment (python 3.11). The dependent packages includes:
Numpy, bct, umap, ot, torch, scipy.

## 2. Usage
In the tutorial, we provided a demo to demonstrate the implementation of NMFGOT. It is quite simple to initialize a NMFGOT model with the following code:
```
sys.path.append('C:/Users/PC/Desktop/NMFGOT/NMFGOT-py/')
from Code import model
from Code import utils

from anndata import AnnData
from utils import ot_mi
import scipy.sparse as sp
import ot

# compute the optimal transport distance
A1 = ot_mi(X1, 10)
A2 = ot_mi(X2, 10)

test_model = model.NMFGOT(microbiome, metabolite, opt_A1, opt_A2)
```
After initializing, run the model is also quite easy: 
```
test_model.run()
```
The result is saved in `test_model.result`, which is a dict, and the major output of NMFGOT, the complete graph S, can be get with easy access:
```
S = test_model.result['S']
```
Noting that the current version of NMFGOT may have some issues, such as running slowly to compute the optimal transport distance. We will solve these issues and update the scripts soon. The matlab version of NMFGOT performs more stable and fast. If you have a problem in using it, please contact [me](chonghua_1983@yeah.net).

## 3. Tutorials
Please refer to [here](https://github.com/chonghua-1983/NMFGOT-py/tree/main/Tutorials) for a simple illustration of the use of NMFGOT model, with clustering and visualization results shown. 
