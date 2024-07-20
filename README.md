# NMFGOT-py
NMFGOT: A Multi-view Learning Framework for the Microbiome and Metabolome Integrative Analysis with Optimal Transport Plan。

This is the Python implementation of the JSNMF algorithm. Note that this implementation can employ GPU to speed up the code. 

## 1. Installation
You can use the following command to install JSNMF:
```
pip install JSNMF-py
```

## 2. Usage
The default number of maximum epochs to run, i.e. the `max_epochs` parameter, is set as 200. So it is quite simple to initialize a NMFGOT model with the following code:
```
from JSNMF.model import JSNMF
test_model = JSNMF(rna,atac)
```
After initializing, run the model is also quite easy: 
```
test_model.run()
```
The result is saved in `test_model.result`, which is a dict, and the major output of JSNMF, the complete graph S, can be get with easy access:
```
S = test_model.result['S']
```
`JSNMF` class also has other methods, you can use the `help` or `?` command for more details explanations of the methods.


## 3. Tutorials
Please refer to [here](https://github.com/chonghua-1983/NMFGOT/tree/main/Tutorials/demo1.py) for a simple illustration of the use of NMFGOT model, with clustering and visualization results shown. 
