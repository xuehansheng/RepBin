# RepBin

This is the source code for xxx paper ["RepBin: Constraint-based Graph Representation Learning for Metagenomic Binning"](xxx).

## Overview
![image](RepBin.jpg)
The RepBin mainly contains two components, (i) graph representation learning that preserves both homophily relations and heterophily constraints (ii) constraint-based graph clustering method that addresses the problems of skewed cluster size distribution.

## Requirement
```
Python 3.6
networkx == 1.11
numpy == 1.18
sklearn == 0.22
pytorch == 1.3.1
```

## Example Usage
To reproduce the experiments on Sim-5G dataset, simply run:
```
python3 main.py --dataset Sim-5G --n_clusters 5 --patience 20
```

## Reference
All readers are welcome to star/fork this repository and use it to reproduce our experiments or train your own data. Please kindly cite our paper:
```bibtex
@inproceedings{Xue2022RepBin,
  title     = {RepBin: Constraint-based Graph Representation Learning for Metagenomic Binning},
  author    = {Xue, Hansheng and Mallawaarachchi, Vijini and Zhang, Yujia and Rajan, Vaibhav and Lin, Yu},
  booktitle = {AAAI},
  year      = {2022}
}
```