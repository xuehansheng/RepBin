# !/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import argparse
import networkx as nx
import time
from loader import *
from utils import sparse_to_tuple
from models import Learning
from models import Graph_Diffusion_Convolution
from evaluation import *
from utils import sample_constraints
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser('Unsup learning model.')
parser.add_argument('--dataset', type=str, default='Sim-5G', help='Dataset string.')
parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters.')
parser.add_argument('--alpha', type=float, default=0.01, help='Teleport probability in graph diffusion convolution operators.')
parser.add_argument('--eps', type=float, default=0.0001, help='Threshold epsilon to sparsify in graph diffusion convolution operators.')

parser.add_argument('--lr', type=float, default=0.005, help='Number of learning rate.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs.')
parser.add_argument('--hid_dim', type=int, default=32, help='Dimension of hidden2.')
parser.add_argument('--batch_size', type=int, default=1, help='Number of batch size.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for L2 loss on embedding matrix.')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping (# of epochs).')
parser.add_argument('--lamb', type=float, default=0.2, help='Weight to balance constraints.')

args = parser.parse_args()

print('----------args----------')
for k in list(vars(args).keys()):
	print('%s: %s' % (k, vars(args)[k]))
print('----------args----------\n')


def main():
	assembly_graph, constraints, ground_truth, Gx = load_data(args.dataset)
	print(assembly_graph.shape, len(constraints), len(ground_truth))
	print(len(list(set([val for key,val in ground_truth.items()]))))
	triplets = sample_constraints(constraints, ground_truth)

	adj = Graph_Diffusion_Convolution(assembly_graph, args.alpha, args.eps)
	adj = sparse_to_tuple(adj)
	feats = assembly_graph.todense()
	model = Learning(feats.shape[0], args.hid_dim, args.n_clusters, args)
	pred_labels = model.train(adj, feats, Gx, triplets, constraints, ground_truth)
	
	print("\nEvaluation:")
	p,r,f1,ari,nmi = evaluate_performance(ground_truth, pred_labels)
	print ("### Precision = %0.4f, Recall = %0.4f, F1 = %0.4f, ARI = %0.4f, NMI = %0.4f"  % (p,r,f1,ari,nmi))


if __name__ == '__main__':
	main()
	print()
