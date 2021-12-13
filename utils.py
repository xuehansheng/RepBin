# !/usr/bin/env python
# -*- coding: utf8 -*-

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import torch.nn as nn
from collections import defaultdict
from itertools import combinations

def sparse_to_tuple(sparse_mx):
	"""Convert sparse matrix to tuple representation."""
	def to_tuple(mx):
		if not sp.isspmatrix_coo(mx):
			mx = mx.tocoo()
		coords = np.vstack((mx.row, mx.col)).transpose()
		values = mx.data
		shape = mx.shape
		return coords, values, shape

	if isinstance(sparse_mx, list):
		for i in range(len(sparse_mx)):
			sparse_mx[i] = to_tuple(sparse_mx[i])
	else:
		sparse_mx = to_tuple(sparse_mx)
	return sparse_mx

def remove_self_loops(edge_index, edge_attr = None):
	"""Removes every self-loop in the graph"""
	mask = edge_index[0] != edge_index[1]
	edge_index = edge_index[:, mask]
	if edge_attr is None:
		return edge_index, None
	else:
		return edge_index, edge_attr[mask]

def preprocess_graph(adj):
	"""adjacency matrix normalization"""
	adj = sp.coo_matrix(adj)
	adj_ = adj + sp.eye(adj.shape[0])
	rowsum = np.array(adj_.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	# return sparse_to_tuple(adj_normalized)
	return adj_normalized

def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def sample_constraints(cons, ground_truth):
	all_pairs = []
	all_pairs_cnt = []
	for line in cons:
		combs = list(combinations(line, 2))
		for pair in combs:
			# if ground_truth[pair[0]] != ground_truth[pair[1]]:
			all_pairs.append([pair[0], pair[1]])
			all_pairs.append([pair[1], pair[0]])
			all_pairs_cnt.append(str(pair[0])+'_'+str(pair[1]))
			all_pairs_cnt.append(str(pair[1])+'_'+str(pair[0]))
	# print("### Number of constraint pairs(sym):", len(all_pairs))
	from collections import Counter
	all_pairs_cnt_dict = Counter(all_pairs_cnt)

	all_pairs_new = []
	for key, val in all_pairs_cnt_dict.items():
		if val > 0:
			node_a = int(key.split('_')[0])
			node_b = int(key.split('_')[1])
			# if adj[node_a,node_b] == 0.0 and adj[node_b,node_a] == 0.0:
			all_pairs_new.append([node_a, node_b])
	# return np.array(all_pairs_new)
	return np.array(all_pairs)
