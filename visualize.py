# !/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import igraph as iG
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
from collections import Counter


def plot(edges, constraints, _labels_dict):
	nodes = list(set([val for line in edges for val in line]))
	n_nodes = len(nodes)
	n_labels = len(list(set([val for key,val in classes.items()])))
	print(n_nodes, n_labels)

	ids = [val for val in range(n_nodes)]
	nodes2id = dict(zip(nodes, ids))
	edges = [[nodes2id[line[0]],nodes2id[line[1]]] for line in edges if line[0] in nodes and line[1] in nodes]
	constraints = [[nodes2id[line[0]],nodes2id[line[1]]] for line in constraints if line[0] in nodes and line[1] in nodes]
	labels_dict = {}
	for key,val in _labels_dict.items():
		tmp = {nodes2id[k]:v for k,v in val.items() if k in nodes}
		labels_dict[key] = tmp
	nodes = [val for val in range(n_nodes)]
	print(len(edges),len(constraints))
	chosed_colors_dict = dict()

	chosed_colors = ['#F5C0C0', '#CD113B', '#01937C', '#2978B5', '#E48257', '#825959', '#F7A440', '#00C1D4', '#907FA4', '#FF577F', '#7579E7', #'#EBEBEB',
	'#7B6079','#BDC3C7','#FF5733','#2E86C1','#239B56','#76448A', '#E67E22','#DAF7A6','#F7DC6F','#85C1E9','#D2B4DE',
	'#F5B7B1','#FFC0CB','#DB7093','#9400D3','#6A5ACD','#4169E1','#4682B4','#708090','#77ACF1', '#F5A962','#346751',
	'#005F99','#D99879','#480032','#008B8B','#2E8B57','#D8E3E7','#CE6262','#214252','#F4ABC4','#595B83','#9AD3BC',
	'#A4B787','#FCF876','#8BCDCD','#1F6F8B','#BEDBBB','#F1D4D4','#0E918C','#B8DE6F','#B2DEEC','#A3D8F4','#EBD4D4','#EBEBEB']
	chosed_colors_dict['RepBin'] = chosed_colors
	chosed_colors_dict['GroundTruth'] = chosed_colors

	# load data
	G = iG.Graph()
	G.add_vertices(nodes)
	edges_pos = [(val[0], val[1]) for val in edges]
	G.add_edges(edges_pos)

	Gi = iG.Graph()
	Gi.add_vertices(nodes)
	edges_neg = [(val[0], val[1]) for val in constraints]
	Gi.add_edges(edges_neg)
	edges_pos = [(val[0], val[1]) for val in edges]
	Gi.add_edges(edges_pos)

	edge_color2 = ['#000000' for i in range(len(edges_pos))]
	edge_color1 = ['#D8E3E7' for i in range(len(edges_neg))]
	edge_color = edge_color1 + edge_color2

	layout = G.layout_fruchterman_reingold()

	# G.vs['label'] = [str(val) for val in range(n_nodes)]
	visual_style = {}
	visual_style["bbox"] = (1000, 1000)
	visual_style["margin"] = 50
	visual_style["vertex_size"] = 12
	visual_style["vertex_label_size"] = 0
	visual_style["edge_curved"] = False
	visual_style["layout"] = layout
	# visual_style["layout"] = G.layout(layout='auto')
	visual_style['edge_color'] = edge_color
	visual_style['label'] = [str(val) for val in range(n_nodes)]


	for key,val in labels_dict.items():
		methd,labels = key,val
		lbled_nodes = [key for key,val in labels.items()]
		chosed_colors = chosed_colors_dict[key]
		# figure settings
		node_colours = []
		for i in range(n_nodes):
			if i in lbled_nodes:
				chosed_color = chosed_colors[int(labels[i])]
				node_colours.append(chosed_color)
			else:
				chosed_color = chosed_colors[-1]
				node_colours.append(chosed_color)

		Gi.vs["color"] = node_colours
		# plot
		iG.plot(Gi, 'figures/Sim10G_'+methd+'.png', **visual_style)


def reLabel(classes, groundTruth):
	gtNodes = [key for key,val in groundTruth.items()]
	labels = [val for key,val in classes.items() if key in gtNodes]
	cnt = Counter(labels)
	cnt = sorted(cnt.items(), key=lambda item:item[1], reverse=True)
	cnt_lbl = [key for key,val in cnt]
	map_lbl = {j:i for i,j in enumerate(cnt_lbl)}
	classes = {key:map_lbl[val] for key,val in classes.items()}
	return classes


if __name__ == '__main__':
	filename = 'Sim-10G'
	labels_dict = {}

	edges = np.load('data/Sim10G_Edges.npy', allow_pickle=True)
	pairwises = np.load('data/Sim10G_Pairwises.npy', allow_pickle=True)
	classes = np.load('data/Sim10G_GroundTruth.npy', allow_pickle=True).item()
	groundTruth = classes
	classes = reLabel(classes, groundTruth)
	labels_dict['GroundTruth'] = classes

	print('---------- visualization ----------')
	plot(edges, pairwises, labels_dict)

