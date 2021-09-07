# !/usr/bin/env python
# -*- coding: utf8 -*-

import re
import numpy as np
import networkx as nx
import scipy.sparse as sp
from collections import Counter
from collections import defaultdict
from Bio import SeqIO
from igraph import *


class BidirectionalError(Exception):
	"""Must set a unique value in a BijectiveMap."""
	def __init__(self, value):
		self.value = value
		msg = 'The value "{}" is already in the mapping.'
		super().__init__(msg.format(value))

class BidirectionalMap(dict):
	"""Invertible map."""

	def __init__(self, inverse=None):
		if inverse is None:
			inverse = self.__class__(inverse=self)
		self.inverse = inverse

	def __setitem__(self, key, value):
		if value in self.inverse:
			raise BidirectionalError(value)

		self.inverse._set_item(value, key)
		self._set_item(key, value)

	def __delitem__(self, key):
		self.inverse._del_item(self[key])
		self._del_item(key)

	def _del_item(self, key):
		super().__delitem__(key)

	def _set_item(self, key, value):
		super().__setitem__(key, value)

def remove_duplicates(constraints):
	for line in constraints:
		line.sort()
	removed_cons = []
	for line in constraints:
		if line not in removed_cons:
			removed_cons.append(line)
		else:
			continue
	print("### Original constraints: {:d}, after removing: {:d}".format(len(constraints),len(removed_cons)))
	return removed_cons


def load_contigs_fasta_markers(file_path, contigs_map):
	raw_data = np.loadtxt(file_path, delimiter=': ', dtype=str)[:,1]
	raw_data = np.array([line.split(', ') for line in raw_data])
	# print(raw_data) #[['NODE_49' 'NODE_5' 'NODE_74' 'NODE_94' 'NODE_67']]
	data = []
	contigs_map_rev = contigs_map.inverse
	for line in raw_data:
		data.append([contigs_map_rev[int(val.split('_')[1])] for val in line])
	# print(data) #[[48, 4, 73, 93, 66]]
	matrix_cons = np.zeros((len(contigs_map_rev),len(contigs_map_rev)))
	for line in data:
		for i in range(len(line)):
			for j in range(i+1, len(line)):
				matrix_cons[line[i],line[j]] = 1.0
				matrix_cons[line[j],line[i]] = 1.0

	return data, sp.csr_matrix(matrix_cons)


def load_contigs(contig_paths):
	paths = {}
	segment_contigs = {}
	node_count = 0
	my_map = BidirectionalMap()

	current_contig_num = ""
	with open(contig_paths) as file:
		name = file.readline()
		path = file.readline()

		while name != "" and path != "":
			while ";" in path:
				path = path[:-2]+","+file.readline()

			start = 'NODE_'
			end = '_length_'
			contig_num = str(int(re.search('%s(.*)%s' % (start, end), name).group(1)))
			segments = path.rstrip().split(",")

			if current_contig_num != contig_num:
				my_map[node_count] = int(contig_num)
				current_contig_num = contig_num
				node_count += 1

			if contig_num not in paths:
				paths[contig_num] = [segments[0], segments[-1]]

			for segment in segments:
				if segment not in segment_contigs:
					segment_contigs[segment] = set([contig_num])
				else:
					segment_contigs[segment].add(contig_num)

			name = file.readline()
			path = file.readline()

	contigs_map = my_map
	contigs_map_rev = my_map.inverse

	return node_count, contigs_map, paths, segment_contigs


def load_assembly_graph(assembly_graph_file, node_count, contigs_map, paths, segment_contigs):
	contigs_map_rev = contigs_map.inverse
	links = []
	links_map = defaultdict(set)
	with open(assembly_graph_file) as file:
		line = file.readline()
		while line != "":
			if "L" in line:
				strings = line.split("\t")
				f1, f2 = strings[1]+strings[2], strings[3]+strings[4]

				links_map[f1].add(f2)
				links_map[f2].add(f1)
				links.append(strings[1]+strings[2]+" "+strings[3]+strings[4])
			line = file.readline()

	edge_list = []
	for i in range(len(paths)):
		segments = paths[str(contigs_map[i])]
		start = segments[0]
		start_rev = ""
		if start.endswith("+"):
			start_rev = start[:-1]+"-"
		else:
			start_rev = start[:-1]+"+"

		end = segments[1]
		end_rev = ""
		if end.endswith("+"):
			end_rev = end[:-1]+"-"
		else:
			end_rev = end[:-1]+"+"

		new_links = []
		if start in links_map:
			new_links.extend(list(links_map[start]))
		if start_rev in links_map:
			new_links.extend(list(links_map[start_rev]))
		if end in links_map:
			new_links.extend(list(links_map[end]))
		if end_rev in links_map:
			new_links.extend(list(links_map[end_rev]))

		for new_link in new_links:
			if new_link in segment_contigs:
				for contig in segment_contigs[new_link]:
					if i!=contigs_map_rev[int(contig)]:
						edge_list.append([i,contigs_map_rev[int(contig)]])

	adj = np.zeros((node_count,node_count))
	for edge in edge_list:
		adj[edge[0],edge[1]] = 1.0
		adj[edge[1],edge[0]] = 1.0
	# print(len(edge_list))
	adj_sp = sp.csr_matrix(adj)
	return adj_sp, edge_list


def load_contigs_fasta(contigs_file, contigs_map):
	contigs_map_rev = contigs_map.inverse
	coverages = {}
	contig_lengths = {}
	seqs = []

	for index, record in enumerate(SeqIO.parse(contigs_file, "fasta")):
		start = 'NODE_'
		end = '_length_'
		contig_num = contigs_map_rev[int(re.search('%s(.*)%s' % (start, end), record.id).group(1))]
		
		start = '_cov_'
		end = ''
		coverage = int(float(re.search('%s(.*)%s' % (start, end), record.id).group(1)))
		
		start = '_length_'
		end = '_cov'
		length = int(re.search('%s(.*)%s' % (start, end), record.id).group(1))
		
		coverages[contig_num] = coverage
		contig_lengths[contig_num] = length
		seqs.append(str(record.seq))
	# print(seqs)


def load_assembly_graph_constraints(datatype):
	file_path = 'dataset/'+datatype+'/'
	contigs_num,contigs_map,paths,segment_contigs = load_contigs(file_path+'contigs.paths')
	# print("### Total number of contigs available: {:d}".format(contigs_num))
	assembly_graph, edges = load_assembly_graph(file_path+'assembly_graph_with_scaffolds.gfa',contigs_num,contigs_map,paths,segment_contigs)
	constraints, matrix_cons = load_contigs_fasta_markers(file_path+'contigs.fasta.markers', contigs_map)
	return assembly_graph, constraints, np.array(edges), contigs_map


def load_ground_truth(dataset, contigs_map):
	filepath = 'dataset/'+dataset+'/ground_truth.csv'
	raw_data = np.loadtxt(filepath, delimiter=',', dtype=str)
	raw_labels = np.unique(raw_data[:,1])

	idx_map_labels = {j:i for i,j in enumerate(raw_labels)}
	contigs_map_rev = contigs_map.inverse
	labels = dict()
	for line in raw_data:
		idx = contigs_map_rev[int(line[0].split('_')[1])]
		label = int(idx_map_labels[line[1]])
		labels[idx] = label
	return labels


def load_data(datatype):
	assembly_graph, constraints, edges, contigs_map = load_assembly_graph_constraints(datatype)
	ground_truth_dict = load_ground_truth(datatype, contigs_map)

	Gx = nx.Graph()
	Gx.add_nodes_from(list(range(assembly_graph.shape[0])))
	for edge in edges:
		Gx.add_edge(edge[0], edge[1])
		Gx.add_edge(edge[1], edge[0])

	assembly_graph, constraints, ground_truth_dict, Gx = filter_isolatedNodes(assembly_graph,constraints,edges,ground_truth_dict,Gx)
	return assembly_graph, constraints, ground_truth_dict, Gx


def filter_isolatedNodes(assembly_graph, constraints, edges, ground_truth_dict, Gx):
	cons = list(set([val for line in constraints for val in line]))
	isoNodes = []
	for idx in range(assembly_graph.shape[0]):
		# if Gx.degree(idx) == 0 and idx not in cons:
		if Gx.degree(idx) == 0:
			isoNodes.append(idx)

	# print(len(isoNodes)) #21570
	nodes = [idx for idx in range(assembly_graph.shape[0])]
	diffNodes = list(set(nodes).difference(set(isoNodes)))
	# print(len(diffNodes)) #20743
	nodeIDsMap = {j:i for i,j in enumerate(diffNodes)}
	# print(nodeIDsMap)
	newEdges = [[nodeIDsMap[line[0]], nodeIDsMap[line[1]]] for line in edges]
	# print(len(edges), len(newEdges))
	newConstraints = []
	for line in constraints:
		temp = []
		for val in line:
			if val in diffNodes:
				temp.append(nodeIDsMap[val])
		newConstraints.append(temp)
	# print(len(constraints), len(newConstraints))

	newGroundTruth = dict()
	for key,val in ground_truth_dict.items():
		if key in diffNodes:
			newGroundTruth[nodeIDsMap[key]] = val
	# print(len(ground_truth_dict), len(newGroundTruth))

	Gx = nx.Graph()
	Gx.add_nodes_from(diffNodes)
	for edge in newEdges:
		Gx.add_edge(edge[0], edge[1])
		Gx.add_edge(edge[1], edge[0])

	adj = np.zeros((len(diffNodes),len(diffNodes)))
	# adj = np.eye(len(diffNodes))
	for edge in newEdges:
		adj[edge[0],edge[1]] = 1.0
		adj[edge[1],edge[0]] = 1.0
	adj_sp = sp.csr_matrix(adj)

	return adj_sp, newConstraints, newGroundTruth, Gx




