# !/usr/bin/env python
# -*- coding: utf8 -*-

import scipy.special
import numpy as np
import networkx as nx
from tabulate import tabulate

from sklearn.metrics import f1_score,adjusted_rand_score,normalized_mutual_info_score

# Get precicion
def getPrecision(mat, k, s, total):
	sum_k = 0
	for i in range(k):
		max_s = 0
		for j in range(s):
			if mat[i][j] > max_s:
				max_s = mat[i][j]
		sum_k += max_s
	return sum_k/total

# Get recall
def getRecall(mat, k, s, total, unclassified):
	sum_s = 0
	for i in range(s):
		max_k = 0
		for j in range(k):
			if mat[j][i] > max_k:
				max_k = mat[j][i]
		sum_s += max_k
	return sum_s/(total+unclassified)

# Get ARI
def getARI(mat, k, s, N):
	t1 = 0	
	for i in range(k):
		sum_k = 0
		for j in range(s):
			sum_k += mat[i][j]
		t1 += scipy.special.binom(sum_k, 2)
	t2 = 0
	for i in range(s):
		sum_s = 0
		for j in range(k):
			sum_s += mat[j][i]
		t2 += scipy.special.binom(sum_s, 2)
	t3 = t1*t2/scipy.special.binom(N, 2)
	t = 0
	for i in range(k):
		for j in range(s):
			t += scipy.special.binom(mat[i][j], 2)
	ari = (t-t3)/((t1+t2)/2-t3)
	return ari

 # Get F1-score
def getF1(prec, recall):
	if prec == 0.0 or recall == 0.0:
		return 0.0
	else:
		return 2*prec*recall/(prec+recall)


def validate_performance(true_labels, pred_labels):
	ReMap = {j:i for i,j in enumerate(np.unique([v for k,v in pred_labels.items()]))}
	ReMap_true = {j:i for i,j in enumerate(np.unique([v for k,v in true_labels.items()]))}
	# print(len(ReMap))
	n_true_labels = len(np.unique([v for k,v in true_labels.items()]))
	n_pred_labels = len(np.unique([v for k,v in pred_labels.items()]))
	# print(np.unique([v for k,v in pred_labels.items()]))
	print(n_true_labels, n_pred_labels)
	ground_truth_count = len(true_labels)
	# print(n_true_labels, n_pred_labels)
	total_binned = 0
	bins_species = [[0 for x in range(n_true_labels)] for y in range(n_pred_labels)]
	for i in pred_labels:
		if i in true_labels:
			total_binned += 1
			bins_species[ReMap[pred_labels[i]]][ReMap_true[true_labels[i]]] += 1

	my_precision = getPrecision(bins_species, n_pred_labels, n_true_labels, total_binned)
	my_recall = getRecall(bins_species, n_pred_labels, n_true_labels, total_binned, (ground_truth_count-total_binned))
	my_ari = getARI(bins_species, n_pred_labels, n_true_labels, total_binned)
	my_f1 = getF1(my_precision, my_recall)

	# print("### Evaluation:")
	print("### Precision = %0.4f  Recall = %0.4f  F1 = %0.4f ARI = %0.4f" % (my_precision, my_recall, my_f1, my_ari))
	return my_precision, my_recall, my_ari, my_f1


from collections import defaultdict
def list_duplicates(seq):
	tally = defaultdict(list)
	for i,item in enumerate(seq):
		tally[item].append(i)
	# return ((key,locs) for key,locs in tally.items() if len(locs)>1)
	return (locs for key,locs in tally.items() if len(locs)>1)


def Clustering(embeds, true_labels, constraints, Gx, n_clusters=5):
	p_all, r_all, ari_all, f1_all = [],[],[],[]
	for t in range(3):
		from sklearn.cluster import KMeans
		estimator = KMeans(n_clusters=n_clusters)
		estimator.fit(embeds)
		pred_labels_ = estimator.labels_
		pred_labels = {i:j for i,j in enumerate(pred_labels_)}

		p, r, ari, f1 = validate_performance(true_labels, pred_labels)
		# print("### Precision = %0.4f Recall = %0.4f F1 = %0.4f ARI = %0.4f" %(p, r, f1, ari))
		p_all.append(p)
		r_all.append(r)
		ari_all.append(ari)
		f1_all.append(f1)
	avg_p,avg_r,avg_ari,avg_f1 = np.mean(p_all),np.mean(r_all),np.mean(ari_all),np.mean(f1_all)
	std_p,std_r,std_ari,std_f1 = np.std(p_all),np.std(r_all),np.std(ari_all),np.std(f1_all)
	
	print()
	print ("### Average (over trials): Precision = %0.4f(%0.4f)  Recall = %0.4f(%0.4f)  F1 = %0.4f(%0.4f) ARI = %0.4f(%0.4f)" 
		% (avg_p,std_p,avg_r,std_r,avg_f1,std_f1,avg_ari,std_ari))
	return pred_labels


def Cluster(embeds, true_labels, n_clusters=5):
	Mf1_all,mf1_all,nmi_all,ari_all = [],[],[],[]
	for t in range(5):
		from sklearn.cluster import KMeans
		estimator = KMeans(n_clusters=n_clusters)
		estimator.fit(embeds)
		pred_labels_ = estimator.labels_
		pred_labels = {i:j for i,j in enumerate(pred_labels_)}

		true_idx = [key for key,val in true_labels.items()]
		true_lbls = [val for key,val in true_labels.items()]
		pred_lbls = [pred_labels[idx] for idx in true_idx]
		# print(len(true_lbls), len(pred_lbls))

		Mf1 = f1_score(true_lbls, pred_lbls, average='macro')
		Mf1_all.append(Mf1)
		mf1 = f1_score(true_lbls, pred_lbls, average='micro')
		mf1_all.append(mf1)
		ari = adjusted_rand_score(true_lbls, pred_lbls)
		ari_all.append(ari)
		nmi = normalized_mutual_info_score(true_lbls, pred_lbls)
		nmi_all.append(nmi)

	avg_Mf1,avg_mf1,avg_ari,avg_nmi = np.mean(Mf1_all),np.mean(mf1_all),np.mean(ari_all),np.mean(nmi_all)
	std_Mf1,std_mf1,std_ari,std_nmi = np.std(Mf1_all),np.std(mf1_all),np.std(ari_all),np.std(nmi_all)

	print()
	print ("### Average (over trials): macro-F1 = %0.4f(%0.4f), micro-F1 = %0.4f(%0.4f), ARI = %0.4f(%0.4f), NMI = %0.4f(%0.4f)" 
		% (avg_Mf1,std_Mf1,avg_mf1,std_mf1,avg_ari,std_ari,avg_nmi,std_nmi))


def validate_ARI_NMI(true_labels, pred_labels):
	# pred_labels = {i:j for i,j in enumerate(pred_labels_)}
	true_idx = [key for key,val in true_labels.items()]
	true_lbls = [val for key,val in true_labels.items()]
	pred_lbls = [pred_labels[idx] for idx in true_idx]

	# Mf1 = f1_score(true_lbls, pred_lbls, average='macro')
	# mf1 = f1_score(true_lbls, pred_lbls, average='micro')
	ari = adjusted_rand_score(true_lbls, pred_lbls)
	nmi = normalized_mutual_info_score(true_lbls, pred_lbls)
	# print ("### macro-F1 = %0.4f, micro-F1 = %0.4f, ARI = %0.4f, NMI = %0.4f"  % (Mf1,mf1,ari,nmi))
	return ari, nmi


def evaluate_performance(true_labels, pred_labels):
	ari, nmi = validate_ARI_NMI(true_labels, pred_labels)
	p, r, _, f1 = validate_performance(true_labels, pred_labels)

	return p,r,f1,ari,nmi


