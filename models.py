# !/usr/bin/env python
# -*- coding: utf8 -*-

import sys
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import glorot_init
from modules import GCN, AvgReadout, Discriminator
from collections import Counter
from evaluation import *
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
import gc
from collections import Counter
from evaluation import validate_performance,validate_ARI_NMI
import time

def Graph_Diffusion_Convolution(A: sp.csr_matrix, alpha: float, eps: float):
	N = A.shape[0]
	# Self-loops
	A_loop = sp.eye(N) + A
	# Symmetric transition matrix
	D_loop_vec = A_loop.sum(0).A1
	D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
	D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
	T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt
	# PPR-based diffusion
	S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)
	# Sparsify using threshold epsilon
	S_tilde = S.multiply(S >= eps)
	# S_tilde = S
	# Column-normalized transition matrix on graph S_tilde
	D_tilde_vec = S_tilde.sum(0).A1
	T_S = S_tilde / D_tilde_vec

	return sp.csr_matrix(T_S)


class Learning:
	def __init__(self, ipt_dim, hid_dim, opt_dim, args):
		self.args = args
		self.model = RepBin(ipt_dim, hid_dim, opt_dim, 'prelu')
		# if device != None:
		self.model = self.model.to(device)

	def train(self, adj, feats, Gx, samples, constraints, ground_truth):
		n_nodes = adj[2][0]
		# adj = torch.FloatTensor(adj)
		adj = torch.sparse.FloatTensor(torch.LongTensor(adj[0].T),torch.FloatTensor(adj[1]),torch.Size(adj[2])).to(device)
		feats = torch.FloatTensor(feats[np.newaxis]).to(device)
		# feats = torch.sparse.FloatTensor(torch.LongTensor(feats_[0].T),torch.FloatTensor(feats_[1]),torch.Size(feats_[2])).to(device)
		samples = torch.LongTensor(samples).to(device)
		# matrix = torch.FloatTensor(matrix_cons[np.newaxis])

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
		b_xent = nn.BCEWithLogitsLoss()
		cnt_wait, best, best_t = 0, 1e9, 0

		print("--------trainable parameters--------")
		print(self.model)
		for parameter in self.model.parameters():
			print(parameter.shape)
		print("--------trainable parameters--------")
		print()

		list_loss,list_losss,list_lossc = [],[],[]
		list_p,list_r,list_f1,list_ari = [],[],[],[]
		for epoch in range(self.args.epochs):
			self.model.train()
			optimizer.zero_grad()
			# corruption
			rnd_idx = np.random.permutation(n_nodes)
			shuf_fts = feats[:,rnd_idx,:].to(device)

			# labels
			lbl_1 = torch.ones(self.args.batch_size, n_nodes)
			lbl_2 = torch.zeros(self.args.batch_size, n_nodes)
			lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

			logits, hidds = self.model(feats, shuf_fts, adj, True, None, None)
			loss_s = b_xent(logits, lbl)
			loss_c = self.model.constraints_loss(hidds, samples)
			loss = self.args.lamb*loss_s + (1-self.args.lamb)*loss_c

			if epoch+1 == 1 or (epoch+1)%100 == 0:
				print("Epoch: {:d} loss={:.5f} loss_s={:.5f} loss_c={:.5f}".format(epoch+1,loss.item(),loss_s.item(),loss_c.item()))

			if loss < best:
				cnt_wait = 0
				best,best_t = loss,epoch
				torch.save(self.model.state_dict(), 'best_model.pkl')
			else:
				cnt_wait+=1
			if cnt_wait == self.args.patience:
				print('Early stopping!')
				break

			loss.backward()
			optimizer.step()
		print('Loading {}-th epoch.'.format(best_t+1))
		self.model.load_state_dict(torch.load('best_model.pkl'))
		self.model.eval()
		embeds, _ = self.model.embed(feats, adj, True)
		print(embeds.shape)
		print("### Optimization Finished!")
		print("### Run K-Means Clustering Algorithm:")
		true_labels = ground_truth
		print(Counter([v for k,v in true_labels.items()]))
		# np.save('Sharon_emb_12.npy', embeds.cpu().detach().numpy())
		# embs = np.load('Sharon_emb_12.npy')
		pred_labels = Clustering(embeds.cpu().detach().numpy(), true_labels, constraints, Gx, n_clusters=self.args.n_clusters)
		pred_labels = Cluster(embeds.cpu().detach().numpy(), true_labels, n_clusters=self.args.n_clusters)


		print("\n### Evaluate the performance of constraints.")
		lbls_idx = [k for k,v in true_labels.items()]
		# CNTs = Counter([val for line in constraints for val in line])
		# cons = list(set([val for line in constraints for val in line])) #80/265 131/413
		cons = [val for line in constraints for val in line if val in lbls_idx]
		print(len(cons)) #804
		# cons = list(set(cons))
		print(len(list(set(cons)))) #223
		print(Counter(cons))
		# cons = [k for k,v in Counter(cons).items() if v>3]
		cons = [k for k,v in Counter(cons).items() if v>3]
		# print(cons)
		print(len(cons))
		print(len(Counter([true_labels[c] for c in cons])),Counter([true_labels[c] for c in cons]))
		n_clusters = len(Counter([true_labels[c] for c in cons]))
		# n_clusters = self.args.n_clusters
		embs = embeds.cpu().detach().numpy()[cons]
		print(embs.shape)
		# embs = embs[cons]
		labels = list(set([true_labels[i] for i in cons]))
		# print(len(labels))
		labels_map = {idx:i for i,idx in enumerate(labels)}
		lbls = {i:true_labels[idx] for i,idx in enumerate(cons)}


		from sklearn.cluster import KMeans
		# kmeans = KMeans(n_clusters=n_clusters)
		kmeans = KMeans(n_clusters=self.args.n_clusters)
		y_pred = kmeans.fit_predict(embs)

		# # print(y_pred)
		print(len(Counter(y_pred)),Counter(y_pred))
		pred_labels = {i:j for i,j in enumerate(y_pred)}
		# pred_labels = {i:y_pred[i] for i,idx in enumerate(cons)}
		p, r, ari, f1 = validate_performance(lbls, pred_labels)
		print("Precision = %0.4f  Recall = %0.4f  F1 = %0.4f ARI = %0.4f" % (p, r, f1, ari))

		init_labels_dict = {cons[i]:y_pred[i] for i in range(len(y_pred))}


		### GCN-Label Propogation
		print()
		print("### Step 2: GCN-based Label Propogation model.")

		idxs = [idx for idx,val in init_labels_dict.items()]
		mask = np.array([True if idx in idxs else False for idx in range(n_nodes)])
		# init_labels = init_labels.argmax(dim=1)
		init_labels = [init_labels_dict[idx] if idx in idxs else 0 for idx in range(n_nodes)]
		init_labels = torch.LongTensor(init_labels).to(device)
		mask = torch.LongTensor(mask).to(device)
		# all_labels = torch.LongTensor(all_labels).to(device)

		listg_loss = []
		listg_p,listg_r,listg_f1,listg_ari = [],[],[],[]
		loss_last = 1e9
		for epoch in range(1000):
			self.model.train()
			optimizer.zero_grad()

			out = self.model.labelProp(feats, adj, True)
			loss = F.cross_entropy(out, init_labels, reduction='none')
			mask = mask.float()
			mask = mask / mask.mean()
			loss *= mask
			loss = loss.mean()

			listg_loss.append(loss.item())

			# loss += self.args.weight_decay * self.model.l2_loss()

			pred = out.argmax(dim=1) #519
			pred_dict = {i:j.item() for i,j in enumerate(pred)}
			p, r, ari, f1 = validate_performance(ground_truth, pred_dict)

			if loss_last-loss < 0.001:
				print('Early stopping!')
				break
			else:
				loss_last = loss
				torch.save(self.model.state_dict(), 'best_model_lp.pkl')

			if epoch+1 == 1 or (epoch+1)%10 == 0:
				print("Epoch: {:d} loss={:.5f}".format(epoch+1,loss.item()))

			loss.backward()
			optimizer.step()
		
		self.model.load_state_dict(torch.load('best_model_lp.pkl'))
		self.model.eval()
		out = self.model.labelProp(feats, adj, True)
		pred = out.argmax(dim=1) #519
		pred_dict = {i:j.item() for i,j in enumerate(pred)}
		# validate_performance(ground_truth, pred_dict)
		# validate_ARI_NMI(ground_truth, pred_dict)
		return pred_dict


class RepBin(nn.Module):
	def __init__(self, n_in, n_h, n_opt, act):
		super(RepBin, self).__init__()
		self.gcn = GCN(n_in, n_h, act)
		# self.gcn_ = GCN(2*n_h, n_h, act)
		self.readout = AvgReadout()
		self.sigm = nn.Sigmoid()
		self.disc = Discriminator(n_h)
		self.gcn2 = GCN(n_h, n_opt, 'prelu')

		# self.layers = nn.Sequential(self.gcn, self.gcn2)

	def forward(self, seq1, seq2, adj, sparse, samp_bias1, samp_bias2):
		h_1 = self.gcn(seq1, adj, sparse)
		# h_1 = self.gcn_(h_1, adj, sparse)
		c = self.readout(h_1)
		c = self.sigm(c)
		h_2 = self.gcn(seq2, adj, sparse)
		# h_2 = self.gcn_(h_2, adj, sparse)
		# print(c.shape, h_1.shape, h_2.shape)
		# h_1 = F.dropout(h_1, 0.3, training=self.training)
		# c = nn.Dropout(p=0.5)(c)
		ret = self.disc(c, h_1, h_2, samp_bias1, samp_bias2)
		# print(ret.shape)
		return ret, h_1.squeeze(0)

	# Detach the return variables
	def embed(self, seq, adj, sparse):
		h_1 = self.gcn(seq, adj, sparse)
		c = self.readout(h_1)
		h_1 = h_1.squeeze(0)
		# return h_1.detach().numpy(), c.detach()
		return h_1, c

	def labelProp(self, seq, adj, sparse):
		h = self.gcn(seq, adj, sparse)
		# print(h.shape)
		h = self.gcn2(h, adj, sparse)
		# h = F.log_softmax(self.gcn2(h, adj, sparse))
		# print(h.shape)
		return h.squeeze(0)
		# return h

	def l2_loss(self):
		loss = None
		for p in self.gcn2.parameters():
			if loss is None:
				loss = p.pow(2).sum()
			else:
				loss += p.pow(2).sum()
		return loss

	def constraints_loss(self, embeds, constraints):
		neg_pairs = torch.stack([constraints[:, 0], constraints[:, 1]], 1)
		p = torch.index_select(embeds, 0, neg_pairs[:,0])
		q = torch.index_select(embeds, 0, neg_pairs[:,1])
		return torch.exp(-F.pairwise_distance(p, q, p=2)).mean()
