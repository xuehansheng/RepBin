# !/usr/bin/env python
# -*- coding: utf8 -*-

import torch
import torch.nn as nn


class GCN(nn.Module):
	def __init__(self, in_ft, out_ft, act, bias=True):
		super(GCN, self).__init__()
		self.fc = nn.Linear(in_ft, out_ft, bias=False)
		# self.act = nn.PReLU() if act == 'prelu' else act
		if act == 'prelu':
			self.act = nn.PReLU() 
		elif act == 'relu':
			self.act = nn.ReLU()
		elif act == 'leakyrelu':
			self.act = nn.LeakyReLU()
		elif act == 'softmax':
			self.act = nn.Softmax()
		elif act == 'sigmoid':
			self.act = nn.Sigmoid()
		elif act == 'identity':
			self.act = nn.Identity()
		
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(out_ft))
			self.bias.data.fill_(0.0)
		else:
			self.register_parameter('bias', None)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	# Shape of seq: (batch, nodes, features)
	def forward(self, seq, adj, sparse=False):
		seq_fts = self.fc(seq)
		if sparse:
			out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
		else:
			out = torch.bmm(adj, seq_fts)
		if self.bias is not None:
			out += self.bias
		
		return self.act(out)


class AvgReadout(nn.Module):
	def __init__(self):
		super(AvgReadout, self).__init__()

	def forward(self, seq):
		return torch.mean(seq, 1)


class Discriminator(nn.Module):
	def __init__(self, n_h):
		super(Discriminator, self).__init__()
		self.f_k = nn.Bilinear(n_h, n_h, 1)

		for m in self.modules():
			self.weights_init(m)

	def weights_init(self, m):
		if isinstance(m, nn.Bilinear):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				m.bias.data.fill_(0.0)

	def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
		c_x = torch.unsqueeze(c, 1)
		# print(c.shape, c_x.shape)
		c_x = c_x.expand_as(h_pl)
		# print(c_x.shape)

		sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
		sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
		# print(sc_1.shape, sc_2.shape)

		if s_bias1 is not None:
			sc_1 += s_bias1
		if s_bias2 is not None:
			sc_2 += s_bias2

		logits = torch.cat((sc_1, sc_2), 1)
		# print(logits.shape)

		return logits