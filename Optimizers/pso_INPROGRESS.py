#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 18:15:56 2018

@author: IGHAZI
"""

import numpy as np
import numpy.random
import sys
import torch
sys.path.append("/home/iot/Public/htmll/PSO/NeuralNet_Optimization_PSO")
from Neural_Net.Utilities import feedforward, calc_cost

class ParticleSwarm_INPROGRESS:
    def __init__(self, cost_func, dim, size=50, chi=0.72984, phi_p=2.05, phi_g=2.05):
        self.cost_func = cost_func
        self.dim = dim
        
        
        self.size = size
        self.chi = chi
        self.phi_p = phi_p
        self.phi_g = phi_g

        X = np.random.uniform(0,1/200,size=(self.size, self.dim))
        self.X=torch.from_numpy(X)
        self.X= self.X.double().cuda()
        V = np.random.uniform(0,1/200,size=(self.size, self.dim))
        self.V = torch.from_numpy(V)
        self.V= self.V.double().cuda()
        self.P = self.X.clone()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        print(self.X.size(),self.V.size(), self.P.size())
        print(self.S.size(), self.g.size())
        

    def optimize(self, epsilon=1e-3, max_iter=100):
        iteration = 0
        while self.best_score > epsilon and iteration < max_iter:
            self.update()
            iteration = iteration + 1
            print(iteration, self.best_score)
        return self.g

    def update(self):
        # Velocities update
        R_p = torch.randn(size=(self.size, self.dim)).double().cuda()
        R_g = torch.randn(size=(self.size, self.dim)).double().cuda()

        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.g - self.X))

        # Positions update
        self.X = self.X + self.V

        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        print(better_scores_idx)
        
        self.P=torch.masked_select(self.X, better_scores_idx)
        self.S= torch.masked_select(self.X, better_scores_idx)
        
        #self.P[better_scores_idx] = self.X[better_scores_idx]
        #self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
    