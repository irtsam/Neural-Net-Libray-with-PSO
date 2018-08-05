# -*- coding: utf-8 -*-

import numpy as np
import numpy.random
import sys
import torch
import matplotlib.pyplot as plt
sys.path.append("/home/iot/Public/htmll/PSO/NeuralNet_Optimization_PSO")
from Neural_Net.Utilities import feedforward, calc_cost
#phi_p=2.05, phi_g=2.05, chi=0.72984


class ParticleSwarm:
    def __init__(self, cost_func, dim, size=50,  phi_p=2.05, phi_g=2.05):
        self.cost_func = cost_func
        self.dim = dim

        self.size = size
        self.phi_p = phi_p
        self.phi_g = phi_g
        print(self.phi_p,self.phi_g)
        self.phi=self.phi_p+self.phi_g
        print(self.phi)
        self.Expl_Factor=1
        #self.chi=2*self.Expl_Factor/np.abs(2-self.phi-np.sqrt(self.phi*(self.phi-4)))
        self.chi= 0.72984
        print(self.chi,"chi")
        

        self.X = np.random.uniform(0,1,size=(self.size, self.dim))
        self.X[0,:]=self.Load_best_particle()
        #V = np.zeros([self.size, self.dim])
        #self.V = V
        self.V = np.random.uniform(0,1/10,size=(self.size, self.dim))

        self.P = self.X.copy()
        self.S = self.cost_func(self.X)
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        self.prev_best=0

    def optimize(self, epsilon=1e-3, max_iter=100,resume=False):
        iteration = 0
        if(resume):
            self.load_optimizer()
        while self.best_score > epsilon and iteration < max_iter:
            self.update()
            iteration = iteration + 1
            print (iteration, self.best_score)
        self.save_optimizer()
        return self.g
    
    
    
    def save_optimizer(self):
        np.savetxt("Particles_Loc", self.X)
        np.savetxt("Particles_Vel", self.V)
        np.savetxt("Particles_best", self.P)
        np.savetxt("Global_best", self.G)
        liss=[self.phi_p,self.phi_g,self.chi]
        np.savetxt("Variables",liss)
    
    def load_optimizer(self):
        self.X=np.loadtxt("Particles_Loc")
        self.V=np.loadtxt("Particles_Vel")
        self.P=np.loadtxt("Particles_best")
        self.G=np.loadtxt("Global_best")
        liss=np.loadtxt("Variables")
        self.chi,self.phi_p,self.phi_g=liss[2],liss[0],liss[1]   
    
    def load_to_CUDA(self):
        self.X=torch.from_numpy(self.X).double().cuda()
        self.V=torch.from_numpy(self.V).double().cuda()
        self.P=torch.from_numpy(self.P).double().cuda()
        self.G=torch.from_numpy(self.G).double().cuda()
        liss=np.loadtxt("Variables")
        liss=torch.from_numpy(liss).double().cuda()
        self.chi,self.phi_p,self.phi_g=liss[2],liss[0],liss[1] 
        
        
    def Load_best_particle(self):
        w1=np.loadtxt("W1_PSO")
        w2=np.loadtxt("W2_PSO")
        w3=np.loadtxt("W3_PSO")
        Weights=[w1,w2,w3]
        vectorZ = self.weights_to_vector(Weights)
        return vectorZ
    
    def weights_to_vector(self,weights):
        w = np.asarray([])
        for i in range(len(weights)):
            v = weights[i].flatten()
            w = np.append(w, v)
        #print(w.shape)
        return w
    def update(self):
        # Velocities update
        R_p = np.random.uniform(size=(self.size, self.dim))
        R_g = np.random.uniform(size=(self.size, self.dim))

        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.g - self.X))

        # Positions update
        self.X = self.X + self.V
        x_max= np.max(self.X)
        v_max= np.max(self.V)
        
        print(x_max, v_max)
        # Best scores
        scores = self.cost_func(self.X)

        better_scores_idx = scores < self.S
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()

# phi_p=1.05, phi_g=1.45

class ParticleSwarm_LBEST(ParticleSwarm):
    def __init__(self, cost_func, dim, neghburs=5, size=50, phi_p=1.05, phi_g=1.45):
        super(ParticleSwarm_LBEST, self).__init__( cost_func, dim, size, phi_p, phi_g)
        self.neghburs=neghburs
        self.G=np.zeros([self.size,self.dim])
        idx=0
        for i in range(self.size//self.neghburs):
            self.G[i,:]=self.g
        print(self.G.shape)
            
    
    def optimizer_reset (self,cudaa=True,vectr=None):
        if cudaa:
            if vectr is None:
                self.X[0,:]=torch.from_numpy(self.Load_best_particle()).double().cuda()
            else:
                self.X[0,:]=vectr
            self.X[1:,:]=((torch.rand(size=(self.size-1, self.dim))*2)-1).double().cuda()
            self.V = (torch.rand(size=(self.size, self.dim)))/100
            self.V= self.V.double().cuda()
            self.P = self.X.clone()
            self.S = self.cost_func(self.X)
            self.g = self.P[self.S.argmin()]
            self.best_score = self.S.min()
            self.G=torch.zeros(self.size,self.dim).double().cuda()
            for i in range(self.size//self.neghburs):
                self.G[i,:]=self.g
    
    def optimize(self, epsilon=1e-3, max_iter=100,load=False,only_best=False,cuda=True):
        iteration = 0
        #while self.best_score > epsilon and iteration < max_iter:
        
        if load:
            self.load_optimizer()
        if cuda:
            self.load_to_CUDA()
        if only_best:
            self.optimizer_reset(cudaa=cuda,vectr=None)
        while iteration < max_iter:
            self.update(max_iter,iteration)
            iteration = iteration + 1
            #if(self.prev_best-self.best_score<5e-06):
            #    self.optimizer_reset(cudaa=cuda,vectr=self.g)
            #if self.best_score-self.prev_best<0.0001:
                #self.decrement_chi()
            self.prev_best=self.best_score
            print(iteration, self.best_score.cpu().numpy())
        self.save_optimizer()
        return self.g
    
    
    """
    def decrement_chi(self):
        self.phi=self.phi_p+self.phi_g
        self.Expl_Factor=self.Expl_Factor-(1/1000)
        self.constriction_calul()
        
    
    def constriction_calul(self):cost= mean_squared_error(y_enc, outpt)
        #self.phi=self.phi_p*R_p + self.phi_g*R_g
        self.chi=2*self.Expl_Factor/np.abs(2-self.phi-np.sqrt(self.phi*(self.phi-4)))
        print(self.chi,"CHI")
    """    
        
    def update(self):
        # Velocities update
        R_p = np.random.uniform(0,1,size=(self.size, self.dim))
        R_g = np.random.uniform(0,1,size=(self.size, self.dim))
        
        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.G - self.X))

        # Positions update
        self.X = self.X + self.V
        
        # Best scores
        scores = self.cost_func(self.X)
        
        

        better_scores_idx = scores < self.S
        #print(better_scores_idx)
        self.P[better_scores_idx] = self.X[better_scores_idx]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        
        idx=0
        for i in range(self.size//self.neghburs):
            if(idx==0 and idx<self.size):
                max_matrix=self.S[self.size-1] #Pick last particle to complete ring
                #print(idx,idx+self.neghburs,i)
                max_matrix= np.append(max_matrix,self.S[0:(self.neghburs+1)]) #Append the next 5 neighbours
                #print(max_matrix.shape)
                assert(max_matrix.shape, (self.neghburs+2))
                #Add support t add into G local best if is at zero
                if(max_matrix.argmin()==0):
                    #If gbest is last element
                    self.G[0:(idx+self.neghburs),:]= self.P[self.size-1]
                else:
                    #else
                    self.G[0:(idx+self.neghburs),:]=self.P[max_matrix.argmin()-1]
                #print(self.P[max_matrix.argmin()].shape,self.G[0:(idx+self.neghburs),:].shape,"Here1")
                idx=idx+self.neghburs
            elif((self.size-idx)<=self.neghburs and idx<self.size):
                max_matrix=self.S[idx-1:]
                
                max_matrix=np.append(max_matrix,self.S[0])
                #print(idx+(max_matrix.argmin()-1))
                #print(max_matrix.shape,"Now")
                if(max_matrix.argmin()==self.neghburs+1):
                    #First element is gbest
                    self.G[idx:,:]= self.P[0]
                    #print("here \too")
                    #print(self.P[0].shape,self.G[idx:,:].shape,"here2")
                else:
                    self.G[idx:,:]=self.P[idx+(max_matrix.argmin()-1)]
                    #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:,:].shape,"here2")
                
                #print(idx,i)
                idx=idx+self.neghburs
            elif(idx<self.size):
                max_matrix=scores[idx-1:idx+(self.neghburs+1)]
                #print(max_matrix.shape,"here")
                assert(max_matrix.shape,self.neghburs+2)
                #add suppot to add to G if the global is from last position
                self.G[idx:idx+self.neghburs,:]=self.P[idx+(max_matrix.argmin()-1)]
                #print(idx,i)
                #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:idx+self.neghburs,:].shape,"here3")
                idx=idx+self.neghburs
                
        assert(self.X.shape,self.G.shape)


        
class ParticleSwarm_LBEST_CUDA(ParticleSwarm_LBEST):
    def __init__(self, cost_func, dim, neghburs=5, size=50, phi_p=1.4960, phi_g=1.4960):
        #super(ParticleSwarm_LBEST_CUDA, self).__init__( cost_func, dim, neghburs, size, phi_p, phi_g)
        
        self.cost_func = cost_func
        self.dim = dim
        self.size = size
        self.phi_p = torch.tensor(phi_p).double().cuda()
        self.phi_g = torch.tensor(phi_g).double().cuda()
        self.Expl_Factor=torch.tensor(1).double().cuda()
        self.phi=(self.phi_p+self.phi_g).double().cuda()
        #self.chi=2/torch.abs(2-self.phi-torch.sqrt(self.phi*(self.phi-4)))
        
        self.chi= torch.tensor(0.72984).double().cuda()
        self.chi_max= torch.tensor(0.72984).double().cuda()
        self.chi_min= torch.tensor(0.4).double().cuda()
        print(self.chi,"chi")
        
        #self.chi,self.phi_p,self.phi_g=self.chi.double.cuda(),self.phi_p.double.cuda(),self.phi_g.double.cuda()

        X = np.random.uniform(-1,1,size=(self.size, self.dim))
        self.X=torch.from_numpy(X)
        self.X= self.X.double().cuda()
        #self.X[0,:]=torch.from_numpy(self.Load_best_particle()).double().cuda()
        V = np.random.uniform(0,1/10,size=(self.size, self.dim))
        self.V = torch.from_numpy(V)
        self.V= self.V.double().cuda()
        self.P = self.X.clone()
        self.S = self.cost_func(self.X)
        self.V_max=torch.tensor(0.5).double().cuda()
        self.V_min=torch.tensor(-0.5).double().cuda()
        self.X_max=torch.tensor(40).double().cuda()
        self.X_min=torch.tensor(-40).double().cuda()
        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        self.neghburs=neghburs
        self.prev_best=self.best_score
        self.G=torch.zeros(self.size,self.dim).double().cuda()
        for i in range(self.size//self.neghburs):
            self.G[i,:]=self.g
        #self.Q=(self.phi_p*self.P+self.phi_g*self.G)/(self.phi_p+self.phi_g)
        #print(self.G.size())
        #print(self.X.size(),self.V.size(), self.P.size())
        #print(self.S.size(), self.g.size())
            
    
    def update(self,max_iter,itera):
        # Velocities update
        R_p = torch.rand(size=(self.size, self.dim)).double().cuda()
        R_g = torch.rand(size=(self.size, self.dim)).double().cuda()
        
        #self.chi=(self.chi_max-self.chi_min)*((max_iter-itera)/max_iter)
        """
        self.V = self.chi * (self.V \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.G - self.X))
        """
        self.V = self.chi * (self.V) \
                + self.phi_p * R_p * (self.P - self.X) \
                + self.phi_g * R_g * (self.G - self.X)
        
        
        # Positions update
        
        #self.V=torch.where(self.V>0.5,-self.V_max,self.V)
        #self.V=torch.where(self.V<-0.5,-self.V_min,self.V)
        #self.X=torch.where(self.X>2.0,self.X_max,self.X)
        #self.X=torch.where(self.X<-2.0,self.X_min,self.X)
        
        self.X = self.X + self.V
        print(self.X.max().cpu().numpy(),self.V.max().cpu().numpy(),self.V.min().cpu().numpy())
        
        #self.V=torch.where(self.V>0.5,-self.V_max,self.V)
        #self.V=torch.where(self.V<-0.5,-self.V_min,self.V)
        #self.X=torch.where(self.X>2.0,self.X_max,self.X)
        #self.X=torch.where(self.X<-2.0,self.X_min,self.X)
        
        print(self.X.max().cpu().numpy(),self.V.max().cpu().numpy(),self.V.min().cpu().numpy())
        # Best scores
        #print(self.X.size(),"X")
        
        scores = self.cost_func(self.X)
        better_scores_idx = scores < self.S
        #print(better_scores_idx)
        
        
        self.P=torch.where(better_scores_idx,self.X,self.P)
        #self.P[better_scores_idx] = self.X[better_scores_idx,:]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        #print(self.X.dtype,self.P.dtype,self.V.dtype,self.G.dtype,better_scores_idx.size(),scores.size(),scores,better_scores_idx)
        idx=0
        for i in range(self.size//self.neghburs):
            if(idx==0 and idx<self.size):
                max_matrix=self.S[self.size-1] #Pick last particle to complete ring
                #print(idx,idx+self.neghburs,i)
                max_matrix= np.append(max_matrix,self.S[0:(self.neghburs+1)]) #Append the next 5 neighbours
                #print(max_matrix.shape)
                assert(max_matrix.shape, (self.neghburs+2))
                #Add support t add into G local best if is at zero
                if(max_matrix.argmin()==0):
                    #If gbest is last element
                    self.G[0:(idx+self.neghburs),:]= self.P[self.size-1]
                else:
                    #else
                    self.G[0:(idx+self.neghburs),:]=self.P[max_matrix.argmin()-1]
                #print(self.P[max_matrix.argmin()].shape,self.G[0:(idx+self.neghburs),:].shape,"Here1")
                idx=idx+self.neghburs
            elif((self.size-idx)<=self.neghburs and idx<self.size):
                max_matrix=self.S[idx-1:]
                
                max_matrix=np.append(max_matrix,self.S[0])
                #print(idx+(max_matrix.argmin()-1))
                #print(max_matrix.shape,"Now")
                if(max_matrix.argmin()==self.neghburs+1):
                    #First element is gbest
                    self.G[idx:,:]= self.P[0]
                    #print("here \too")
                    #print(self.P[0].shape,self.G[idx:,:].shape,"here2")
                else:
                    self.G[idx:,:]=self.P[idx+(max_matrix.argmin()-1)]
                    #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:,:].shape,"here2")
                
                #print(idx,i)
                idx=idx+self.neghburs
            elif(idx<self.size and (self.size-idx)>self.neghburs):
                max_matrix=scores[idx-1:idx+(self.neghburs+1)]
                #print(max_matrix.shape,"here")
                assert(max_matrix.shape,self.neghburs+2)
                #add suppot to add to G if the global is from last position
                self.G[idx:idx+self.neghburs,:]=self.P[idx+(max_matrix.argmin()-1)]
                #print(idx,i)
                #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:idx+self.neghburs,:].shape,"here3")
                idx=idx+self.neghburs
                
        assert(self.X.shape,self.G.shape)




class ParticleSwarm_BreakNN_CUDA(ParticleSwarm_LBEST_CUDA):
    def __init__(self, cost_func, dim, layer_dim,prev_vect=None,neghburs=5, size=50, phi_p=3.05, phi_g=3.05):
        super(ParticleSwarm_BreakNN_CUDA, self).__init__( cost_func, dim, neghburs, size, phi_p, phi_g)
        self.layer_dim=layer_dim
        self.X_static=self.X
        if prev_vect is not None:
            for i in range(self.size):
                self.X_static[i,:]=prev_vect
        #self.constant_matrix=            
            
    
    def layer_select(self):
        X_temp=self.X
        self.X=self.X_static
        layr_dim0=self.layer_dim[0]
        layr_dim1=self.layer_dim[1]
        self.X[:,layr_dim0:layr_dim1]=X_temp[:,layr_dim0:layr_dim1]
        
    def update(self,max_iter,iteration):
        # Velocities update
        #self.chi=(self.chi_max-self.chi_min)*((max_iter-itera)/max_iter)
        R_p = torch.rand(size=(self.size, self.dim)).double().cuda()
        R_g = torch.rand(size=(self.size, self.dim)).double().cuda()
        lr1,lr2=self.layer_dim[0],self.layer_dim[1]
        self.V=torch.where(self.V>0.5,(-self.V_max)/5,self.V)
        self.V=torch.where(self.V<-0.5,(-self.V_min)/5,self.V)
        self.X=torch.where(self.X>40.0,self.X_max,self.X)
        self.X=torch.where(self.X<-4.0,self.X_min,self.X)
        #self.layer_select()
        
        self.V[:,lr1:lr2] = self.chi * (self.V[:,lr1:lr2]) \
                + self.phi_p * R_p[:,lr1:lr2] * (self.P[:,lr1:lr2] - self.X[:,lr1:lr2]) \
                + self.phi_g * R_g[:,lr1:lr2] * (self.G[:,lr1:lr2] - self.X[:,lr1:lr2])

        # Positions update
        self.X[:,lr1:lr2] = self.X[:,lr1:lr2] + self.V[:,lr1:lr2]
        print(self.X.max().cpu().numpy(),self.V.max().cpu().numpy())
        # Best scores
        #print(self.X.size(),"X")
        
        scores = self.cost_func(self.X)
        better_scores_idx = scores < self.S
        #print(better_scores_idx)
        
        
        self.P=torch.where(better_scores_idx,self.X,self.P)
        #self.P[better_scores_idx] = self.X[better_scores_idx,:]
        self.S[better_scores_idx] = scores[better_scores_idx]

        self.g = self.P[self.S.argmin()]
        self.best_score = self.S.min()
        #print(self.X.dtype,self.P.dtype,self.V.dtype,self.G.dtype,better_scores_idx.size(),scores.size(),scores,better_scores_idx)
        idx=0
        for i in range(self.size//self.neghburs):
            if(idx==0 and idx<self.size):
                max_matrix=self.S[self.size-1] #Pick last particle to complete ring
                #print(idx,idx+self.neghburs,i)
                max_matrix= np.append(max_matrix,self.S[0:(self.neghburs+1)]) #Append the next 5 neighbours
                #print(max_matrix.shape)
                assert(max_matrix.shape, (self.neghburs+2))
                #Add support t add into G local best if is at zero
                if(max_matrix.argmin()==0):
                    #If gbest is last element
                    self.G[0:(idx+self.neghburs),:]= self.P[self.size-1]
                else:
                    #else
                    self.G[0:(idx+self.neghburs),:]=self.P[max_matrix.argmin()-1]
                #print(self.P[max_matrix.argmin()].shape,self.G[0:(idx+self.neghburs),:].shape,"Here1")
                idx=idx+self.neghburs
            elif((self.size-idx)<=self.neghburs and idx<self.size):
                max_matrix=self.S[idx-1:]
                
                max_matrix=np.append(max_matrix,self.S[0])
                #print(idx+(max_matrix.argmin()-1))
                #print(max_matrix.shape,"Now")
                if(max_matrix.argmin()==self.neghburs+1):
                    #First element is gbest
                    self.G[idx:,:]= self.P[0]
                    #print("here \too")
                    #print(self.P[0].shape,self.G[idx:,:].shape,"here2")
                else:
                    self.G[idx:,:]=self.P[idx+(max_matrix.argmin()-1)]
                    #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:,:].shape,"here2")
                
                #print(idx,i)
                idx=idx+self.neghburs
            elif(idx<self.size and (self.size-idx)>self.neghburs):
                max_matrix=scores[idx-1:idx+(self.neghburs+1)]
                #print(max_matrix.shape,"here")
                assert(max_matrix.shape,self.neghburs+2)
                #add suppot to add to G if the global is from last position
                self.G[idx:idx+self.neghburs,:]=self.P[idx+(max_matrix.argmin()-1)]
                #print(idx,i)
                #print(self.P[idx+(max_matrix.argmin()-1)].shape,self.G[idx:idx+self.neghburs,:].shape,"here3")
                idx=idx+self.neghburs
                
        assert(self.X.shape,self.G.shape)

