import os
import numpy as np
import cv2
import random
from Neural_Net.Utilities import *
from Dataset_Loaders.Deveanagri_Handwritten_Character_Dataset_Loader import *
from sklearn.utils import shuffle
import time
import torch
try:
    import visdom
    viss=True
except ImportError:
    viss=False


def Train(X,y, X_t, y_t,resume):

    X_copy, y_enc = X.copy(),y.copy()
    X_copy=X_copy/255
    #X_copy=X_copy-0.5
    epochs = 2000
    batch = 50
    s= np.arange(X_copy.shape[1])
    np.random.shuffle(s)
    
    X_copy, y_enc = X_copy[:,s],y_enc[:,s]
    X_copy=np.transpose(X_copy)
    
    
    if(resume):
        w1=np.loadtxt("W1")
        w2=np.loadtxt("W2")
        w3=np.loadtxt("W3") 
    else:
        w1, w2, w3 = init_weights(64,90, 50, 10)
        #w1, w2, w3 = init_weights(1024,400, 100, 46)
    print(w3.shape)

    alpha = 0.8
    eta = 0.1
    dec = 0.00001
    
    Loss=[] 
    
    delta_w1_prev = np.zeros(w1.shape)
    delta_w2_prev = np.zeros(w2.shape)
    delta_w3_prev = np.zeros(w3.shape)
    ttl_cost=0
    if viss:
          vis = visdom.Visdom()
          loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1)).cpu(),
                       opts=dict(xlabel='SGD',
                                 ylabel='Loss',
                                 title='Training Loss for Normalized Data with Momentum',
                                 legend=['Loss']))
    j=0
    t1=time.time()
    for i in range(epochs):
        total_cost = []
        #shuffle = np.random.permutation(y_copy.shape[0])
        #X_copy, y_enc = X_copy[shuffle], y_enc[:,shuffle]
        #eta/=(1+dec*i) 
        Loss.append(ttl_cost/batch)
        ttl_cost=0
        mini = np.array_split(range(y_enc.shape[1]),batch)

        for step in mini:
            #Feedforward the model
            a1,z2,a2,z3,a3,z4,a4 = feedforward(X_copy[step],w1,w2,w3)
            cost = calc_cost(y_enc[:,step],a4)
            
            ttl_cost=ttl_cost+cost
            
            total_cost.append(cost)

            #Back Propagation
            grad1,grad2,grad3 = calc_grad(a1,a2,a3,a4,z2,z3,z4,y_enc[:,step],w1,w2,w3)
            delta_w1,delta_w2,delta_w3 = eta*grad1,eta*grad2,eta*grad3
            w1 -= delta_w1 #+ alpha*delta_w1_prev  #to be clarified. Have to sum the deltaw1prev and delta w1 and then multiply with alpha for gradient descent
            w2 -= delta_w2 #+ alpha*delta_w2_prev
            w3 -= delta_w3 #+ alpha*delta_w3_prev

            delta_w1_prev,delta_w2_prev,delta_w3_prev = delta_w1, delta_w2, delta_w3

        
        print('epoch #',i,"Loss",ttl_cost/batch)
        if viss:
            vis = visdom.Visdom()

            loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss for FCN',
                                     legend=['Loss']))
            j=j+1  
            vis.line(
                        X=torch.ones((1, 1)).cpu() * j,
                        Y=torch.Tensor([ttl_cost /batch]).unsqueeze(0).cpu(),
                        win=loss_window,
                        update='append')
    
    t2=time.time()
    print(t2-t1)
    np.savetxt("Loss",total_cost)
    np.savetxt("W1", w1)
    np.savetxt("W2", w2)
    np.savetxt("W3", w3)
    Test(X_t,y_t)
    
    return 1

def Test(X_t,y_t):
    w1=np.loadtxt("W1")
    w2=np.loadtxt("W2")
    w3=np.loadtxt("W3")
    
    X_t=np.transpose(X_t)
    y_pred = predict(X_t,w1,w2,w3)
    y_t= np.argmax(y_t,axis=0)
    acc = np.sum(y_t == y_pred, axis=0)/ X_t.shape[0]
    print('training accuracy', acc*100)
    
global X
if __name__ == '__main__':
    path="/home/iot/Public/htmll/PSO/NeuralNet_Optimization_PSO/DevanagariHandwrittenCharacterDataset"
    #data= Devan_data_loader(path, 'Train')
    #X_Tr,Y_Tr=data.getdata('Train')
    #X_Ts,Y_Ts=data.getdata('Test')
    #Train(X_Tr,Y_Tr,X_Ts,Y_Ts,resume=False)
    #Test(X_Ts,Y_Ts)
    global X
    num_classes = 10
    mnist = sklearn.datasets.load_digits(num_classes)
    X, X_test, y, y_test = sklearn.cross_validation.train_test_split(mnist.data, mnist.target)
    #Revert the below back
    #num_inputs = X.shape[1]
    num_inputs=64
    y_true = np.zeros((len(y), num_classes))
    for i in range(len(y)):
        y_true[i, y[i]] = 1

    y_test_true = np.zeros((len(y_test), num_classes))
    for i in range(len(y_test)):
        y_test_true[i, y_test[i]] = 1
        
    # Set up
    shape = (num_inputs, 64, 32, num_classes)
    X, y_true,X_test, y_test_true=X.T,y_true.T,X_test.T,y_test_true.T
    print(X.shape, y_true.shape, X_test.shape, y_test_true.shape)
 
   
    Train(X, y_true,X_test, y_test_true,resume=False)
