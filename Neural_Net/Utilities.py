

""" The following define functions to bes used as utilities for defining our
neural network class. This is based off of Andrew Ng's machine learning course"""
    
import os
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.metrics import mean_squared_error
global cuda
cuda=True
global w1,w2,w3
def sigmoid (z):
    """Takes as input a np array, number etc and returns sigmoid of the
    matrix"""
    return 1/(1+np.exp(-z))

def sigmoid_gradient(z):
    """Takes as input a np array, number etc and returns gradient of the
    matrix"""
    s= sigmoid(z)
    return s*(1-s)

    
def enc_one_hot(y,num_labels):
    """Takes as input a 1D vector of classification labels
    and returns a data_size*classes vector for output according to the commmon
    one_hot_encoding process"""
    print(y.shape)
    one_hot=np.zeros((num_labels,y.shape[0]))
    counts= y.shape[0]
    #print(y[2,1])
    #k = one_hot.shape
    #print(k)
    for i in range(0,counts):
        a= y[i,0]
        a=a.astype(int)
        #print(a)
        one_hot[a,i] = 1
        #print(one_hot)
    return one_hot


# output is the predicted output
def calc_cost(y_enc, outpt):
    """Logistic Regression loss function. Requires a label and calculated ouput"""
    global cuda
    if cuda:
        device = torch.device("cuda")
        #y_enc,outpt = torch.from_numpy(y_enc), torch.from_numpy(outpt)
        y_enc,outpt= y_enc.to(device), outpt.to(device)
        cost = f.cross_entropy(outpt, y_enc)
        print("here")
    else:
        #print("Else Here")
        t1= -y_enc*np.log(outpt)
        t2= (1-y_enc)*np.log(1-outpt)
        cost=np.sum(t1-t2)
    return cost

def add_bias_unit(X,where):
    """Add bias during forward & backward propagation by specifying the
    where item"""
    global cuda
    if cuda:
        if where == 'column':
            X_new= torch.ones((X.shape[0], X.shape[1]+1)).cuda().double()
            X_new[:,1:] = X
        elif where == 'row':
            X_new= torch.ones((X.shape[0] + 1, X.shape[1])).cuda().double()
            X_new[1:, :] = X
    else:
        if where == 'column':
            X_new= np.ones((X.shape[0], X.shape[1]+1))
            X_new[:,1:] = X
        elif where == 'row':
            X_new= np.ones((X.shape[0] + 1, X.shape[1]))
            X_new[1:, :] = X
    return X_new

def init_weights(n_features, n_hidden, n_hidden1, n_output):
    """Randomly initializes weights. Uses Xavier initialization"""
    global w1, w2,w3
    w1= np.random.uniform((-1/n_features),(1/n_features), size=(n_hidden*(n_features + 1)))
    w1= w1.reshape(n_hidden, n_features+1)
    w2= np.random.uniform((-1/n_hidden),(1.0/n_hidden), size=(n_hidden1*(n_hidden + 1)))
    w2= w2.reshape(n_hidden1, n_hidden +1)
    w3= np.random.uniform((-1/n_hidden1),(1.0/n_hidden1), size=(n_output*(n_hidden1 + 1)))
    w3= w3.reshape(n_output, n_hidden1 +1)
    return [w1, w2, w3]

def feedforward( x,w1,w2,w3):
    """Feedforward through two layers of neurons"""
    #Add Bias to the input
    #Column within the row is just a byte of data
    #So we need to add column vector of ones
    global cuda
    if cuda:
        device = torch.device("cuda")
        x,w1,w2,w3 = torch.from_numpy(x), torch.from_numpy(w1),torch.from_numpy(w2),torch.from_numpy(w3)
        #w1,w2,w3 = torch.from_numpy(w1),torch.from_numpy(w2),torch.from_numpy(w3)
        x,w1,w2,w3 = x.to(device), w1.to(device),w2.to(device),w3.to(device)
        sigm= nn.Sigmoid()	
        a1= add_bias_unit (x, where='column')
        a1=torch.t(a1).double()
        a1=a1.to(device)
        z2= torch.mm(w1,a1)
        a2= sigm(z2)
        # Since we transposed we have to add bias units as a row
        a2= add_bias_unit(a2, where= 'row')
        z3= torch.mm(w2,a2)
        a3= sigm(z3)
        a3= add_bias_unit(a3, where='row')
        z4=torch.mm(w3,a3)
        a4=sigm(z4)
    else:
        a1= add_bias_unit (x, where='column')
        z2= w1.dot(a1.T)
        a2= sigmoid(z2)
        # Since we transposed we have to add bias units as a row
        a2= add_bias_unit(a2, where= 'row')
        z3= w2.dot(a2)
        a3= sigmoid(z3)
        a3= add_bias_unit(a3, where='row')
        z4=w3.dot(a3)
        a4=sigmoid(z4)
    return [a1, z2, a2, z3 ,a3 ,z4, a4]

def feedforward_PSO( x,w1,w2,w3,y_enc):
    """Feedforward through two layers of neurons"""
    #Add Bias to the input
    #Column within the row is just a byte of data
    #So we need to add column vector of ones
    global cuda
    if cuda:
        device = torch.device("cuda")
        #x,w1,w2,w3 = torch.from_numpy(x), torch.from_numpy(w1),torch.from_numpy(w2),torch.from_numpy(w3)
        #w1,w2,w3 = torch.from_numpy(w1),torch.from_numpy(w2),torch.from_numpy(w3)
        x,w1,w2,w3 = x.to(device), w1.to(device),w2.to(device),w3.to(device)
        sigm= nn.Sigmoid()	
        a1= add_bias_unit (x, where='column')
        a1=torch.t(a1).double()
        a1=a1.to(device)
        z2= torch.mm(w1,a1)
        a2= sigm(z2)
        # Since we transposed we have to add bias units as a row
        a2= add_bias_unit(a2, where= 'row')
        z3= torch.mm(w2,a2)
        a3= sigm(z3)
        a3= add_bias_unit(a3, where='row')
        z4=torch.mm(w3,a3)
        a4=sigm(z4)
        loss=nn.MSELoss()
        #loss= nn.CrossEntropyLoss()
        #y_enc,a4=y_enc.long(),a4.long()
        cost= loss(y_enc,a4)
    else:
        a1= add_bias_unit (x, where='column')
        z2= w1.dot(a1.T)
        a2= sigmoid(z2)
        # Since we transposed we have to add bias units as a row
        a2= add_bias_unit(a2, where= 'row')
        z3= w2.dot(a2)
        a3= sigmoid(z3)
        a3= add_bias_unit(a3, where='row')
        z4=w3.dot(a3)
        a4=sigmoid(z4)
        cost= mean_squared_error(y_enc, a4)
    return cost



def predict(x,w1,w2,w3):
    a1,z2,a2,z3,a3,z4,a4= feedforward(x,w1,w2,w3)
    if cuda:
        a4= a4.data.cpu().numpy()
    #print(a4.shape)
    y_pred= np.argmax(a4,axis=0)
    return y_pred




#Backpropagation Starts here
def calc_grad(a1,a2,a3,a4,z2,z3,z4,y_enc,w1,w2,w3):
    """Implements the backpropagation functions"""
    delta4= a4-y_enc
    #print(a4)
    #print(delta4)
    # Add bias unit as dimension of z3 doesn't incorporate bias unit which was added in next step i.e. the a step
    z3= add_bias_unit(z3,where='row')
    #print(a4.shape)
    #print(y_enc.shape)
    #print(w3.shape)
    delta3= w3.T.dot(delta4)*sigmoid_gradient(z3)
    delta3= delta3[1:,:]
    z2= add_bias_unit(z2, where= 'row')
    
    delta2= w2.T.dot(delta3)*sigmoid_gradient(z2)
    delta2= delta2[1:,:]
    #print(delta2.shape)
    #print(a1.shape, a2.shape, a3.shape, a4.shape, z2.shape, z3.shape, z4.shape)

    grad1= delta2.dot(a1)
    grad2= delta3.dot(a2.T)
    grad3= delta4.dot(a3.T)

    return grad1, grad2, grad3



    
