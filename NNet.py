import random
import numpy as np

np.random.seed(1234)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def relerr(y,ycap):
    return np.sum(np.abs(y-ycap),axis = 0)

def cost_prime(y,ycap):
    return ycap-y

def GetNet(sizes):
    if(len(sizes)<2):
        return None
    W = []
    b = []
    for i in range(len(sizes)-1):
        W.append(np.random.random((sizes[i],sizes[i+1])))
        b.append(np.random.random((sizes[i+1],)))
    return {"W":W,"b":b,"sizes":sizes}

def Forward(net,x):
    z = np.array(x).dot(net["W"][0]) + net["b"][0]
    a = sigmoid(z)
    for i in range(1,len(net["W"])):
        z = a.dot(net["W"][i]) + net["b"][i]
        a = sigmoid(z)
    return a

def feed_forward(net,x):
    Z = [np.array(x)]
    A = [sigmoid(np.array(x))]
    z = np.array(x).dot(net["W"][0]) + net["b"][0]
    Z.append(z)
    a = sigmoid(z)
    A.append(a)
    for i in range(1,len(net["W"])):
        z = a.dot(net["W"][i]) + net["b"][i]
        a = sigmoid(z)
        Z.append(z)
        A.append(a)
    return A,Z

def back_prop(net,A,Z,y):
    d = []
    L = len(net["sizes"])
    for i in range(L):
        d.append(np.zeros((net["sizes"][i],)))
    for j in range(net["sizes"][L-1]):
        d[L-1][j] = cost_prime(y[j],A[L-1][j])*sigmoid_prime(Z[L-1][j])
    for l in reversed(range(L-1)):
        temp = net["W"][l].dot(d[l+1])
        for j in range(net["sizes"][l]):
            d[l][j] = temp[j]*sigmoid_prime(Z[l][j])
    return d

def train_epoch(net,x,y,lr = 1.0,mbs = -1,write=False,ERR = relerr):
    N = len(x)
    L = len(net["sizes"])
    if(mbs > N):
        return None
    if(mbs <= 0):
        mbs = N
    cnt = 0
    while cnt < N:
        minix = []
        miniy = []
        for i in range(mbs):
            minix.append(x[cnt])
            miniy.append(y[cnt])
            cnt+=1
            if(cnt >= N):
                break;
        gradW = []
        gradb = []
        for i in range(L-1):
            gradW.append(np.zeros((net["sizes"][i],net["sizes"][i+1])))
            gradb.append(np.zeros((net["sizes"][i+1],)))
        for index in range(len(minix)):
            A,Z = feed_forward(net,minix[index])
            #if(write):
            #    print(ERR(miniy[index],A[L-1]))
            delta = back_prop(net,A,Z,miniy[index])
            for l in range(L-1):
                gradb[l] += lr*delta[l+1]/float(len(minix))
            for l in range(L-1):
                for k in range(net["sizes"][l]):
                       for j in range(net["sizes"][l+1]):
                           gradW[l][k][j] += lr*A[l][k]*delta[l+1][j]/float(len(minix))
        for l in range(L-1):
            net["b"][l] -= gradb[l]
        for l in range(L-1):
            for k in range(net["sizes"][l]):
                for j in range(net["sizes"][l+1]):
                    net["W"][l][k][j] -= gradW[l][k][j]


def Train(net,x,y,epochs,lr = 1.0,mbs = -1,write=False,ERR = relerr):
    for i in range(epochs):
        train_epoch(net,x,y,lr,mbs,write,ERR)

net = GetNet((2,3,1))
x = [[1.0,1.0],
     [1.0,0.0],
     [0.0,1.0],
     [0.0,0.0]]
y = [[0.0],[1.0],[1.0],[0.0]]
Train(net,x,y,10000,lr = 0.5,write = True)
print(x)
print(np.heaviside(Forward(net,x)-0.5,0))
