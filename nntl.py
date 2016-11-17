# # Neural_network_two_layer

import numpy as np
import heapq
from sklearn.metrics import confusion_matrix


class pca():

    def run(self,imox_mtx,k):

        imox_mtx = self.imox_mtx
        k = self.k
        imox_sctr = np.dot(imox_mtx.T, imox_mtx)
        imox_eigval, imox_eigvec = np.linalg.eig(imox_sctr)
        per = np.real(np.sum(imox_eigval[:k])/ np.sum(imox_eigval))
        print  'percentage: %.4f'%(per), ', number of features: %d' %(k)
    
        max_pos = heapq.nlargest(k, xrange(len(imox_eigval)), key=imox_eigval.__getitem__)
        transmtx = imox_eigvec[:,max_pos]  



class nntl():
    
    def __init__(self, X_train, Y_train, step_size = 0.1, reg = 0.001, h_size = 10, niter = 10000):
        
        self.X = X_train
        self.Y = Y_train
        
        self.step_size = step_size
        self.reg = reg
        self.h = h_size
        self.niter = niter
        
        self.ndata = np.shape(self.X)[0]
        self.ndim = np.shape(self.X)[1]
        self.nclass = len(np.unique(self.Y))
   
        self.wt_ini = 0.02
        
        self.wt = self.wt_ini * np.random.randn(self.ndim, self.h)
        self.b = np.zeros(self.h)
        
        self.wt2 = self.wt_ini * np.random.randn(self.h, self.nclass)
        self.b2 = np.zeros(self.nclass)
  
        
    def model(self):
        
        wt = self.wt
        b = self.b
        wt2 = self.wt2
        b2 = self.b2
        loss = 3
        
        for i in range(self.niter):
 
            hidden_layer = np.maximum(0, np.dot(self.X, wt) + b)
 
            scores = np.dot(hidden_layer, wt2) + b2
            exp_scores = np.exp(scores)         
            probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
            corect_logprobs = -np.log(probs[range(self.ndata),self.Y])
  
            data_loss = np.sum(corect_logprobs)/self.ndata

            reg_loss = .5 * self.reg * np.sum((wt*wt)) + .5 *self.reg*np.sum((wt2*wt2))
            loss_new = data_loss + reg_loss
            
            if (i%100 == 0 or i == self.niter-1):
                print 'iteration: %d, loss: %f' %(i, loss_new)
                #print np.sum(wt2), np.sum(b2)
            """    
            if (abs(loss-loss_new)/loss < 0.001 and i > 1):
                print 'break condition match: ', (loss-loss_new)/loss, loss
                break
            """
            loss = loss_new
            
            grads = {} 
            dscores = probs
            dscores[range(self.ndata),self.Y] -= 1
            dscores /= self.ndata         
 
            grads['wt2'] = np.dot(hidden_layer.T, dscores)
            grads['b2'] = np.sum(dscores, axis = 0)
            dhidden = np.dot(dscores, wt2.T)
            dhidden[hidden_layer <=0] = 0
            grads['wt'] = np.dot(self.X.T, dhidden)
            grads['b'] = np.sum(dhidden, axis = 0)

            grads['wt2'] += self.reg * wt2
            grads['wt'] += self.reg * wt

            wt2 = wt2 - self.step_size * grads['wt2']
            wt = wt - self.step_size * grads['wt']
            b2 = b2 - self.step_size * grads['b2']
            b =  b - self.step_size * grads['b']
        
        self.wt2, self.wt = wt2, wt
        self.b2, self.b = b2, b

        return loss
        
    def predict(self, X_test):
            
        wt = self.wt
        b = self.b
        wt2 = self.wt2
        b2 = self.b2
        
        hidden_layer = np.maximum(0, np.dot(X_test,wt) + b)
        scores = np.dot(hidden_layer, wt2) + b2
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
        Y_pred = probs.argmax(axis = 1)
        
        return Y_pred
        
        
    def accuracy(self, Y_testset, Y_pred):
        
        accu_array = [1 if Y_testset[i:i+1]==Y_pred[i:i+1] else 0 for i in range(len(Y_testset)) ]
        accu = sum(accu_array)/np.double(len(Y_testset))
        return accu

