import numpy as np
class LogLoss():

    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #  Where N can be any integer >=1
    #Output: A single floating point value.
    def eval(self, Y, Yhat):
        eps = 1e-6
        J = -(Y*np.log(Yhat+eps) + (1-Y)*np.log(1-Yhat+eps))
        return np.mean(J)
    
    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #Output: An N by K matrix
    def gradient(self, Y, Yhat):
        eps = 1e-6
        grad = (Y - Yhat) / (Yhat * (1-Yhat) + eps) #dJ/dYhat
        
        return grad