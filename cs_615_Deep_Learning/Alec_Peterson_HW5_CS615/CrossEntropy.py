import numpy as np
class CrossEntropy():

    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #  Where N can be any integer >=1
    #Output: A single floating point value.
    def eval(self, Y, Yhat):
        eps = 1e-6
        J = -np.sum(Y * np.log(Yhat + eps), axis=0)
        return np.mean(J)
    
    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #Output: An N by K matrix
    def gradient(self, Y, Yhat):
        eps = 1e-6
        grad = -(Y / (Yhat + eps))
        return grad