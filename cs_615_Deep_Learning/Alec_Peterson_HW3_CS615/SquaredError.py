import numpy as np
class SquaredError():

    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #  Where N can be any integer >=1
    #Output: A single floating point value.
    def eval(self, Y, Yhat):
        J = (Y - Yhat) * (Y - Yhat)
        return np.mean(J)
    
    #Input: Y is an N x K matrix of target values.
    #Input: Yhat is an N x K matrix of estimated values.
    #Output: An N by K matrix
    def gradient(self, Y, Yhat):
        grad = -2 * (Y - Yhat) # dJ/dYhat
        return grad