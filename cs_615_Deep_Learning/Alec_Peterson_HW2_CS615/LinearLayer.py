# CS615 WI 2024

import numpy as np
from Layer import Layer

class LinearLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        #TODO
        super().__init__()
  
    
    # Input: dataIn, an NxK matrix
    # Output: An NxK matrix
    def forward(self, dataIn):
        #TODO
        self.setPrevIn(dataIn)
        
        Y = dataIn
        
        self.setPrevOut(Y)
        return Y
    
    # Input: None
    # Output: Either an N x D matrix, or an N x (D x D) tensor
    def gradient(self):
        N = self.getPrevOut().shape[0] # number of observations

        K = self.getPrevOut().shape[1]
        # grad = np.identity(K)

        # grad_tensor = np.tile(grad, (N, 1)).reshape(N, grad.shape[0], grad.shape[1])

        
        grad_tensor_shortcut = np.ones((N, K))
        
        return grad_tensor_shortcut
    
    def backward(self, gradIn):
        return gradIn * self.gradient()