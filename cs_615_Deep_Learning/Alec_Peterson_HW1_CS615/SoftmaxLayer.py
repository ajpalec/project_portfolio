# CS615 WI 2024

import numpy as np
from Layer import Layer

class SoftmaxLayer(Layer):
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
        
        num = np.exp(dataIn - np.max(dataIn))
        denom = (num.sum(axis=1) # sum each row
                 .reshape(dataIn.shape[0], -1) # reshape to # of rows as dataIn
                )
        
        Y = num / denom
        
        self.setPrevOut(Y)
        
        return Y
    
    # We'll worry about this later...
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass