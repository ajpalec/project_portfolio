# CS615 WI 2024

import numpy as np
from Layer import Layer

class LogisticSigmoidLayer(Layer):
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
        
        Y = 1 / (1 + np.exp(-dataIn))
        
        self.setPrevOut(Y)
        
        return Y
    
    # Input: None
    # Output: Either an N x D matrix, or an N x (D x D) tensor
    def gradient(self):
        gz = self.getPrevOut()
        grad = gz * (1 - gz)
        
#         grad_tensor = np.array(
#             [np.diag(row) for row in grad]
#         )
        
#         return grad_tensor
        
        grad_tensor_shortcut = grad
        
        return grad_tensor_shortcut
    
    def backward(self, gradIn):
        return gradIn * self.gradient() # faster implementation