# CS615 WI 2024

import numpy as np
from Layer import Layer

class TanhLayer(Layer):
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
        
        Y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
        
        self.setPrevOut(Y)
        
        return Y
    
    # We'll worry about this later...
    def gradient(self):
        gz = self.getPrevOut()
        grad_tensor_shortcut = grad = 1 - (gz**2)
        
        # grad_tensor = np.array(
        #     [np.diag(row) for row in grad]
        # )
        return grad_tensor_shortcut
    
    def backward(self, gradIn):
        return gradIn * self.gradient() # faster implementation