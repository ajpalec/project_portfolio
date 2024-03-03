# CS615 WI 2024

import numpy as np
from Layer import Layer

class ReLULayer(Layer):
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
        
        # If less than 0, set to 0, otherwise keep the same
        Y = np.where(dataIn < 0, 0, dataIn)
        
        self.setPrevOut(Y)
        
        return Y
    
    # Input: None
    # Output: Either an N x D matrix, or an N x (D x D) tensor
    def gradient(self):
        
        # Make tensor with N x (D x D) matrices
        # Iterate over each row (N rows) from input data
        # Make binary array - gradient is 0 where input is negative, else 1 where input is positive
        # Make a square (D x D) matrix with this binary array along the diagonal
        # Tensor is made up of these D x D matrices
        
        z = self.getPrevIn()
        # grad_tensor = np.array(
        #     [np.diag(np.where(row < 0, 0, 1)) for row in z]
        # )
        # return grad_tensor
        
        grad_tensor_shortcut = np.array([np.where(row < 0, 0, 1) for row in z])
        return grad_tensor_shortcut
    
    def backward(self, gradIn):
        return gradIn * self.gradient() # faster implementation