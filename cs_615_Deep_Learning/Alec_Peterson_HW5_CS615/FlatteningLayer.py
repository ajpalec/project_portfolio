# CS615 WI 2024

import numpy as np
from Layer import Layer

class FlatteningLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self):
        super().__init__()
  
    # Input: dataIn, an Q x Q matrix
    # Output: A 1 x (Q*Q) matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        Y = dataIn.reshape(1, -1)
        
        self.setPrevOut(Y)
        
        return Y
    
    def gradient(self):

        # Gradient reshape mostly handled by backward()
        # Just providing the shape of the input
        inp_shape = self.getPrevIn().shape  
        
        return inp_shape
    
    # Reshape incoming gradient to match the input shape
    def backward(self, gradIn):

        new_grad = gradIn.reshape(self.gradient())
  
        return new_grad