# CS615 WI 2024

import numpy as np
from Layer import Layer
# np.random.seed(seed=1)

class FullyConnectedLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Output: sizeOut, the number of feature for the data coming out
    def __init__(self, sizeIn, sizeOut):
        #TODO
        super().__init__()
        
        self.__weights = np.random.uniform(low=-10e-4,
                                           high=10e-4,
                                           size=(sizeIn, sizeOut)
                                          )
        self.__biases = np.random.uniform(low=-10e-4,
                                          high=10e-4,
                                          size=(1, sizeOut)
                                         )
    
    #Input: None
    #Output: The sizeIn x sizeOut weight matrix
    def getWeights(self):
        return self.__weights
    
    #Input: The sizeIn x sizeOut weight matrix.
    #Output: None
    def setWeights(self, weights):
        self.__weights = weights
        
    #Input: The 1 x sizeOut bias vector
    #Output: None
    def getBiases(self):
        return self.__biases
    
    #Input: None
    #Output: The 1 x sizeOut bias vector
    def setBiases(self, biases):
        self.biases = biases
  
    
    # Input: dataIn, an NxD matrix
    # Output: An NxK matrix
    def forward(self, dataIn):
        #TODO
        self.setPrevIn(dataIn)
        
        Y = dataIn @ self.__weights + self.__biases
        
        self.setPrevOut(Y)
        return Y
    
    # Input: None
    # Output: Either an N x D matrix, or an N x (D x D) tensor
    def gradient(self):
        
        grad_tensor_shortcut = grad = self.getWeights().T # Simplified
        # N = self.getPrevIn().shape[0] # number of observations
        # grad_tensor = np.tile(grad, (N, 1)).reshape(N, grad.shape[0], grad.shape[1])
        # # return grad_tensor
        return grad_tensor_shortcut
    
    def backward(self, gradIn):
        return gradIn @ self.gradient() # faster implementation