# CS615 WI 2024

import numpy as np
from Layer import Layer
# np.random.seed(seed=1)

class FullyConnectedLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Output: sizeOut, the number of feature for the data coming out
    def __init__(self, sizeIn, sizeOut, XavierInit=True):
        super().__init__()
        
        if XavierInit:
            self.__weights = np.random.uniform(
                low=-np.sqrt(6/(sizeIn + sizeOut)), # Xavier initialization
                high=np.sqrt(6/(sizeIn + sizeOut)),  # Xavier initialization
                size=(sizeIn, sizeOut)
            )
            self.__biases = np.random.uniform(
                low=-np.sqrt(6/(sizeIn + sizeOut)), # Xavier initialization
                high=np.sqrt(6/(sizeIn + sizeOut)), # Xavier initialization
                size=(1, sizeOut)
            )

            self.__Xavier_sw = np.zeros((sizeIn, sizeOut))
            self.__Xavier_rw = np.zeros((sizeIn, sizeOut))
            self.__Xavier_sb = np.zeros((1, sizeOut))
            self.__Xavier_rb = np.zeros((1, sizeOut))
        else:
            self.__weights = np.random.uniform(
                low=-10e-4,
                high=10e-4,
                size=(sizeIn, sizeOut)
            )
            self.__biases = np.random.uniform(
                low=-10e-4,
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

    def getXavier(self):
        return self.__Xavier_sw, self.__Xavier_rw, self.__Xavier_sb, self.__Xavier_rb
  
    
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

    def updateWeights(self, 
                      gradIn,
                      t, # epoch
                      p_1 = 0.9,
                      p_2 = 0.999,
                      eta=0.0001,
                      delta=10e-8, 
                      optimizer="Adam"):
        
        if optimizer == "Adam":
            sw = self.__Xavier_sw
            rw = self.__Xavier_rw
            sb = self.__Xavier_sb
            rb = self.__Xavier_rb
            
            dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]
            dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]

            self.__Xavier_sw = (p_1 * sw) + (1 - p_1) * dJdW
            self.__Xavier_rw = (p_2 * rw) + (1 - p_2) * (dJdW * dJdW)

            self.__Xavier_sb = (p_1 * sb) + (1 - p_1) * dJdb
            self.__Xavier_rb = (p_2 * rb) + (1 - p_2) * (dJdb * dJdb)

            self.__weights -= eta * (self.__Xavier_sw / (1-p_1**t)) / (np.sqrt(self.__Xavier_rw / (1-p_2)**t) + delta)
            self.__biases -= eta * (self.__Xavier_sb / (1-p_1**t)) / (np.sqrt(self.__Xavier_rb / (1-p_2)**t) + delta)

        else:
            dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]

            dJdW = (self.getPrevIn().T @ gradIn)/gradIn.shape[0]

            self.__weights -= eta * dJdW
            self.__biases -= eta * dJdb
