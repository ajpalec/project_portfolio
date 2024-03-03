# CS615 WI 2024

import numpy as np
from Layer import Layer
# np.random.seed(seed=1)

class ConvolutionalLayer(Layer):
    # Input: sizeIn, the number of features of data coming in
    # Output: sizeOut, the number of feature for the data coming out
    def __init__(self, kernelHeight, kernelWidth):
        super().__init__()
        

        self.__kernel = np.random.uniform(
            low=-10e-4,
            high=10e-4,
            size=(kernelHeight, kernelWidth)
        )

    #Input: None
    #Output: The kernel matrix
    def getKernel(self):
        return self.__kernel
    
    #Input: The kernel matrix
    #Output: None
    def setKernel(self, kernel):
        self.__kernel = kernel

    def crossCorrelate2D(self, dataIn, matrixIn):
        X = dataIn

        H, W = X.shape # image of of height H and width W
        M_rows, M_cols = matrixIn.shape # Kernel or dJ/dF

        # Create empty feature map of output size
        Y = np.zeros((H - M_rows + 1, W - M_cols + 1))

        # Replace zeros with sliding window dot product at each location
        for i in range(H - M_rows + 1):
            for j in range(W - M_cols + 1):
                Y[i, j] = np.sum(X[i:i+M_rows, j:j+M_cols] * matrixIn)

        return Y

    # Input: dataIn, an H x W matrix
    # Output: an M x M (typically) feature map
    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        K = self.__kernel
        
        Y = self.crossCorrelate2D(dataIn, K)
        
        self.setPrevOut(Y)
        return Y
    
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass

    def updateWeights(self, gradIn, eta=0.0001):
    
        dJdK = self.crossCorrelate2D(self.getPrevIn(), gradIn)

        self.__kernel -= eta * dJdK