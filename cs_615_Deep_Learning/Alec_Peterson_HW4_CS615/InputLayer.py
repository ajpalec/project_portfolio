# CS615 WI 2024

import numpy as np
from Layer import Layer
class InputLayer(Layer):
    # Input: dataIn, an NxD matrix
    # Output: None
    def __init__(self, dataIn):
        super().__init__()
        # Vector of averages for each column
        self.meanX = np.mean(dataIn, axis=0)
        
        # Vector of standard deviations for each column, where 0s turned to 1s
        # self.stdX = np.where(np.std(dataIn, axis=0, ddof=1) == 0,
        #                      1,
        #                      np.std(dataIn, axis=0, ddof=1)
        #                     )
        std_calc = np.std(dataIn, axis=0, ddof=1)
        self.stdX = np.where(std_calc == 0, 1, std_calc)
    
    # Input: dataIn, an NxD matrix
    #Output: An NxD matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        
        dataZscored= (dataIn - self.meanX) / self.stdX
        
        self.setPrevOut(dataZscored)
        
        return dataZscored
    
    # We'll worry about this later...
    def gradient(self):
        pass
    
    def backward(self, gradIn):
        pass