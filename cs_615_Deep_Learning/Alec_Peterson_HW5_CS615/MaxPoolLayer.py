# CS615 WI 2024

import numpy as np
from Layer import Layer

class MaxPoolLayer(Layer):
    # Input: None
    # Output: None
    def __init__(self, poolsize, stride):
        super().__init__()

        self.__poolsize = poolsize
        self.__stride = stride
  
    
    # Input: dataIn, an M x M matrix
    # Output: A Q x Q matrix
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        D, E = dataIn.shape # Get shape of sample
        Q = self.__poolsize
        S = self.__stride

    
        Y = np.zeros(((D - Q)//S + 1, (E - Q)//S + 1))
        for r in range(0, D - Q + 1, S):
            for c in range(0, E - Q + 1, S):
                i = r // S
                j = c // S
                Y[i, j] = np.max(dataIn[r:r+Q, c:c+Q])
        
        self.setPrevOut(Y)
        
        return Y
    
    def gradient(self):
        F_inp = self.getPrevIn()
        D, E = F_inp.shape
        Q = self.__poolsize
        S = self.__stride

        max_loc = []

        for r in range(0, D - Q + 1, S):
            for c in range(0, E - Q + 1, S):
                pool_slice = F_inp[r:r+Q, c:c+Q]
                slice_max = np.max(pool_slice)
                i, j = np.where(pool_slice == slice_max)
                max_loc.append((r + i[0], c + j[0]))
                    
        return max_loc
    
    def backward(self, gradIn):
        F_inp = self.getPrevIn()

        new_grad = np.zeros(F_inp.shape)
        max_locs = self.gradient()
        fg_elems = gradIn.reshape(1, -1)[0]
        seen = []
        for max_loc, fg_elem in zip(max_locs, fg_elems):
            if max_loc not in seen:
                seen.append(max_loc)
                new_grad[max_loc] = fg_elem

        return new_grad