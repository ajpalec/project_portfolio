#main.py
import InputLayer, FullyConnectedLayer, LogisticSigmoidLayer
import numpy as np
import pandas as pd

#Import data
np.random.seed(seed=1)
## Read in CSV, excluding first two columns
X = (pd.read_csv("./KidCreative.csv")
     .iloc[:, 2:]
     .to_numpy()
    )


# Given input X
L1 = InputLayer.InputLayer(X)
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1], 1)
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
layers = [L1, L2, L3]

#forwards!
h = X
for i in range(len(layers)):
    h = layers[i].forward(h)
    

Yhat = h

print("Output result for first observation is:")
print(Yhat[0])
print()
print("Total output result is:")
print(Yhat)