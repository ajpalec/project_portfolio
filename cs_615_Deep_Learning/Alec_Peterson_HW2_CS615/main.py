#main.py
import InputLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
import numpy as np
import pandas as pd

#Import data
np.random.seed(seed=1)
## Read in CSV, excluding first two columns
df = pd.read_csv("./KidCreative.csv")
X = df.iloc[:, 2:].to_numpy()
    

Y = df.iloc[:, 1].to_numpy().reshape(-1,1) # 2nd column (index=1) is target Y

## Given input X
L1 = InputLayer.InputLayer(X)
L2 = FullyConnectedLayer.FullyConnectedLayer(X.shape[1], 1)
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L4 = LogLoss.LogLoss()
layers = [L1, L2, L3, L4]

#forwards!
h = X
for i in range(len(layers) - 1):
    h = layers[i].forward(h)

#backwards!
grad = layers[-1].gradient(Y,h) # Gradient of log-loss

print(f"Mean gradient across observations of {layers[-1]} is:")
print(grad.mean(axis=0))
print()
      
for i in range(len(layers)-2,0,-1):
    print(f"Mean gradient across observations of {layers[i]} is:")
    grad = layers[i].backward(grad)
    print(grad.mean(axis=0))
    print()