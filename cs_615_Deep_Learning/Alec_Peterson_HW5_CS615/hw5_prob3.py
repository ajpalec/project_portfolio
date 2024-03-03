import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Generate data
## Vertical white stripe
im0 = np.zeros((40, 40))
im0[:, 23] = 1

## Horizontal white stripe
im1 = np.zeros((40, 40))
im1[10, :] = 1

X = np.stack([im0, im1])

y = np.array([[[0]],
              [[1]]])

# 2) Train neural network
from ConvolutionalLayer import ConvolutionalLayer
from MaxPoolLayer import MaxPoolLayer
from FlatteningLayer import FlatteningLayer
from FullyConnectedLayer import FullyConnectedLayer
from LogisticSigmoidLayer import LogisticSigmoidLayer
from LogLoss import LogLoss


np.random.seed(0)
L1 = ConvolutionalLayer(kernelHeight=9, kernelWidth=9) # output: N x 32 x 32
L2 = MaxPoolLayer(poolsize=4, stride=4) # output: N x 8 x 8
L3 = FlatteningLayer() # output: N x 1 x (8x8) = N x 1 x 64
L4 = FullyConnectedLayer(sizeIn=64, sizeOut=1) # output: N x 1 x 1
L5 = LogisticSigmoidLayer() # output: N x 1 x 1
L6 = LogLoss() # output: N x 1 x 1

initial_kernel = L1.getKernel().copy()
print(f"Initial kernel is: \n{initial_kernel}")
print()

layers = [L1, L2, L3, L4, L5, L6]

epochs = 20
eta = 0.5
logloss_lst = []


for epoch in range(epochs):
    logloss_samples = []
    for n in range(X.shape[0]):
        h = X[n]
        for i in range(len(layers) - 1):
            h = layers[i].forward(h)

        logloss = layers[-1].eval(y[n], h)
        logloss_samples.append(logloss) # average these after all samples processed for an epoch

        grad = layers[-1].gradient(y[n], h)

        for i in range(len(layers)-2,-1,-1):
            newgrad = layers[i].backward(grad)
                
            if(isinstance(layers[i],FullyConnectedLayer)):
                layers[i].updateWeights(gradIn = grad, t=epochs+1, eta=eta, 
                                        # optimizer="Adam",
                                        optimizer="GD"
                                        )

            if(isinstance(layers[i],ConvolutionalLayer)):
                layers[i].updateWeights(gradIn = grad, eta=eta)
            
            grad = newgrad

    logloss_lst.append(np.mean(logloss_samples))


graph_dict = {"epochs": [epoch for epoch in range(epochs)], 
              "logloss": logloss_lst}

final_kernel = L1.getKernel().copy()
print(f"Final kernel is: \n{final_kernel}")

df_graph = pd.DataFrame(graph_dict)

# Make plot of log-loss vs. epochs

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_graph["epochs"], df_graph["logloss"], color="blue", linestyle="solid", label="Log Loss Error", linewidth=2, marker="o")

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Log Loss", fontsize=16)
ax.set_title(f"Log Loss vs. Epochs for Synthetic Images, $\eta$ = {eta}", 
             fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()
plt.show()