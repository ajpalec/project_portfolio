import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

np.random.seed(0)
from ConvolutionalLayer import ConvolutionalLayer
from MaxPoolLayer import MaxPoolLayer
from FlatteningLayer import FlatteningLayer
from FullyConnectedLayer import FullyConnectedLayer
from SoftmaxLayer import SoftmaxLayer
from CrossEntropy import CrossEntropy

###-------------------------FUNCTIONS-------------------------###
'''
Take image from input filepath string, a 40x40 array based on pixel values
'''

def process_image(filepath):
    from PIL import Image
    import numpy as np
    
    im = Image.open(filepath)
    im_resized = im.resize((40, 40))
    
    return np.array(im_resized)

"""
One-hot encode labels present in @y_labels, giving a column for each
"""
def one_hot_encode(y_labels, num_classes):
    encoded_labels = np.zeros((len(y_labels), num_classes))
    for i, y_label in enumerate(y_labels):
        encoded_labels[i, y_label] = 1
    return encoded_labels

###-------------------------SCRIPT-------------------------###

# 1) Load and resize images, create labels
## Filenames of "happy" face images
happy_filenames = sorted([filename for filename in os.listdir("./yalefaces") if re.search("(happy)", filename)])

X = []
for name in happy_filenames:
    filepath = f"./yalefaces/{name}"
    X.append(process_image(filepath))

X = np.array(X)
X_norm = X / 255 # "Normalize" pixel values

## Create y labels
y_labels = [i for i in range(len(happy_filenames))]
y_labels_encoded = one_hot_encode(y_labels, len(y_labels))

# 2) Create Neural Network
L1 = ConvolutionalLayer(kernelHeight=9, kernelWidth=9) # output: N x 32 x 32
L2 = MaxPoolLayer(poolsize=4, stride=4) # output: N x 8 x 8
L3 = FlatteningLayer() # output: N x 1 x (8x8) = N x 1 x 64
L4 = FullyConnectedLayer(sizeIn=64, sizeOut=14) # output: N x 1 x 14
L5 = SoftmaxLayer() # output: N x 14 x 14
L6 = CrossEntropy()

initial_kernel = L1.getKernel().copy()
print(f"Initial kernel is: \n{initial_kernel}")
print()
layers = [L1, L2, L3, L4, L5, L6]

epochs = 20
eta = 0.001
cross_entropy_lst = []

for epoch in range(epochs):
    cross_entropy_samples = []

    for i in range(X_norm.shape[0]):
        h = X_norm[i]
        for i in range(len(layers) - 1):
            h = layers[i].forward(h)

        cross_entropy = layers[-1].eval(y_labels_encoded[i], h)
        cross_entropy_samples.append(cross_entropy) # average these after all samples processed for an epoch

        grad = layers[-1].gradient(y_labels_encoded[i], h)

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
    cross_entropy_lst.append(np.mean(cross_entropy_samples))


graph_dict = {"epochs": [epoch for epoch in range(epochs)], 
              "cross_entropy_loss": cross_entropy_lst}

final_kernel = L1.getKernel().copy()
print(f"Final kernel is: \n{final_kernel}")

df_graph = pd.DataFrame(graph_dict)

# Make plot of log-loss vs. epochs

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_graph["epochs"], df_graph["cross_entropy_loss"], color="green", linestyle="solid", label="Cross Entropy Error", linewidth=2, marker="o")


ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Cross Entropy Loss", fontsize=16)
ax.set_title(f"Cross Entropy Loss vs. Epochs for Happy Face Images, $\eta$ = {eta}", 
             fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()
plt.show()