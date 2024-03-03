# CS 615
# Homework 3 Problem 2 - Plotting Gradient Descent
# Alec Peterson
# ap3842@drexel.edu


####----------LIBRARIES----------####
import numpy as np
import pandas as pd


####----------FUNCTIONS----------####
def calc_J(w1, w2):
    J = (w1 - 5*w2 - 2)**2

    return J

def calc_dJdw(w1, w2):
    dJdw1 = 2*w1 - 10*w2 - 4
    dJdw2 = -10*w1 + 50*w2 + 20

    dJdw = np.array([dJdw1, dJdw2])

    return dJdw

####----------SCRIPT----------####

# Gradient descent
eta = 0.01
W = np.array([0, 0])

err_dict = {"epochs": [],
            "w1": [],
            "w2": [],
            "J": [],
            }

for epochs in range(100):
    err_dict["epochs"].append(epochs)
    err_dict["w1"].append(W[0])
    err_dict["w2"].append(W[1])

    J = calc_J(W[0], W[1])
    err_dict["J"].append(J)

    W = W - eta * calc_dJdw(W[0], W[1])

df = pd.DataFrame(err_dict)


# Plot 3D line
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

ax = plt.figure(figsize=(12, 8)).add_subplot(projection="3d")


X = df.loc[:, ["w1","w1"]].to_numpy()
Y = df.loc[:, ["w2", "w2"]].to_numpy()
Z = df.loc[:, ["J", "J"]].to_numpy()

ax.plot_surface(X, Y, Z,
                edgecolor="royalblue",
                lw=1,
                )

ax.set_xlabel("w1", fontsize=14)
ax.set_ylabel("w2", fontsize=14)
ax.set_zlabel("J", fontsize=14)

plt.show()
