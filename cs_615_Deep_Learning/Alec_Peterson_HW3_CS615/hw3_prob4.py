# CS 615
# Homework 3 Problem 4 - Linear Regression (Neural Network)
# Alec Peterson
# ap3842@drexel.edu
####----------LIBRARIES----------####
import InputLayer, FullyConnectedLayer, SquaredError
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


####----------FUNCTIONS----------####
'''
Shuffle data to training and validation (test) sets with training set proportion of @train_size_prop
Assumes that target labels are last column
'''
def shuffle_split(dat, train_size_prop):
    dat_shuffled = dat.copy()
    np.random.shuffle(dat_shuffled)
    
    # split
    train_size_last_index = int(train_size_prop * dat_shuffled.shape[0])
    X_shuffled_train = dat_shuffled.copy()[:train_size_last_index, :-1]
    y_shuffled_train = dat_shuffled.copy()[:train_size_last_index, -1].reshape(-1, 1)
    
    X_shuffled_test = dat_shuffled.copy()[train_size_last_index:, :-1]
    y_shuffled_test = dat_shuffled.copy()[train_size_last_index:, -1].reshape(-1, 1)
    
    return X_shuffled_train, X_shuffled_test, y_shuffled_train, y_shuffled_test

'''
Calculate symmetric mean absolute percent error
'''
def calc_SMAPE(y_inp, y_pred):
    num = np.abs(y_inp - y_pred) 
    denom = (np.abs(y_inp) + np.abs(y_pred))

    smape = np.sum(num / denom) / len(y_inp)

    return smape

####----------SCRIPT----------####
#Import data
np.random.seed(seed=0)
## Read in CSV, excluding first two columns
df = pd.read_csv("./medical.csv")
dat = df.to_numpy()

# Shuffle data
X_train, X_val, y_train, y_val = shuffle_split(dat, 2/3)

# Neural network for linear regression
L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(X_train.shape[1], 1)
L3 = SquaredError.SquaredError()

layers= [L1, L2, L3]

# Train model
epochs = 0
MSE_change_train = 1e6
MSE_dict = {"epochs": [],
            "MSE_train": [],
            "MSE_val": []
            }
MSE_train_prev = 0

num_epochs = 100000
while (epochs < num_epochs and np.abs(MSE_change_train) > 10e-10):

    MSE_dict["epochs"].append(epochs)
    if epochs > 0:
        MSE_train_prev = MSE_train

    # Forward train
    h = X_train
    for i in range(len(layers) - 1):
        h = layers[i].forward(h)
    # print("Forward on train completed")

    
    ## Calculate MSE for training
    Yhat_train = L2.getPrevOut()
    MSE_train = L3.eval(y_train, Yhat_train)
    MSE_dict["MSE_train"].append(MSE_train)

    FCL_prev_in_train = L2.getPrevIn().copy() # Reset gradient calculation later

    # Forwards - validation (for MSE calculation only)
    h_val = X_val
    for i in range(len(layers) - 1):
        h_val = layers[i].forward(h_val)

    ## Calculate MSE for validation
    Yhat_val = L2.getPrevOut()
    MSE_val = L3.eval(y_val, Yhat_val)
    MSE_dict["MSE_val"].append(MSE_val)

    ## Reset FCL PrevIn for update weights calculation
    L2.setPrevIn(FCL_prev_in_train)

    #backwards!
    grad = layers[-1].gradient(y_train,h)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
            
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)):
            eta = math.pow(10, -4)
            layers[i].updateWeights(grad, eta)
        
        grad = newgrad

    if epochs == 0:
        MSE_change_train = MSE_train

    else:
        MSE_change_train = MSE_train - MSE_train_prev


    # if (epochs % 5000 == 0):
    #     print(f"For epoch = {epochs}")
    #     print(f"   MSE_train = {MSE_train: e}")
    #     print(f"   MSE_val = {MSE_val: e}")
    #     print(f"   MSE_change_train is {MSE_change_train: e}")
    
    epochs += 1

# Plot
df_MSE = pd.DataFrame(MSE_dict)

# Make plot of log-loss vs. epochs
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_MSE["epochs"], df_MSE["MSE_train"], color="blue", linestyle="solid", label="Training Set", linewidth=2)
ax.plot(df_MSE["epochs"], df_MSE["MSE_val"], color="green", linestyle="dashed", label="Validation Set", linewidth=2)

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=16)
ax.set_title(f"Mean Squared Error vs. Epochs for Linear Regression NN, LR = {eta}", 
             fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()

# Calculate RMSE and SMAPE
RMSE_train = np.sqrt(MSE_train)
RMSE_val = np.sqrt(MSE_val)

SMAPE_train = calc_SMAPE(y_train, Yhat_train)
SMAPE_val = calc_SMAPE(y_val, Yhat_val)

print(f"Training set RMSE is {RMSE_train: 0.2f}")
print(f"Validation set RMSE is {RMSE_val: 0.2f}")

print(f"Training set SMAPE is {SMAPE_train: 0.4f}")
print(f"Validation set SMAPE is {SMAPE_val: 0.4f}")

# Show plot
plt.show()

    