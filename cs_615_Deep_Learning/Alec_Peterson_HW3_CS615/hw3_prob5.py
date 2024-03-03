# CS 615
# Homework 3 Problem 5 - Logistic Regression (Neural Network)
# Alec Peterson
# ap3842@drexel.edu


####----------LIBRARIES----------####
import InputLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
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
Calculate accuracy metric for binary classification
'''
def calc_accuracy(y_inp, y_pred):
    num_correct = len(np.where(y_pred == y_inp)[0])
    
    accuracy = num_correct / len(y_inp)

    return accuracy

####----------SCRIPT----------####
#Import data
np.random.seed(seed=0)
## Read in CSV, excluding first two columns
df = pd.read_csv("./KidCreative.csv")
X = df.iloc[:, 2:].to_numpy() # features start and 2nd (index=2) column
Y = df.iloc[:, 1].to_numpy().reshape(-1, 1) # 1st (index=1) column is target label

dat = np.concatenate((X, Y), axis=1)

X_train, X_val, y_train, y_val = shuffle_split(dat, 2/3)

L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(X_train.shape[1], 1)
L3 = LogisticSigmoidLayer.LogisticSigmoidLayer()
L4 = LogLoss.LogLoss()
layers = [L1, L2, L3, L4]


epochs = 0
err_change_train = 1e6
err_dict = {"epochs": [],
            "log_loss_train": [],
            "log_loss_val": []
            }
log_loss_train = 0

num_epochs = 100000

while (epochs < num_epochs and np.abs(err_change_train) > 10e-10):

    err_dict["epochs"].append(epochs)
    log_loss_train_prev = log_loss_train

    # Forward train
    h = X_train
    for i in range(len(layers) - 1):
        h = layers[i].forward(h)
    
    ## Calculate log-loss for training
    Yhat_train = L3.getPrevOut() # Logistic Sigmoid probability
    log_loss_train = L4.eval(y_train, Yhat_train) # Log-Loss calculation
    err_dict["log_loss_train"].append(log_loss_train)


    ## Save values for gradient calculation / update weights
    FCL_prev_in_train = L2.getPrevIn().copy() # Reset for updating weights later
    Sigmoid_prev_out_train = L3.getPrevOut().copy()

    # Forwards - validation (for error calculation only)
    h_val = X_val
    for i in range(len(layers) - 1):
        h_val = layers[i].forward(h_val)

    ## Calculate log-loss for validation
    Yhat_val = L3.getPrevOut()
    log_loss_val = L4.eval(y_val, Yhat_val)
    err_dict["log_loss_val"].append(log_loss_val)


    ## Reset FCL PrevIn and LogisticSigmoid PrevOut for gradient calculation
    L2.setPrevIn(FCL_prev_in_train)
    L3.setPrevOut(Sigmoid_prev_out_train)

    #backwards!
    grad = layers[-1].gradient(y_train,h)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
            
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)):
            eta = math.pow(10,-4)
            layers[i].updateWeights(grad, eta)
        
        grad = newgrad

    if epochs == 0:
        err_change_train = log_loss_train

    else:
        err_change_train = log_loss_train - log_loss_train_prev


    # if (epochs % 5000 == 0):
    #     print(f"For epoch = {epochs}")
    #     print(f"   log_loss_train = {log_loss_train: e}")
    #     print(f"   log_loss_val = {log_loss_val: e}")
    #     print(f"   log_loss_change_train is {err_change_train: e}")
    
    epochs += 1

# Make dataframe from tabulated log-loss error metrics
df_ll = pd.DataFrame(err_dict)

# Make plot of log-loss vs. epochs
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_ll["epochs"], df_ll["log_loss_train"], color="blue", linestyle="solid", label="Training Set", linewidth=2)
ax.plot(df_ll["epochs"], df_ll["log_loss_val"], color="green", linestyle="dashed", label="Validation Set", linewidth=2)

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Log-Loss", fontsize=16)
ax.set_title(f"Log-Loss vs. Epochs for Logistic Regression NN, LR = {eta}", 
             fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()

# Make class predictions for train and test
Yhat_train_class_pred = np.where(Yhat_train > 0.5, 1, 0)
Yhat_val_class_pred = np.where(Yhat_val > 0.5, 1, 0)

accuracy_train = calc_accuracy(y_train, Yhat_train_class_pred)
accuracy_val = calc_accuracy(y_val, Yhat_val_class_pred)

print(f"Accuracy for training set is: {accuracy_train: 0.4}")
print(f"Accuracy for validation set is: {accuracy_val: 0.4}")

# Show plot

plt.show()

