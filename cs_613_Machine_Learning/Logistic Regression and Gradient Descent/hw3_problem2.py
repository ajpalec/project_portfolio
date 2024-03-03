'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Linear Regression
Problem 2, Closed Form Linear Regression
'''

import numpy as np
import matplotlib.pyplot as plt
###---------------FUNCTIONS---------------###
'''
Shuffle input data matrix @dat, splits into training set of proportion @train_size_prop and remainder into validation set.

Assumes all but last column are continuous variables, and last column is class labels.
'''

def shuffle_split(dat, train_size_prop):
    np.random.seed(seed=0)
    dat_shuffled = dat.copy()
    np.random.shuffle(dat_shuffled)
    
    # split
    train_size_last_index = int(train_size_prop * dat_shuffled.shape[0])
    X_shuffled_train = dat_shuffled.copy()[:train_size_last_index, :-1]
    y_shuffled_train = dat_shuffled.copy()[:train_size_last_index, -1]
    
    X_shuffled_test = dat_shuffled.copy()[train_size_last_index:, :-1]
    y_shuffled_test = dat_shuffled.copy()[train_size_last_index:, -1]
    
    return X_shuffled_train, X_shuffled_test, y_shuffled_train, y_shuffled_test

'''
Mean-center data by subtracting column mean from values in each column of input matrix X
'''
def mean_center(X):
    X_mean = np.mean(X, axis=0)
    
    X_mc = X - X_mean
    
    return X_mc

'''
Mean-center then make unit standard deviation (divide values in each mean-centered column by standard deviation)

The standard deviation for a field could be all 0's, and so dividing by this would give np.nan (Not a number).
np.nan_to_num() would turn these nan's to 0's, avoiding calculation problems
'''

def zscore(X):
    X_mc = mean_center(X)
    
    X_zscore = X_mc / np.std(X, axis=0, ddof=1)
    
    return np.nan_to_num(X_zscore)

'''
Adds onto matrix a dummy column of 1's. Used to create dummy "bias" feature
'''
def add_bias(X):
    b = np.ones(X.shape[0]).reshape(-1,1)
    
    X_b = np.concatenate((X, b), axis=1)
    
    return X_b

'''
Pre-processes data as required to create input data matrix X

1) Z-scores data
2) Adds bias feature
'''
def pre_process(X):
    X_numeric_scaled = zscore(X.astype(float))
    
    X_processed_b = add_bias(X_numeric_scaled)
    
    return X_processed_b

'''
Calculate w coefficients for logistic regresion
'''
def calc_y_pred(X, w):
    y_pred = 1 / (1 + np.exp(np.dot(-X, w)))
    
    return y_pred

'''
Calculate derivative of loss function for logistic regression
'''
def calc_deriv(X, y, y_pred):
    N = X.shape[0]
    dJ_dw = (1/N)*(X.T @ (y_pred - y))
    
    return dJ_dw

'''
Calculate mean log loss (J) given reference set @y and prediction set @y_pred
'''
def calc_mean_log_loss(y, y_pred):
    tol = 10e-6
    # Dot products sum up over all samples. Adding together is equivalent to iterating through and performing same formula for each set of samples.
    J = -(y@np.log(y_pred+tol) + (1-y)@np.log(1 - y_pred+tol))
    
    return J / len(y)

'''
Train logistic regression through iterative gradient descent for learning rate @eta, keeping track of mean log-loss (J) and change in mean log-loss for both training set and validation set. Coefficients w are calculated using pre-processed training data @X_train_pp and model is updated until mean log-loss change reaches @convergence_criteria.

Return predicted probabilities for train set, validation set, as well as data related to training the model e.g. J, change in J, and epochs for both training and validation set.
'''
def train_logistic_regression(X_train_pp, y_train, X_val_pp, y_val, eta, convergence_criteria):
    
    data = []
    w = np.ones(X_train_pp.shape[1]) # initialize
    J_train = 0
    J_val = 0
    J_train_change = 1000
    epochs=0
    
    while np.abs(J_train_change) > convergence_criteria:
        J_last_train = J_train
        J_last_val = J_val
        
        y_pred_train = calc_y_pred(X_train_pp, w)
        y_pred_val = calc_y_pred(X_val_pp, w)
        
        J_train = calc_mean_log_loss(y_train, y_pred_train)
        J_val = calc_mean_log_loss(y_val, y_pred_val)
        
        dJ_train_dw = calc_deriv(X_train_pp, y_train, y_pred_train)
        w = w + eta*(-dJ_train_dw)

        J_train_change = (J_train - J_last_train)
        J_val_change = (J_val - J_last_val)
        epochs += 1
        
        data.append([eta, 
                     epochs, 
                     J_train, 
                     J_val, 
                     J_train_change, 
                     J_val_change
                    ])
        
        
    return y_pred_train, y_pred_val, np.array(data)

'''
Given class labels @y and class label predictions @y_pred, calculate error metrics: Precision, Recall, F-measure (F1) and Accuracy.

Only works for binary classification.
'''
def classification_error_metrics(y, y_pred):
    TP_count = 0
    FP_count = 0
    TN_count = 0
    FN_count = 0
    for i in range(len(y_val)):
        if y[i] == 1:
            # Both are 1 i.e. true positive
            if y_pred[i] == y[i]:
                TP_count += 1
            # y_pred is 0 if not equal, thus a false negative
            else:
                FN_count +=1
        else:
            # Both are 0 i.e. true negative
            if y_pred[i] == y[i]:
                TN_count += 1
            # y_pred is 1 if not equal, thus a false positive
            else:
                FP_count +=1
                
    # Calculate Error Metrics
    precision = TP_count / (TP_count + FP_count)
    recall = TP_count / (TP_count + FN_count)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP_count + TN_count) / len(y)
    
    return {"Precision": precision, "Recall": recall, "F1": f1, "Accuracy": accuracy}

###---------------SCRIPT---------------###
# Load data
import numpy as np
import pandas as pd

dat_filepath = "./spambase.data"

dat = np.genfromtxt(dat_filepath, 
                              dtype=float,
                              delimiter=",", 
                              # names=True,
                              encoding="utf8")

# Shuffle split, pre-process data
X_train, X_val, y_train, y_val = shuffle_split(dat, 2/3)

X_train_pp = pre_process(X_train)
X_val_pp = pre_process(X_val)

eta = 10

y_pred_train, y_pred_val, model_data = train_logistic_regression(X_train_pp, y_train, X_val_pp, y_val, eta=eta, convergence_criteria=0.001)

# Make plot of log-loss vs. epochs
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(model_data[:, 1], model_data[:, 2], color="blue", marker="o", linestyle="solid", label="Training Set");
ax.plot(model_data[:, 1], model_data[:, 3], color="green", marker="o", linestyle="dashed", label="Validation Set");

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Mean Log-Loss", fontsize=16)
ax.set_title(f"Mean Log-Loss vs. Epochs for Logistic Regression Model, Learning Rate = {eta}", 
             fontsize=18)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()

plt.show()

# Make classification predictions
y_pred_classes_val = (y_pred_val >= 0.50).astype(int)

error_metrics_dict = classification_error_metrics(y_val, y_pred_classes_val)

for key in error_metrics_dict:
    print(f"{key} = {error_metrics_dict[key]: .3f}")
