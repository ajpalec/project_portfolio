import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import plotly.express as px

#--------------------LR FUNCTIONS--------------------#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.maximum(epsilon, y_pred)
    y_pred = np.minimum(1 - epsilon, y_pred)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient_descent(X, y, learning_rate, num_epochs):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)
    bias = 0
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        training_loss = np.mean(log_loss(y, y_pred))
        training_losses.append(training_loss)

        dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
        db = (1/num_samples) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db
        z_val = np.dot(X_val, weights) + bias
        y_pred_val = sigmoid(z_val)
        validation_loss = np.mean(log_loss(y_val, y_pred_val))
        validation_losses.append(validation_loss)

    return weights, bias, training_losses, validation_losses

def classify(X, weights, threshold=0.5):
    z = np.dot(X, weights)
    y_pred = sigmoid(z)
    return (y_pred >= threshold).astype(int)

def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, f_measure, accuracy

#--------------------ROC AND FUNCTIONS--------------------#
def calc_TPR(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # print(f"tp are {tp}")
    fn = np.sum((y_true == 1) & (y_pred == 0))
    # print(f"fn are {fn}")
    
    return tp / (tp + fn)
    

def calc_FPR(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # print(f"fp are {fp}")
    tn = np.sum((y_true == 0) & (y_pred == 0))
    # print(f"tn are {tn}")
    
    return fp / (fp + tn)


def make_roc_df(X_inp, weights_inp, y_true_inp):
    pred_dict = {"threshold": [],
                 "tpr": [],
                 "fpr": [],
                 "accuracy": [],
                 "f1-measure": [],
                 "precision": [],
                 "recall": []
                }
    
    # thresholds = [0.1, 0.4, 0.7, 0.9]
    thresholds = [i/100 for i in range(0, 100+1)] # 0 to 1, by 0.01 steps
    # print(thresholds)
    for thr in thresholds:
        pred_dict["threshold"].append(thr)
        # print(f"threshold is {thr}")
        y_pred_calc = classify(X = X_inp, 
                               weights = weights_inp, 
                               threshold = thr)
        
        tpr = calc_TPR(y_true_inp, y_pred_calc)
        pred_dict["tpr"].append(tpr)
        # print(f"TPR is {tpr}")
        fpr = calc_FPR(y_true_inp, y_pred_calc)
        pred_dict["fpr"].append(fpr)
        # print(f"FPR is {fpr}")
        precision, recall, f_measure, accuracy = evaluate(y_true_inp, y_pred_calc)
        pred_dict["accuracy"].append(accuracy)
        pred_dict["f1-measure"].append(f_measure)
        pred_dict["precision"].append(precision)
        pred_dict["recall"].append(recall)
        # print()
    
    return pd.DataFrame(pred_dict)

def calc_AUC(df_roc_inp):
    
    df_roc_inp_sorted = df_roc_inp.sort_values(["fpr"])
    
    auc = np.trapz(y=df_roc_inp_sorted["tpr"], 
                   x=df_roc_inp_sorted["fpr"],
                  )
    
    return auc

def roc_curves_LR(X_train_inp, y_train_inp, X_val_inp, y_val_inp, learning_rate_lst, num_epochs):
    
    df_graph_lst = []
    auc_dict = {"learning_rate": [],
                "AUC": []
               }
    
    for l_rate in learning_rate_lst:
        weights, bias, training_losses, validation_losses = gradient_descent(X_train_inp, 
                                                                             y_train_inp, 
                                                                             l_rate, 
                                                                             num_epochs)
        
        calculate_weight_impact(X_train_inp, weights, l_rate)
        df_graph_inp = make_roc_df(X_val_inp, weights, y_val_inp)
        df_graph_inp["learning_rate"] = l_rate
                
        df_graph_lst.append(df_graph_inp)
        
        auc_dict["learning_rate"].append(l_rate)
        auc_dict["AUC"].append(calc_AUC(df_graph_inp))
        
    df_graph = pd.concat(df_graph_lst)
    df_graph.to_csv('table_of_error_metrics.csv')
    df_auc = pd.DataFrame(auc_dict)
    
    fig = px.line(df_graph,
                  title=f"ROC for Logistic Regression with varied Learning Rates and {num_epochs} Epochs",
                  x="fpr",
                  y="tpr",
                  hover_data = ["threshold", "tpr", "fpr", "learning_rate"],
                  color="learning_rate",
                  template="plotly_white",
                  # markers=True,
                  width=800,
                  height=600,
                  labels={"tpr": "TPR", "fpr": "FPR", "learning_rate": "Learning Rate"}
                 )
    
    fig.show()
    
    
    
    
    
    return df_graph, df_auc, fig    

def calculate_weight_impact(X_train, weights, l_rate):
    # weights_array = np.array(list(zip))
    weights_df = pd.DataFrame({'Weight': weights})
    # print(X_train)
    # weights_df.to_csv('weights' + str(l_rate) + '.csv')
    # print(weights_df)

    weights_df['Absolute_Weight'] = weights_df['Weight'].abs()
    sorted_weights = weights_df.sort_values(by='Absolute_Weight', ascending=False)

    # Print the sorted DataFrame
    print(sorted_weights)
    sorted_weights.to_csv('weights' + str(l_rate) + '.csv')



# Data is shuffled

train_path = "./pre_processing/output/training_data_resampled_encoded.csv"
val_path = "./pre_processing/output/validation_data_encoded.csv"

#train_path = "training_data_resampled_encoded.csv"
#val_path = "validation_data_encoded.csv"

bank_training = pd.read_csv(train_path)
bank_validation = pd.read_csv(val_path)

# Step 4: Standardize the features
X_train = bank_training.iloc[:, :-1].values
y_train = bank_training.iloc[:, -1].values
X_val = bank_validation.iloc[:, :-1].values
y_val = bank_validation.iloc[:, -1].values

# Add a bias to X_Train and X_val
ones_column = np.ones((X_train.shape[0], 1))
X_train = np.c_[ones_column, X_train]
ones_column = np.ones((X_val.shape[0], 1))
X_val = np.c_[ones_column, X_val]

print(X_train)
df_graph, df_auc, roc_figure = roc_curves_LR(X_train, y_train, X_val, y_val, [0.01, 0.1, 1, 10], 100)

print()
print(df_auc)
