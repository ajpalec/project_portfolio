'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 4 - Linear Discriminant Analysis

'''

import numpy as np

###---------------FUNCTIONS---------------###
'''
Shuffle input data matrix @dat, splits into training set of proportion @train_size_prop and remainder into validation set.

Assumes all but last column are continuous variables, and last column is class labels.
'''

def shuffle_split(dat, train_size_prop):
    from math import ceil
    
    np.random.seed(seed=0)
    dat_shuffled = dat.copy()
    np.random.shuffle(dat_shuffled)
    
    # split
    train_size_last_index = ceil(train_size_prop * dat_shuffled.shape[0])
    X_shuffled_train = dat_shuffled.copy()[:train_size_last_index, :-1]
    y_shuffled_train = dat_shuffled.copy()[:train_size_last_index, -1]
    
    X_shuffled_test = dat_shuffled.copy()[train_size_last_index:, :-1]
    y_shuffled_test = dat_shuffled.copy()[train_size_last_index:, -1]
    
    return X_shuffled_train, X_shuffled_test, y_shuffled_train, y_shuffled_test

'''
The standard deviation for a field could be all 0's, and so dividing by this would give np.nan (Not a number).
np.nan_to_num() would turn these nan's to 0's, avoiding calculation problems
'''

def zscore(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0, ddof=1)
    
    X_mc = X - X_mean
    
    X_zscore = X_mc / X_std
    
    return np.nan_to_num(X_zscore)


'''
(Written for scalability to multiple classes / beyond binary classification)

Determines unique class labels from @Y_total
For each unique class label, finds indices in dataset corresponding to sample with that class label.

Stores in dictionary @label_index_dict, with each key as label and each array of indices as value.

Returns @label_index_dict for future use/filtering of corresponding X data matrices.
'''
def get_label_indices(Y, Y_total):
    unique_labels = np.unique(Y_total)
    
    label_index_dict = {}
    for label in unique_labels:
        label_index_dict[label] = np.where(Y == label)[0]
    
    return label_index_dict

'''
(Written for scalability to multiple classes / beyond binary classification)

Using indices corresponding to class label keys in @label_index_dict, filters X down to samples corresponding to that class label (via indices)

Stores in dictionary @X_dict, with label as key and corresponding filtered data matrix as value.
'''
def get_data_for_labels(X, label_index_dict):
    X_dict = {}
    for label in label_index_dict:
        X_dict[label] = X[label_index_dict[label], :]
        
    return X_dict

'''
Given dictionary with class labels as keys and filtered corresponding data matrices as values, calculates "closed-form" linear discriminant analysis coefficients @w.

Suited for binary classification.
1) Means from each label (0, 1) stored in list @means.
2) Covariance matrix for each label (0, 1) stored in list @covs.
3) Computes eigenvalues and eigenvectors within-class and between-class scatter matrices.
4) Selects eigenvector using corresponding index of non-zero eigenvalue.
'''
def LDA(X_by_label_dict):
    means = [np.mean(X_by_label_dict[label], axis=0).reshape(1,-1) for label in X_by_label_dict]
    covs = [np.cov(X_by_label_dict[label].T) for label in X_by_label_dict]
    
    S_b = (means[0]-means[1]).T @ (means[0] - means[1])
    S_w = covs[0] + covs[1]
    decomp_input = np.linalg.inv(S_w) @ S_b
    
    eig_values, eig_vectors = np.linalg.eig(decomp_input)
    
    ind = int(np.where(eig_values > 0)[0][0])
    w = eig_vectors[:, ind]
    
    return np.real(w)

'''
'''


'''
Given pre-processed input dataset @X_input_pp, LDA-projected means from training dataset @X_tr_proj_means, and LDA-calculated coefficients @w
1) Projects input matrix using @w to give X_input_proj
2) Iterate through elements of X_input_proj and checks if element is closer to mean corresponding to class 0, or mean corresponding to class 1, then labels accordingly and adds predicted class to list y_pred
3) Return prediction list as type float as dataset for assignment treated all as float datatype
'''
def label_predictions(X_input_pp, X_tr_proj_means, w):
    X_input_proj = X_input_pp @ w
    y_pred = []
    
    for i in range(len(X_input_proj)):
        dist_0_mean = np.abs(X_input_proj[i] - X_tr_proj_means[0])
        dist_1_mean = np.abs(X_input_proj[i]  - X_tr_proj_means[1])
        
        if dist_0_mean < dist_1_mean:
            y_pred.append(0.0)
        else:
            y_pred.append(1.0)
            
    return np.array(y_pred).astype(float)

'''
Given class labels @y and class label predictions @y_pred, calculate error metrics: Precision, Recall, F-measure (F1) and Accuracy.

Only works for binary classification.
'''
def classification_error_metrics(y, y_pred):
    TP_count = 0
    FP_count = 0
    TN_count = 0
    FN_count = 0
    for i in range(len(y)):
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
dat = np.genfromtxt("./spambase.data", 
                              dtype=float,
                              delimiter=",", 
                              # names=True,
                              encoding="utf8")

# Shuffle split, pre-process data
X_dat_scaled = zscore(dat[:, :-1])
dat_scaled = np.concatenate((X_dat_scaled, dat[:, -1].reshape((-1,1))), axis=1)

X_train, X_val, y_train, y_val = shuffle_split(dat_scaled, 2/3)


# 1)
# Make dictionary with each class label as a key, and each value an array of indices of samples that have that training label
tr_label_index_dict = get_label_indices(y_train, dat_scaled[:, -1])

# Make dictionary with each class label as a key, and each value an array of features for samples that had that label (obtained from tr_label_index_dict)
X_by_label_dict_tr = get_data_for_labels(X_train, tr_label_index_dict)

# Calculate coefficient matrix using LDA closed-form
w = LDA(X_by_label_dict_tr)

# Calculate means from projected training values
X_tr_proj_means = [np.mean(X_by_label_dict_tr[label] @ w) for label in X_by_label_dict_tr]

# Label based on how close projected value is to means for projected training set
y_tr_pred = label_predictions(X_train, X_tr_proj_means, w)
y_val_pred = label_predictions(X_val, X_tr_proj_means, w)

# Calculate and print accuracy for training set
y_tr_pred_accuracy = classification_error_metrics(y_train, y_tr_pred)["Accuracy"]

print(f"Accuracy of LDA for training set: {y_tr_pred_accuracy: .3f}")

# Calculate and print error metrics for validation set
val_error_metrics_dict = classification_error_metrics(y_val, y_val_pred)

print()
print("Validation set classification metrics:")
for key in val_error_metrics_dict:
    print(f"   {key} = {val_error_metrics_dict[key]: .3f}")