'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 5 - Decision Trees
Problem 2, Decision Tree
'''

###-------------------------CLASSES-------------------------###
'''
Node data structure for binary search tree i.e. only has left and right children. Feature attribute is feature index, while label is string label for examination purposes.
'''
class Node:
    def __init__(self, feature, label):
        self.left = None
        self.right = None
        self.feature = feature # used to initialize
        self.class_estimate = None
        self.label = label
        
    def get_class(self):
        return self.class_estimate

###-------------------------FUNCTIONS-------------------------###
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
For input continuous single-dimensional array @arr, turn data into binary categorical data (as integer) based on mean of array. If value is < array mean, make 0 else make 1.

Intended for training data pre-processing.

Returns 1-dimensional array to enable concatenation.
'''
def binarize_tr(arr):
    arr_mean = np.mean(arr)
    
    binary_arr = np.where(arr < arr_mean, 0, 1).reshape(-1,1)
    
    return binary_arr

'''
For input continuous single-dimensional array @arr_val, turn data into binary categorical data (as integer) based on mean of array from training set @tr_mean. If value is < training array mean, make 0 else make 1.

Intended for validation data pre-processing.

Returns 1-dimensional array to enable concatenation.
'''
def binarize_val(arr_val, tr_mean):
    binary_arr_val = np.where(arr_val < tr_mean, 0, 1).reshape(-1, 1)
    
    return binary_arr_val

'''
Function for identifying indices for elements of input 1-dimensional array @arr that are equal to input @val.

Returns list of indices that can be later passed on to filter down other arrays (e.g. if @arr is class labels y, use to filter down y and corresponding values for given label in X).
'''
def ind_where_true(arr, val):
            
    ind_lst = [i for i in range(len(arr)) if arr[i] == val]
            
    return ind_lst

'''
Calculate weight-averaged entropy for input feature array @feat_arr and class labels Y.

1) Calculates H of each unique input category of X from given probabilities of those categories for Y. For log calculations, uses tolerance of 1e-9 to prevent log(0) errors.

2) Calculate weight-averaged entropy accordingly by scaling H_i by proportion of class labels. Intermediate entropy values stored in dictionary before being summed.

Returns weight-averaged entropy for the feature.
'''

def calc_E(feat_arr, Y):
    tol = 1e-9
    E_dict = {} # dictionary to track terms summed for weight-averaged entropy calculation
    
    x_unique = np.unique(feat_arr)
    y_unique = np.unique(Y)
    
    for val in x_unique:
        filt_ind = ind_where_true(feat_arr, val)
        
        Y_filt = Y[filt_ind]
        
        H = 0
        for label in y_unique:
            prob_label = len(Y_filt[Y_filt == label]) / len(Y_filt) # Probability of label in filtered Y corresponding to x_unique
            H += -(prob_label) * np.emath.logn(len(y_unique), prob_label+tol)

        E_dict[val] = (len(Y_filt) / len(Y)) * H
            
    E_feat = np.array(list(E_dict.values())).sum()
    
    return E_feat

'''
Given input training matrix @X, class labels @Y, and list of feature indices @features, determine the best feature for decision tree algorithm based on lowest weight-averaged entropy.

Returns index of "best" attribute.
'''
def choose_attribute(X, Y, features):
    comparison_dict = {}
    for feat_ind in features:
        E_feat = calc_E(X[:, feat_ind], Y)
        comparison_dict[feat_ind] = E_feat
    
    min_E = np.min(list(comparison_dict.values()))
    
    best = [feat_ind for feat_ind, E_feat in comparison_dict.items() if E_feat == min_E][0]
    
    return best       

'''
Calculates mode, or most frequently occuring value in input @arr.
'''
def mode(arr):
    val_uniq, val_counts = np.unique(arr, return_counts=True)
    
    max_ind = np.where(val_counts == np.max(val_counts))
    mode = val_uniq[max_ind][0]
    
    return mode

'''
Estimates class label based on highest probability (proportion) of input list of class labels @Y.
'''

def estimate_class(Y):
    val_uniq, val_counts = np.unique(Y, return_counts=True)
    
    max_prob = -1
    prob_dict = {}
    for i in range(len(val_uniq)):
        prob_val = val_counts[i] / len(Y)
        prob_dict[val_uniq[i]] = prob_val
        
        if prob_val > max_prob:
            max_prob = prob_val
    
    for val_uniq in prob_dict:
        if prob_dict[val_uniq] == max_prob:
            return val_uniq

'''
Recursive algorithm for decision tree learning.

Base cases create leaf nodes that give class estimates. 
a) Gives @default if no more samples in @X
b) Gives only remaining class label if @Y only has a single class label left
c) Gives highest probablity class in @Y if no more features are available.

Otherwise, determines best feature (@best) to evaluate in decision nodes based on lowest weight-averaged entropy of remaining features in @features. Makes left child branch where best feature is True, and makes right child branch where best feature is False. Reduces feature list to evaluate (removes @best from available indices) and recursively calls function until base cases are reached.
'''
def myDT(X, Y, features, default):
    if len(X) == 0:
        leaf = Node(None, "leaf - default")
        leaf.class_estimate = default
        return leaf
    
    elif len(np.unique(Y)) == 1:
        leaf = Node(None, "leaf - Y remain")
        leaf.class_estimate = np.unique(Y)[0]
        return leaf
    
    elif len(features) == 0:
        leaf = Node(None, "leaf - feat_done")
        leaf.class_estimate = estimate_class(Y)
        class_estimate = estimate_class(Y)
        return leaf
    
    else:
        best = choose_attribute(X, Y, features)
        tree = Node(best, "decision")
        
        T_filt = ind_where_true(X[:, best], 1)
        X_T, Y_T = X[T_filt, :], Y[T_filt], 
        
        sel = [feat_ind for feat_ind in features if feat_ind != best]
        left_child = myDT(X_T, Y_T, sel, mode(Y))
        tree.left = left_child
        
        
        F_filt = ind_where_true(X[:, best], 0)
        X_F, Y_F = X[F_filt, :], Y[F_filt]
        right_child = myDT(X_F, Y_F, sel, mode(Y))
        tree.right = right_child
        
        return tree

'''
For input binary categorical X sample array @sample_arr, recursively evalutes binary categorical feature values against decision tree @tree.

Evaluates each feature against node in tree. If feature is True (1) proceeds left, otherwise proceeds right.

Continues until base case is reached where class estimate is present in Node (i.e. only for leaf node).

Returns predicted class label for sample.
'''
def pred_sample(sample_arr, tree):
    if tree.class_estimate != None:
        return tree.get_class()
    else:
        if sample_arr[tree.feature] == 1:
            result = pred_sample(sample_arr, tree.left)
        else:
            result = pred_sample(sample_arr, tree.right)
            
    return result


'''
Iterates through samples in pre-processed (binary categorical) @X and gets predictions from decision tree @tree. Calls pred_sample() for each sample.

Returns array of predicted class labels.
'''
def pred_from_X(X, tree):
    y_pred = np.array([pred_sample(sample, tree) for sample in X])
    
    return y_pred

'''
Returns confusion matrix based on actual class labels @y and prediction class labels @y_pred.

For multi-class classification for n classes, returns n x n matrix.

Diagonals contain counts of where actual label == predicted label. Off-diagonal elements are counts of where predicted label does not match actual label.

Read as Actual on y-axis, Predicted on x-axis.
'''
def confusion_matrix(y, y_pred):

    uniq_labels = list(np.unique(y))
    
    confusion_matrix = []
    for act_label in uniq_labels:
        confusion_row = []
        
        filt_ind = ind_where_true(y, act_label)
        y_pred_filt = y_pred[filt_ind]
        for pred_label in uniq_labels:
            count = 0
            for val in y_pred_filt:
                if val == pred_label:
                    count += 1
            confusion_row.append(count)
        confusion_matrix.append(confusion_row)
    
                
    return np.array(confusion_matrix)

'''
Given input confusion matrix @cf and predicted class labels @y_pred, calculates accuracy.

Accuracy for multi-class classification is total number of elements where predicted class label == actual class label. This can be calculated by summing counts in diagonal of confusion matrix.
'''
def calc_accuracy(cf, y_pred):
    
    acc = np.diag(cf).sum() / len(y_pred)
    
    return acc

###-------------------------SCRIPT-------------------------###

# Load data
import numpy as np
dat_filepath = "./CTG.csv"

dat_raw = np.genfromtxt(dat_filepath, 
                    dtype=float,
                    delimiter=",",
                    skip_header=2,
                    # names=True,
                    encoding="utf8")


cols_select = [i for i in range(dat_raw.shape[1])]
cols_select.remove(dat_raw.shape[1]-1 - 1) # remove 'CLASS' column per assignment instructions

dat = dat_raw[:, cols_select]

# Shuffle and Split data
X_train, X_val, y_train, y_val = shuffle_split(dat, 2/3)

# Make continuous data into binary categorical

## Training data
### Apply binarize_tr() function to all columns
X_train_bin = np.apply_along_axis(func1d=binarize_tr, axis=0, arr=X_train)[:, 0, :] # Output is 3D, selecting appropriate columns to get back to original 2D

## Validation data
### Initialize using first column (0-index), then build rest of columns
X_tr_means = np.mean(X_train, axis=0)
X_val_bin = binarize_val(X_val[:, 0], X_tr_means[0])

for i in range(1, X_val.shape[1]):
    bin_arr = binarize_val(X_val[:, i], X_tr_means[i])
    X_val_bin = np.concatenate((X_val_bin, bin_arr), axis=1)
    
# Train decision tree
feature_indices = [col for col in range(X_train_bin.shape[1])]
DT = myDT(X_train_bin, y_train, feature_indices, mode(y_train))

# Make predictions on validation set, then evaluate confusion matrix and accuracy of validation set
y_val_pred = pred_from_X(X_val_bin, DT)

cf_val = confusion_matrix(y_val, y_val_pred)

acc_val = calc_accuracy(cf_val, y_val_pred)

print(f"Confusion matrix for validation set:\n {cf_val}")
print(f"Accuracy for validation set (rounded): {acc_val: .2f}")
