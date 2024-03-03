'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 6 - Probabilistic Models
Problem 2, Naive Bayes Classifier
'''
import numpy as np

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
Calculate probability of label in the array
'''
def calc_prob(arr, label):
    filt_ind = ind_where_true(arr, label)
    
    num_true = len(arr[filt_ind])
    
    return num_true / len(arr)

'''
Calculate probability of feature value @x_label in feature array @x_arr given that class label @y_label is True 
'''
def prob_x_given_y(x_arr, y_arr, x_label, y_label):
    ind_y = ind_where_true(y_arr, y_label)
    ind_x = ind_where_true(x_arr, x_label)
    common_ind = list(set(ind_y).intersection(set(ind_x)))
    
    y_filt_arr = y_arr[ind_y]
    x_filt_arr = x_arr[common_ind]
    
    p_x_given_y = len(x_filt_arr) / len(y_filt_arr)
    
    return p_x_given_y

'''
Calculate log of numerator, proportional to probability of class label given evidence for Naive Bayes
'''
def calc_num_prob_y_given_features(X_mat, y_arr, x_labels, y_label):
    tol = 1e-12 # avoid log zero
    cum_log_prob = 0
    for i in range(len(x_labels)):
        cum_log_prob += np.log(
            prob_x_given_y(X_mat[:, i], y_arr, x_labels[i], y_label) + tol
        )

    log_of_num = np.log(calc_prob(y_arr, y_label)) + cum_log_prob
    
    return log_of_num


'''
Given @sample array with same features as @X_mat, determine most likely class label based on highest log of numerator in Naive Bayes classifier formula (calls calc_num_prob_y_given_features())
'''
def pred_class_sample(X_mat, y_arr, sample):
    y_uniq = np.unique(y_arr)
    prob_num_lst = []
    
    for y_label in y_uniq:
        log_prob_class_num = calc_num_prob_y_given_features(X_mat, y_arr, list(sample), y_label)
        prob_num_lst.append(log_prob_class_num)
        
    class_pred_ind = np.argmax(prob_num_lst)
    class_pred = y_uniq[class_pred_ind]
    
    return class_pred

'''
Predict class labels for alll samples in @X_inp through Naive Bayes classifier given training set @X_tr, training labels @y_tr
'''
def pred_class(X_tr, y_tr, X_inp):
    
    y_pred = [pred_class_sample(X_tr, y_tr, sample) for sample in X_inp]
    
    return np.array(y_pred)
    

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

# Predict on validation set, print error metrics
y_val_pred = pred_class(X_train_bin, y_train, X_val_bin)

cf_val = confusion_matrix(y_val, y_val_pred)
acc_val = calc_accuracy(cf_val, y_val_pred)

print()
print(f"Accuracy of Naive Bayes Classifier for validation set (rounded) is {acc_val: 0.3f}")

print()
print("Confusion matrix for validation set is:")
print(cf_val)
