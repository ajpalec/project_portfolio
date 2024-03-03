'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 6 - Probabilistic Models
Problem 3, Naive Bayes Classifier on Additional Dataset (yalefaces)
'''
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

###-------------------------FUNCTIONS-------------------------###
'''
Take image from input filepath string, return a flattened numpy array based on pixel values
'''

def process_image(filepath):
    from PIL import Image
    import numpy as np
    
    im = Image.open(filepath)
    im_resized = im.resize((40, 40))
    
    return np.array(im_resized).flatten()

'''
Build a matrix of flattened numpy arrays from images located in directory yalefaces. Expects this directory to be co-located with file.
'''
def build_data_matrix():
    import os
    
    img_filenames = sorted(
        os.listdir("./yalefaces")
    )# get filenames of all images besides Readme.txt
    
    dat=[]
    
    for name in img_filenames:
        if name != "Readme.txt":
            filepath = f"./yalefaces/{name}"
            
            dat.append(process_image(filepath))
        
    data_matrix = np.vstack(np.array(dat))
        
    return data_matrix

'''
Get class labels from list of strings @str_lst, parsing regex @pattern corresponding to the class label. Build list of parsed string labels, str_labels.

From str_labels, convert to integers.

Return array of class label integers
'''

def class_labels_from_str_lst(str_lst, pattern):
    import re
    
    str_labels = []
    
    for txt in str_lst:
        match = re.search(pattern, txt)
        
        if match != None:
            str_labels.append(match.group())
    
    # Gives integer encoding of unique class labels for input categorical array
    class_labels = np.unique(str_labels, return_inverse=True)[1]
            
    return class_labels

'''
Shuffle input data matrix @X and @y, splitting based on proportion to go to training data @train_size_prop.

Keeps proportion of class labels consistent across training set and validation set (to ensure validation set will contain desired proportion of class labels to verify against).
'''

def shuffle_by_class_split(X, y, train_size_prop):
    from math import ceil
    np.random.seed(seed=0)
    uniq_classes = np.unique(Y)
    
    train_i = []
    val_i = []
    
    # Build list of indices where samples have given class label. Adds first 2/3 of shuffled indices to train_i, and last 1/3 to val_i.
    for label in uniq_classes:

        filt_ind = np.array(ind_where_true(Y, label))

        shuffled_ind = filt_ind.copy()
        np.random.shuffle(shuffled_ind)


        train_size_last_index = ceil(train_size_prop * (len(shuffled_ind)-1))


        train_i += list(shuffled_ind[:train_size_last_index])

        val_i += list(shuffled_ind[train_size_last_index:])
    
    # Training and validation sets filtered down based on lists indices built earlier
    X_train, X_val = X[train_i], X[val_i]
    y_train, y_val = y[train_i], y[val_i]
    
    return X_train, X_val, y_train, y_val

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
# Import data matrix from images in ".yalefaces" directory, pre-process to 40 x 40 then to 1 x 1600 arrays. 
dat = build_data_matrix() # X matrix, n samples x 1600 features

# Get array of class labels based on file names in ".yalefaces" directory
img_filenames = sorted(
        os.listdir("./yalefaces")
    )# get filenames of all images besides Readme.txt
Y = class_labels_from_str_lst(img_filenames, "subject(\d{2,4})")

# Split data into training and validation sets, keeping proportion of each class label consistent across training and validation sets
X_train, X_val, y_train, y_val = shuffle_by_class_split(dat, Y, 2/3)

# Predict on validation set, print error metrics
y_val_pred = pred_class(X_train, y_train, X_val)

cf_val = confusion_matrix(y_val, y_val_pred)
acc_val = calc_accuracy(cf_val, y_val_pred)

print()
print(f"Accuracy of Naive Bayes Classifier for validation set (rounded) is {acc_val: 0.3f}")

print()
print("Confusion matrix for validation set is:")
print(cf_val)