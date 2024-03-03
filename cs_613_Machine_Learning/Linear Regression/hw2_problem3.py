'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Linear Regression
Problem 3, Cross-Validation
'''

import numpy as np
import pandas as pd

###----------FUNCTIONS----------###
'''
Makes training and validation sets for input data matrix @dat into S cross-validation folds. 

Iterates S times, setting seed to value of iteration index. Divides dataset into S folds, with last fold being slightly larger and containing remainder of data due to potential rounding of indices used to slice dataset. The i-th fold is the validation set, and concatentates other folds together to form training set.

Stores arrays corresponding to training and validation X and y into a dictionary, assuming last column is y as per problem requirements.
'''
def cv_shuffle_split(dat, S):
    cv_dict = {}
    for i in range(1, S+1):

        # Shuffle data
        # np.random.seed(seed=i)
        dat_shuffled = dat.copy()
        np.random.shuffle(dat_shuffled)
        
        # Define test set fold

        fold_size = int(np.around(len(dat_shuffled) / S, 0))
        
        ## Label start and stop indices for test fold
        ind_start_val = (i-1) * fold_size


        if i != S:
            ind_end_val = i * fold_size
        else:
            ind_end_val = len(dat_shuffled)

    
        dat_val = dat_shuffled[ind_start_val:ind_end_val, :]
        
        # Define train set made up of all other folds
        pre_val_slice = dat_shuffled[:ind_start_val, :]
        post_val_slice = dat_shuffled[ind_end_val:, :]

        dat_train = np.concatenate((pre_val_slice,
                                    post_val_slice))
        
        
        
        cv_dict[i] = {"X_train": dat_train[:, :-1],
                      "y_train": dat_train[:, -1],
                      "X_val": dat_val[:, :-1],
                      "y_val": dat_val[:, -1]
                     }
    
    return cv_dict

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
Take a binary feature array (i.e. column of a data matrix) and get encoded array of integers corresponding to each category
'''

def get_encoding(dat, dat_subset, bin_feat_ind):
    binary_vals = np.unique(dat[:, bin_feat_ind])
    
    bin_feat_arr = dat_subset[:, bin_feat_ind]
    
    binary_encoding = (bin_feat_arr == binary_vals[0]).astype(int)
    
    return binary_encoding

'''
Take a categorical feature array (i.e. column of a data matrix) and get encoded array of integers corresponding to each category.
Then return a dictionary of binary-encoded arrays for each category.

Example: for category column with unique values ["north", "south", "east", "west"], will return dictionary with boolean-like array of integers corresponding to columns "is_east", "is_north", "is_south", "is_west". Unique values are sorted, and so corresponding boolean->integer arrays are similarly ordered.
'''
def one_hot_encode(dat, dat_subset, cat_feat_ind): 
    unique_cats = np.unique(dat[:, cat_feat_ind]) # dat_subset might not contain all values, so needs dat for reference
    # np.unique() returns sorted values
    
    feature_dict_onehot = {}
    
    for cat in unique_cats:
        onehot_arr = []
        for elem in dat_subset[:, cat_feat_ind]:
            if elem == cat:
                onehot_arr.append(1)
            else:
                onehot_arr.append(0)
        feature_dict_onehot[cat] = np.array(onehot_arr)
        
    return feature_dict_onehot

'''
Adds onto matrix a dummy column of 1's. Used to create dummy "bias" feature
'''
def add_bias(X):
    b = np.ones(X.shape[0]).reshape(-1,1)
    
    X_b = np.concatenate((X, b), axis=1)
    
    return X_b

'''
Pre-processes data as required to create input data matrix X for closed-form linear regression.

1) Selects numeric columns corresponding to "age", "bmi", and "children". Number of children is considered continuous/numerical.
2) Z-scores numeric features
3) Encodes binary features "sex" and "smoker"
4) One-hot encodes categorical feature "region"
5) Concatentaes columns for numeric features, binary features, and one-hot encoded categorical feature
6) Adds dummy bias feature
'''
def pre_process(dat, X, colnames_dict):
    # numeric features
    X_numeric = X[:, [colnames_dict["age"], colnames_dict["bmi"], colnames_dict["children"]] ].astype(float)
    X_numeric_scaled = zscore(X_numeric.astype(float))
    
    
    # binary encodings
    sex_encoding = get_encoding(dat, X, colnames_dict["sex"])
    smoker_encoding = get_encoding(dat, X, colnames_dict["smoker"])
    
    # one-hot encoding
    regions_onehot = one_hot_encode(dat, X, colnames_dict["region"])
    
    # Concatenate
    X_processed = np.concatenate((X_numeric_scaled,
                                  sex_encoding.reshape((-1,1)),
                                  smoker_encoding.reshape((-1,1)),
                                  np.array(list(regions_onehot.values())).T
                             ), axis=1)
    
    X_processed_b = add_bias(X_processed)
    
    return X_processed_b

'''
Calculates linear regression coefficient matrix w. Calculates pseudo-inverse due to sparsity issues during testing.
'''
def calc_w(X, y):
    w_a = np.linalg.pinv(np.dot(X.T, X))
    
    w_b = np.dot(X.T, y)
    
    return np.dot(w_a, w_b)

'''
Given input dictionary @cv_dict containing cross-validation training and validation X and y, and column names dictionary @colnames_dict:
1) Generate closed-form linear model from each X_training and y_training set
2) From w coefficients calculated from linear model, predict validation set @y_val_pred using input X_val
3) Calculate root-mean square error of linear model on validation set and report out in dictionary
'''
def calc_rmse_folds(dat, cv_dict, colnames_dict):
    se_dict = {}
    # Find closed-form linear regression model
    for key in cv_dict.keys():
        X_train_processed_b = pre_process(dat, cv_dict[key]["X_train"], colnames_dict)
        
        w_train = calc_w(X_train_processed_b, cv_dict[key]["y_train"])
        
        X_val_processed_b = pre_process(dat, cv_dict[key]["X_val"], colnames_dict)
        y_val_pred = np.dot(X_val_processed_b, w_train)
        
        se = np.square(cv_dict[key]["y_val"] - y_val_pred)
        
        se_dict[key] = se
    
    se_arr = np.concatenate([se_dict[key] for key in se_dict.keys()])
    se_arr_non_null = pd.Series(se_arr).dropna().to_numpy()
    
    mse = se_arr.sum() / len(se_arr)
    rmse = np.sqrt(mse)
        
    return rmse

'''
Process data per Problem 3 requirements. For each of 20 iterations:
1) Set seed to iteration #
2) Shuffle/split data matrix @dat into @S folds
3) After generating closed-form linear model, calculate RMSE of validation set.
   Input indices matched to column names via colnames_dict
   Add RMSE to list
4) Calculate then return mean RMSE and standard deviation of RMSEs
'''

def report_rmses(dat, S, colnames_dict):
    np.seterr(invalid="ignore")
    rmse_list = []
    for i in range(1, 20+1):
        np.random.seed(seed=i)
        # print(f"Iter. {i}")
        cv_dict = cv_shuffle_split(dat, S)
        rmse_folds = calc_rmse_folds(dat, cv_dict, colnames_dict)
        # print(rmse_folds)
        
        rmse_list.append(rmse_folds)
        # print(rmse_list)
    
    rmse_arr = np.array(rmse_list)
    rmse_mean = np.mean(rmse_arr)
    # print(rmse_mean)
    rmse_std = np.std(rmse_arr, 
                      ddof=1
                     )
        
    return rmse_mean, rmse_std

###----------SCRIPT----------###
# Load data
df = pd.read_csv("./insurance.csv")
dat = df.to_numpy()

## Get dictionary of column name: index
colnames_list = np.genfromtxt("./insurance.csv", 
                              dtype=None,
                              delimiter=",", 
                              names=True,
                              encoding="utf8").dtype.names

colnames_dict = {name: index for index, name in enumerate(colnames_list[:-1])}

# Perform calculations
S=3
rmse_mean, rmse_std = report_rmses(dat, S, colnames_dict)
print(f"Mean RMSE after 20 iterations for {S}-fold cross-validation is {rmse_mean: .2f}, with standard deviation {rmse_std: .2f}")

S=223
rmse_mean, rmse_std = report_rmses(dat, S, colnames_dict)
print(f"Mean RMSE after 20 iterations for {S}-fold cross-validation is {rmse_mean: .2f} with standard deviation {rmse_std: .2f}")

S=len(dat)
rmse_mean, rmse_std = report_rmses(dat, S, colnames_dict)
print(f"Mean RMSE after 20 iterations for {S}-fold cross-validation is {rmse_mean: .2f} with standard deviation {rmse_std: .2f}")