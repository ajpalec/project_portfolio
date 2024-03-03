'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Linear Regression
Problem 2, Closed Form Linear Regression
'''

import numpy as np
import pandas as pd

###----------FUNCTIONS----------###
'''
Shuffle input data matrix @dat, then split 2/3 for training set and 1/3 for test set.
Assumes all but last column are independent variables, and last column is dependent variable.
'''

def shuffle_split(dat):
    np.random.seed(seed=0)
    dat_shuffled = dat.copy()
    np.random.shuffle(dat_shuffled)
    
    # split
    train_size_last_index = int(2/3 * dat_shuffled.shape[0])
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
Calculates root mean square error of predictions
'''
def calc_rmse(y, y_pred):
    se = np.square(y - y_pred)
    mse = se.sum() / len(y)
    rmse = np.sqrt(mse)
    
    return rmse

'''
Calculates symmetric mean absolute percentage error (proportion i.e. does not multiply by 100%)
'''
def calc_smape(y, y_pred):
    num = np.abs(y - y_pred)
    denom = np.abs(y) + np.abs(y_pred)
    smape = (num / denom).sum() / len(num)
    
    return smape


###----------SCRIPT----------###

# Load data
df = pd.read_csv("./insurance.csv")
dat = df.to_numpy()

colnames_list = np.genfromtxt("./insurance.csv", 
                              dtype=None,
                              delimiter=",", 
                              names=True,
                              encoding="utf8").dtype.names

colnames_dict = {name: index for index, name in enumerate(colnames_list[:-1])}

# Shuffle and split
X_train, X_test, y_train, y_test = shuffle_split(dat)


# Process training data
X_train_processed_b = pre_process(dat, X_train, colnames_dict)

# Calculate closed-form linear regression model
w_train = calc_w(X_train_processed_b, y_train)

# Calculate prediction error metrics
## Train set
y_train_pred = np.dot(X_train_processed_b, w_train)

train_rmse = calc_rmse(y_train, y_train_pred)
train_smape = calc_smape(y_train, y_train_pred)

## test set
X_test_processed_b = pre_process(dat, X_test, colnames_dict)
y_test_pred = np.dot(X_test_processed_b, w_train)

test_rmse = calc_rmse(y_test, y_test_pred)
test_smape = calc_smape(y_test, y_test_pred)

# Print results
print(f"Closed-form linear regression model for training set had RMSE of {train_rmse: .2f} and SMAPE of {train_smape: .2f} i.e. {np.around(train_smape,2)*100:.0f}%\n")

print(f"Using linear model developed from test (validation) set gave RMSE of {test_rmse: .2f} and SMAPE of {test_smape: .2f} i.e. {np.around(test_smape,2)*100:.0f}%")