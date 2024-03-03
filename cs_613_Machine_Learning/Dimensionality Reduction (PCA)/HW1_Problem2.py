'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Dimensionality Reduction

'''

###----------FUNCTIONS----------###

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
Returns mean-centered matrix from input matrix X
'''

def mean_center(X):
    X_mean = np.mean(X, axis=0)
    
    X_mc = X - X_mean
    
    return X_mc

'''
Performs principal component analysis on input matrix X. First mean-centers data, then calculates covariance matrix, then performs singular value decomposition to find principal components.
Returns matrices U, S, and V-transpose. Principal components lie in the columns of U.
'''
def calculate_PCA_svd(X):
    X_mc = mean_center(X)
    X_cov = np.cov(X_mc.T)
    U, S, V_T = np.linalg.svd(X_cov)
    
    return U, S, V_T

###----------SCRIPT----------###
dat = build_data_matrix()

U, S, V_T = calculate_PCA_svd(dat)

PC_2D_T = U[:, :2] # transpose of first 2 principal components

dat_mc = mean_center(dat)

dat_projected = np.dot(dat_mc, PC_2D_T)


plt.scatter(x=dat_projected[:, 0],
            y=dat_projected[:, 1],
           )

plt.show()