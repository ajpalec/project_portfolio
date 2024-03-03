'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Dimensionality Reduction

'''
import numpy as np

###----------FUNCTIONS----------###

'''
Returns mean-centered matrix from input matrix X
'''

def mean_center(X):
    X_mean = np.mean(X, axis=0)
    
    X_mc = X - X_mean
    
    return X_mc

'''
Performs principal component analysis on input matrix X. First mean-centers data, then calculates covariance matrix, then finds eigenvalues and eigenvectors.
'''
def calculate_PCA_eig(X):
    X_mc = mean_center(X)
    X_cov = np.cov(X_mc.T)
    eig_values, eig_vectors = np.linalg.eig(X_cov)
    
    
    return eig_values, eig_vectors

###----------SCRIPT----------###


X = np.array([[0, 1],
             [0, 0],
             [1, 1],
             [0, 0],
             [1, 1],
             [1, 0],
             [1, 0],
             [1, 1],
             [2, 0],
             [2, 1]
             ])

Y = np.array([[1],
             [1],
             [1],
             [1],
             [1],
             [0],
             [0],
             [0],
             [0],
             [0]])

eig_values, eig_vectors = calculate_PCA_eig(X)

pc1 = eig_vectors[:, 0] # corresponds to eig_value 0.556
pc2 = eig_vectors[:, 1] # corresponds to eig_value 0.267

print(f"Part d) - Principal components for X are {pc1} and {pc2}")


pc1_projection = np.dot(mean_center(X), pc1.T)
pc1_projection_rounded = np.around(pc1_projection, decimals=2)

print(f"Part e) - 1-D projection of X onto first principal component is (with rounded values): {pc1_projection_rounded}")