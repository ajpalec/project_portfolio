'''
Alec Peterson
ap3842@drexel.edu
CS 613 Fall 2023
Homework 1 - Dimensionality Reduction

'''

###----------FUNCTIONS----------###

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2

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

'''
A) Takes image from ./yalefaces/subject02.centerlight, resizes to 40x40 image, turns into numpy array, then flattens to 1600-length array.

B) Mean-centers the flattened array based on provided means from entire dataset, @dat_means.

C) Projects the array generated from A) onto k-indexed principal components from pc_matrix.

D) Reconstructs image from the projected array using transpose k-indexed principal components, undoes the mean-centering, then reshapes to 40x40 array.

E) Annotates image with which k was used

Returns 8-bit mapped numpy array of the annotated image
'''
def gen_im_arr(pc_matrix, dat_means, k):
    im = Image.open("./yalefaces/subject02.centerlight")
    im_arr = np.array(im.resize((40,40))).flatten()
    im_arr_mc = im_arr - dat_means
    
    im_projected = np.dot(im_arr_mc, pc_matrix[:(k+1)].T)
    im_reconstructed_mc = np.dot(im_projected, pc_matrix[:(k+1)])
    
    im_reconstructed = im_reconstructed_mc + dat_means
    
    im_recon_reshaped = im_reconstructed.reshape((40, 40))
    
    im_PIL = Image.fromarray(np.uint8(im_recon_reshaped))
    
    draw = ImageDraw.Draw(im_PIL)
    annotation = f"k={k}"
    draw.text((3, 30),
              annotation,
              align="left"
             )
    
    im_PIL_annotated_ar = np.array(im_PIL)
    
    
    return np.uint8(im_PIL_annotated_ar)

###----------SCRIPT----------###
dat = build_data_matrix()
dat_means = np.mean(dat, axis=0)
U, S, V_T = calculate_PCA_svd(dat)
pc_matrix = U.T


out = cv2.VideoWriter("eigenfaces.avi",
                      cv2.VideoWriter_fourcc(*'XVID'),
                      15,
                      (40,40),
                      False);

for k in range(0, 1600):
    image = gen_im_arr(pc_matrix, dat_means, k)
    out.write(image)

out.release()

