Alec Peterson
ap3842@drexel.edu
CS 613 Machine Learning, Fall 2023
Homework 5 - Decision Trees

This readme.txt describes contents for Homework 5 for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW5_CS613.pdf 
* Contains the PDF writeup for this assignment.


2) hw5_problem2.py
* Contains script that loads data located at file path "./CTG.csv".
* Shuffles then splits data into training and validation sets
* Turns the training and validation input variable data matrices into binary categorical variables based on less-than the mean (made to 0) or greater-than-or-equal-to the mean (made to 1). Mean of training set used to evaluate/binarize validation set.
* Creates a binary search tree by recursively choosing features that give the lowest entropy, then creating a decision node for that feature with left branch if feature value is True for samples, right branch if feature value is False for samples. This continues until base cases are reached (no more samples, only one unique class label, or no more features), with leaf nodes providing class estimates based on remaining class label highest probability.
* Calculates and prints to terminal accuracy for validation set, and n x n confusion matrix where n is the number of unique classes.

2) hw5_problem3.py
* Contains script that loads data located at file path "./yalefaces".
* Pre-processes images in "./yalefaces" into 40 x 40 data matrices, and further flattens these to 1 x 1600 vectors.
* Shuffles then splits data into training and validation sets, ensuring to keep same proportion of class labels across training and validation sets.
* Turns the training and validation input variable data matrices into binary categorical variables based on less-than the mean (made to 0) or greater-than-or-equal-to the mean (made to 1). Mean of training set used to evaluate/binarize validation set.
* Creates a binary search tree by recursively choosing features that give the lowest entropy, then creating a decision node for that feature with left branch if feature value is True for samples, right branch if feature value is False for samples. This continues until base cases are reached (no more samples, only one unique class label, or no more features), with leaf nodes providing class estimates based on remaining class label highest probability.
* Calculates and prints to terminal accuracy for validation set, and n x n confusion matrices for both training set and validation set, where n is the number of unique class labels.