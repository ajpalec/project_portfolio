Alec Peterson
ap3842@drexel.edu
CS 613 Machine Learning, Fall 2023
Homework 6 - Probabilistic Models

This readme.txt describes contents for Homework 6 for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW6_CS613.pdf 
* Contains the PDF writeup for this assignment.


2) hw6_problem2.py
* Contains script that loads data located at file path "./CTG.csv".
* Shuffles then splits data into training and validation sets
* Turns the training and validation input variable data matrices into binary categorical variables based on less-than the mean (made to 0) or greater-than-or-equal-to the mean (made to 1). Mean of training set used to evaluate/binarize validation set.
* Calculates probability of validation set through Naive Bayes classifier, using training set features and labels to determine probabilities of class labels given evidence.
* Calculates and prints to terminal accuracy for validation set, and n x n confusion matrix where n is the number of unique classes.

2) hw6_problem3.py
* Contains script that loads data located at file path "./yalefaces".
* Pre-processes images in "./yalefaces" into 40 x 40 data matrices, and further flattens these to 1 x 1600 vectors.
* Shuffles then splits data into training and validation sets, ensuring to keep same proportion of class labels across training and validation sets.
* Calculates probability of validation set through Naive Bayes classifier, using training set features and labels to determine probabilities of class labels given evidence.
* Calculates and prints to terminal accuracy for validation set, and n x n confusion matrix where n is the number of unique classes.