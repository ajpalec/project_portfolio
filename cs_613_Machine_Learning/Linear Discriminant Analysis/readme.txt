Alec Peterson
ap3842@drexel.edu
CS 613 Machine Learning, Fall 2023
Homework 4 - Linear Discriminant Analysis

This readme.txt describes contents for Homework 4 for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW4_CS613.pdf 
* Contains the PDF writeup for this assignment, including classification metrics / statistics for Part 1.


2) hw4_problem.py
* Contains script that loads data located at file path "./spambase.data", z-scores (zero mean, unit standard deviation) the dataset.
* Shuffles then splits data into training and validation sets
* Performs linear discriminant analysis, calculating closed-form (analytical) solution involving eigenvector decomposition
* Calculates and prints to terminal accuracy for training set, and classification metrics for validation set including precision, recall, F-measure, and accuracy