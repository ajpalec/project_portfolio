Alec Peterson
ap3842@drexel.edu
CS 613 Machine Learning, Fall 2023
Homework 3 - Logistic Regression and Gradient-Based Learning

This readme.txt describes contents for Homework 3 for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW3_CS613.pdf 
* Contains the LaTeX-generated PDF writeup for this assignment, including shown work for theory questions for Part 1, classification metrics and generated mean log-loss vs. epochs figure for Part 2


2) hw2_problem3.py
* Contains script that shuffles and splits data located at file path "./spambase.data"
* Shuffles then splits data into training and validation sets
* Pre-processes input data for both training and validation sets
* Trains logistic regression classifier model with gradient descent until difference in mean log-loss is less than or equal to 0.001, using a learning rate of 10
* Generates figure of mean-log loss vs. epoch for training and validation sets
* Calculates classification metrics for precision, recall, F-measure, and accuracy and prints to terminal