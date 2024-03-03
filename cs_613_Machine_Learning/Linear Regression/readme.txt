Alec Peterson
ap3842@drexel.edu
CS 613 Machine Learning, Fall 2023
Homework 2 - Linear Regression

This readme.txt describes contents for Homework 2 for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW2_CS613.pdf 
* Contains the PDF writeup for this assignment, including shown work for theory questions for Part 1, metrics and pre-processing for Part 2, and metrics for Part 3.


2) hw2_problem2.py
* Contains script that shuffles and splits data located at file path "./insurance.csv" into training and validation (test) datasets. It then calculates closed-form linear regression from the training dataset, then calculates root-mean square error (RMSE) and symmetric mean absolute percentage error (SMAPE) of this model on validation (test) set.
* Prints formatted string with calculated values
* For simplicity, reads into data using pandas library, but then translates this to a Numpy array. All subsequent processing used numpy in spirit of the assignment.
    * Much experimentation was done with np.genfromtxt() but this gave back a "structured" array due disparate data types that could not be easily converted into a numpy array.
    * While I could have gone through and iterated to get arrays for each of the features, I would have to call each of these individually and that wouldn't scale well.

3) hw2_problem3.py
* Per problem 3 requirements, iterates 20 times and calculates mean RMSE and standard deviation of RMSE's generated from S-fold cross-validation.
* Prints mean and standard deviation of RMSE's for S = 3, S = 223, and S = (length of dataset)