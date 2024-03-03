Alec Peterson
ap3842@drexel.edu
CS 615 Deep Learning, Winter 2024
Homework 3 - Learning and Basic Architectures

This readme.txt describes contents for Homework 3 for class CS 615 Deep Learning, Winter 2023 at Drexel University.

Contents are as follows, with only code files as required for assignment:

1) Alec_Peterson_HW3_CS615.pdf 
* Contains the PDF writeup for this assignment.


2) Python files containing neural network classes, which inherit from abstract class Layer from Layer.py:
* InputLayer.py - z-scores input datamatrix
* LogisticSigmoidLayer.py - Logistic sigmoid activation function
* FullyConnectedLayer.py - Dense layer taking input with D dimensions and turning to K dimensions after multiplying by weights and adding biases.
	* Implements method to update weights, per assignment Problem 3

3) Python files for objective functions:
* SquaredError.py
* LogLoss.py


4) Python files for problems 2, 4, and 5 and associated input data files
* hw3_prob2 - produces 3D line graph visualizing gradient descent
* hw3_prob4 - linear regression neural network architecture, printing out training and validation RMSE and SMAPE to console and showing MSE vs. Epochs plot
	* input is ./medical.csv
* hw3_prob5 - logistic regression neural network architecture, printing out training and validation accurcy to console and showing Log-Loss vs. Epochs plot
	* input is ./KidCreative.csv