Alec Peterson
ap3842@drexel.edu
CS 615 Deep Learning, Winter 2024
Homework 1 - Forward Propagation

This readme.txt describes contents for Homework 1 for class CS 615 Deep Learning, Winter 2023 at Drexel University.

Contents:

1) Alec_Peterson_HW1_CS615.pdf 
* Contains the PDF writeup for this assignment, including shown work for theory questions for Part 1, and requested output of neural network architecture for Part 3.


2) Python files containing neural network classes, which inherit from abstract class Layer from Layer.py:
* InputLayer.py - z-scores input datamatrix
* LinearLayer.py - linear activation function
* ReLULayer.py - ReLU activation function
* LogisticSigmoidLayer.py - Logistic sigmoid activation function
* TanhLayer.py - Hyperbolic tangent activation function
* SoftmaxLayer.py - Softmax activation function
* FullyConnectedLayer.py - Dense layer taking input with D dimensions and turning to K dimensions after multiplying by weights and adding biases


3) main.py
* Contains main function that when run, creates neural network architecture of Input -> Fully Connected -> Logistic Sigmoid layer
( Imports co-located data file from KidCreative.csv using all but first 2 columns (per homework directions)
* Prints output with explanatory text to console - first observation, then whole output