Alec Peterson
ap3842@drexel.edu
CS 615 Deep Learning, Winter 2024
Homework 5

This readme.txt describes contents for Homework 5 submission for class CS 615 Deep Learning, Winter 2024 at Drexel University.


Contents are as follows, with only code files as required for assignment:

1) Alec_Peterson_HW5_CS615.pdf 
* Contains the PDF writeup for this assignment.


2) Python files containing neural network classes, which inherit from abstract class Layer from Layer.py:
* ConvolutionalLayer.py - performs 2D cross-correlation on an input image and initialized kernel
* MaxPoolLayer.py - Given a pooling size and stride, determine the max in each "window" of an input feature map
* FlatteningLayer.py - Flattens an input 2D matrix to a 1D array
* FullyConnectedLayer.py - Dense layer taking input with D dimensions and turning to K dimensions after multiplying by weights and adding biases.
* LogisticSigmoidLayer.py - Logistic sigmoid activation function
* SoftmaxLayer.py - Softmax activation function

3) Python files for objective functions:
* LogLoss.py
* CrossEntropy.py


4) Python files for: Problem 2-5, and Problem 6 and associated input data files
* hw5_prob3.py - Produces plots per problems 3 in assignment and prints initial and final kernels to console
* hw5_prob4.py - Produces plots per problems 4 in assignment and prints initial and final kernels to console
    * In order to run, a co-located directory called "yalefaces" is expected with images per http://cvc.cs.yale.edu/cvc/projects/yalefaces/yalefaces.html