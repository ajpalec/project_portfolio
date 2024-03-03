####----------LIBRARIES----------####
import InputLayer, FullyConnectedLayer, SoftmaxLayer, CrossEntropy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

####----------FUNCTIONS----------####
"""
One-hot encode labels present in @y_labels, giving a column for each
"""
def one_hot_encode(y_labels, num_classes):
    encoded_labels = np.zeros((len(y_labels), num_classes))
    for i, y_label in enumerate(y_labels):
        encoded_labels[i, y_label] = 1
    return encoded_labels

"""
Generate data frame of error metrics for training and validation sets.
Specifically for cross entropy for a multi-class classification.
Used to compare different learning rates, @eta_inp, and epochs, @num_epochs_inp.
"""
def gen_df_err(X_train_inp, X_val_inp, y_train_encoded_inp, y_val_encoded_inp, eta_inp, num_epochs_inp, optimizer="Adam"):
    np.random.seed(seed=0)

    L1 = InputLayer.InputLayer(X_train_inp)
    L2 = FullyConnectedLayer.FullyConnectedLayer(sizeIn=X_train_inp.shape[1], sizeOut=10)
    L3 = SoftmaxLayer.SoftmaxLayer()
    L4 = CrossEntropy.CrossEntropy()

    layers = [L1, L2, L3, L4]

    epochs = 0
    # err_change_train = 1e6
    err_dict = {"epochs": [],
                "cross_entropy_train": [],
                "cross_entropy_val": []
                }
    cross_entropy_train = 0

    while (epochs < num_epochs_inp):

        err_dict["epochs"].append(epochs)

        # Forwards - validation (for error calculation only)
        h_val = X_val_inp
        for i in range(len(layers) - 1):
            h_val = layers[i].forward(h_val)

        ## Calculate cross entropy for validation
        Yhat_val = L3.getPrevOut()
        cross_entropy_val = L4.eval(y_val_encoded_inp, Yhat_val)
        err_dict["cross_entropy_val"].append(cross_entropy_val)

        # Forward train
        h = X_train_inp
        for i in range(len(layers) - 1):
            h = layers[i].forward(h)
        
        ## Calculate cross entropy for training
        Yhat_train = L3.getPrevOut() # Softmax probability
        cross_entropy_train = L4.eval(y_train_encoded_inp, Yhat_train) # Cross Entropy calculation
        err_dict["cross_entropy_train"].append(cross_entropy_train)

        #backwards!
        grad = layers[-1].gradient(y_train_encoded_inp, h)
        for i in range(len(layers)-2,0,-1):
            newgrad = layers[i].backward(grad)
                
            if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)):
                layers[i].updateWeights(gradIn = grad, t=epochs+1, eta=eta_inp, optimizer=optimizer)
            
            grad = newgrad

        epochs += 1

    df_err = pd.DataFrame(err_dict)
    df_err["eta"] = str(eta_inp)
    
    return df_err

"""
Get class label prediction from softmax probabilities
"""
def get_class_labels(y_classes_prob):
    class_labels = np.argmax(y_classes_prob, axis=1)
    return class_labels

'''
Calculate accuracy metric for classification
(Num correct predictions / Total predictions)
'''
def calc_accuracy(y_inp, y_pred):
    num_correct = len(np.where(y_pred == y_inp)[0])
    
    accuracy = num_correct / len(y_inp)

    return accuracy

####----------MAIN----------####

###-------PART 1: Import data-------###

# Import data
df_train = pd.read_csv('mnist_train_100.csv', 
                       names=["y"] + [f"feat_{i}" for i in range(784)]
                       )
df_val = pd.read_csv('mnist_valid_10.csv',
                     names=["y"] + [f"feat_{i}" for i in range(784)]
                     )

## Get X and y for training and validation
X_train = df_train.iloc[:, 1:].to_numpy()
y_train = df_train.iloc[:, 0].to_numpy()

X_val = df_val.iloc[:, 1:].to_numpy()
y_val = df_val.iloc[:, 0].to_numpy()

## Encode y labels
y_train_encoded = one_hot_encode(y_train, 10)
y_val_encoded = one_hot_encode(y_val, 10)

###-------PART 2: Compare Learning Rates-------###
## Generate error data frame for different learning rates
df_lst=[]

for i in range(-4, -1+1, 1):
    df_lst.append(gen_df_err(X_train, X_val, y_train_encoded, y_val_encoded, math.pow(10,i), 30+1))

df_err_graph = pd.concat(df_lst)

# Plot the error data frame
## Group the data by 'eta'
grouped = df_err_graph.groupby('eta')

## Define the linetypes
linetypes = ['-', '--', ':']

## Plot the lines with different linetypes
for i, (eta, group) in enumerate(grouped):
    plt.plot(group['epochs'], 
             group['cross_entropy_train'], 
             label=f'$\eta$={eta}', 
             linestyle=linetypes[i % len(linetypes)]
             )

# Add labels and legend
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Train')
plt.title('Cross Entropy (Training) vs. Epochs with Adam Optimizer for Different $\eta$')
plt.legend()
plt.grid()

## Show the plot
plt.show()

###-------PART 3: Use chosen learning rate-------###
# Use finalized learning rate to train model
np.random.seed(seed=0)

L1 = InputLayer.InputLayer(X_train)
L2 = FullyConnectedLayer.FullyConnectedLayer(X_train.shape[1], 10)
L3 = SoftmaxLayer.SoftmaxLayer()
L4 = CrossEntropy.CrossEntropy()

layers = [L1, L2, L3, L4]

epochs = 0
err_dict = {"epochs": [],
            "cross_entropy_train": [],
            "cross_entropy_val": []
            }
cross_entropy_train = 0

num_epochs = 10

while (epochs < num_epochs):

    err_dict["epochs"].append(epochs)
    cross_entropy_train_prev = cross_entropy_train

    # Forward train
    h = X_train
    for i in range(len(layers) - 1):
        h = layers[i].forward(h)
    
    ## Calculate cross entropy for training
    Yhat_train = L3.getPrevOut() # Softmax probability
    cross_entropy_train = L4.eval(y_train_encoded, Yhat_train) # Cross Entropy calculation
    err_dict["cross_entropy_train"].append(cross_entropy_train)


    ## Save values for gradient calculation / update weights
    FCL_prev_in_train = L2.getPrevIn().copy() # Reset for updating weights later
    Softmax_prev_out_train = L3.getPrevOut().copy()

    # Forwards - validation (for error calculation only)
    h_val = X_val
    for i in range(len(layers) - 1):
        h_val = layers[i].forward(h_val)

    ## Calculate cross entropy for validation
    Yhat_val = L3.getPrevOut()
    cross_entropy_val = L4.eval(y_val_encoded, Yhat_val)
    err_dict["cross_entropy_val"].append(cross_entropy_val)


    ## Reset FCL PrevIn and LogisticSigmoid PrevOut for gradient calculation
    L2.setPrevIn(FCL_prev_in_train)
    L3.setPrevOut(Softmax_prev_out_train)

    #backwards!
    grad = layers[-1].gradient(y_train_encoded,h)
    for i in range(len(layers)-2,0,-1):
        newgrad = layers[i].backward(grad)
            
        if(isinstance(layers[i],FullyConnectedLayer.FullyConnectedLayer)):
            eta = math.pow(10,-2)
            layers[i].updateWeights(gradIn = grad, t=epochs+1, eta=eta)
        
        grad = newgrad
    
    epochs += 1

# Graph training and validation error
df_graph = pd.DataFrame(err_dict)

# Make plot of log-loss vs. epochs
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_graph["epochs"], df_graph["cross_entropy_train"], color="blue", linestyle="solid", label="Training Set", linewidth=2)
ax.plot(df_graph["epochs"], df_graph["cross_entropy_val"], color="green", linestyle="dashed", label="Validation Set", linewidth=2)

ax.set_xlabel("Epochs", fontsize=16)
ax.set_ylabel("Cross Entropy", fontsize=16)
ax.set_title(f"Cross Entropy vs. Epochs, $\eta$ = {eta}", 
             fontsize=16)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
ax.grid()
plt.show()

# Calculate and print training accuracy to console
y_pred_tr = get_class_labels(L3.getPrevOut())
y_train_acc = calc_accuracy(y_train, y_pred_tr)
print(f"Training accuracy is {y_train_acc:.3f}")

 # Forward val
h = X_val
for i in range(len(layers) - 1):
    h = layers[i].forward(h)

y_pred_val = get_class_labels(L3.getPrevOut())
y_val_acc = calc_accuracy(y_val, y_pred_val)
print(f"Validation accuracy is {y_val_acc:.3f}")