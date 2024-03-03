####----------LIBRARIES----------####
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####----------FUNCTIONS----------####
def J(x_1_inp, w_1_inp):

    try:
        J = (1/4)*(x_1_inp * w_1_inp)**4 - (4/3)*(x_1_inp * w_1_inp)**3 + (3/2)*(x_1_inp*w_1_inp)**2

        return J

    except OverflowError:
        print("Overflow in calculation of J")
        return np.inf
    
    except:
        print("Error in calculation of J")
        return np.NaN
    
def dJ_dw1(x_1_inp, w_1_inp):
    try:
        dJ_dw1 = (x_1_inp**4 * w_1_inp**3) - (x_1_inp**3 * w_1_inp**2) + 3*(x_1_inp**2 *w_1_inp)
        return dJ_dw1
    except OverflowError:
        print("Overflow in calculation of dJ_dw1")
        return np.inf
    except:
        print("Error in calculation of dJ_dw1")
        return np.NaN
    
def gd_J_calc(x_1_inp, w_1_inp, eta_inp, epochs_inp):
    w = w_1_inp

    J_vals = []
    w_vals = []

    for i in range(epochs_inp+1):
        w = w - eta_inp * dJ_dw1(x_1_inp, w)
        w_vals.append(w)
        J_vals.append(J(x_1_inp, w))

    return J_vals, w_vals

def Adam(x_1_inp, w_1_inp, p1_inp, p2_inp, eta_inp, delta_inp, epochs_inp):
    s, r = 0, 0

    w = w_1_inp

    J_vals = [J(x_1_inp, w_1_inp)]
    w_vals = [w_1_inp]

    for t in range(1, epochs_inp+1):
        grad = dJ_dw1(x_1_inp, w_1_inp)
        s = (p1_inp * s) + (1 - p1_inp) * grad
        r = (p2_inp * r) + (1 - p2_inp) * (grad * grad)
        
        w = w - eta_inp * (s / (1-p1_inp**t)) / (np.sqrt(r / (1-p_2)**t) + delta_inp)
        w_vals.append(w)
        J_vals.append(J(x_1_inp, w))

    return J_vals, w_vals

####----------SCRIPT----------####
###-------PROBLEM 1-------###
w_1 = np.arange(-2, 5+0.1, 0.1)
x_1 = np.ones(w_1.shape)
J_calc = J(x_1, w_1)

# Create a new figure
fig1 = plt.figure()

# Plot
plt.plot(w_1, J_calc)
plt.xlabel('w$_1$')
plt.ylabel('J')
plt.title('Plot of Objective Function J vs. w$_1$ for x$_1$ = 1')
plt.grid()
plt.suptitle("Problem 2: Visualizing an Objective Function", 
             fontweight="bold")
# Show the plot
# plt.show()

###-------PROBLEM 2-------###
w_vals_input = [-1, 0.2, 0.9, 4]

J_vals_graph = {}
w_vals_graph = {}

x_1_graph = 1
eta = 0.1
epochs = 100

for w_val in w_vals_input:
    J_vals_graph[w_val], w_vals_graph[w_val] = (gd_J_calc(x_1_graph, w_val, eta, epochs))

epochs_graph = [epoch for epoch in range(epochs+1)]
final_J_vals = [J_vals_graph[key][-1] for key in J_vals_graph]
final_w_vals = [w_vals_graph[key][-1] for key in w_vals_graph]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, key in enumerate(J_vals_graph):
    row = i // 2
    col = i % 2
    axs[row, col].plot(np.array(epochs_graph).flatten(), np.array(J_vals_graph[key]).flatten())
    
    axs[row, col].set_xlabel('Epochs')
    axs[row, col].set_ylabel('J')
    axs[row, col].set_title(f'Plot of J vs. Epochs during GD for w$_1$ = {key}')

    axs[row, col].text(0.5, 0.9, f'Final w$_1$ = {final_w_vals[i]:.2e}', 
                       transform=axs[row, col].transAxes, ha='left'
                       )
    axs[row, col].text(0.5, 0.8, f'Final J = {final_J_vals[i]:.2e}', 
                       transform=axs[row, col].transAxes, ha='left'
                       )


plt.suptitle("Problem 3: Exploring Model Initialization Effects", 
             fontweight="bold")

plt.tight_layout()
# plt.show()

###-------PROBLEM 4-------###
eta_vals_input = [0.001, 0.01, 1.0, 5.0]

J_vals_graph = {}
w_vals_graph = {}

x_1_graph = 1
w_1_graph = 0.2
epochs = 100

for eta_val in eta_vals_input:
    J_vals_graph[eta_val], w_vals_graph[eta_val] = (gd_J_calc(x_1_graph, w_1_graph, eta_val, epochs))

final_J_vals = [J_vals_graph[key][-1] for key in J_vals_graph]

final_w_vals = [w_vals_graph[key][-1] for key in w_vals_graph]

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

for i, key in enumerate(J_vals_graph):
    row = i // 2
    col = i % 2
    axs[row, col].plot(np.array(epochs_graph).flatten(), np.array(J_vals_graph[key]).flatten())
    
    axs[row, col].set_xlabel('Epochs')
    axs[row, col].set_ylabel('J')
    axs[row, col].set_title(f'Plot of J vs. Epochs during GD for $\eta$ = {key}')

    axs[row, col].text(0.5, 0.9, f'Final w$_1$ = {final_w_vals[i]:.2e}', 
                       transform=axs[row, col].transAxes, ha='left'
                       )
    axs[row, col].text(0.5, 0.8, f'Final J = {final_J_vals[i]:.2e}', 
                       transform=axs[row, col].transAxes, ha='left'
                       )

plt.suptitle("Problem 4: Explore Learning Rate Effects", 
             fontweight="bold")
plt.tight_layout()
# plt.show()

###-------PROBLEM 5-------###
w_1 = 0.2
x_1 = 1
p_1 = 0.9
p_2 = 0.999
eta = 5
delta = 10e-8 

J_vals_graph, w_vals_graph = Adam(x_1, w_1, p_1, p_2, eta, delta, 100)

epochs_graph = [epoch for epoch in range(100+1)]

plt.figure()
plt.plot(epochs_graph, J_vals_graph)
plt.xlabel('Epochs')
plt.ylabel('J')
plt.title('Plot of Objective Function J vs. Epochs for x$_1$ = 1, using ADAM Optimizer')

plt.suptitle("Problem 5: Adaptive Learning Rate", 
             fontweight="bold")
plt.grid()
plt.show()