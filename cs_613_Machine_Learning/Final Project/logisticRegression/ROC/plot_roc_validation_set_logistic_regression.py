import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_roc_val = pd.read_csv("./logistic_regression_optimized_val_ROC_table.csv")

model_name = "Logistic Regression"
colors = {"Random Forest": "blue",
          "Logistic Regression": "green",
          "Naive Bayes": "magenta",
         }

# Make plot of ROC
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_roc_val["fpr"], df_roc_val["tpr"], 
        color=colors[model_name], 
        linestyle="solid",
        label=model_name
       );

# Find optimal threshold based on largest difference between TPR and FPR
df_roc_val_sorted = df_roc_val.sort_values(["diff"], ascending=False).reset_index()

# Get threshold, round to 2 decimal places
optimal_threshold = np.around(df_roc_val_sorted.loc[0, ["threshold"]]["threshold"], 2)

# Get fpr and tpr for optimal threshold
optimal_threshold_fpr = df_roc_val_sorted.loc[0, ["fpr"]]["fpr"]
optimal_threshold_tpr = df_roc_val_sorted.loc[0, ["tpr"]]["tpr"]

# Plot point for optimal threshold
ax.plot(optimal_threshold_fpr, 
        optimal_threshold_tpr, 
        color=colors[model_name],
        marker="o")

# Calculate AUC
def calc_AUC(df_roc_inp):
    
    df_roc_inp_sorted = df_roc_inp.sort_values(["fpr"])
    
    auc = np.trapz(y=df_roc_inp_sorted["tpr"], 
                   x=df_roc_inp_sorted["fpr"],
                  )
    
    return auc

auc_plot = np.around(calc_AUC(df_roc_val), decimals=2)

# Label graph
ax.set_xlabel("FPR", fontsize=16)
ax.set_ylabel("TPR", fontsize=16)
ax.set_title(f"ROC for Optimized {model_name} (AUC = {auc_plot: .2f})", 
             fontsize=18)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
font = {
        # "family": "Arial",
        # "weight": "bold",
        "size": 14}

# Mess with text placement depending on the optimal threshold point
plt.text(0.31, 0.90, 
         f"Optimal Threshold = {optimal_threshold}", font)
plt.show()

