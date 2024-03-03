import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_roc_val = pd.read_csv("rand_forest_optimized_val_ROC_table.csv")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(df_roc_val["fpr"], df_roc_val["tpr"], color="blue", 
        # marker="o", 
        linestyle="solid",
        label="Random Forest" 
       );
# ax.plot(model_data[:, 1], model_data[:, 3], color="green", marker="o", linestyle="dashed", label="Validation Set");

ax.plot(0.3072314, 0.967033, color="blue", marker="o")
ax.set_xlabel("FPR", fontsize=16)
ax.set_ylabel("TPR", fontsize=16)
ax.set_title(f"ROC for Optimized Random Forest (AUC = 0.90)", 
             fontsize=18)

ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)

ax.legend(fontsize=14)
# ax.grid()
font = {
        # "family": "Arial",
        # "weight": "bold",
        "size": 14}

plt.text(0.31, 0.90, "Optimal Threshold = 0.11", font)
plt.show()