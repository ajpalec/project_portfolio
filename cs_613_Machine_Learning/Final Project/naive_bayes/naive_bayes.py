import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import auc, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os
#Load pre-processed training and validation data

# df_train_encoded_resampled = pd.read_csv("../pre_processing/output/training_data_resampled_encoded.csv", sep=";")
# df_val_encoded = pd.read_csv("../pre_processing/output/validation_data_encoded.csv", sep=";")
# print(df_train_encoded_resampled)
# print (df_val_encoded)

def get_training_validation_data():
    df_train_encoded_resampled = pd.read_csv("../pre_processing/output/training_data_resampled_encoded.csv", sep=";")
    df_val_encoded = pd.read_csv("../pre_processing/output/validation_data_encoded.csv", sep=";")

    training = df_train_encoded_resampled.to_numpy()
    validation = df_val_encoded.to_numpy()
    
    rows, dim = np.shape(training)
    validation_target, validation_features = validation[:, dim-1:], validation[:, :dim-1]
    training_target, training_features = training[:, dim-1:], training[:, :dim-1]

    return training_target, training_features, validation_target, validation_features

def calculate_class_priors(training_targets):
    #classes = np.unique(np.array(training_targets).flatten())
    classes = np.unique(training_targets)

    class_priors = {}
    for c in classes:
        class_counts = np.where(np.array(training_targets) == c)[0]
        class_priors[c] = len(class_counts) / len(training_targets)

    return class_priors



def calculate_naive_probabilities(training_features, training_targets):
    classes = np.unique(training_targets)
    num_classes = len(classes)
    num_features = training_features.shape[1]

    class_feature_probs = []

    for c in classes:
        #class_indices, count = np.where(training_targets == c)
        class_indices = np.where(np.array(training_targets) == c)[0]
        class_feature_probs_c = []

        for i in range(num_features):
            # get rows for the corresponding class
            # if all(v == 0 for v in training_features[class_indices, i]):
            #     feature_values = [0, 1]
            #     feature_probs=[1, 0]
            # elif all(v == 1 for v in training_features[class_indices, i]):
            #     feature_values = [0, 1]
            #     feature_probs=[0, 1]
            # else:
            feature_values, feature_counts = np.unique(training_features[class_indices, i], return_counts=True)
            feature_probs = feature_counts / np.sum(feature_counts)
            class_feature_probs_c.append((feature_values, feature_probs))
           
        class_feature_probs.append((c, class_feature_probs_c))

    return class_feature_probs

''' start alec '''
#--------------------ROC AND FUNCTIONS--------------------#
def evaluate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return precision, recall, f_measure, accuracy

def calc_TPR(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # print(f"tp are {tp}")
    fn = np.sum((y_true == 1) & (y_pred == 0))
    # print(f"fn are {fn}")
    
    return tp / (tp + fn)
    

def calc_FPR(y_true, y_pred):
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # print(f"fp are {fp}")
    tn = np.sum((y_true == 0) & (y_pred == 0))
    # print(f"tn are {tn}")
    
    return fp / (fp + tn)

def make_roc_df(y_val_prob, y_true_inp):
    pred_dict = {"threshold": [],
                 "tpr": [],
                 "fpr": [],
                 "accuracy": [],
                 "f1-measure": [],
                 "precision": [],
                 "recall": []
                }
    
    thresholds = [i/100 for i in range(0, 100+1)] # 0 to 1, by 0.01 steps
    
    for thr in thresholds:
        pred_dict["threshold"].append(thr)
        y_pred_calc = np.where(y_val_prob >= thr, 1, 0)
        
        tpr = calc_TPR(y_true_inp, y_pred_calc)
        pred_dict["tpr"].append(tpr)
        # print(f"TPR is {tpr}")
        fpr = calc_FPR(y_true_inp, y_pred_calc)
        pred_dict["fpr"].append(fpr)
        # print(f"FPR is {fpr}")
        precision, recall, f_measure, accuracy = evaluate(y_true_inp, y_pred_calc)
        pred_dict["accuracy"].append(accuracy)
        pred_dict["f1-measure"].append(f_measure)
        pred_dict["precision"].append(precision)
        pred_dict["recall"].append(recall)
    
    return pd.DataFrame(pred_dict)

# Calculate AUC
def calc_AUC(df_roc_inp):
    
    df_roc_inp_sorted = df_roc_inp.sort_values(["fpr"])
    
    auc = np.trapz(y=df_roc_inp_sorted["tpr"], 
                   x=df_roc_inp_sorted["fpr"],
                  )
    
    return auc

''' end alec '''

def calculate_validation_predicted_targets(training_class_priors, training_naive_probabilities, validation_features, validation_targets):
    # find the highest log probability
    # log2(P)y |x)~ log2(P(y)) + Sum of log2(P(xi|y)) / log2(P(x)))
    classes = np.unique(validation_targets)
    num_features = validation_features.shape[1]
    predicted_targets = []

    for sample in validation_features:
        #for each row in validation features
        sample = np.array(sample)
        posterior_validation_class_feature_probs = {}

        for _, class_probs in enumerate(training_naive_probabilities):
            c = class_probs[0]

            # this is the class prior probability
            log_of_class_posterior_prob_for_class = np.log2(training_class_priors[c])
            # i is the feature/column. Need to compute posteriors for each class. For that get the probability
            # of the value in the validation feature from training set, take log, add them to get sum of logs of probabilies for each class
            # predicted class is then a max of that sum + log of prior
            for i, (feature_values, feature_probs) in enumerate(class_probs[1]):
                #print(f"Training class {c}, Feature/column {i}:")
                #print(f"-->Feature Values: {feature_values},  Feature Probabilities: {feature_probs}")
                value = sample[i] #  column i
                index = np.where(feature_values == value)
                
                feature_in_class_prob = feature_probs[index]
                if any(feature_in_class_prob) == False:
                    feature_in_class_prob = 0
                #print(f"-->Validation feature value: {value}, Found in training probability: {feature_in_class_prob}, log: {np.log(feature_in_class_prob + 1e-10)}")
                
                # for each value in each feature in the class 
                log_of_class_posterior_prob_for_class += np.log2(feature_in_class_prob + 1e-10)
                
            posterior_validation_class_feature_probs[c] = log_of_class_posterior_prob_for_class
        predicted_class = max(zip(posterior_validation_class_feature_probs.values(), posterior_validation_class_feature_probs.keys()))[1]
        #print(f"-->Predicted class: {predicted_class}")
        predicted_targets.append(predicted_class)

    return predicted_targets

# with thresholds
def calculate_validation_posterior_probabilities(training_class_priors, training_naive_probabilities, validation_features, validation_targets):
    # find the highest log probability
    # log2(P)y |x)~ log2(P(y)) + Sum of log2(P(xi|y)) / log2(P(x)))
    classes = np.unique(validation_targets)
    num_features = validation_features.shape[1]
    predicted_posteriors = []

    for sample in validation_features:
        #for each row in validation features
        sample = np.array(sample)
        posterior_validation_class_feature_probs = {}

        for _, class_probs in enumerate(training_naive_probabilities):
            c = class_probs[0]

            # this is the class prior probability
            posterior_prob_for_class = training_class_priors[c]
            # i is the feature/column. Need to compute posteriors for each class. For that get the probability
            # of the value in the validation feature from training set, take log, add them to get sum of logs of probabilies for each class
            # predicted class is then a max of that sum + log of prior
            for i, (feature_values, feature_probs) in enumerate(class_probs[1]):
                # print(f"Training class {c}, Feature/column {i}:")
                # print(f"-->Feature Values: {feature_values},  Feature Probabilities: {feature_probs}")
                value = sample[i] #  column i
                index = np.where(feature_values == value)
                
                feature_in_class_prob = feature_probs[index]
                if any(feature_in_class_prob) == False:
                    feature_in_class_prob = 0
                # print(f"-->Validation feature value: {value}, Found in training probability: {feature_in_class_prob}, log: {np.log(feature_in_class_prob + 1e-10)}")
                
                # for each value in each feature in the class 
                posterior_prob_for_class *= feature_in_class_prob
                
            posterior_validation_class_feature_probs[c] = posterior_prob_for_class
        
        #print(f"-->posterior_validation_class_feature_probs[{c}]/: {posterior_validation_class_feature_probs[c]}, posterior_prob_for_class: {posterior_prob_for_class},  Total Probabilities: {total_posterior_probs}")

        total_posterior_probs = sum(posterior_validation_class_feature_probs.values())
        for cl in posterior_validation_class_feature_probs:
            posterior_validation_class_feature_probs[cl] /= total_posterior_probs

        class_1_prob = posterior_validation_class_feature_probs[1]
        class_2_prob = posterior_validation_class_feature_probs[0]
        predicted_posteriors.append(class_1_prob)

    return np.array(predicted_posteriors)


def calc_AUC(df_roc_inp):

    df_roc_inp_sorted = df_roc_inp.sort_values(["fpr"])

    auc = np.trapz(y=df_roc_inp_sorted["tpr"], 
                   x=df_roc_inp_sorted["fpr"],
                  )

    return auc

# Metrics
def get_metrics(predicted_targets, validation_targets):
    validation_targets = validation_targets.flatten()
    predicted_targets = np.array(predicted_targets)
    true_targets_total = [p == v for p, v in zip(predicted_targets, validation_targets)]
    false_targets_total = [p != v for p, v in zip(predicted_targets, validation_targets)]
 
    classes = np.unique(validation_targets)

    false_positives = np.logical_and(validation_targets == 0, predicted_targets == 1)
    false_positives_cnt = np.sum(false_positives)
    false_negatives_cnt = np.sum(np.logical_and(validation_targets == 1, predicted_targets == 0))
    true_positives = np.logical_and(validation_targets == 1, predicted_targets == 1)
    true_positives_cnt = np.sum(true_positives)
    true_negatives_cnt = np.sum(np.logical_and(validation_targets == 0, predicted_targets == 0))

    #accuracy = np.mean(validation_targets == predicted_targets)
    # Precision – percentage of things that were classified as positive and actually were positive
    precision = true_positives_cnt / (true_positives_cnt + false_positives_cnt)
    # Recall – the percentage of true positives correctly identified
    recall = true_positives_cnt / (true_positives_cnt + false_negatives_cnt)
    # f-measure  - the weighted harmonic mean of precision and recall
    f_measure = 2 * (precision * recall) / (precision + recall)
    # the persentage of times we are correct
    accuracy = (true_positives_cnt + true_negatives_cnt)/(true_negatives_cnt+true_positives_cnt+false_positives_cnt+false_negatives_cnt) 

    TPR = true_positives_cnt/(true_positives_cnt + false_negatives_cnt)
    FPR = false_positives_cnt/(false_positives_cnt + true_negatives_cnt)

    # initialize the confusion matrix
    conf_matrix = np.zeros((len(classes), len(classes)))

    # loop across the different combinations of actual / predicted classes
    for i in range(len(classes)):
        for j in range(len(classes)):
           # count the number of instances in each combination of actual / predicted classes
           conf_matrix[i, j] = np.sum(( np.array(predicted_targets) == classes[i]) & (validation_targets == classes[j]))
    print (f"accuracy:{accuracy},recall:{recall},f measure:{f_measure}, precision: {precision}, TPR: {TPR}, FPR:{FPR}")
    return TPR, FPR, accuracy, precision, f_measure, recall, conf_matrix



#  Read the preprocessed data
df_train_encoded_resampled = pd.read_csv("../CS-613/Project/pre_processing/output/training_data_resampled_encoded.csv")
df_val_encoded = pd.read_csv("../CS-613/Project/pre_processing/output/validation_data_encoded.csv")

print(df_train_encoded_resampled)
print (df_val_encoded)

training = df_train_encoded_resampled.to_numpy()
validation = df_val_encoded.to_numpy()

rows, dim = np.shape(training)
validation_target, validation_features = validation[:, dim-1:], validation[:, :dim-1]
training_target, training_features = training[:, dim-1:], training[:, :dim-1]
   
# 1. Calculate Class Priors
class_priors = calculate_class_priors(training_target)
#print("Class Priors:", class_priors)

# 2. Calculate Naive Probabilities P(xi | y)
class_feature_probs = calculate_naive_probabilities(training_features, training_target)

# 3. Calculate predicted targets P(y| X) and y=argmax P(y|X) = argmax log2(P(X1|y)) + log2(P(X2|y)).. +log2(P(y))
# on the validation set
predicted_target = calculate_validation_predicted_targets(class_priors, class_feature_probs, validation_features, validation_target)

# 4. Calc metrix
TPR, FPR, accuracy, precision, f_measure, recall, conf_matrix = get_metrics(predicted_target, validation_target)

# 5. run nb using thresholds and get metrics 
predicted_targets_prob  = calculate_validation_posterior_probabilities(class_priors, class_feature_probs, validation_features, validation_target)


''' alec '''
df_roc_val = make_roc_df(predicted_targets_prob, validation_target)
# df_graph = pd.concat(df_roc_val)
df_roc_val.to_csv('table_of_error_metrics.csv')

df_roc_val["diff"] = df_roc_val["tpr"] - df_roc_val["fpr"]

model_name = "Naive Bayes"
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
       )

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


auc = calc_AUC(df_roc_val)
auc_plot = np.around(auc, decimals=2)

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
plt.text(0.25, 0.7, 
         f"Optimal Threshold = {optimal_threshold}", font)
plt.show()
plt.savefig("nb_roc.png")
