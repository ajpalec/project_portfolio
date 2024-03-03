CS 613 Machine Learning, Fall 2023
Final Project - ML for Bank Telemarketing Campaign
Team Members:
* Alec Peterson (ap3842@drexel.edu)
* Jenny Boroda (jb4655@drexel.edu)
* Yifan Wang (yw827@drexel.edu)

This readme.txt describes contents final project for class CS 613 Machine Learning, Fall 2023 at Drexel University.

Contents:

1) data
* Directory containing subdirectories and data files from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
* bank-additional subdirectory contains data source file with selected features made available by Moro et al., bank-additional-full.csv
* pre_processing subdirectory contains Jupyter notebooks for exploratory data analysis (EDA) and pre-processing of data.
    * output subdirectory contains pre-processing notebook output .csv files for training (resampled) and testing set

2) logisticRegression
* Code, files, output tables (.csv) and output figures of of logistic regression algorithm testing and tuning
* LR_with_cv.ipynb is Jupyter notebook showing 5-fold cross validation for LR algorithm testing different learning rates
* ROC subdirectory contains files for plotting ROC curves
	* plot_roc_validation_set_logistic_regression.py contains script to plot ROC curve for optimized LR implementation
	* logistic_regression_optimized_val_ROC_table.csv contains TPR and FPR data for ROC curve
	* lr_roc.png is resulting ROC curve
* weights subdirectory contains analysis of contributing feature weights for logistic regression at different learning rates
* table_of_error_metrics.csv is classification error metrics such as precision, recall, F1, accuracy

3) naive_bayes
* Code, files, output tables (.csv) and output figures of Naive Bayes algorithm testing
* naive_bayes_10Dec2023.ipynb is Jupyter notebook for Naive Bayes implementation to generate predicted labels on validation set
* naive_bayes.py is script with more streamlined implementation
* nb_roc.png is generated ROC curve from validation set
* table_of_error_metrics.csv is classification error metrics such as precision, recall, F1, accuracy

4) random_forest
* Code, files, output tables (.csv) and output figures of random forest algorithm testing and tuning
* random_forest_for_project_proba.ipynb is Jupyter notebook containing code for implementation of random forest algorithm and error metric evaluation
* rand_forest_grid_search_results.csv contain the results of grid search of random forest hyper parameters and 5-fold cross-validation
* rand_forest_grid_groupby_mean.csv contains average AUC across folds for each set of hyperparameters (input dataframe corresponds to rand_forest_grid_search_results.csv)
* rand_forest_optimized_val_ROC_table.csv contains TPR and FPR data for selected for optimal probability threshold
* roc_plot_rf.py can be executed to produce ROC graph shown in rf_roc.png

5) Latex
* LatEx file for final project report, and accompanying figure picture files