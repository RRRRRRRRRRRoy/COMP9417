##############################################################################################################################
##############################################################################################################################
# This file is for COMP9417 Group project
# Using Otto Product dataset to do the training and testing
# Data Download link
# Source: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
#
# This is a old question you can also find some information for other guys resources
# Source: https://github.com/wepe/Kaggle-Solution/tree/master/Otto%20Group%20Product%20Classification%20Challenge
# Source: https://github.com/amaltarghi/Otto-Group-Product-Classification-Challenge
# Source: https://github.com/zhouhaozeng/kaggle-otto-classification
##############################################################################################################################
##############################################################################################################################

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import log_loss

##############################################################################################################################
# Data Processing and Checking
# Using the 50000 data to do the training
# Using the extra 11878 data to do the testing
##############################################################################################################################
# Using pandas reading data
# !!!!!!!!!!!!!!! Here change the location to your data location !!!!!!!!!!!!!!!
# Downloading from Kaggle
# Source: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
raw_train_data = pd.read_csv("D:\\train.csv")
raw_test_data = pd.read_csv("D:\\test.csv")
print(raw_train_data.shape)
print(raw_test_data.shape)

# using iloc provided by pandas to cutting the data into training set and test set
# SOurce: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
Train_data = raw_train_data.iloc[:, 1:95]
labels = set(Train_data.iloc[:, 93:94].target)
Valid_x = np.array(raw_test_data.iloc[:, 1:94])
Total_x = np.array(Train_data.iloc[:, 0:93])
Total_y = np.array(Train_data.iloc[:, 93:94])

# Checking the current information of the dataset
# Do not forget to use set() to delete the same information
print(f"The current has the following label {labels}")
print(f"The shape the Total_X {Total_x.shape}")
print(f"The shape the Total_Y {Total_y.shape}")
print(f"The shape the Valid_X {Valid_x.shape}")
print(f"Check Nan in Total_X {set(np.isnan(Total_x).any(axis=1))}")
# print(f"Check Nan in Train_Y {np.isnan(Train_Y).any(axis=1)}")
print(f"Check Nan in Valid_X {set(np.isnan(Valid_x).any(axis=1))}")

# setting the randseed and using shuffle to mess up the dataset
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
np.random.seed(10)
Total_X, Total_Y = shuffle(Total_x, Total_y)

# In this part using the previous 50000 to do the training and the last data to do the test check the score
# previous 50000 for training 11878 for testing
# Training Data
Train_X = Total_X[:50000]
Train_Y = Total_Y[:50000]
# Testing Data
Test_X = Total_X[50000:]
Test_Y = Total_Y[50000:]
# Checking the shape of the data
print()
print(f"The shape of Training X {Train_X.shape}")
print(f"The shape of Training Y {Train_Y.shape}")
print(f"The shape of Testing Y {Test_X.shape}")
print(f"The shape of Testing Y{Test_Y.shape}")

##############################################################################################################################
# The normal decision tree model
# Here setting the random_state as 0
# Source:https://scikit-learn.org/stable/modules/tree.html
##############################################################################################################################
# How to use metrics
# Source: https://scikit-learn.org/stable/modules/model_evaluation.html
# How to get the training accuracy
# Source: https://datascience.stackexchange.com/questions/28426/train-accuracy-vs-test-accuracy-vs-confusion-matrix
##############################################################################################################################
# Using the basic DecisionTree to do the training
DT_normal_model = DecisionTreeClassifier(random_state=0)
DT_normal_model.fit(Train_X, Train_Y.ravel())
DT_norm_predict = DT_normal_model.predict(Test_X)
DT_norm_predict_proba = DT_normal_model.predict_proba(Test_X)
# print(DT_norm_predict)
normal_DT_report = metrics.classification_report(Test_Y, DT_norm_predict)
log_loss_result_DT = log_loss(Test_Y, DT_norm_predict_proba)
print()
print("Here is the report of DecisionTree")
print(normal_DT_report)
print(
    f"Here is the log loss value of Decision Tree model {log_loss_result_DT}")
print()
##############################################################################################################################
# Using the Boosting method to boost decision tree model
# Here using the AdaBoostClassifier to optimize DT
# Source:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
##############################################################################################################################
DT_Ada_model = AdaBoostClassifier(DecisionTreeClassifier(
), n_estimators=200, learning_rate=0.1, random_state=0)
DT_Ada_model.fit(Train_X, Train_Y.ravel())
DT_Ada_predict = DT_Ada_model.predict(Test_X)
DT_Ada_predict_proba = DT_Ada_model.predict_proba(Test_X)
Ada_DT_report = metrics.classification_report(Test_Y, DT_Ada_predict)
Ada_DT_log_loss = log_loss(Test_Y, DT_Ada_predict_proba)
print("Here is the report of Adaboosted DecisionTree model")
print(Ada_DT_report)
print(
    f"Here is the log loss value of Adaboosted Decision Tree model {Ada_DT_log_loss}")

##############################################################################################################################
# Only using AdaBoostClassifier to do the classification
# setting the n_estimators=200
# # Source:https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
##############################################################################################################################
Ada_Model = AdaBoostClassifier(n_estimators=200, random_state=0)
Ada_Model.fit(Train_X, Train_Y.ravel())
Ada_Model_predict = Ada_Model.predict(Test_X)
Ada_Model_predict_proba = Ada_Model.predict_proba(Test_X)
Ada_Model_report = metrics.classification_report(Test_Y, Ada_Model_predict)
Ada_Model_loss = log_loss(Test_Y, Ada_Model_predict_proba)
print("Here is the report of Adaboost model")
print(Ada_Model_report)
print(
    f"Here is the log loss value of Adaboost model {Ada_Model_loss}")

##############################################################################################################################
# Using the prediction result to draw the line graph to do the comparison
##############################################################################################################################
ground_truth_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                     "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
DT_result_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                  "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
Ada_DT_result_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                      "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
Ada_result_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                   "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}

##############################################################################################################################
# retriving the data from the prediction result
##############################################################################################################################
print()
print("Here is the result collection")
for item in Test_Y:
    ground_truth_dict[item[-1]] += 1
print(f"ground_truth_dict: {ground_truth_dict}")

for item in DT_norm_predict:
    DT_result_dict[item] += 1
print(f"DT_result_dict: {DT_result_dict}")

for item in DT_Ada_predict:
    Ada_DT_result_dict[item] += 1
print(f"Ada_DT_result_dict: {Ada_DT_result_dict}")

for item in Ada_Model_predict:
    Ada_result_dict[item] += 1
print(f"Ada_result_dict: {Ada_result_dict}")
print()
##############################################################################################################################
# Draw the line graph
# Notice: To tell the difference setting the linestyle and color
# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# Color Source:
# Source: https://matplotlib.org/stable/gallery/color/named_colors.html
##############################################################################################################################
plt.figure(figsize=(8, 8))
l1 = plt.plot(ground_truth_dict.keys(), ground_truth_dict.values(),
              color='red', linestyle='--', label='ground truth')
l2 = plt.plot(DT_result_dict.keys(), DT_result_dict.values(),
              color='green', linestyle='--', label='DT normal')
l3 = plt.plot(Ada_DT_result_dict.keys(), Ada_DT_result_dict.values(),
              color='black', linestyle='--', label='DT AdaBoostClassifier')
l4 = plt.plot(Ada_result_dict.keys(), Ada_result_dict.values(),
              color='blue', linestyle='--', label='AdaBoostClassifier')
plt.title('The Prediction VS Ground Truth')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.savefig('./Adaboost_DT_comparison.jpg')
plt.show()

##############################################################################################################################
# How to draw the confusion matirx
# Source:https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
##############################################################################################################################
# Draw confusion matrix for Decision Tree model
DT_Confusion = confusion_matrix(DT_norm_predict, Test_Y)
plt.figure(figsize=(8, 8))
plt.title('Decision Tree')
plt.imshow(DT_Confusion, cmap=plt.cm.Blues)
plt.savefig('./DT_Classifier_confusion.jpg')
plt.show()

# Draw confusion matrix for Adaboost Decision Tree model
Ada_DT_Confusion = confusion_matrix(DT_Ada_predict, Test_Y)
plt.figure(figsize=(8, 8))
plt.title('Adaboosted Decision Tree')
plt.imshow(Ada_DT_Confusion, cmap=plt.cm.Blues)
plt.savefig('./Adaboost_DT_Classifier_confusion.jpg')
plt.show()

# Draw confusion matrix for Adaboost model
Ada_Confusion = confusion_matrix(Ada_Model_predict, Test_Y)
plt.figure(figsize=(8, 8))
plt.title('Adaboosted Classifier')
plt.imshow(Ada_Confusion, cmap=plt.cm.Blues)
plt.savefig('./Adaboost_Classifier_confusion.jpg')
plt.show()
