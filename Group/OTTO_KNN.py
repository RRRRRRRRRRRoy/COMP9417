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
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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
print()

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
print(f"The shape of Training X {Train_X.shape}")
print(f"The shape of Training Y {Train_Y.shape}")
print(f"The shape of Testing Y {Test_X.shape}")
print(f"The shape of Testing Y {Test_Y.shape}")
print()

##############################################################################################################################
# Using different training the dataset to find the best n_neighbour value
# The current best value is n_neighbour=7
##############################################################################################################################
# Manully checking which parameter n_neighbour can performs best
checking_dict = dict()
for n in range(1, 25):
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    # fit the knn classifier
    knn_classifier.fit(Train_X, Train_Y.ravel())
    # using knn to do the prediction and calculate the log loss value
    predict_y_n = knn_classifier.predict(Test_X)
    # Using proba to calculate the log loss
    # log loss is a kind of probability
    predict_y_proba_n = knn_classifier.predict_proba(Test_X)
    acc_score = accuracy_score(Test_Y, predict_y_n)
    log_loss_n = log_loss(Test_Y, predict_y_proba_n)
    print(
        f"Current n_neighbour is {n}, the log_loss is {log_loss_n}, the accuracy score is {acc_score}")
    checking_dict.update({n: acc_score})

# How to sort the dictionary based on the value
# SOurce: https://blog.csdn.net/tangtanghao511/article/details/47810729
sorted_dict = sorted(checking_dict.items(),
                     key=lambda kv: (kv[1], kv[0]), reverse=True)
# Reverse can help to get the best performs value in the first location
best_n_neighbour = sorted_dict[0][0]
best_accuracy = sorted_dict[0][1]
print(
    f"The current best acc is {best_accuracy}, and the best n neighbours is {best_n_neighbour}")
##############################################################################################################################
# Using the best n_neighbors to retrain the KNN model
# KNN parameter n_neighbour understand
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# Source: https://blog.csdn.net/joshuajinxiaoshuai/article/details/80349264
##############################################################################################################################
# Using the best parameter to retrain the model
knn_classifier = KNeighborsClassifier(n_neighbors=best_n_neighbour)
# fit the knn classifier
knn_classifier.fit(Train_X, Train_Y.ravel())
# Getting the prediction result
predict_y = knn_classifier.predict(Test_X)
# Used to calculate the training accuracy
predict_y_train = knn_classifier.predict(Train_X)
# Used to calculate the log loss value
predict_y_proba = knn_classifier.predict_proba(Test_X)
##############################################################################################################################
# Print the metrics report of KNN model
# How to use metrics
# Source: https://scikit-learn.org/stable/modules/model_evaluation.html
# How to get the training accuracy
# Source: https://datascience.stackexchange.com/questions/28426/train-accuracy-vs-test-accuracy-vs-confusion-matrix
##############################################################################################################################
normal_KNN_report = metrics.classification_report(Test_Y, predict_y)
Train_acc = accuracy_score(Train_Y, predict_y_train)
log_loss_result = log_loss(Test_Y, predict_y_proba)
print("The report of KNN")
print(normal_KNN_report)
print(f"The log loss of KNN is {log_loss_result}")
print(f"The Training accuracy of KNN is {Train_acc}")
##############################################################################################################################
# Using the prediction result to draw the line graph to do the comparison
##############################################################################################################################
ground_truth_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                     "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
knn_result_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                   "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
##############################################################################################################################
# retriving the data from the prediction result
##############################################################################################################################
# Getting the data collection from test y
for item in Test_Y:
    ground_truth_dict[item[-1]] += 1
# print(ground_truth_dict)
# Getting the data collection from predict y
for item in predict_y:
    knn_result_dict[item] += 1
# print(knn_result_dict)
plt.figure(figsize=(8, 8))
l1 = plt.plot(ground_truth_dict.keys(),
              ground_truth_dict.values(), 'r--', label='ground truth')
l2 = plt.plot(knn_result_dict.keys(), knn_result_dict.values(),
              'g--', label='KNN_prediction')
plt.title('The Prediction VS Ground Truth')
plt.xlabel('Classes')
plt.ylabel('Number')
plt.legend()
plt.show()
##############################################################################################################################
# How to draw the confusion matirx
# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# Source: https://blog.csdn.net/diyong3604/article/details/101184214
##############################################################################################################################
confusion_KNN = confusion_matrix(predict_y, Test_Y)
print(confusion_KNN)
plt.figure(figsize=(8, 8))
plt.title('K-Nearest Neighbour')
plt.imshow(confusion_KNN, cmap=plt.cm.Blues)
plt.show()
