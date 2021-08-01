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
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn import metrics
from sklearn.metrics import log_loss, confusion_matrix
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
print()
print(f"The shape of Training X {Train_X.shape}")
print(f"The shape of Training Y {Train_Y.shape}")
print(f"The shape of Testing Y {Test_X.shape}")
print(f"The shape of Testing Y{Test_Y.shape}")
print()

##############################################################################################################################
# The normal model training in naive_bayes
# Source: https://scikit-learn.org/stable/modules/naive_bayes.html
# Gaussian Naive Bayes: for the dataset which satisfied the Gaussian distribution
# Multinomial Naive Bayes: for the dataset which is categorical(has different classes)
# Bernoulli Naive Bayes: for the dataset which is satisfied binary distribution
##############################################################################################################################
# Setting Gaussian Naive Bayes model
GNB_model = GaussianNB()
# Setting Multinomial Naive Bayes model
MNB_model = MultinomialNB()
# Setting Bernoulli Naive Bayes model
BNB_model = BernoulliNB()
# Using these 3 models fit the training data
GNB_model.fit(Train_X, Train_Y.ravel())
MNB_model.fit(Train_X, Train_Y.ravel())
BNB_model.fit(Train_X, Train_Y.ravel())

##############################################################################################################################
# Using the current models to predict the data
# these 3 is used to calculate the accuracy training_accuracy and the log loss
##############################################################################################################################
# Calculating the result of Gaussian
Y_pred_GNB = GNB_model.predict(Test_X)
y_pred_GNB_Train = GNB_model.predict(Train_X)
Y_pred_proba_GNB = GNB_model.predict_proba(Test_X)

# Calculating the result of Multinomial
Y_pred_MNB = MNB_model.predict(Test_X)
y_pred_MNB_Train = MNB_model.predict(Train_X)
Y_pred_MNB_proba = MNB_model.predict_proba(Test_X)

# Calculating the result of Bernoulli
Y_pred_BNB = BNB_model.predict(Test_X)
y_pred_BNB_Train = BNB_model.predict(Train_X)
Y_pred_BNB_proba = BNB_model.predict_proba(Test_X)

##############################################################################################################################
# Getting the report for 3 model
# Metrics report function
# Source: https://scikit-learn.org/stable/modules/model_evaluation.html
# Log loss(which is a probability using the proba to predict)
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
##############################################################################################################################
normal_GNB_report = metrics.classification_report(Test_Y, Y_pred_GNB)
log_loss_result_GNB = log_loss(Test_Y, Y_pred_proba_GNB)

normal_MNB_report = metrics.classification_report(Test_Y, Y_pred_MNB)
log_loss_result_MNB = log_loss(Test_Y, Y_pred_MNB_proba)

normal_BNB_report = metrics.classification_report(Test_Y, Y_pred_BNB)
log_loss_result_BNB = log_loss(Test_Y, Y_pred_BNB_proba)

# Calculating the training accuracy for 3 models
GNB_Train_acc = accuracy_score(Train_Y, y_pred_GNB_Train)
MNB_Train_acc = accuracy_score(Train_Y, y_pred_MNB_Train)
BNB_Train_acc = accuracy_score(Train_Y, y_pred_BNB_Train)

# print the result
print("The report of GaussianNB")
print(normal_GNB_report)
print(f"The log loss of GaussianNB {log_loss_result_GNB}")
print(f"The Train Accuracy of GaussianNB {GNB_Train_acc}")
print()
print("The report of MultinomialNB")
print(normal_MNB_report)
print(f"The log loss of MultinomialNB {log_loss_result_MNB}")
print(f"The Train Accuracy of MultinomialNB {MNB_Train_acc}")
print()
print("The report of BernoulliNB")
print(normal_BNB_report)
print(f"The log loss of BernoulliNB {log_loss_result_BNB}")
print(f"The Train Accuracy of BernoulliNB {BNB_Train_acc}")

##############################################################################################################################
# Using the prediction result to draw the line graph to do the comparison
##############################################################################################################################
# Setting the data collection for each training model
checking_test_label = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                       "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
Test_GNB_counter = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                    "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
Test_MNB_counter = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                    "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
Test_BNB_counter = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                    "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}

# Getting the data collection for each training model
for item in Test_Y:
    checking_test_label[item[-1]] += 1
# print(checking_test_label)

for item in Y_pred_GNB:
    Test_GNB_counter[item] += 1
# print(Test_GNB_counter)

for item in Y_pred_MNB:
    Test_MNB_counter[item] += 1
# print(Test_MNB_counter)

for item in Y_pred_BNB:
    Test_BNB_counter[item] += 1
# print(Test_BNB_counter)

##############################################################################################################################
# Draw the line graph
# Notice: To tell the difference setting the linestyle and color
# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# Color Source:
# Source: https://matplotlib.org/stable/gallery/color/named_colors.html
##############################################################################################################################
plt.figure(figsize=(8, 8))
l1 = plt.plot(checking_test_label.keys(),
              checking_test_label.values(), 'r--', label='ground truth')
l2 = plt.plot(Test_GNB_counter.keys(), Test_GNB_counter.values(),
              'g--', label='GaussianNB')
l3 = plt.plot(Test_MNB_counter.keys(), Test_MNB_counter.values(),
              'b--', label='MultinomialNB')
l4 = plt.plot(Test_BNB_counter.keys(), Test_BNB_counter.values(),
              'y--', label='BernoulliNB')
plt.title('The Prediction VS Ground Truth')
plt.xlabel('Classes')
plt.ylabel('Number')
plt.legend()
plt.savefig('./NB_comparison.jpg')
plt.show()

##############################################################################################################################
# How to draw the confusion matirx
# Source:https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
##############################################################################################################################
confusion_GNB = confusion_matrix(Y_pred_GNB, Test_Y)
confusion_MNB = confusion_matrix(Y_pred_MNB, Test_Y)
confusion_BNB = confusion_matrix(Y_pred_BNB, Test_Y)
# Draw the confusion matrix for Gaussian
plt.figure(figsize=(4, 4))
plt.title('Gaussian Naive Bayesian')
plt.imshow(confusion_GNB, cmap=plt.cm.Blues)
plt.savefig('./GNB_confusion.jpg')
plt.show()

# Draw the confusion matrix for Multinomial
plt.figure(figsize=(4, 4))
plt.title('Multinomial Naive Bayesian')
plt.imshow(confusion_MNB, cmap=plt.cm.Blues)
plt.savefig('./MNB_confusion.jpg')
plt.show()

# Draw the confusion matrix for Bernoull
plt.figure(figsize=(4, 4))
plt.title('Bernoull Naive Bayesian')
plt.imshow(confusion_BNB, cmap=plt.cm.Blues)
plt.savefig('./BNB_confusion.jpg')
plt.show()
