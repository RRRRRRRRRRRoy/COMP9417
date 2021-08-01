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
import xgboost as xgb
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
##############################################################################################################################
# Data Processing and Checking
# Using the 50000 data to do the training
# Using the extra 11878 data to do the testing
##############################################################################################################################
# Using pandas reading data
# Here change the location to your data location
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
num_class = len(list(labels))

# Checking the current information of the dataset
# Do not forget to use set() to delete the same information
print()
print(f"The number of Classes {num_class}")
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
# Training data
Train_X = Total_X[:50000]
Train_Y = Total_Y[:50000]
# Testing data
Test_X = Total_X[50000:]
Test_Y = Total_Y[50000:]
print()
print(f"The shape of Training X {Train_X.shape}")
print(f"The shape of Training Y {Train_Y.shape}")
print(f"The shape of Testing Y {Test_X.shape}")
print(f"The shape of Testing Y{Test_Y.shape}")


##############################################################################################################################
# Here the XGBoost is only accept number label
# Using a list change these Class_X label to numbers
##############################################################################################################################
Train_Y_num = list()
for item in Train_Y:
    Train_Y_num.append(int(item[-1][-1])-1)
# print(Train_Y_num)

##############################################################################################################################
# XGBoost is similar to Tensorflow before, using change the value type to DMatrix
# Source: https://xgboost.readthedocs.io/en/latest/python/python_api.html
# You can also find some code written like sklearn
##############################################################################################################################
Training_data_xgb = xgb.DMatrix(Train_X, Train_Y_num)
Testing_data_xgb = xgb.DMatrix(Test_X)
Training_data_xgb_X = xgb.DMatrix(Train_X)
# Notice: The parameters set as mlogloss the result is the probability
print()
params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss", "num_class": num_class}
print(f"The parameter of XGBoost are {params}")


##############################################################################################################################
# Using XGBoost to train the model
# The current boost round is 200
# SOurce: https://www.programcreek.com/python/example/99829/xgboost.train
# This process you can also write in sklearn style
# Chinese Source: https://www.cnblogs.com/pinard/p/11114748.html
# 10   acc: 0.77
# 50   acc: 0.80
# 100   acc: 0.81
# 120   acc: 0.81
# 130   acc: 0.82
# 180   acc: 0.82
# 300   acc: 0.82
# 1000   acc: 0.82
##############################################################################################################################
# number of iterations - initial 20,
XGBoost_model = xgb.train(params, Training_data_xgb, 200)
# Getting the prediction ---> accuracy
predict_Y_xgb = XGBoost_model.predict(Testing_data_xgb)
# Getting the Training accuracy
predict_train_xgb = XGBoost_model.predict(Training_data_xgb_X)

##############################################################################################################################
# the current result is probability therefore select the largest result to get the index
# Using the where and max to select
# Source: https://numpy.org/doc/stable/reference/generated/numpy.where.html
# Source: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html
##############################################################################################################################
result_list = list()
for item in predict_Y_xgb:
    result_list.append(np.where(item == max(item))[0].tolist()[0]+1)
# print(result_list)

Train_acc_list = list()
for item in predict_train_xgb:
    Train_acc_list.append(np.where(item == max(item))[0].tolist()[0]+1)
# print(Train_acc_list)

##############################################################################################################################
# Change the Class_X label to the number label to do the prediction
##############################################################################################################################
ground_true_label_Y = [int(item[-1][-1]) for item in Test_Y]
Train_acc_label_Y = [int(item[-1][-1]) for item in Train_Y]
# print(ground_true_label_Y)

# 130 0.4691932772102278
# 180 0.46355069379757285
# 200 0.46340078275644003
log_loss_result = log_loss(Test_Y, predict_Y_xgb)
print(log_loss_result)

# Using the metrics and the label list to get the result
normal_xgboost_predict = metrics.classification_report(
    ground_true_label_Y, result_list)
train_acc_xgboost = accuracy_score(Train_acc_label_Y, Train_acc_list)
print()
print("The metrics report of XGBoost:")
print(normal_xgboost_predict)
print(f"The log loss result of XGBoost is {log_loss_result}")
print(f"The Train Accuracy of XGBoost {train_acc_xgboost}")
print()

##############################################################################################################################
# Using the prediction result to draw the line graph to do the comparison
##############################################################################################################################
ground_truth_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                     "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}
xgboost_result_dict = {"Class_1": 0, "Class_2": 0, "Class_3": 0, "Class_4": 0,
                       "Class_5": 0, "Class_6": 0, "Class_7": 0, "Class_8": 0, "Class_9": 0}

# Counting the result in the ground truth list
for item in ground_true_label_Y:
    ground_truth_dict[f"Class_{item}"] += 1

# Counting the result in the prediction list
for item in result_list:
    xgboost_result_dict[f"Class_{item}"] += 1

##############################################################################################################################
# Draw the line graph
# Notice: To tell the difference setting the linestyle and color
# Source: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
# Color Source:
# Source: https://matplotlib.org/stable/gallery/color/named_colors.html
##############################################################################################################################
plt.figure(figsize=(8, 8))
l1 = plt.plot(ground_truth_dict.keys(),
              ground_truth_dict.values(), 'r--', label='ground truth')
l2 = plt.plot(xgboost_result_dict.keys(),
              xgboost_result_dict.values(), 'g--', label='xgboost_prediction')
plt.title('The Prediction VS Ground Truth')
plt.xlabel('Classes')
plt.ylabel('Number')
plt.legend()
plt.show()

##############################################################################################################################
# Draw the confusion matrix
# Source:https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
##############################################################################################################################
confusion_xgboost = confusion_matrix(result_list, ground_true_label_Y)
plt.figure(figsize=(8, 8))
plt.title('XGBoost Confusion Matrix')
plt.imshow(confusion_xgboost, cmap=plt.cm.Blues)
plt.show()


##############################################################################################################################
# Generating the csv result
##############################################################################################################################
# Valid_x = np.array(raw_test_data.iloc[:, 1:94])
# Total_x = np.array(Train_data.iloc[:,0:93])
# Total_y = np.array(Train_data.iloc[:,93:94])
# print(Total_x.shape)
# print(Total_y.shape)

# Total_aaa = list()
# for item in Total_y:
#     Total_aaa.append(int(item[-1][-1])-1)
# Total_training = xgb.DMatrix(Total_x,Total_aaa)
# Total_test = xgb.DMatrix(Valid_x)
# total_xgb = xgb.train(params, Total_training, 200)
# total_prediction = gbm.predict(Total_test)

# submission = pd.DataFrame({ "id": raw_test_data["id"]})
# i = 0
# for num in range(1,10):
#     col_name = str("Class_{}".format(num))
#     submission[col_name] = total_prediction[:,i]
#     i = i + 1
# submission.to_csv('./otto.csv', index=False)
