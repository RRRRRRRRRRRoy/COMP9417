from anytree import Node, RenderTree
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd
import itertools

###############################################################################################
###############################################################################################
# Question2 b
###############################################################################################
###############################################################################################
# Entering these data from the Exam instruction
# For X in data4 which length is 7
# Therefore, the total situation in dataset X which is 2^7=128.
# This is because the current feature only has 0 or 1 which are 2 situations
data1 = ["010", "011", "100", "111"]
data2 = ["011", "100", "110", "111"]
data3 = ["0100", "0101", "0110", "1000", "1100", "1101", "1110", "1111"]
data4 = ["1000000", "1000001", "1000101"]


def train_perceptron(data_list: list):
    score_list = list()
    iterations = 10000
    learning_rate = 1.0
    for data in data_list:
        # Generating all situation of 0/1 combination 2^n
        # Generating the traning y which contains only 1
        # Notice here the same length with X
        Generating_X_list = list(
            itertools.product([0, 1], repeat=len(data[-1])))
        X = np.asarray(Generating_X_list)

        y = np.zeros(len(X))
        # print(y)

        # For those features in the list X setting the result as 1
        # This means these features are positive features
        for item in data:
            # Change the data to the ten-based
            # Source Convert binary to ten-based
            # https://www.delftstack.com/howto/python/convert-binary-to-int-python/
            y[int(item, 2)] = 1
        # print(y)

        # After getting the true class of y then setting the value 0 as -1
        for index in range(len(y)):
            if y[index] == 0:
                y[index] = -1
        # print(y)

        # Notice: the iteration is 10000 and the learning rate is 1.0
        perceptron_classifier = Perceptron(
            max_iter=iterations, eta0=learning_rate, random_state=0)
        perceptron_classifier.fit(X, y)
        score = perceptron_classifier.score(X, y)
        score_list.append(score)
    return score_list


def check_score_list(score_list: list):
    result_list = list()
    for score in score_list:
        if score <= 0.5:
            result_list.append("No")
        else:
            result_list.append("Yes")
    return result_list


def analysis_result(result_list: list):
    for index in range(len(result_list)):
        if result_list[index] == "No":
            print(f"The current data{index} is not linear separable")
        else:
            print(f"The current data{index} is linear separable")
    print()
    print("Dataset          Linearly Separable (Yes/No)")
    for index in range(len(result_list)):
        print(f"    {index}                 {result_list[index]}")


score_list = train_perceptron([data1, data2, data3, data4])
result_list = check_score_list(score_list)
print()
print("Here is the result of Question2 B")
analysis_result(result_list)
print()
