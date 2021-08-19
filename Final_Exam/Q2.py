from sklearn.linear_model import Perceptron
import numpy as np
import itertools


# Entering these data from the Exam instruction
data1 = ["010", "011", "100", "111"]
data2 = ["011", "100", "110", "111"]
data3 = ["0100", "0101", "0110", "1000", "1100", "1101", "1110", "1111"]
data4 = ["1000000", "1000001", "1000101"]

# For X in data4 which length is 7
# Therefore, the total situation in dataset X which is 2^7=128.
# This is because the current feature only has 0 or 1 which are 2 situations


def train_perceptron(data):
    # Generating all situation of 0/1 combination 2^n
    X = np.asarray(list(itertools.product([0, 1], repeat=len(data[-1]))))
    # Generating the traning y which contains only 1
    # Notice here the same length with X
    y = np.zeros(len(X))
    print(y)

    # For those features in the list X setting the result as 1
    # This means these features are positive features
    for item in data:
        # Change the data to the ten-based
        print(int(item, 2))
        y[int(item, 2)] = 1
    print(y)

    # After getting the true class of y then setting the value 0 as -1
    for index in range(len(y)):
        if y[index] == 0:
            y[index] = -1
    print(y)


score1 = train_perceptron(data1)
