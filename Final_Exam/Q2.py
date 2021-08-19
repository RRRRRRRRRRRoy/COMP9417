from sklearn.linear_model import Perceptron
import numpy as np
import itertools


# Entering these data from the Exam instruction
data1 = ["010", "011", "100", "111"]
data2 = ["011", "100", "110", "111"]
data3 = ["0100", "0101", "0110", "1000", "1100", "1101", "1110", "1111"]
data4 = ["1000000", "1000001", "1000101"]

print(len(data1[0]))
print(len(data2[0]))
print(len(data3[0]))
print(len(data4[0]))
X = np.asarray(list(itertools.product([0, 1], repeat=len(data4[-1]))))
y = -1 * np.ones(len(X))
print(X)
print(y)
