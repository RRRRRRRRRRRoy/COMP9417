import numpy as np

arr = np.arange(15).reshape(3, 5)

print(arr)
print(arr.T)
print(arr.T.T)
print(arr.dot(arr.T))
