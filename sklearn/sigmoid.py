import cv2
import numpy as np


# 激活函数
def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


l0 = np.random.randint(0, 10, 784)
print(l0)

w0 = np.random.normal(0, 1, (16, 784))
# 偏执
b0 = np.random.normal(0, 1, (16,))
l1 = w0.dot(l0)+b0  # x*k+b
l1 = sigmoid(l1)

w1 = np.random.normal(0, 1, (16, 16))
b1 = np.random.normal(0, 1, (16,))
l2 = w1.dot(l1)+b1
l2 = sigmoid(l2)

w2 = np.random.normal(0, 1, (10, 16))
b2 = np.random.normal(0, 1, (10,))
l3 = w2.dot(l2)+b2
l3 = sigmoid(l3)

print(l3)
print(l3.shape)

# img9 = np.array(cv2.imread("2.jpg", 0), dtype="float64")
#
# l3 = sigmoid(13)
# print(l3)
# print(l3.shape)
# print(img9)