import cv2
import numpy as np


# 激活函数
def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))


# l0 = 总共784个 0-9之间随机数
l0 = np.random.randint(0, 10, 784)

# 正态分布：normal
# 参数1：正态分布的均值，对应着这个分布的中心。loc=0说明以Y轴为对称轴的正态分布
# 参数2：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
# 参数3：int或者整数元组
# w0 = 16行784列矩阵
w0 = np.random.normal(0, 1, (16, 784))

# 偏执，
# b0 = 一维数组16个元素
b0 = np.random.normal(0, 1, (16,))

# dot矩阵相乘累加，w0中784列每个数 * l0中每个数 再累加 最后得到16个数，+b0表示这16个数分别+b0中对应数
# l1 = 一维数组16个元素
l1 = w0.dot(l0)+b0  # x*k+b
print('激活前：', l1)
# sigmoid：将线性l1 转变 为非线性，返回值在0-1之间，0-0.5表示低 0.5-1表示高，主要起到分类作用
l1 = sigmoid(l1)
print('激活后：', l1)

w1 = np.random.normal(0, 1, (16, 16))
b1 = np.random.normal(0, 1, (16,))
l2 = w1.dot(l1)+b1
print('激活前1：', l2)
l2 = sigmoid(l2)
print('激活后1：', l2)

w2 = np.random.normal(0, 1, (10, 16))
b2 = np.random.normal(0, 1, (10,))
l3 = w2.dot(l2)+b2
print("激活前2：", l3)
l3 = sigmoid(l3)
print("激活后2：", l3)


# imread：读取图片，第二个参数：0灰度模式加载图片、1彩色图片(默认)、-1包括alpha(透明度通道)
img9 = np.array(cv2.imread("2.jpg", 0), dtype="float64")
print(img9)
