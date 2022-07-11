import numpy as  np
import cv2

img = cv2.imread("mell.jpg", 0)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 图像增强

# 灰度直方图
# 线性变换

# 每个数*2
a = 2
o = float(a)*img
# >255转换为255
o[0 > 255] = 255

o = np.round(0)
# 转换整数
o = o.astype(np.uint8)
#
cv2.imshow('img', img)
cv2.imshow('enhance', 0)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 伽马运算

# 检测边缘

cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
