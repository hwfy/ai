# 图像增强
# 由于某种因素造成图像质量退化，希望获得对比度高、细节丰富、可读性好的图像，通过图像增强改善视觉质量，让观察者更清晰分析信息
# 用例：https://blog.csdn.net/weixin_36070282/article/details/113979026

import numpy as np
import cv2

# 读取图片
# img类型uint8，尺寸(566h, 915w)
img = cv2.imread("dog.jpg", 0)
cv2.imshow('image', img)
# 按键关闭图片
cv2.waitKey(0)
cv2.destroyAllWindows()

# 灰度直方图
# 线性变换：是实现图像增强的一种方式，公式：a*I(r,c)+b
# 当a=1, b=0时，即为原图的副本。
# 当a=2, b=0时，将原图的灰度级范围扩大两倍，提高对比度。
# 当a=0.5, b=0时，将原图的灰度级范围缩小两倍，降低对比度

# 每个数*2（提高对比度）
a = 2
# a必须转换float类型才能和img相乘
o = float(a)*img
# >255转换为255
o[o > 255] = 255
# 取整，小数四舍五入，当整数部分以0结束时，round函数一律是向下取整；另外np.ceil()向上,np.floor()向下取整
o = np.round(o)
# 转换成0-255
o = o.astype(np.uint8)
# 显示增强后的图像
cv2.imshow('enhance', o)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 直方图正规化：选择线性变换系数带有一定的盲目性，而直方图可以自动确定a和b的值

# 求出 img 最大最小值
imax = np.max(img)
imin = np.min(img)
# 最小灰度级和最大灰度级
omin, omax = 0, 255
# 求a和b
a = float(omax - omin)/(imax - imin)
b = omin - a*imin
print('a={0}, b={1}'.format(a, b))
# 线性变换
o = a*img + b
o = o.astype(np.uint8)
# 显示增强后的图片
cv2.imshow('enhance', o)
cv2.waitKey(0)
cv2.destroyAllWindows()
