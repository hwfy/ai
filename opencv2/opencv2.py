import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import sleep

# os.path.exsits(path)


def readImg():
    # opencv 读取图片通道颜色顺序为BGR（blue、green、red）
    img = cv2.imread('1.jpg')
    # BGR 转换 RGB 通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PIL 读取图片通道颜色顺序为RGB（red、green、blue）
    img2 = Image.open('1.jpg')  # 可带中文路径
    img_cv2 = np.array(img2)  # 转换成opencv格式

    # 显示图片
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_cv2)
    plt.show()
    plt.clf()

# 保存图片
# cv2.imwrite('11.jpg', img)


# 转换为灰度图
def convertGRAY():
    # 方法1
    img = cv2.imread('1.jpg', 0)
    # 方法2
    img2 = cv2.imread('1.jpg')
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 显示图片
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_gray)
    plt.show()
    plt.clf()


# 拆分3个通道
def splitImg():
    # 方法1
    img = cv2.imread('1.jpg')
    b, g, r = cv2.split(img)
    print(img.shape)
    # 方法2
    bb, gg, rr = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # 验证是否一致
    print(b - bb)


# 截取图片某些区域
def clippingImg():
    img = cv2.imread('1.jpg')
    print(img.shape)

    img_1 = img[0:200, 0:200]
    plt.imshow(img_1)
    plt.show()
    plt.clf()


# 线性变换
def linearTransform():
    img = cv2.imread('1.jpg')
    img = 255 + img

    print(np.max(img))  # 获取最大值、np.min()获取图片最小值、np.mean()获取图片均值

    plt.imshow(img)
    plt.show()
    plt.clf()


# 线性动画
def linearAnimation():
    img = cv2.imread('dog.jpg')

    # 模糊处理
    img_blur = cv2.blur(img, (1, 100))  # 横向、纵向模糊

    i = 0
    while 1:
        img2 = img + i

        # 高度、宽度、深度
        h, w, d = img2.shape
        img3 = cv2.resize(img2, (w+i, h-i))

        # 选择区域
        # img_roi = img[0:400+i, 0: 400+i]
        # # 调整颜色
        # img_roi = 255 - img_roi
        # # 放到指定位置
        # img[0:400+i, 100:500+i] = img_roi

        # x = 300 - i
        # if x < 0:
        #     x = 0
        #
        # y = 400 + i
        #
        # # 选择清晰区域
        # img_roi = img[x:y:, x:y]
        #
        # c = i
        # if c > 255:
        #     c = 0
        #
        # if c > 200:
        #     img_roi = c + img_roi
        #
        # # 放到模糊区域
        # img_blur[x:y:, x:y] = img_roi


        cv2.imshow('2', img3)
        i += 1

        # 检测键盘按键
        key = cv2.waitKey(1)

        if key == 27:  # 27对应esc
            cv2.destroyAllWindows()  # 关掉所有窗口
            break

        sleep(0.05)


# 尺寸调整
def resizeImg(rate=2):
    img = cv2.imread("dog.jpg")
    # 图片缩放
    # img1 = cv2.resize(img, (100, 50))

    # 等比例缩放
    h, w, d = img.shape  # 高度、宽度、深度
    hh, ww = int(h * rate), int(w * rate)

    print("原始高宽：", h, w)
    print("调整高宽：", hh, ww)

    img1 = cv2.resize(img, (ww, hh))

    plt.imshow(img1)
    plt.show()


# gamma变换
def gamma():
    img = cv2.imread("dog.jpg")
    img = img / 255  # rgb通道数值压缩到0-1之间

    img_gamma = np.power(img, 0.1)

    plt.imshow(img_gamma)
    plt.show()


# 区域处理
def regionalProcessing():
    img = cv2.imread('dog.jpg')

    # 选择区域
    img_roi = img[50:400, 600: 800]
    # 调整颜色
    img_roi = 255 - img_roi
    # 放到指定位置
    img[50:400, 100:300] = img_roi

    plt.imshow(img)
    plt.show()


# 模糊图像
def blurImg():
    img = cv2.imread('dog.jpg')

    # 模糊处理
    img_blur = cv2.blur(img, (1, 100))  # 横向、纵向模糊
    # plt.imshow(img_blur)
    # plt.show()

    # 选择清晰区域
    img_roi = img[:600, :400]
    # 放到模糊区域
    img_blur[:600, :400] = img_roi

    plt.imshow(img_blur)
    plt.show()


# 二值化 找轮廓
# 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，也就是将整个图像呈现出明显的只有黑和白的视觉效果。
def binarizationImg():
    im = cv2.imread('flower.jpg')
    # 转换灰度图
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # 将灰度图二值化，这里的10，原图中小于10的位置用0代替，大于10的用255代替
    ret, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)  # 当前二值化的方式

    # 寻找边缘点

    # 第一个参数：二值化图像

    # 第二个参数：轮廓检索模式
    # RETR_LIST: 提取所有轮廓、但不创建任何父子关系
    # RETR_EXTERNAL 返回最外边的轮廓、所有子轮廓被忽略
    # RETR_CCOMP 返回所有轮廓并分为两级组织结构
    # RETR_TREE 返回所有轮廓，创建一个完整的组织结构，会告诉你谁是爷爷、父亲、儿子、孙子等

    # 第三个参数：轮廓近似方法
    # cv2.CHAIN_APPROX_NONE 边界所有点会被存储
    # cv2.CHAIN_APPROX_SIMPLE 压缩轮廓，将轮廓冗余点去掉、比如四边形只存储四个点
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 对原图描边

    # 第一个参数：图片，可以原图或其他
    # 第二个参数：轮廓列表
    # 第三个参数：对轮廓（第二个参数）的索引，若要全部绘制可设为-1
    # 第四个参数：轮廓的颜色
    # 第五个参数Cont：轮廓的厚度
    cv2.drawours(im, contours, -1, (255, 0, 0), 3)

    # 显示
    plt.imshow(im)
    plt.show()


if __name__ == "__main__":
    binarizationImg()