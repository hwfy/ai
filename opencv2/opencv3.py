# 实现风格迁移
# 示例：https://www.jianshu.com/p/fa4b9da2b0bb

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time
import functools
import tensorflow_hub as hub

# 设置画布大小
mpl.rcParams['figure.figsize'] = (12, 12)
# 关闭网格线
mpl.rcParams['axes.grid'] = False


# tensor转换图片
def tensor_to_image(tensor):
    # tensor中颜色区间0-1，而RGB颜色区间为0-255，因此需要*255
    tensor = tensor*255
    # tensor的类型为float32，因此转换为整型
    tensor = np.array(tensor, dtype=np.uint8)
    # np.ndim：返回数组的维度，这里是4（1, 316, 512, 3）
    if np.ndim(tensor) > 3:
        # assert：断言如果值为true继续执行，否则抛出异常
        assert tensor.shape[0] == 1
        # image是三维的，只取（316，512，3）
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


# 从远程下载图片，并命名为YellowLabradorLooking_new.jpg 保存在C盘...\.keras\datasets
content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
style_path = tf.keras.utils.get_file('kandinsky5.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
print('图片保存在：', content_path)


# 读取图片，并转换为tensor
# 参考：https://zhuanlan.zhihu.com/p/118242485
def load_img(path_to_img):
    max_dim = 512
    # 函数用于读取文件，相当于python的open()函数，二进制，dtype=string
    img = tf.io.read_file(path_to_img)
    # 一般要配合tf.image.decode_image()函数对图片解码，得到宽*高*色深，范围0-255，shape=(566, 915, 3), dtype=uint8
    img = tf.image.decode_image(img, channels=3)
    # 将一个uint类型的tensor转换为float类型,该方法会自动对数据进行归一化处理,将数据缩放到0-1
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 减掉最后一维色深，转为浮点类型，得到宽*高，[566. 915.]
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # 获取图片长端，915.0
    long_dim = max(shape)
    # 以长端为比例缩放，让图片成为512x???，0.55956286
    scale = max_dim / long_dim
    # [316 512]
    new_shape = tf.cast(shape*scale, tf.int32)
    # 实际缩放图片，shape=(316, 512, 3), dtype=float32
    img = tf.image.resize(img, new_shape)
    # 再扩展一维，（1，长，宽，色深），shape=(1, 316, 512, 3), dtype=float32
    # tensor的类型都是4维，第1维是1，数值范围0-1
    img = img[tf.newaxis, :]
    return img


# 显示tensor图像
def imshow(image, title=None):
    # 如果是tensor四维向量，则移除大小为1的维度，image变成 宽*高*色深
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)

    if title:
        plt.title(title)


# https://hub.tensorflow.google.cn/google/magenta/arbitrary-image-stylization-v1-256/1
# 由于url下载会异常，因此从本地路径加载
hub_module = hub.load(r'./arbitrary-image-stylization-v1-256')
# 加载原图和风格图
content_image = load_img('dog.jpg')
style_image = load_img('./train_data/MN/689.jpg')

# 风格迁移
stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

# 显示最终图片
plt.subplot(1, 3, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 3, 2)
imshow(style_image, 'Style Image')

plt.subplot(1, 3, 3)
imshow(stylized_image, 'Finish Image')
plt.show()
