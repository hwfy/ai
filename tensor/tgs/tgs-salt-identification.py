import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 进度条
from tqdm import tqdm
# 可以对多个容器的不同对象做循环迭代（链接多个不同类型的矩阵）
# 例如：for item in chain([1, 2, 3], ['a', 'b', 'c']):
from itertools import chain

# pip install scikit-image
# 读取图片
from skimage.io import imread, imshow, concatenate_images
# 修改图片尺寸
from skimage.transform import resize
# 根据相邻情况从左到右，从上到下划分区域,并对区域建立索引
from skimage.morphology import label
# 数据分割
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers import Lambda, RepeatVector, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, GlobalMaxPool2D
from keras.layers import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 图片宽度
IMG_WIDTH = 128
# 图片高度
IMG_HEIGHT = 128
# 图片深度，灰度图（单通道）
IMG_CHANNELS = 1
# 获取当前文件绝对路径
local = os.getcwd()
# 拼接路径：/input/train，
# path.join拼接的路径兼容任何平台
TRAIN_PATH = os.path.join(local, 'input', 'train')
TEST_PATH = os.path.join(local, 'input', 'test')
# 保存模型的命名
WEIGHT_FILE = 'model.h5'

# 展示这5个图片
ids = ['1f1cc6b3a4', '5b7c160d0d', '6c40978ddf', '7dfdf6eeb8', '7e5a6e5013']
plt.figure(figsize=(20, 10))
for j, img_name in enumerate(ids):
    q = j+1
    # 加载图片
    img = load_img(os.path.join(TRAIN_PATH, 'images', img_name+'.png'))
    img_mask = load_img(os.path.join(TRAIN_PATH, 'masks', img_name+'.png'))

    # 绘制子图：1行12列，位置为基数
    plt.subplot(1, 2*(1+len(ids)), q*2-1)
    plt.imshow(img)
    # 绘制子图：1行12列，位置为偶数
    plt.subplot(1, 2*(1+len(ids)), q*2)
    plt.imshow(img_mask)

plt.show()


# Get and resize train images and masks
def get_data(path, train=True):
    # 遍历目录，获取所有文件名
    # walk返回：三元组(root,dirs,files)，root文件夹的本身的地址、dirs目录的名字(不包括子目录)、files文件名(不包括子目录)
    ids = next(os.walk(path + "/images/"))[2]
    # zeros：用0填充数组
    X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm(enumerate(ids), total=len(ids)):
        # 加载图像，灰度
        img = load_img(path + '/images/' + id_, color_mode="grayscale")
        # 图像转换数组
        x_img = img_to_array(img)
        # 重置大小
        # mode=constant: 用常量0填充
        # preserve_range=True：保持原来的模式，skimage.transform.resize会顺便把图片的像素归一化缩放到(0,1)区间内
        x_img = resize(x_img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, color_mode="grayscale"))
            # np.uint8: x>255则x-256，x<0则x+256
            mask = np.uint8(resize(mask, (IMG_HEIGHT, IMG_WIDTH, 1), mode='constant', preserve_range=True))

        # Save images
        # squeeze 去掉维度为1的维度，例如（4000，128，128，1），变成（4000，128，128）
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X


X, y = get_data(TRAIN_PATH, train=True)
# (4000, 128, 128, 1) float32 (4000, 128, 128, 1) float32
print(X.shape, X.dtype, y.shape, y.dtype)

# 划分训练集和测试集
# test_size：划分比例
# random_state：随机种子，保持每一次划分数据不变
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=2021)
# (2000, 128, 128, 1) (2000, 128, 128, 1) (2000, 128, 128, 1) (2000, 128, 128, 1)
print(X_train.shape, X_valid.shape, y_train, y_valid)


# 构建U-Net网络模型
def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    # Conv2D卷积层
    # filters：卷积核个数，16、32、64
    # kernel_size：卷积核大小，3*3
    # kernel_initializer：权重初始化器, he_normal（he正态分布初始化器）
    # padding: same 即使通过kernel_size缩小了维度，但是四周会填充0，保持原先的维度
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    # BatchNormalization：对每一批数据进行归一化
    if batchnorm:
        x = BatchNormalization()(x)
    # 激活函数
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    # 对于神经网络单元，按照一定的概率将其暂时从网络中丢弃，主要为了解决过拟合
    # 什么是过拟合？在机器学习的模型中，如果模型的参数太多，而训练样本又太少，训练出来的模型很容易产生过拟合的现象。
    # 过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低
    # 参数p：单个神经元被丢弃的概率，例如dropout选择0.5，那么这一层神经元经过dropout后，1000个神经元中会有大约500个的值被置为0
    p1 = Dropout(dropout * 0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # expansive path
    # Conv2DTranspose：转置卷积层，前面一系列卷积操作之后特征图分辨率很小了，利用转置卷积来提升特征图的大小
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    # concatenate：链接矩阵
    # 为何要链接 转置卷积和卷积层 ？
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    # 创建输出层：1个卷积核，1*1尺寸
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    # 创建模型
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# 初始化一个keras tensor
input_img = Input((IMG_HEIGHT, IMG_WIDTH, 1), name='img')
# 获取unet模型
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
# 编辑模型，设置优化器和损失函数
# binary_crossentropy：二元交叉熵，交叉熵一般用于求目标与预测值之间的差距
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
# model.compile(optimizer=Adam(), loss=tf.keras.losses.BinaryCrossentropy(from_logits = False), metrics=["accuracy"])
model.summary()

# # 参数参考：https://www.cnblogs.com/Renyi-Fan/p/13703325.html
# callbacks = [
#     # patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
#     EarlyStopping(patience=10, verbose=1),
#     # 学习率动态调整
#     # factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
#     # patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
#     # cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
#     # min_lr：学习率的下限
#     ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#     # WEIGHT_FILE：模型名称
#     # verbose：信息展示模式，0/1
#     # save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
#     # save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
#     ModelCheckpoint(WEIGHT_FILE, verbose=1, save_best_only=True, save_weights_only=True)
# ]
# # 训练模型
# # callbacks：在训练过程中的适当时机被调用，实时保存训练模型以及训练参数
# # validation_data：在每个epoch之后或者每⼏个epoch,验证⼀次验证集,⽤来及早发现过拟合、超参数等设置问题，⽅便我们及时调整
# results = model.fit(X_train, y_train, batch_size=32, epochs=10, callbacks=callbacks,
#                     validation_data=(X_valid, y_valid))

# 仅读取权重，load_model()读取网络、权重
model.load_weights(WEIGHT_FILE)
# 评估模型：输入数据和标签,输出损失和精确度
# 通常参数为：x_test特征, y_test标签；有时候不知道为啥参数为 y_train和y_test ？
model.evaluate(X_valid, y_valid, verbose=1)

# 模型预测：输入测试集,输出预测结果（numpy类型）
# 这里不知道为啥预测训练集X_train ？ 一般都只预测X_valid看是否和y_valid一致
preds_train = model.predict(X_train, verbose=1)
preds_val = model.predict(X_valid, verbose=1)

# astype：转换uint8（0-255）
# 预测结果为0-1之间的小数，转换以后变成0或1，转换二进制数据以后图片更加清晰
# [[[[0.30350286][0.39646116][0.51261187]]]] 转换以后 [[[[0][0][1]]]]
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


# 展示预测结果
def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='seismic')
    if has_mask:
        ax[0].contour(y[ix].sequeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Seismic')

    ax[1].imshow(y[ix].squeeze())
    ax[1].set_title('Salt')

    ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), color='k', levels=[0.5])
    ax[2].set_title('Salt Predicted')

    ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[3].contour(y[ix].squeeze(), colors='k', levels=[0, 5])
    ax[3].set_title('Salt Predicted binary')

    plt.show()


# Check if training data looks all right
plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)
