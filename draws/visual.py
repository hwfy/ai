# 可视化
import numpy as np
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    # 数据可视化
    # x轴 年份
    x3_1 = np.array(['2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'])
    # y1 出境游客
    y1 = np.array([11659.2, 12786, 13535, 14563, 16199, 17026, 6542, 11952])
    # y2 入境游客
    y2 = np.array([12849, 13382, 13845, 13928, 14119, 15248, 3584, 8426])
    # 解决中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.scatter(x3_1, y1)
    plt.scatter(x3_1, y2)
    # 画x轴标题
    plt.xlabel("年份", fontsize=15)
    plt.ylabel('人数（万人次）', fontsize=15)
    plt.title('出入境游客', fontsize=20)
    plt.plot(x3_1, y1, label='出境游客')
    plt.plot(x3_1, y2, label='入境游客')
    plt.legend()
    plt.show()