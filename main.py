# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    data = np.arange(9)
    data = data.reshape(3, 3)
    # reshape 转换多行多列
    print(data)
    # 取行（左边包含:右边不包含）, 取列（左边包含:右边不包含）
    # 如果左边为单个数字，例如data[2,:]，表示只取第3行（下标0开始）
    # 如果右边为单个数字，例如data[:,3]，表示只取第4列（下标0开始）
    print(data[:2, :])

    print(np.uint8([1, 2, 259, 255, 0, -2]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
