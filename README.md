## 常用命令
conda -V  查看版本  
conda info -e  查看所有环境  
conda create -n name python==3.7  创建环境  
conda activate name 激活环境  

conda install jupyter  
jupyter-notebook  

pip uninstall pyzmq 再执行 pip install pyzmq==19.0.2 修改jupyter-notebook通信问题

## 国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

## 解决 Requirement already satisfied: keras
pip install --target=d:\anaconda3\envs\ai\lib\site-packages keras

## 解决 Could not load dynamic library 'cudart64_110.dll'; dlerror:cudart64_110.dll not found
https://cn.dll-files.com/cudart64_110.dll.html  
下载以后放在C:\Windows\System32

## 视频监控项目
https://github.com/ultralytics/yolov5  
https://github.com/EricLee2021-72324/handpose_x

## 建模流程
数据探索 - 特征工程（标准差、平方） - 算法比较 - 算法选择 - 算法模型参数优化 - 模型融合

## 常用数据标记工具
Labelimg  
Labelme  
RectLabel  
OpenCV/CVAT  
LableBox  
Boobs

## 图像识别任务
分类：属于什么类别  
定位：左上角右上角坐标  
语义分割：将像素按照图像中表达含义的不同进行分割，识别像素点轮廓，不同类别用不同颜色区分  
检测模型：  
swin transformer：新的算法，类似yolov5  
SegNet：全卷积网络，常用于场景理解  
U-Net：全卷积网络，常用于医学图像分割  
OCR：光学文字识别，常用于提取图片文字、财务对账、支票检查  
