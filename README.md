## 常用命令
conda -V  查看版本  
conda info -e  查看所有环境  
conda create -n name python==3.7  创建环境  
conda activate name 激活环境  

# python网页版开发环境，相对于IDE来说，它可以单独执行某段代码
conda install jupyter  
jupyter-notebook  

pip uninstall pyzmq 再执行 pip install pyzmq==19.0.2 修改jupyter-notebook通信问题

## 配置国内镜像源，解决pip安装缓慢问题
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
定位：左上角右下角坐标  
语义分割：将像素按照图像中表达含义的不同进行分割，识别像素点轮廓，不同类别用不同颜色区分  
模型检测网络：swin transformer新的算法、SegNet常用于场景理解、U-Net常用于医学图像分割、OCR常用于提取图片文字、财务对账、支票检查等


## yolov5图像识别步骤

### 一、图片打标 
##### 1、安装打标工具  
``` pip install labelImg ```  
##### 2、命令行运行工具  
```  labelimg ```  

![打标界面](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/apply.png)
注意右边选择"YOLO"，最后导出的文件才是yolov支持的txt格式  

##### 3、保存标签文件  
![打标存储目录](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/save_dir.png)  
存储目录一般命名为"labels"  

### 二、划分训练集和测试集
##### 1、修改 yolov/labelme/gen_train_val_txt.py文件
![划分测试集和训练集](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/cfg_train_val.png) 
input_path：原始图片目录  
output_path：训练集和测试集输出路径  
txt_path：上一步打标文件路径  

##### 2、运行脚本生成train.txt和val.txt  
```python gen_train_val_txt.py```  

![查看测试集和训练集](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/gen_train_val.png)

### 三、开始训练
##### 1、yolov配置
![yolov配置](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/cfg_yolov.png)
path：打标文件目录  
train：上一步生成的训练集文件  
val：上一步生成的测试集文件  
test：不需要配置

##### 2、训练文件配置
![yolov训练配置](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/cfg_yolov_train.png)
cfg：引入官方模型文件  
epochs：训练轮数，有GPU情况下设置300、100都行，不然设置几十即可，为了速度而忽略模型准确性，我设置10  
batch-size：每一轮处理的图片数，越大训练越快但耗费内存越高，有时候会提示：内存不足，就要调小点，我设置8  

##### 3、在项目目录下执行  
```python train.py```  

##### 4、查看训练准确性  
![查看训练结果](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/train_result.png)  

### 四、验证结果
##### 1、验证配置
![yolov验证配置](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/cfg_yolov_detect.png)
weight：这里需要配置刚训练出的模型，选best.pt  
source：一般在命令行输入图片路径，也可以填写在配置里，为0代表用摄像头  

##### 2、在项目目录下执行  
```python detect.py --source ..\VOC2007\images\000021.jpg```  

![yolov验证](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/run_detect.png)

##### 3、查看验证结果  
![yolov验证结果](https://raw.githubusercontent.com/hwfy/ai/master/yolov/labelme/show_detect.png)  
结果并不准确，主要是训练的模型精确度不高，训练时候可以调高轮数epochs，如果还是不准确使用labelme多打一些标签，再重新训练模型 

### 五、验证视频
##### 1、下载视频
①先安装下载工具  
``` pip install you-get```  

②下载视频  
```you-get 视频网址```  

③修改视频后缀  
下载下来是flv格式，将后缀修改为yolov支持的.mp4  

##### 2、导出模型
```python export.py --imgsz 320```  
  
--imgsz：减小到320，可以提高验证速度  
--weights：可以用自己训练的模型文件  

在命令行下执行如果没有生成onnx文件，那就使用"conda activate 环境名"，切换到安装了yolov的环境下，再执行上面命令，在当前目录下生成了静态模型yolov5s.onnx


##### 3、开始验证 
```python detect.py --weights yolov5s.onnx --view-img --imgsz 320 --source ./七夕节吃掉半个月工资是种什么体验.mp4```  

--weights：上一步生成的验证模型  
--view-img：可以边预测边显示结果  
--imgsz：必须和上一步大小对应  
--source：you-get下载的视频文件

