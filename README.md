# 常用命令
conda -V  查看版本  
conda info -e  查看所有环境  
conda create -n name python==3.7  创建环境  
conda activate name 激活环境  

conda install jupyter  
jupyter-notebook  

pip uninstall pyzmq 再执行 pip install pyzmq==19.0.2 修改jupyter-notebook通信问题

# 国内镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 解决 Requirement already satisfied: keras
pip install --target=d:\anaconda3\envs\ai\lib\site-packages keras

# 解决 Could not load dynamic library 'cudart64_110.dll'; dlerror:cudart64_110.dll not found
https://cn.dll-files.com/cudart64_110.dll.html
下载以后放在C:\Windows\System32

# 视频监控项目
https://github.com/ultralytics/yolov5
https://github.com/EricLee2021-72324/handpose_x



