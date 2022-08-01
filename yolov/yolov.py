import torch

# Model 加载模型是典型的用例，但这也可以用于加载其他对象，例如分词器、损失函数

# repo_or_dir：如果第三个参数source是github，则对应格式为repo_owner/repo_name[:tag_name]，例如'pytorch/vision:0.10'。
# 如果未指定tag_name，则默认分支假定为main(如果存在)，否则为master。
# 如果source是local，那么它应该是本地目录的路径，默认github

# model：在repo/dir的 hubconf.py 中定义的可调用(入口点)的名称
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
# 下面图片打不开
# img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
# 使用本地路径即可
img = r'C:\Users\v_vyangfan\.cache\torch\hub\ultralytics_yolov5_master\data\images\zidane.jpg'

# Inference 自动识别出图片中的人物、领带等
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()
