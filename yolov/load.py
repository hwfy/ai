import torch
model=torch.load(yolov.py)#这里已经不需要重构模型结构了，直接load就可以
model.eval()