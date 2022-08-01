import torch

# 保存整个model的状态
# torch.save(model, "mymodel.pth")

# 加载整个模型
model = torch.load("yolov5s.pt")  # 这里已经不需要重构模型结构了，直接load就可以
model.eval()

# 比如模型训练的时候是在GPU上进行并保存的，测试的时候却想在CPU上进行训练：
# model = torch.load(PATH, map_location='cpu')

# 转为GPU需要指明在哪块GPU上，例如转到第0块GPU：
# model = torch.load(PATH, map_location=lambda storage, loc: storage.cuda(0))

# 不同块GPU的转换，例如第1块转到第0块：
# model = torch.load(PATH, map_location={'cuda:1':'cuda:0'})


# 只保存模型权重参数，不保存模型结构，速度快，占空间少
# torch.save(model.state_dict(), "mymodel.pth")

# 这里需要重构模型结构，My_model
# model = My_model(*args, **kwargs)
# 这里根据模型结构，调用存储的模型参数
# model.load_state_dict(torch.load("mymodel.pth"))
