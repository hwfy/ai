import os
from glob import glob

input_path = r'D:\Projects\Python\AiProject\yolovTest\MyVOC\images'
output_path=r'D:\Projects\Python\AiProject\yolovTest\MyVOC'
txt_path = r'D:\Projects\Python\AiProject\yolovTest\MyVOC\labels'

imgtype=['png','jpg','jpeg','bmp']

os.makedirs(output_path,exist_ok=True)
result=[]
txt_list = os.listdir(txt_path)
local = os.getcwd()
mlist=[]
for t in imgtype:
	mm=glob(os.path.join(input_path,'*.%s'%t))
	for m in mm:
		if m.split('\\')[-1].split('.')[0]+'.txt' in txt_list:
			mlist.append(m)



rate = 0.8
total = len(mlist)
n =  int(total*rate)
print('Found txt:%d'%total)
train_txt = '\n'.join(mlist[:n])
val_txt = '\n'.join(mlist[n:])


with open(os.path.join(output_path,'train.txt'),'w',encoding='utf8') as f:
	f.write(train_txt)

with open(os.path.join(output_path,'val.txt'),'w',encoding='utf8') as f:
	f.write(val_txt)
