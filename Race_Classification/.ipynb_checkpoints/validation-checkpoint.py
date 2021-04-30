import torch
import os
from network.resnet import *
from utils.data_setup import *
import numpy as np
from torch.utils.data import *
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from utils.utils import *
from utils.MyDataSet import *
from torchvision import transforms
from collections import OrderedDict
import shutil
from pathlib import Path
transform = transforms.Compose([
    	#transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225								])
])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = net_set()
print(model)
pre = torch.load('./model/vgg16_PRE_50_13step.ckpt')
#new_dict = OrderedDict()
#for k, v in pre.items():
#print(k, v)
#    name = k[7:]
#   new_dict[name] = v

model.load_state_dict(pre)
print(model)
model.eval()
dataset = mydataset('/home/liuting/Race_img_no_grid/', transform=transform)
print(len(dataset))
#print(dataset[100])
dataloader = DataLoader(dataset, batch_size=1,
            shuffle=True,
            num_workers=4)
#data = iter(dataloader)
#print(data.next())

#============================================================
#model.eval()
file_obj = open('./file.txt','w')
cor = 0
for i, (img, label, name) in enumerate(dataloader):
    #if i > 5:
    #break
    #print(name)
    
    if not os.path.exists(name[0]):
        print(name[0], "not exists")
        
    #print(img, label, name)
    img = img.to(device)
    label = label.to(device)
    #print(label)
    outputs = model(img)
    _, preds = torch.max(outputs, 1)
    #cor += torch.sum(preds == label.data)
    for i in range(len(preds)):
        if preds[i] == label[i]:
            cor += 1
        else:
            file_obj.write(name[i] + ' ------> ')
            shutil.copy(name[i], '/home/liuting/wrong')
            _, last = os.path.split(name[i])
            newpa = os.path.join('/home/liuting/wrong', last)
            #print(preds[i].item())
            print(newpa,(str(preds[i].item()) + last) )
            tempath = os.path.join('/home/liuting/wrong', 
                (str(preds[i].item()) + last))
            os.rename(newpa, tempath)
            file_obj.write(tempath + '\n')


    #print(name)
file_obj.close()
acc = float(cor) / len(dataset)
print(acc)

