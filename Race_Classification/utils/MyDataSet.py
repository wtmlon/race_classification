import numpy as np
import scipy.misc as sm
import os
from torch.utils.data import *
import linecache
import torch
from torchvision import transforms
#202599
if torch.cuda.is_available():
    use = True
else:
    use = False

'''
path = '../Race_img_no_grid/'
y = os.path.join(path, 'YELLOW')
w = os.path.join(path, 'WHITE')
b = os.path.join(path, 'BLACK')


Ylistdir = os.listdir(os.path.join(path, 'YELLOW'))
Blistdir = os.listdir(os.path.join(path, "BLACK"))
Wlistdir = os.listdir(os.path.join(path, 'WHITE'))

lb = len(Blistdir)
lw = len(Wlistdir)
ly = len(Ylistdir)


Y_wor_list = Val_for(Ylistdir)
B_wor_List = Val_for(Blistdir)
W_wor_list = Val_for(Wlistdir)

'''
class mydataset(Dataset):
    def __init__(self, dirs, transform=None):
        self.transform = transform
        self.path = dirs
        self.y = os.path.join(self.path, 'YELLOW')
        self.w = os.path.join(self.path, 'WHITE')
        self.b = os.path.join(self.path, 'BLACK')


        self.Ylistdir = os.listdir(os.path.join(self.path, 'YELLOW'))
        self.Blistdir = os.listdir(os.path.join(self.path, "BLACK"))
        self.Wlistdir = os.listdir(os.path.join(self.path, 'WHITE'))

        self.lb = len(self.Blistdir)
        self.lw = len(self.Wlistdir)
        self.ly = len(self.Ylistdir)       
        self.lenth = self.lb + self.lw + self.ly
        #se

    def __getitem__(self, index):
        if index < self.lb:
            imgpath = os.path.join(self.b, self.Blistdir[index])
            img = sm.imread(imgpath)
            label = 0
            names = imgpath
        elif index >= self.lb and index < self.lb + self.lw:
            imgpath = os.path.join(self.w, self.Wlistdir[index - self.lb])
            img = sm.imread(imgpath)
            label = 1
            names = imgpath
        else:
            imgpath = os.path.join(self.y, self.Ylistdir[index - self.lb - self.lw])
            img = sm.imread(imgpath)
            label = 2
            names = imgpath
        if self.transform != None:
            img = self.transform(img)
            #label = transform(label)
        return img, label, names

        '''
        img = sm.imread(os.path.join(self.dirs, 'img_align_celeba/%06d.jpg' % (index + 1)))
        line = linecache.getline(os.path.join(self.dirs, 'list_attr_celeba.txt'), index + 3).split()[1:]
        return img, line
        '''
    def __len__(self):
        return self.lenth


if __name__ == '__main__':
    transform = transforms.Compose([
    	#transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225								])
])

    dataset = mydataset('/home/liuting/Race_img_no_grid/', transform)
    print(len(dataset))
    #print(dataset[100])
    dataloader = DataLoader(dataset, batch_size=12,
            shuffle=False,
            num_workers=4)
    #data = iter(dataloader)
    #print(data.next())
    for i, (img, label, name) in enumerate(dataloader):
        if i > 5:
            break
        print(img.shape)
        print(label)
        print(name)
