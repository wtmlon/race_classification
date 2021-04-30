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
from torch.backends import cudnn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_r18 = net_set()
#pre = torch.load('./model/Res18_PRE_15_10step300.ckpt')
#model_r18.load_state_dict(pre)
print(model_r18)
transform = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        #transforms.RandomGrayscale(),
        
		#transforms.CenterCrop(256),
		transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform1 = transforms.Compose([
    	#transforms.CenterCrop(256),
	transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225								])
])

data_dir = '/home/liuting/RACE_IMG1'
#data_dir = '/home/liuting/DAF/dataset/'
#image_datasets = {x: get_Folder(os.path.join(data_dir, x),
#				transform)
#					for x in ['train', 'val']}

image_datasets = {'train': get_Folder(os.path.join(data_dir, 'train'), transform), 
        'val': get_Folder(os.path.join(data_dir, 'val'), transform1)}
#image_datasets = get_Folder((data_dir), transform)
dataloaders = {x: DataLoader(image_datasets[x], batch_size=96,
						shuffle=True, 
						num_workers=4)
					for x in ['train', 'val']}


class_names = image_datasets['train'].classes
print(class_names)


def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
	since = time.time()
	
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		for phase in ['train', 'val']:
			if phase == 'train':		
				scheduler.step()
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:
				inputs = inputs.to(device)
				labels = labels.to(device)
				#print(labels)
				optimizer.zero_grad()
 
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)
					if phase == 'train':
						loss.backward()
						optimizer.step()

				running_loss += loss.item() * inputs.size()[0]
				running_corrects += torch.sum(preds == labels.data)
		
			epoch_loss = running_loss / len(image_datasets[phase])
			epoch_acc = running_corrects.double() / len(image_datasets[phase])

			print(' {} Loss: {:.4f} Acc: {:.4f}'.format(
				phase, epoch_loss, epoch_acc))

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()
	
	time_use = time.time() - since
	print('Train complete in {:.0f}m {:.0f}s'.format(
			time_use // 60, time_use % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	model.load_state_dict(best_model_wts)
	return model


#model_r18 = model_r18.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_r18 = optim.SGD(model_r18.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_r18, step_size=13, gamma=0.1)

model_r18 = train_model(model_r18, criterion, optimizer_r18, exp_lr_scheduler, 
					num_epochs=30)
#visualize_model(model=model_r18, dataloader=dataloaders['train'], classes = class_names,
#                        device=device,
#                        path='./utils/showpic/Rest_pic.jpg')
#visualize_model(model=model_r18, dataloader=dataloaders['val'], classes = class_names,
#                        device=device, 
#                        path='./utils/showpic/Resv_pic.jpg')

torch.save(model_r18.state_dict(), './model/vgg16_PRE_50_13step.ckpt')

			
				
