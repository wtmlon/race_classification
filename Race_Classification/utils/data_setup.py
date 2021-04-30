import torch
import os
from torchvision.datasets import *
from torch.utils.data import *
from torchvision import *
def get_Folder(local, transform=None):
	#
	Folder = ImageFolder(local, transform)
	return Folder

if __name__ == "__main__":
	transform = transforms.Compose([
		#transforms.CenterCrop(256),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
	folder = get_Folder('/home/liuting/RACE_IMG', transform)
	data_loader = DataLoader(folder,
				batch_size=128,
				shuffle=True,
				num_workers=4)
	print(folder.class_to_idx)
	print(len(data_loader))
	diter = iter(data_loader)
	print(diter.next())
