import matplotlib.pyplot as plt
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

def visualize_model(model, num_images=6, dataloader=None, classes=None, device=None, path=None):
    

    was_training = model.training
    model.eval()
    image_so_far = 0

    
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)


            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


            for j in range(inputs.size()[0]):
                image_so_far += 1
                ax = plt.subplot(num_images // 2, 2, image_so_far)
                ax.axis('off')
                ax.set_title('predict: {}'.format(classes[preds[j]]))
                a = inputs.cpu().data[j].permute(1, 2, 0)
                #print(a.shape)
                plt.imshow(a)


                if image_so_far == num_images:
                    model.train(mode=was_training)
                    plt.savefig(path, dpi=500)
                    return


            model.train(mode=was_training)
            plt.savefig(path, dpi=500)

def net_set():

    model_r18 = resnet18(pretrained=True)
    cudnn.benchmark = True
    num_ftrs = model_r18.fc.in_features
    model_r18.fc = nn.Linear(num_ftrs, 3)
    model_r18 = nn.DataParallel(model_r18.cuda(1), device_ids=[1])
    return model_r18


def Val_for(listdir=None):
    lists = []
    

