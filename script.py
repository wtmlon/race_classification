import os
import numpy as np
from PIL import Image
from PIL import ImageFile

path = r'F:\Race_Classification\RADB\train'
dirs = os.listdir(path)
mean = np.zeros((218, 178, 3))
cnt = 0
for dir_ in dirs:
    pics = os.listdir(os.path.join(path, dir_))
    #print(len(pics))
    for pic in pics:
        pic_path = os.path.join(os.path.join(path, dir_), pic)
        try:
            img = Image.open(pic_path)
        except:
            continue

        width, height = img.size
        matrix = np.asarray(img)
        mean = mean + matrix
        '''
        for x in range(width):
                for y in range(height):
                    r,g,b = img.getpixel((x,y))	
                    mean[x][y][0] += r
                    mean[x][y][1] += g
                    mean[x][y][2] += b
        '''
        #print(mean)
        cnt += 1

mean = mean / float(cnt)
mean = np.uint8(mean)
print('cnt = ', cnt)
print('mean = ', mean)            
for dir_ in dirs:
    pics = os.listdir(os.path.join(path, dir_))
    #print(len(pics))
    for pic in pics:
        pic_path = os.path.join(os.path.join(path, dir_), pic)
        try:
            img = Image.open(pic_path)
        except:
            continue

        width, height = img.size
        '''
        for x in range(width):
                for y in range(height):
                    r,g,b = img.getpixel((x,y))	
                    r = r - mean[x][y][0]
                    g = g - mean[x][y][1]
                    b = b - mean[x][y][2]
                    img.putpixel((x,y),(r, g, b))
        '''
        matrix = np.asarray(img)
        matrix = matrix - mean
        img = Image.fromarray(matrix)
        img.save(pic_path)

'''
img = Image.open('C:\\000014.jpg')

width, height = img.size
'''


    