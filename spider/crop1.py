# coding=gbk
import os
import face_recognition
from PIL import Image
from PIL import ImageFile
import threading
ImageFile.LOAD_TRUNCATED_IMAGES = True
'''
def func(img, midx, midy):
    width, height = img.size
    if width < 178 or height < 218:
        continue
    if midx + 88 < width and midx - 89 >= 0:
        left = midx - 89
        right = midx + 88
    else if midx + 88 >= width:
        right = width - 1
        #left = midx -(178 - (right - midx) - 1)
        left = - 177 + right
    else if midx - 89 < 0:
        left = 0
        right = left + 177
    if midy + 108 < height and midy - 109 >= 0:
        top = midy - 109
        bottom = midy + 108
    else if midy + 108 >= height:
        bottom = height - 1
        #left = midx -(178 - (right - midx) - 1)
        top = - 217 + bottom
    else if midy - 109 < 0:
        top = 0
        bottom = left + 217
'''
'''
try:
                img = Image.open(pic_path)
            except:
                continue
            width, height = img.size
            if width < 179 or height < 219:
                continue
            if width > height:
                height_n = 224
                width_n = float(width) * (224.0 / float(height))
            else:
                width_n = 224
                height_n = float(height) * (224.0 / float(width))
            
            img = img.resize((int(width_n), int(height_n)), Image.ANTIALIAS)
            width, height = img.size
            #print(width, height)
            new_pic_path = os.path.join(new_path, pic_dir)
            if not os.path.exists(new_pic_path):
                    os.makedirs(new_pic_path)
            try:
                img.save(new_pic_path + '\\' + pic, quality = 100)
            except:
                continue
'''
def process_img(path, new_path, cnt):
    dirs = os.listdir(path)
    for pic_dir in dirs:
        print(pic_dir)
        dir_path = os.path.join(path, pic_dir)
        pics = os.listdir(dir_path)
        for pic in pics:
            pic_path = os.path.join(dir_path, pic)
            
            
            #image = face_recognition.load_image_file(pic_path)
            #face_locations = face_recognition.face_locations(image)
            #if len(face_locations) == 0 or len(face_locations) > 1:
            #    continue
            img = Image.open(pic_path)
            #print(face_locations)
            '''
            for face_location in face_locations:
                top, right, bottom, left = face_location
                midx = (right + left) // 2
                midy = (top + bottom) // 2
                img = Image.open(pic_path)
                width, height = img.size
            
                if midx + 88 < width and midx - 89 >= 0:
                    left = midx - 89
                    right = midx + 89
                elif midx + 89 >= width:
                    right = width - 1
                    #left = midx -(178 - (right - midx) - 1)
                    left = - 178 + right
                elif midx - 89 < 0:
                    left = 0
                    right = left + 178
                if midy + 109 < height and midy - 109 >= 0:
                    top = midy - 109
                    bottom = midy + 109
                elif midy + 109 >= height:
                    bottom = height - 1
                    top = - 218 + bottom
                elif midy - 109 < 0:
                    top = 0
                    bottom = top + 218
                #print(top, bottom, left, right)
                '''
            new_pic_path = os.path.join(new_path, pic_dir)
            if not os.path.exists(new_pic_path):
                os.makedirs(new_pic_path)
            if len(img.split()) == 4:
                # ����split��merge��ͨ�����ĸ�ת��Ϊ����
                r, g, b, a = img.split()
                toimg = Image.merge("RGB", (r, g, b))
                toimg.save(new_pic_path + '\\' + pic)
            else:
                try:
                    img.save(new_pic_path + '\\' + pic)
                except:
                    continue
                '''
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                #pil_image.show()
                width, height = pil_image.size
                if width == 178 and height == 218:
                    pil_image.save(new_path + '\\' + 'crawl_{:0>7d}.jpg'.format(cnt), quality = 100)
                    cnt = cnt + 1
                '''
            
        print('Finish......!')
 
def lock_test(path, new_path, cnt):
    mu = threading.Lock()
    if mu.acquire(True):
        process_img(path, new_path, cnt * 20000)
        mu.release()
 
if __name__ == '__main__':
    paths = [r'F:\Race_Classification\spider\starName_cauca']
    new_paths = [r'F:\Race_Classification\spider\starName_cauca_crop']
    for i in range(len(paths)):
        my_thread = threading.Thread(target=lock_test, args=(paths[i], new_paths[i], i))
        my_thread.start()