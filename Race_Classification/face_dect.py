import cv2
import os


print ('loading...')
OPCV_PATH="/home/liuting/opencv"
color = (0, 0, 0)

def getlist(path):
    lists = os.listdir(path)
    return path, lists

def findface(src, path):
    image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(image, image) 
    
    classfier = cv2.CascadeClassifier(OPCV_PATH + "/data/haarcascades/haarcascade_frontalface_alt.xml")
    
    
    divisor = 8
    h = image.shape[1]
    w = image.shape[0]
    minSize = (int(w / divisor), int(h / divisor)) 
    maxSize = minSize
    faceRects = classfier.detectMultiScale(image, 1.2, 2, cv2.CASCADE_SCALE_IMAGE, (100, 100), (100, 100))
    b = 0
    if len(faceRects) > 0:  
        for faceRect in faceRects: 
            #b = 1
            x, y, w, h = faceRect
            #cv2.rectangle(src, (x, y), (x + w, y + h), color)
            #print(x,y,w,h)
            if y + h > src.shape[0]:
                y = y - ((y + h) - src.shape[1])
                print('done')
            if x + w > src.shape[1]:
                x = x - ((x + w) - src.shape[0])
                print('done')
            src = src[y: y + h, x: x + w]
            #print(src.shape)
            #print(w, h)
    if src.shape[0]==103 and src.shape[1]==103:
        cv2.imwrite(path, src, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    else:
        print('worng')
        print(src.shape)

if __name__ == '__main__':

    fn = '/home/liuting/RACE_IMG/train/WHITE/'
    dirs, lis = getlist(fn)
    for i in range(len(lis)):
        addr = os.path.join(dirs, lis[i])
        my_img = cv2.imread(addr)
        out = os.path.join('/home/liuting/RACE_IMG_crop/train/WHITE/', lis[i])

        findface(my_img, out)

    #cv2.waitKey()
    #cv2.destroyAllWindows()
