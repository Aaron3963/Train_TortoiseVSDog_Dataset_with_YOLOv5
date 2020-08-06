from cv2 import cv2
import os
import numpy as np
from math import ceil
from random import shuffle

#Parameters
rowNum = 3
colNum = 4
IMAGE_SIZE = 256
GAP_SIZE = 2
timeInterval = 3000

#Datasets
imageDir = 'D:\QiuChengTong\\coco81\\images\\train2017'
labelDir = 'D:\QiuChengTong\\coco81\\labels\\train2017'

classNames=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush', 'tortoise']
# Add tortoise as NO.80

#Random Color Frame
numClass = len(classNames)
colors = []
for i in range(numClass):
    colors.append((np.random.randint(0, 255), np.random.randint(0, 255),np.random.randint(0, 255)))

#Get Dir
for dirpath, dirnames, filenames in os.walk(imageDir):
    if dirpath == imageDir:
        imageFileList = filenames  
imageNum = len(imageFileList)
print('images:{}'.format(imageNum))

for dirpath,dirnames,filenames in os.walk(labelDir):
    if dirpath == labelDir:
        txtFileList=filenames
total_txts=len(txtFileList)
print('labels:{}'.format(total_txts))

#Window setup
windowName = "Dataset Display"
cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
windowHeight = rowNum*(IMAGE_SIZE)+ (rowNum)*(GAP_SIZE)
windowWidth = colNum*(IMAGE_SIZE)+ (rowNum)*(GAP_SIZE)
cv2.resizeWindow(windowName,windowWidth,windowHeight)

flashNum = ceil(imageNum/(rowNum*colNum))
background = np.zeros((windowHeight,windowWidth,3), np.uint8)  #Black background
background.fill(255)

for n in range(flashNum):
    for rowIndex in range(rowNum):
        for colIndex in range(colNum):
        
        #Show Individual Image
            index=n*rowNum*colNum+rowIndex*colNum+colIndex
            imgFile=imageFileList[index]
            file_path=os.path.join(imageDir,imgFile)
            img = cv2.imread(file_path)

            width = img.shape[1]
            height = img.shape[0]

        #Check for Resize
            scaleFactor = IMAGE_SIZE / max(width, height)
            if scaleFactor < 1:
                h = ceil(scaleFactor * height)
                w = ceil(scaleFactor * width)
                img = cv2.resize(img, (w, h))
            else:
                h =height
                w =width  

            file_name,_= os.path.splitext(imgFile) 
            txt_file=file_name+'.txt'        
            file_path=os.path.join(labelDir,txt_file)
            if os.path.exists(file_path):
                file=open(file=file_path,mode='r',encoding='utf-8')
            else:
                print("image {} has no corresponding label".format(imgFile))
                break

            for line in file:
                strList = line.split()
                frameColor = colors[int(strList[0])]
                x1=int(float(strList[1])*w)-int(float(strList[3])*w/2.0)
                x2=int(float(strList[1])*w)+int(float(strList[3])*w/2.0)
                y1=int(float(strList[2])*h)-int(float(strList[4])*h/2.0)
                y2=int(float(strList[2])*h)+int(float(strList[4])*h/2.0)

                #Rectangle
                cv2.rectangle(img, (x1,y1),(x2,y2),frameColor, 2)
                cv2.rectangle(img, (x1,y1),(x1+20,y1+10),frameColor, thickness=-1)  
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img,strList[0],(x1,y1+10), font,0.5,(255,255,255), thickness=1,lineType=cv2.LINE_AA)

            #Spread Image
            imageOutput = np.full((IMAGE_SIZE,IMAGE_SIZE,3), 128)

            hStart = int((IMAGE_SIZE-h)/2)
            hEnd = hStart + h
            wStart = int((IMAGE_SIZE-w)/2) 
            wEnd = wStart + w

            imageOutput[hStart:hEnd, wStart:wEnd] = img
            cv2.putText(imageOutput, imgFile, (int(IMAGE_SIZE/2)-50,15),font,0.5,(255,255,255),thickness=1,lineType=cv2.LINE_AA)

            grid = background
            grid[rowIndex*(IMAGE_SIZE+GAP_SIZE):rowIndex*(IMAGE_SIZE+GAP_SIZE) + IMAGE_SIZE, 
                 colIndex*(IMAGE_SIZE+GAP_SIZE):colIndex*(IMAGE_SIZE+GAP_SIZE) + IMAGE_SIZE] = imageOutput

    result = grid[0:772,0:1030] 
    cv2.imshow(windowName, result)

    if cv2.waitKey(timeInterval) == 27:
        break

#Save Screenshot
screenshot = 'dataset_' + str(n) + '.png'
cv2.imwrite(screenshot,result)

print('Images Displayed: {}'.format(index))
print('Saved Current Image')
cv2.destroyAllWindows()

