import cv2 as cv
import numpy as np
from tkinter import filedialog
import os
import math
from tkinter import *


#this functionis use to resize img
def Resize(img, size):
    img = img[:,1:-1]
    resized_image = cv.resize(img, (size, size)) 
    return resized_image

#this function is use to equalize the img histogram
def Equalization(img):
    equ = cv.equalizeHist(img)
    return equ

#this function is use to cut the useless bottom part of image
def CroppingLowerRegion(img):
    height, width = img.shape[:2]
    x = height-1
    stop = True
    lastMean = 0
    #while we aren't in the horizontal middle of the image or the lung start isn't find
    while(x > (height/2) and stop):
        y = 20
        lastMean = 0
        #we move on the ligne
        while(y < width/2 and stop):
            #we take the mean of 5 pixel
            batche = img[x,y:y+5]
            mean = cv.mean(batche)
            #if the gradient beetween the lastMean and the actual is greater than 50 we stop
            if(lastMean - mean[0] > 50):
                stop = False
                break
            else:
                lastMean = mean[0]
            y+= 5
        x -= 2;
    img = img[0:x,0:width-1]
    return img

def LungBoundary(img):
    height, width = img.shape[:2]
    #we cut the image into 2 image
    imgLeft = img[0:height,0:int(width/2)]
    imgRight = img[0:height,int(width/2):width-1]

    #we create a range of equidistant lines
    r = range(height-1,0, -int((height)/50))


    i = 0;
    
    #Left Variable
    lastYLeft=-50
    arrayLeft = []
    mostRealLeft =-50

    #Right Variable
    lastYRight=-50
    arrayRight = []
    mostRealRight =-50

    #for each lines
    for x in r:
        #leftPart
        y = 20
        lastMean = 0 
        heightLeft, widthLeft = imgLeft.shape[:2]
        while(y < widthLeft-6):
            batche = imgLeft[x,y:y+5] 
            mean = cv.mean(batche)
            #if the gradient beetween the two batch is greater than 15 
            if(lastMean - mean[0] > 15):
                #if lastYLeft isn't initialize we initialize is value
                if(lastYLeft < 0):
                    lastYLeft = y
                    mostRealLeft = y
                    break
                else:
                    if(mostRealLeft == -50):
                        mostRealLeft = y
                    #we compare the new possible y with the last find and keep the most close of the y of the previous line
                    elif(abs(mostRealLeft - lastYLeft) > abs(y - lastYLeft)):
                        mostRealLeft = y;
            lastMean = mean[0]
            y+= 5
        #imgLeft[x-3:x+3,mostRealLeft-3:mostRealLeft+3] = 255
        arrayLeft.append((mostRealLeft,x))
        if(mostRealLeft != -50):
            lastYLeft = mostRealLeft
        mostRealLeft = -50;


        #RighPart same as for left part
        lastMean = 0 
        heightRight, widthRight = imgRight.shape[:2]
        y = widthRight-1;
        while(y > 6):
            batche = imgRight[x,y-5:y] 
            mean = cv.mean(batche)
            if(lastMean - mean[0] > 15):
                if(lastYRight < 0):
                    lastYRight = y
                    mostRealRight = y
                    break
                else:
                    if(mostRealRight == -50):
                        mostRealRight = y-5
                    elif(abs(mostRealRight - lastYRight) > abs(y-5 - lastYRight)):
                        mostRealRight = y-5;
            lastMean = mean[0]
            y-= 5
        #imgRight[x-3:x+3,mostRealRight-3:mostRealRight+3] = 255
        arrayRight.append((mostRealRight + widthRight,x))
        if(mostRealRight != -50):
            lastYRight = mostRealRight
        
        mostRealRight = -50;

        i += 1
    return arrayLeft,arrayRight


#create a mash of the lung area with theleft and right part of the picture
def LungArea(LungAreaImg, left, right):
    height, width = LungAreaImg.shape[:2]
    lastCoord = (0, height-1)

    for x in left:
        cv.rectangle(LungAreaImg, x, lastCoord, 255, -1)
        lastCoord = (0, x[1])
    cv.rectangle(LungAreaImg, (0, 0), (x[0], x[1]), 255, -1)

    lastCoord = (width-1, height-1)
    for x in right:
        cv.rectangle(LungAreaImg, x, lastCoord, 255, -1)
        lastCoord = (width-1, x[1])
    cv.rectangle(LungAreaImg, (width-1, 0), (x[0], x[1]), 255, -1)
    
    LungAreaImg = cv.bitwise_not(LungAreaImg)
    
    return LungAreaImg


#compare the lung area w=before and after the otsu threshold to determine if the lung is infected or not
def Compare(crop, LungAreaImg, thresholdImg):
    thresholdMask = cv.bitwise_and(thresholdImg, LungAreaImg)
    #cropLung = cv.bitwise_and(crop, LungAreaImg)
    cropThreshold = cv.bitwise_and(crop, thresholdMask)
    #cv.imshow('Lung'+imgagePath,cropThreshold)

    nbPixelArea = cv.countNonZero(LungAreaImg)
    nbPixelThreshold = cv.countNonZero(thresholdMask)

    return (((nbPixelArea - nbPixelThreshold) / nbPixelArea)  < 0.62),cropThreshold

#if we want to process on directory
def dirProcess():
    path = filedialog.askdirectory()
    if(os.path.isdir(path)):
        master.destroy()
        os.chdir(path)
        listDir = os.listdir(path)
        nbTrue = 0
        n = 0
        for imgagePath in listDir:
            if(os.path.isfile(imgagePath)):
                img = cv.imread(imgagePath,0)
                if(img is not None):
                    #cv.imshow('image'+imgagePath,img)

                    resized_image = Resize(img,800)
                    equ = Equalization(resized_image)
                    crop = CroppingLowerRegion(equ)

                    height, width = crop.shape[:2]
                    
                    left, right = LungBoundary(crop)
                    ret, thresholdImg = cv.threshold(crop, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
                    thresholdImg = cv.bitwise_not(thresholdImg)

                    LungAreaImg = np.zeros((height, width, 1), np.uint8)
                    LungAreaImg = LungArea(LungAreaImg, left, right)

                    if(Compare(crop, LungAreaImg, thresholdImg)[0]):
                        print(imgagePath + ": clean")
                        nbTrue += 1
                    else:
                        print(imgagePath + ": infected")

                    n += 1
                else:
                    print(imgagePath + "is not a picture")
        if(n != 0):
            print("in "+path+": "+ str(nbTrue * 100 / n)+"% clean")
        else:
            print("no pucture in "+path)
        print("Press Enter to quit ...")
        input() 

#if we want to process only one picture
def fileProcess():
    imgagePath = filedialog.askopenfilename()
    if(os.path.isfile(imgagePath)):
        img = cv.imread(imgagePath,0)
        if(img is not None):
            master.destroy()
            resized_image = Resize(img,800)
            equ = Equalization(resized_image)
            crop = CroppingLowerRegion(equ)

            height, width = crop.shape[:2]
            
            left, right = LungBoundary(crop)
            ret, thresholdImg = cv.threshold(crop, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
            thresholdImg = cv.bitwise_not(thresholdImg)

            LungAreaImg = np.zeros((height, width, 1), np.uint8)
            LungAreaImg = LungArea(LungAreaImg, left, right)

            result = Compare(crop, LungAreaImg, thresholdImg)
            if(result[0]):
                print(imgagePath + ": clean")
            else:
                print(imgagePath + ": infected")
            
            cv.imshow(imgagePath+" before",img)
            cv.imshow(imgagePath+" after",result[1])
            cv.waitKey(0)
            cv.destroyAllWindows()
            print("Press Enter to quit ...")
            input() 
        else:
            print(imgagePath + " is not a picture")
            print("Press Enter to quit ...")
            input() 


#script
master = Tk()

b1 = Button(master, text="Process Picture", command=fileProcess)
b1.pack()

b2 = Button(master, text="Process Directory", command=dirProcess)
b2.pack()

master.mainloop()
