import cv2
import numpy as np
from colortool import rgb2hsv

def ClickColor(event, x, y, flags, param):
    global lowerBound, upperBound, img
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = np.mean(np.mean(np.array(img[y-5:y+5, x-5:x+5]), 0), 0)
        diff = [10, 50, 70]
        bgr = np.uint8(np.expand_dims(np.expand_dims(bgr, 0), 0))
        hsvMin = list(cv2.cvtColor(bgr,  cv2.COLOR_BGR2HSV)[0, 0, :])
        lowerBound = np.array([max(min(x-diff[i], 255), 0) for i,x in enumerate(hsvMin)])
        upperBound = np.array([max(min(x+diff[i], 255), 0) for i,x in enumerate(hsvMin)])
        print lowerBound, upperBound

lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)

#open camera
cam = cv2.VideoCapture(0)
ret = False
while not ret:
    ret, img=cam.read()
cv2.imshow("cam",img)

# attatch callback
cv2.setMouseCallback("cam", ClickColor)

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)