import cv2    
import numpy as np

#Turns img black and white with gausian blur and threshold
def makeBW(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	ret,thresh1 = cv2.threshold(blur,40,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) #_INV+cv2.THRESH_OTSU
	return thresh1

#Turns img gray with gausian blur
def makeGray(img, threshHoldValue):
	imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(imgray,(5,5),0)
	ret,thresh = cv2.threshold(blur,threshHoldValue,255,0)
	return thresh
	
#Finds the contours in an img
def getContours(thresh):
	max_area =0
	image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for i in range(len(contours)):
		cnt=contours[i]
		area = cv2.contourArea(cnt)
		if(area>max_area):
			max_area=area
			ci=i
	cnt = contours[ci]
	return cnt, contours
	
#Makes a drawing of the contours
def getDrawingContour(cnt, img, howDraw):
	hull=cv2.convexHull(cnt)
	drawing=np.zeros(img.shape,np.uint8)
	
	moments = cv2.moments(cnt)
	if moments['m00']!=0:
		cx = int(moments['m10']/moments['m00']) # cx = M10/M00
		cy = int(moments['m01']/moments['m00']) # cy = M01/M00
	
	centr=(cx,cy)
	if(howDraw == 0):
		cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
		cv2.drawContours(drawing,[hull],0,(0,0,255),2)
	elif(howDraw == 1):
		cv2.drawContours(drawing,[cnt],0,(0,255,0),-1)
	return hull, drawing, centr

#Excludes the black from an img
def excloseBlack(drawing, img, imgShape):
	for i in range(imgShape[0]):
		for j in range(imgShape[1]):
			if(drawing[i,j] == 255):
				img[i,j] = [255]
	return img
	
#Draw the circles of the contours
def drawContours(cnt, centr, img):
	cv2.circle(img,centr,5,[0,0,255],2)		
	cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
	hull = cv2.convexHull(cnt,returnPoints = False)
	
	#if(1):
	defects = cv2.convexityDefects(cnt,hull)
	mind = 0
	maxd=0
	#print(defects.shape[0])
	#i=0
	for i in range(defects.shape[0]):
		s,e,f,d = defects[i,0]
		start = tuple(cnt[s][0])
		end = tuple(cnt[e][0])
		far = tuple(cnt[f][0])
		dist = cv2.pointPolygonTest(cnt,centr,True)
		#cv2.line(img,start,end,[0,255,0],2)
		cv2.circle(img,far,5,[0,0,255],-1)
	#print (i)
	i=0
	return(img)
	