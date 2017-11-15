#importing libraries
import cv2               
import numpy as np
import _mysql
from matplotlib import pyplot as plt
import Filter, Features, Matching, Database
import json, codecs

imgOriginal = cv2.imread('img/handIR.jpg')
cap = cv2.VideoCapture(0)

while True:
	ret,imgOriginal = cap.read()
	#Img options
	imgShape = imgOriginal.shape
	img = imgOriginal.copy()
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	thresh = Filter.makeGray(img, 70)
	#thresh = filter.makeBW(img)
	
	#Contour of image
	cnt, contours = Filter.getContours(thresh)
	hull, drawing, centr = Filter.getDrawingContour(cnt, img, 1)
	#img = Filter.drawContours(cnt, centr, img)
	
	#Get cutout of the hand
	im_gray = Filter.makeBW(drawing)
	img = Filter.excloseBlack(im_gray ,img, imgShape)	
	#print(imgOriginal[100,100])
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#img = filter.makeGray(img, 10)
	#cv2.imshow("thresh", thresh)
	#cv2.imshow('output',drawing)
	cv2.imshow('input',img)
	#print(len(cnt))
	k = cv2.waitKey(10)
	if k == 99:
		print("c is pressed")
		kp, des = Features.printSift(gray,img)
		#print(len(kp), len(des))
		#Features.printCorner(gray, img)
		#Features.printSurf(gray, img)
		#Features.printFast(gray, img)
		#Features.printBrief(gray, img)
		#Features.printORB(gray, img)
		Database.insertHandValues(2, kp, des)
	if k == 112:
		print("p is pressed")
		kp,des = Database.getHandValues(2)
		#Matching.orbMatch(imgGray)
		#Matching.bfmMatch()
		Matching.flannMatch('img/sift_keypoints', kp, des)
		
	if k == 27: #escape key
		Database.stopDB()
		break