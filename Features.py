import cv2    
import numpy as np
import MySQLdb
import json, codecs
import hashlib

def printSift(gray, img):
	counter = 1
	
	sift = cv2.xfeatures2d.SIFT_create()
	hKP, hDes = sift.detectAndCompute(gray,None)
	#print(des)
	des2 = 0
	for kp, des in zip(hKP, hDes):#get length of kp
		x = int(kp.pt[1])
		y = int(kp.pt[0])
		#print(x)
		if((img[x - 2,y][0] == 255 or img[x, y-2][0] == 255 or img[x + 2, y][0] == 255 or img[x, y + 2][0] == 255) == False):
			counter+=1
			
	newKp = []
	newDes = []
	counter = 0
	for kp, des in zip(hKP, hDes):
		x = int(kp.pt[1])
		y = int(kp.pt[0])
		#print(x)
		if((img[x - 2,y][0] == 255 or img[x, y-2][0] == 255 or img[x + 2, y][0] == 255 or img[x, y + 2][0] == 255) == False):
			newKp.append(kp)
			newDes.append(des)
			counter+=1
	query = ('(%s)' % (hDes))
	hash_object = hashlib.md5(query.encode())
	print(hash_object.hexdigest())
	cv2.imwrite('img/sift_keypoints.jpg',img)
	return newKp,newDes
	
def printCorner(gray, img):
	corners = cv2.goodFeaturesToTrack(gray,500,0.001,10)
	corners = np.int0(corners)
	#print(img.shape)
	for i in corners:
		y,x = i.ravel()
		if((img[x - 2,y][0] == 255 or img[x, y-2][0] == 255 or img[x + 2, y][0] == 255 or img[x, y + 2][0] == 255) == False):
			#print("Wrong ")
		#else:
			cv2.circle(img,(y,x),3,255,-1)
		
	cv2.imwrite('img/corner_keypoints.jpg',img)
	
def printSurf(gray, img):
	# Create SURF object. You can specify params here or later.
	# Here I set Hessian Threshold to 400
	surf = cv2.xfeatures2d.SURF_create(400)
	
	# Find keypoints and descriptors directly
	kp, des = surf.detectAndCompute(gray,None)

	surf.setHessianThreshold(2500)
	kp, des = surf.detectAndCompute(gray, None)

	surf.setUpright(True)
	kp = surf.detect(img, None)
	
	surf.setExtended(True)
	kp, des = surf.detectAndCompute(gray,None)
	img2 = cv2.drawKeypoints(gray, kp,None,(255,0,0),4)
	#print(surf.descriptorSize())
	cv2.imwrite('img/Surf_keypoints.jpg',img2)
	
def printFast(gray, img):
	fast = cv2.FastFeatureDetector_create()
	kp = fast.detect(gray, None)
	#print(kp)
	img2 = cv2.drawKeypoints(gray,kp,None,color=(255,0,0))
	cv2.imwrite('img/Fast_keypoints.jpg',img2)
	# Disable nonmaxSuppression
	"""fast.setNonmaxSuppression(0)
	kp = fast.detect(img,None)
	#print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )
	img3 = cv2.drawKeypoints(gray, kp, None, color=(255,0,0))
	cv2.imwrite('img/fast_false.png',img3)"""
	
def printBrief(gray, img):
	# Initiate FAST detector
	star = cv2.xfeatures2d.StarDetector_create()
	# Initiate BRIEF extractor
	brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
	# find the keypoints with STAR
	kp = star.detect(gray,None)
	# compute the descriptors with BRIEF
	kp, des = brief.compute(gray, kp)
	img2 = cv2.drawKeypoints(gray,kp,None,color=(255,0,0))
	"""for i in range(len(kp)):
		x = int (kp[i].pt[0])
		y = int (kp[i].pt[1])
		img2[y,x] = [0,0,255]
		print( "x =", x, "y =", y )"""
	cv2.imwrite('img/brief.png',img2)
	
def printORB(gray, img):
	# Initiate ORB detector
	orb = cv2.ORB_create(5000)
	# find the keypoints with ORB
	kp = orb.detect(gray,None)
	# compute the descriptors with ORB
	kp, des = orb.compute(gray, kp)
	# draw only keypoints location,not size and orientation
	img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
	cv2.imwrite('img/ORB.png',img2)