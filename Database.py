import MySQLdb

db=MySQLdb.connect(host="Localhost",user="root",passwd="",db="Hillson")
x = db.cursor()
"""def getDB():
	db=MySQLdb.connect(host="Localhost",user="root",passwd="",db="Hillson")
	return db"""
	
def insertHandValues(uID, hKP,hDes):
	counter =0
	"""for kp, des in zip(hKP, hDes):
		sift_row = {"User_id":uID,"featureId":counter, "Ds": des.tolist(), "Keypoints":kp.pt }
		#print(sift_row)"""
	try:
		query = ('INSERT INTO handPoints (User_id, featureId, Keypoints, Ds) VALUES (%s,%s,"%s", "%s")' % (uID, counter,hKP, hDes))
		#print(query)
		x.execute(query)
		#x.execute('INSERT INTO handPoints VALUES (?,?,?,?)',(sift_row))
		db.commit()
		counter+=1
	except (MySQLdb.Error, MySQLdb.Warning) as e:
		db.rollback()
		print(e)

		
def getHandValues(uID):
	try:
		sql = "SELECT Keypoints,Ds FROM handPoints \
		WHERE User_id = '%s'" % (uID)
		x.execute(sql)
		results = x.fetchall()
		kp = results[0][0]
		des = results[0][1]
		return(kp,des)
	except (MySQLdb.Error, MySQLdb.Warning) as e:
		db.rollback()
		print(e)
def stopDB():
	db.close()