from skimage import feature
import numpy as np
# from sklearn.svm import LinearSVC
from sklearn import svm
import cv2
import datetime

import utils

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist

def get_match_label(train_data, train_imgnames, test_imgpath):
    desc = LocalBinaryPatterns(8, 1)
    data = []
    labels = []
    for index, hist in enumerate(train_data):
        data.append(hist)
        labels.append(utils.get_label(train_imgnames[index]))

    starttime = datetime.datetime.now()
    svc_model = svm.SVC(C=100.0, cache_size=1000, random_state=42)
    svc_model.fit(data, labels)
    endtime = datetime.datetime.now()
    print ('train', (endtime - starttime).seconds)

    testImage = cv2.imread(test_imgpath)
    gray = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    testHist = get_hist(test_imgpath)
    prediction = svc_model.predict(testHist.reshape(1, -1))
    return prediction[0]
    

def get_hist(imgpath):
    desc = LocalBinaryPatterns(24, 8)
    image = cv2.imread(imgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    return hist
