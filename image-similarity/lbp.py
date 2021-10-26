from skimage import feature
import numpy as np
# from sklearn.svm import LinearSVC
from sklearn import svm
import cv2
import os
import datetime
import random

# model = LinearSVC(C=100.0, random_state=42)
model = svm.SVC(C=100.0, cache_size=1000, random_state=42)

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

def match(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))
    print(prediction[0])
    # display the image and the prediction
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)

def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg','.png')):
                img_paths.append(path+'/'+filename)
    return img_paths

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

for imagePath in get_img("D://tmp//siamese//left//left"):
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# print('hist', hist)
	# extract the label from the image path, then update the
	# label and data lists
	label = 'a' + str(random.randint(0,500))
	# print('label', label)
	labels.append(label)
	data.append(hist)


# train a Linear SVM on the data
# model = LinearSVC(C=100.0, random_state=42)
starttime = datetime.datetime.now()
model.fit(data, labels)
endtime = datetime.datetime.now()
print ('train', (endtime - starttime).seconds)

starttime = datetime.datetime.now()
match('database-tea-1/d1.jpg')
endtime = datetime.datetime.now()
print ('match', (endtime - starttime).seconds)
