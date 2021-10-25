import os

import cv2
import h5py
import numpy as np

from extract_cnn_vgg16_keras import VGGNet

import utils

'''
 Returns a list of filenames for all jpg images in a directory.
'''


def get_imlist(database):
    databasePath = os.path.join(utils.DATABASES, database)
    os.makedirs(databasePath, exist_ok=True)
    return [os.path.join(databasePath, f) for f in os.listdir(databasePath) if utils.allowed_file(f)]


'''
 Extract features and index the images
'''


def init():
    for database in os.listdir(utils.DATABASES):
        init_database(database)


def init_database(database):
    index = os.path.join(utils.MODELS, 'vgg_featureCNN_' + database + '.h5')
    img_list = get_imlist(database)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()
    for i, img_path in enumerate(img_list):
        norm_feat = model.vgg_extract_feat(
            img_path)  # 修改此处改变提取特征的网络
        img_name = os.path.split(img_path)[1]
        feats.append(norm_feat)
        names.append(img_name)
        print("extracting feature from image No. %d , %d images in total" %
              ((i + 1), len(img_list)))

    feats = np.array(feats)
    # print(feats)
    # directory for storing extracted features
    # output = args["index"]
    output = index
    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    # h5f.create_dataset('dataset_2', data = names)
    h5f.create_dataset('dataset_2', data=np.string_(names))
    h5f.close()


def match(database, queryImgPath):
    index = os.path.join(utils.MODELS, 'vgg_featureCNN_' + database + '.h5')
    # read in indexed images' feature vectors and corresponding image names
    h5f = h5py.File(index, 'r')
    feats = h5f['dataset_1'][:]
    imgNames = h5f['dataset_2'][:]
    h5f.close()

    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")

    # init VGGNet16 model
    model = VGGNet()

    # extract query image's feature, compute simlarity score and sort
    queryVec = model.vgg_extract_feat(queryImgPath)  # 修改此处改变提取特征的网络
    # print(queryVec.shape)
    # print(feats.shape)
    print('--------------------------')
    # print(queryVec)
    # print(feats.T)
    print('--------------------------')
    scores = np.dot(queryVec, feats.T)
    # scores = np.dot(queryVec, feats.T)/(np.linalg.norm(queryVec)*np.linalg.norm(feats.T))
    rank_ID = np.argsort(scores)[::-1]
    rank_score = scores[rank_ID]
    # print (rank_ID)
    print(rank_score)

    # number of top retrieved images to show
    maxres = 5  # 检索出三张相似度最高的图片
    imglist = []
    for i, index in enumerate(rank_ID[0:maxres]):
        if rank_score[i].item() > 0.8:
            imglist.append(imgNames[index].decode(
                'UTF-8') + ":" + str(rank_score[i].item()))
            # print(type(imgNames[index]))
            print("image names: " +
                  str(imgNames[index]) + " scores: %f" % rank_score[i])
    print("top %d images in order are: " % maxres, imglist)

    return imglist

def preDeal(img_path):
    objectDetect(img_path)
    linesPalm(img_path)

def zeroPaddingResizeCV(img, size=(224, 224), interpolation=None):
    isize = img.shape
    ih, iw = isize[0], isize[1]
    h, w = size[0], size[1]
    scale = min(w / iw, h / ih)
    new_w = int(iw * scale + 0.5)
    new_h = int(ih * scale + 0.5)
 
    img = cv2.resize(img, (new_w, new_h), interpolation)
    new_img = np.zeros((h, w, 3), np.uint8)
    new_img[(h-new_h)//2:(h+new_h)//2, (w-new_w)//2:(w+new_w)//2] = img

    return new_img

def linesPalm(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,60,65,apertureSize = 3)
    edges = cv2.bitwise_not(edges)
    # cv2.imwrite("palmlines.jpg", edges)
    # palmlines = cv2.imread("palmlines.jpg")
    # img = cv2.addWeighted(palmlines, 0.3, image, 0.7, 0)
    cv2.imwrite(img_path, edges)
    

def objectDetect(img_path):
    print(img_path)

    # step1：加载图片
    image = cv2.imread(img_path)

    # 缩放图片
    image = zeroPaddingResizeCV(image)

    # 转成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=- 1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=- 1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    # show image
    # cv2.imshow("first", gradient)
    # cv2.waitKey()

    # step3：去除图像上的噪声。首先使用低通滤泼器平滑图像（9 x 9内核）,这将有助于平滑图像中的高频噪声。
    # 低通滤波器的目标是降低图像的变化率。如将每个像素替换为该像素周围像素的均值。这样就可以平滑并替代那些强度变化明显的区域。
    # 然后，对模糊图像二值化。梯度图像中不大于90的任何像素都设置为0（黑色）。 否则，像素设置为255（白色）。
    # blur and threshold the image
    blurred = cv2.blur(gradient, (9,  9))
    _, thresh = cv2.threshold(blurred,  90,  255, cv2.THRESH_BINARY)
    # SHOW IMAGE
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey()

    # step4:在上图中我们看到主体区域有很多黑色的空余，我们要用白色填充这些空余，使得后面的程序更容易识别昆虫区域，
    # 这需要做一些形态学方面的操作。
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,  25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # show image
    # cv2.imshow("closed1", closed)
    # cv2.waitKey()

    # step5:从上图我们发现图像上还有一些小的白色斑点，这会干扰之后的轮廓的检测，要把它们去掉。分别执行4次形态学腐蚀与膨胀。
    # perform a series of erosions and dilations
    closed = cv2.erode(closed,  None, iterations=4)
    closed = cv2.dilate(closed,  None, iterations=4)
    # show image
    # cv2.imshow("closed2", closed)
    # cv2.waitKey()

    (cnts, _) = cv2.findContours(closed.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    # ....................................................注意opencv3用法
    box = np.int0(cv2.boxPoints(rect))

    # draw a bounding box arounded the detected barcode and display the image
    # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)#                                       去除就没有绿色
    # cv2.imshow("Image", image)
    # cv2.imwrite("contoursImage2.jpg", image)
    # cv2.waitKey(0)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = image[y1:y1+hight, x1:x1+width]
    print(y1)
    print(y1+hight)
    print(x1)
    print(x1+width)
    # cv2.imshow('cropImg', cropImg)
    cv2.imwrite(img_path, cropImg)
    # cv2.waitKey(0)
