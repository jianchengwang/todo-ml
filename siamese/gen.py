import cv2
import numpy as np
import os
import random

outdir = 'C://Users//Administrator//.keras//right1//'


def get_img(input_Path):
    img_paths = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_paths.append(path+'/'+filename)
    return img_paths


def rotate_bound(imgpath):
    # 读取原图像
    image = cv2.imread(imgpath)

    angle = random.randint(20, 100)
    if angle % 2 == 0:
        angle = -angle

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    output = cv2.warpAffine(image, M, (nW, nH))
    filename = imgpath[-9:]
    print(filename)
    savepath = outdir + filename
    print(savepath)
    cv2.imwrite(savepath, output)


for imgpath in get_img("C://Users//Administrator//.keras//left"):
    print(imgpath)
    rotate_bound(imgpath)
