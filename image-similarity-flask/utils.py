from os.path import join, dirname
from os import environ, makedirs
from dotenv import load_dotenv
import cv2
import numpy as np
import uuid

CLIENT_SECRET_KEY = '123456'
DATA = '/data'
DATABASES = '/data/databases'
MODELS = '/data/models'
FEATURES = '/data/features'
TEMPDIR = '/data/tmp'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif', 'bmp'])

def init_config():
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    global CLIENT_SECRET_KEY
    global DATA
    global DATABASES
    global MODELS
    global FEATURES
    global TEMPDIR
    CLIENT_SECRET_KEY = environ.get("CLIENT_SECRET_KEY")
    DATA = environ.get("DATA")
    DATABASES = join(DATA, 'databases')
    MODELS = join(DATA, 'models')
    FEATURES = join(DATA, 'features')
    TEMPDIR = join(DATA, 'tmp')
    makedirs(DATABASES, exist_ok=True)
    makedirs(MODELS, exist_ok=True)
    makedirs(FEATURES, exist_ok=True)
    makedirs(TEMPDIR, exist_ok=True)


def check_clientSecretKey(clientSecretKey):
    return clientSecretKey and clientSecretKey == CLIENT_SECRET_KEY


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def get_label(imagePath):
    # imagePath like 1-aaaaaa.jpg
    return imagePath.split('-')[0]


def rotate_bound(imgpath, angle):
    # 读取原图像
    image = cv2.imread(imgpath)

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
    output = cv2.warpAffine(image, M, (nW, nH), borderValue=[255, 255, 255])
    filename = str(uuid.uuid1()) + "." + imgpath.split(".")[1]
    savePath = join(TEMPDIR, filename)
    cv2.imwrite(savePath, output)
    return savePath

def rotate_img1(image_path, angle):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]
    a, b = w / 2, h / 2
    M = cv2.getRotationMatrix2D((a, b), angle, 1)
    image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[255, 255, 255])
    filename = str(uuid.uuid1()) + "." + image_path.split(".")[1]
    savePath = join(TEMPDIR, filename)
    cv2.imwrite(savePath, image)
    return savePath


def result(data=None, msg='', status=200):
    return {
        "data": data,
        "msg": msg
    }, status, {'ContentType': 'application/json'}


def badRequest(msg='Bad Request'):
    return result(msg=msg, status=400)


def notAuth(msg='Not Auth'):
    return result(msg=msg, status=401)


def notPermission(msg='Not Permission'):
    return result(msg=msg, status=403)


def notFound(msg='Not Found'):
    return result(msg=msg, status=404)


def serveError(msg='Serve Error'):
    return result(msg=msg, status=500)
