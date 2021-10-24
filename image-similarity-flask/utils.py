from os.path import join, dirname
from os import environ, makedirs
from dotenv import load_dotenv

CLIENT_SECRET_KEY = 'test'
DATA = '/data'
DATABASES = '/data/databases'
MODELS = '/data/models'
TEMPDIR = '/data/tmp'
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'jfif', 'bmp']) // opencv imwrite only support png or jpg
ALLOWED_EXTENSIONS = set(['png', 'jpg'])


def init_config():
    dotenv_path = join(dirname(__file__), '.env')
    load_dotenv(dotenv_path)

    global CLIENT_SECRET_KEY
    global DATA
    global DATABASES
    global MODELS
    global TEMPDIR
    CLIENT_SECRET_KEY = environ.get("CLIENT_SECRET_KEY")
    DATA = environ.get("DATA")
    DATABASES = join(DATA, 'databases')
    MODELS = join(DATA, 'models')
    TEMPDIR = join(DATA, 'tmp')
    makedirs(DATABASES, exist_ok=True)
    makedirs(MODELS, exist_ok=True)
    makedirs(TEMPDIR, exist_ok=True)


def check_clientSecretKey(clientSecretKey):
    return clientSecretKey and clientSecretKey == CLIENT_SECRET_KEY


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


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
