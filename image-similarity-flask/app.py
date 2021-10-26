from flask import Flask, request
import logging
import os
import utils
import core


app = Flask(__name__)
utils.init_config()
core.init()

logging.basicConfig(level=logging.DEBUG)


@app.route('/upload', methods=['POST'])
def upload():
    clientSecretKey = request.form['clientSecretKey']
    print(clientSecretKey)
    if not utils.check_clientSecretKey(clientSecretKey):
        return utils.notAuth(msg='Illegal clientSecretKet.')

    database = request.form['database']
    if not database:
        return utils.badRequest(msg='Illegal database.')

    uploaded_files = request.files.getlist("files")
    app.logger.info('file length: {}', len(uploaded_files))
    for file in uploaded_files:
        if file and utils.allowed_file(file.filename):
            savePath = os.path.join(utils.DATABASES, database, file.filename)
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            app.logger.info('savePath: {}', savePath)
            file.save(savePath)
            # core.preDeal(savePath)
        else:
            app.logger.info('error file: {}', file.filename)
    core.init_database(database)
    return utils.result(msg='Successed.')


@app.route('/match', methods=['POST'])
def match():
    clientSecretKey = request.form['clientSecretKey']
    print(clientSecretKey)
    if not utils.check_clientSecretKey(clientSecretKey):
        return utils.notAuth(msg='Illegal clientSecretKet.')

    database = request.form['database']
    if not database:
        return utils.badRequest(msg='Illegal database.')

    uploaded_file = request.files.getlist("files")[0]
    if uploaded_file and utils.allowed_file(uploaded_file.filename):
        savePath = os.path.join(utils.TEMPDIR, uploaded_file.filename)
        os.makedirs(os.path.dirname(savePath), exist_ok=True)
        uploaded_file.save(savePath)
        # core.preDeal(savePath)
        imglist = core.match(database, savePath)
        return utils.result(msg='Successed.', data=imglist)
    else:
        return utils.badRequest(msg='Illegal file.')


if __name__ == '__main__':
    app.run(debug=True)
