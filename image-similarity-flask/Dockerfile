FROM python:3.7
WORKDIR /opt/image-similarity-flask
COPY requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple
COPY . .
CMD ["gunicorn", "app:app", "-c", "./gunicorn.conf.py"]