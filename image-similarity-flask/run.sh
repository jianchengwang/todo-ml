#!/bin/bash

# docker build . -t image-similarity-flask:v0.0.1
# docker tag image-similarity-flask:v0.0.1 jianchengwang/image-similarity-flask
# docker login
# docker push jianchengwang/image-similarity-flask
docker run --rm -e CLIENT_SECRET_KEY='Wjc123456' -e DATA='/data' -v $(pwd)/data:/data --name image-similarity-flask -p 5000:5000 jianchengwang/image-similarity-flask
docker run --rm --env-file ./env.list -v $(pwd)/data:/data --name image-similarity-flask -p 5000:5000 jianchengwang/image-similarity-flask
