#!/bin/sh
docker-compose run web python3 manage.py makemigrations imageupload
docker-compose run web python3 manage.py migrate imageupload
mkdir -p ./images/segmentation
mkdir -p ./images/style
