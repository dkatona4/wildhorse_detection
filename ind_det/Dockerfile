FROM python:3.7.9
WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
COPY ind_det.py ind_det.py
COPY model/inception_BEST.h5 model/inception_BEST.h5
RUN pip install --no-cache-dir -r requirements.txt

