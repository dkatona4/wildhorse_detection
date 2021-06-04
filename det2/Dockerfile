FROM python:3.8
RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-dev
WORKDIR /usr/src/app
COPY requirements.txt requirements.txt
COPY horse_detection.py horse_detection.py
COPY model/model_final_f10217.pkl model/model_final_f10217.pkl
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html

