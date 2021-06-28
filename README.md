# Wildhorse Recognition
AI project for identifying przewalski horses as individual
## Useful information
IMPORTANT: We used several drone footage and photos to train our model. 

## Installation & Running step by step:
1) Clone repository https://github.com/dkatona4/wildhorse_detection.git
2) Download models from: https://vocs.unideb.hu/nextcloud/index.php/s/H7BGxq6HWfrbkXE
3) Copy model_final_f10217.pkl file into wildhorse_detection/det2/model folder
4) Copy inception_BEST.h5 file into wildhorse_detection/ind_det/model folder
5) Create a folder with AI_storage name into wildhorse_detection/ folder.
6) Run "docker-compouse up" command 
7) Copy a picture into the AI_storage folder. (The picture name must be "temp.jpg")
8) "sudo docker-compose run det2 python horse_detection.py && sudo docker-compose run ind_det python ind_det.py"
9) You will have 3 new file in the AI_storage folder: result.json, temp_black.jpg, temp_crop.jpg
10) The result.json file will contain the id of the individual which was the closest according to picture.

## Workflow explaination:
1) Copy a picture with temp.jpg name into AI_storage folder. This image will be our processed image.
2) With this command "sudo docker-compose run det2 python horse_detection.py" the program will automatically detect the objects (we used the detectron2 zoo model). From these objects the script will take out one, which has the the largest area (in pixel). The algorithm will crop the picture around the detected individual.
> IMPORTANT: This is a limitation because the individual we want to identify must always take the largest area in the image.
3) After the detection we use the detectron2 maskRCNN function to black out the background. This feature could be usefull for futher training.
4) Run the following command "sudo docker-compose run ind_det python ind_det.py". The program will give us the top20 closest picture (knn method). To do that we used the previously generated AI_data/prediction.csv file.

## Current state

- Currently we have 9 trained individuals
- We are trying to improve the model precision.

## Some results
