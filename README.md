# Wildhorse Detection

### Installation & Running step by step:
1) Clone repository https://github.com/dkatona4/wildhorse_detection.git
2) Download models from: https://vocs.unideb.hu/nextcloud/index.php/s/H7BGxq6HWfrbkXE
3) Copy model_final_f10217.pkl file into wildhorse_detection/det2/model folder
4) Copy inception_BEST.h5 file into wildhorse_detection/ind_det/model folder
5) Create a folder with AI_storage name into wildhorse_detection/ folder.
6) Run "docker-compouse up" command 
7) Copy a picture into the AI_storage folder. (The picture name must be "temp.jpg")
8) "sudo docker-compose run det2 python horse_detection.py && sudo docker-compose run ind_det python ind_det.py"
9) You will have 3 new file in the AI_storage folder: result.json, temp_black.jpg, temp_crop.jpg
