import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import PIL                                                 
from sklearn.neighbors import NearestNeighbors

# Read image and add plus dimension
def read_image_expand_dim(image):
    img = PIL.Image.open(image).resize((600,400))                 # Read Image
    img_array = np.expand_dims(np.asarray(img), axis=(0))         # Expand dimension
    return img_array

# Generate json
def create_json(json_header1,json_header2,elements_indexes):
    counter = 0
    json_knn='\"' + json_header1 + '\": [ {'
    file_knn='\"' + json_header2 + '\": [ {'
    for det_ind,pic_name in zip(prediction_class[elements_indexes],picture_name[elements_indexes]):
        counter += 1
        if counter == n_neighbors:
            json_knn+='\"' + str(counter) +'\":\"' + str(int(det_ind)) + '\" } ]'
            file_knn+='\"' + str(counter) +'\":\"' + pic_name + '\" } ]'
        else:
            json_knn+='\"' + str(counter) +'\":\"' + str(int(det_ind)) + '\", '
            file_knn+='\"' + str(counter) +'\":\"' + pic_name + '\", '
    return json_knn,file_knn

# Define paramters path
prediction_csv_path = "/AI_data/prediction.csv"
model_path = "model/inception_BEST.h5"
img_path_crop = "/AI_storage/temp_crop.jpg"
img_path_black = "/AI_storage/temp_black.jpg"
output_path = "/AI_storage/result.json"
n_neighbors = 20

# Give values to the variables
prediction_csv = np.genfromtxt(prediction_csv_path, delimiter=',', dtype="|U200")
prediction_vector = prediction_csv[:,0:8].astype(float)
prediction_class = prediction_csv[:,8].astype(int)
picture_name = prediction_csv[:,9]

# Read Images
img_crop = read_image_expand_dim(img_path_crop)
img_black = read_image_expand_dim(img_path_black)

# Load model and calculate preditcion array
model = tf.keras.models.load_model(model_path, compile=False)
prediction_crop = model.predict(img_crop)
prediction_black = model.predict(img_black)

# Calculate knn
nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(prediction_vector)
distances_crop, indices_crop = nbrs.kneighbors(prediction_crop)
distances_black, indices_black = nbrs.kneighbors(prediction_black)

# Generate JSON files
json_knn_crop,file_knn_crop = create_json("knn_results_crop","file_name_knn_crop",indices_crop[0])
json_knn_black,file_knn_black = create_json("knn_results_black","file_name_knn_black",indices_black[0])

# Simple file write
f = open(output_path, "w")
f.write('{' + json_knn_crop + ',' + file_knn_crop + ',' + json_knn_black + ',' + file_knn_black + '}')
f.close()



