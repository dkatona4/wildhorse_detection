import torch, torchvision
import numpy as np
import cv2

# Import some common detectron2 utilities
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

# Format boxes
def xyxy_to_xywh(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    w = x2-x1
    h = y2-y1
    return [x1,y1,w,h]

input_file_path = "/AI_storage/temp.jpg"
output_crop_path = "/AI_storage/temp_crop.jpg"
output_black_path = "/AI_storage/temp_black.jpg"

img = cv2.imread(input_file_path)

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"  # Run on CPU
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "model/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(img)

# Find a biggest object in the picture
counter = 0
max_index = 0
for box in outputs["instances"].pred_boxes.tensor.cpu().numpy():
    area = (box[2]-box[0])*(box[3]-box[1])
    if counter == 0:
        max = area
        max_index = counter
    elif max < area:
        max = area
        max_index = counter
    counter+=1

# Simple crop
box = outputs['instances'].pred_boxes[max_index]
box = box.tensor.cpu().numpy()[0]
box = xyxy_to_xywh(box)
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
cropped = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

cv2.imwrite(output_crop_path,cropped)

# Blackout the background
background_color = (0,0,0,255)
mask = outputs['instances'].pred_masks[max_index]
img[mask.cpu()!=True,:] = background_color
black = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]

cv2.imwrite(output_black_path,black)