# create image to visualise the predictions of the EITNet model in a mock environment from input data

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader

from dataset_creator import EITDataset
from network_models import EITNet

# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available
print(f"Using {device} device")

# use opencv to create a mock image of the environment
def create_env():
    # create a white background
    img = np.ones((512, 512, 3), np.uint8) * 255

    # outline with black
    cv2.rectangle(img, (0, 0), (511, 511), (0, 0, 0), 1)
    
    # draw 3 circles equally spaced along the vertical axis not touching the border
    cv2.circle(img, (256, 256), 10, (0, 0, 0), -1)
    cv2.circle(img, (256, 128), 10, (0, 0, 0), -1)
    cv2.circle(img, (256, 384), 10, (0, 0, 0), -1)
    
    # draw 2 rectangles centred on the edges of left and right sides
    cv2.rectangle(img, (0, 256-128), (24, 256+128), (0, 0, 0), -1)
    cv2.rectangle(img, (512-24, 256-128), (511, 256+128), (0, 0, 0), -1)
    
    return img

def draw_prediction(img, prediction): # draw predicted object on the image
    # shape: 0=square, 1=heart, 2=pill, 3=circle, 4=triangle
    # position: 0=bottom, 1=middle, 2=top
    # orientation: 0=v1, 1=v2
    
    shape, position, orientation = prediction
    
    # draw shape in the predicted position and orientation
    # if shape == 0: # square
    #     if position == 0:
    #         if orientation == 0:
                
                
    
    
        
    
    

if __name__ == '__main__':

    # create env
    img = create_env()
    
    # load model
    model = EITNet().to(device)
    model_path = '4M25_modelling/trained_models/model_20250317_162245_13'
    model.load_state_dict(torch.load(model_path))
    print('Loaded model from', model_path)
    model.eval()

    # load data
    dataset = EITDataset('Readings/Funky/')
    dataloader = DataLoader(dataset, batch_size=1)
    
    # make predictions
    for data in dataloader:
        readings, labels = data
        readings = readings.to(device)
        labels = labels.to(device)
        
        # gt
        shape_labels = labels[:, 0]
        position_labels = labels[:, 1]
        orientation_labels = labels[:, 2]
        shape_logits, position_logits, orientation_logits = model(readings)
            
        # predictions
        _, shape_pred = torch.max(shape_logits, 1)
        _, position_pred = torch.max(position_logits, 1)
        _, orientation_pred = torch.max(orientation_logits, 1)
        
        # draw predictions
        draw_prediction(img, (shape_pred, position_pred, orientation_pred))
    
    
