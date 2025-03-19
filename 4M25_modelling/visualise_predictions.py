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
    cv2.rectangle(img, (0, 256-128), (24, 256+128), (12, 104, 15), -1)
    cv2.rectangle(img, (512-24, 256-128), (511, 256+128), (12, 104, 15), -1)
    
    return img

def draw_prediction(img, prediction): # draw predicted object on the image
    # shape: 0=square, 1=heart, 2=pill, 3=circle, 4=triangle
    # position: 0=bottom, 1=middle, 2=top
    # orientation: 0=v1, 1=v2
    
    shape, position, orientation = prediction
    
    # draw true shape in the predicted position and orientation using png files of shapes
    if shape == 0: # square
        shape_img = cv2.imread('4M25_modelling/images/square.png')
        if orientation == 1:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_CLOCKWISE)
    elif shape == 1: # heart
        shape_img = cv2.imread('4M25_modelling/images/heart.png', cv2.IMREAD_UNCHANGED)
        if orientation == 0:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation == 1:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_180)
    elif shape == 2: # pill
        shape_img = cv2.imread('4M25_modelling/images/pill.png', cv2.IMREAD_UNCHANGED)
        if orientation == 1:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_CLOCKWISE)
    elif shape == 3: # circle
        shape_img = cv2.imread('4M25_modelling/images/circle.png', cv2.IMREAD_UNCHANGED)
    elif shape == 4: # triangle
        shape_img = cv2.imread('4M25_modelling/images/triangle.png', cv2.IMREAD_UNCHANGED)
        if orientation == 0:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_CLOCKWISE)
        if orientation == 1:
            shape_img = cv2.rotate(shape_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if position == 0: # bottom
        y = 384
    elif position == 1: # middle
        y = 256
    elif position == 2: # top
        y = 128
        
    x = 256 # Middle vertically

    # Resize the shape image to 128x128
    shape_img = cv2.resize(shape_img, (173, 355), interpolation=cv2.INTER_AREA)
    
    # crop the shape image to 128x128 arond the centre
    shape_img = shape_img[shape_img.shape[0]//2-64:shape_img.shape[0]//2+64, shape_img.shape[1]//2-64:shape_img.shape[1]//2+64]
    
    # color the shape image as blue
    shape_img[:, :, 0] = 0
    shape_img[:, :, 1] = 0
    shape_img[:, :, 2] = 255
    
    # Overlay the shape image on the environment image with middle of the shape at (x, y) for all channels
    for c in range(3):
        img[y-64:y+64, x-64:x+64, c] = shape_img[:, :, c] * (shape_img[:, :, 3] / 255.0) + img[y-64:y+64, x-64:x+64, c] * (1.0 - shape_img[:, :, 3] / 255.0)
        
        
    
    
    return img

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
   
    # get a random sample from the dataset
    sample = dataset[100]
    readings = sample[0].to(device)
    labels = sample[1]
    print('Sample:', sample)
    
    # plot the input readings for the sample
    plt.plot(readings)
    plt.xlabel('Channel Number')
    plt.ylabel('Voltage, mV')
    plt.title('Input Readings')
    plt.show()
    
    # get the predicted object
    with torch.no_grad():
        labels = labels.to(device)
        shape_labels = labels[0]
        position_labels = labels[1]
        orientation_labels = labels[2]
                
        shape_logits, position_logits, orientation_logits = model(readings.unsqueeze(0))
        
        _, shape_pred = torch.max(shape_logits, 1)
        _, position_pred = torch.max(position_logits, 1)
        _, orientation_pred = torch.max(orientation_logits, 1)
        
        print('Predicted:', shape_pred.item(), position_pred.item(), orientation_pred.item())
        print('Actual:', shape_labels.item(), position_labels.item(), orientation_labels.item())
        
    # draw the predicted object on the image
    output = draw_prediction(img, (shape_pred.item(), position_pred.item(), orientation_pred.item()))
    
    # display the image
    plt.imshow(output)
    plt.axis('off')
    
    plt.show()
    

    
