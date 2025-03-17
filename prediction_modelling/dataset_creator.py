import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io

# map shape strings to indices (e.g., 'cube' -> 0, 'cylinder' -> 1, etc.)
def build_shape_dict(file_list):
    shapes = set() # Use a set to avoid duplicates
    for file in file_list: # loop through all files to get unique shapes
        # Assuming filename format: shape_position_orientation_trial.mat
        base = os.path.splitext(os.path.basename(file))[0]
        parts = base.split('_')
        if parts[1] == 'Control':
            continue
        shapes.add(parts[1]) # add shape to set
    shape_to_idx = {shape: idx for idx, shape in enumerate(sorted(shapes))} # build map shape strings to indices
    # save the mapping to a file
    with open('prediction_modelling/shape_to_idx.txt', 'w') as f:
        for shape, idx in shape_to_idx.items():
            f.write(f"{shape}: {idx}\n")
    return shape_to_idx

class EITDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        # data_folder: path to the folder containing .mat files
        # transform: a function to apply to each sample (optional)
        self.folder = data_folder
        self.transform = transform
        
        # Get a list of .mat files in the folder
        self.file_paths = [os.path.join(data_folder, f) 
                           for f in os.listdir(data_folder) if f.endswith('.mat')]
        
        # Build a mapping for object shapes
        self.shape_to_idx = build_shape_dict(self.file_paths)
        
        # Lists to hold data samples and labels
        self.samples = []  # each sample will be a 1024-dimensional vector
        self.labels = []   # labels: (shape, position, orientation)

        # Process each file
        for file in self.file_paths:
            # Load .mat file
            mat = scipy.io.loadmat(file)
            data = mat['plotthis_right']  # extract the 11x1024 matrix from the .mat file
            
            # Extract label information from the filename
            # Example filename: "cube_2_1_1.mat" => shape: "cube", position: 2, orientation: 1, trial: 1
            base = os.path.splitext(os.path.basename(file))[0]
            parts = base.split('_')
            
            # ignore control data
            if parts[1] == 'Control':
                continue
            
            shape_str = parts[1]
            position = int(parts[2])
            orientation = int(parts[3])
            
            # Convert shape string to an index
            shape_num = self.shape_to_idx[shape_str]
            
            # For each row (reading) in the 11x1024 matrix, add the sample and label
            for reading in data:
                self.samples.append(reading)
                self.labels.append((shape_num, position, orientation))

        # Convert lists to numpy arrays (optional)
        self.samples = np.array(self.samples)  # shape: (n_samples, 1024)
        self.labels = np.array(self.labels)      # shape: (n_samples, 3)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get the sample and label at the given index
        x = self.samples[idx]
        y = self.labels[idx]
        
        # Apply transformation if provided (e.g., normalization, converting to torch tensor)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.tensor(x, dtype=torch.float)
        
        # convert label to torch tensor
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# test usage:
if __name__ == "__main__":
    data_folder = "Readings/Funky/"
    
    # Create dataset instance
    dataset = EITDataset(data_folder)
    
    # Create DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate over the data
    for batch in dataloader:
        inputs, labels = batch
        print("Input batch shape:", inputs.shape)  # Expected shape: [batch_size, 1024]
        print("Labels batch shape:", labels.shape)   # Expected shape: [batch_size, 3]
        break  # Remove break to iterate over entire dataset
