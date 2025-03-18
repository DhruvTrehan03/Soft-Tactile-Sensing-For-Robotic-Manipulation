# pytorch models

import torch
import torch.nn as nn
import torch.optim as optim

class EITNet(nn.Module):
    def __init__(self, input_dim=1024, num_shapes=5, num_pos=3, num_orient=2):
        super(EITNet, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # Ouput heads for each task
        self.shape_head = nn.Linear(64, num_shapes)
        self.pos_head = nn.Linear(64, num_pos)
        self.orient_head = nn.Linear(64, num_orient)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # output each raw logits
        shape_logits = self.shape_head(x)
        pos_logits = self.pos_head(x)
        orient_logits = self.orient_head(x)
        
        return shape_logits, pos_logits, orient_logits
    
    # Example usage:
if __name__ == "__main__":
    # Assuming you have determined the number of shapes from your dataset
    num_shapes = 5  # Example value
    model = EITNet(input_dim=1024, num_shapes=num_shapes)
    
    # Example input (batch of 32 readings)
    batch_size = 32
    dummy_input = torch.randn(batch_size, 1024)
    
    # Forward pass
    shape_logits, position_logits, orientation_logits = model(dummy_input)
    print("Shape logits:", shape_logits.shape)          # Expected: [32, num_shapes]
    print("Position logits:", position_logits.shape)      # Expected: [32, 3]
    print("Orientation logits:", orientation_logits.shape)  # Expected: [32, 2]
    
    # Define loss functions and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy labels for demonstration (replace with your actual labels)
    shape_labels = torch.randint(0, num_shapes, (batch_size,))
    position_labels = torch.randint(0, 3, (batch_size,))
    orientation_labels = torch.randint(0, 2, (batch_size,))
    
    # Compute losses for each head
    loss_shape = criterion(shape_logits, shape_labels)
    loss_position = criterion(position_logits, position_labels)
    loss_orientation = criterion(orientation_logits, orientation_labels)
    
    # Combine losses (here we simply sum them)
    loss = loss_shape + loss_position + loss_orientation
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("Training step complete, combined loss:", loss.item())