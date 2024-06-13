import os
import torch
from model import LidarCenterNet

# Set the path to the pre-trained model
model_path = "RL_SB3_CARLA/model/model_seed1_39.pth"

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create an instance of LidarCenterNet
lidar_center_net = LidarCenterNet(config, device, backbone='transFuser')

# Load the pre-trained model weights
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    lidar_center_net.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pre-trained model weights from:", model_path)
else:
    print("No pre-trained model found at:", model_path)

# Set the model to evaluation mode
