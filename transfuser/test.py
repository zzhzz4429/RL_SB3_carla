import torch
import os

# Specify the path to the saved .pt file
pt_file_path = "/home/ascc304/transfuser/logdir/transfuser/fused_features_epoch1_batch0_sample5.pt"

# Check if the file exists
if os.path.exists(pt_file_path):
    # Load the tensor from the .pt file
    fused_features = torch.load(pt_file_path)
    
    # Print the tensor
    print("Fused Features Tensor:")
    print(fused_features)
    
    # Print the tensor shape
    print("Tensor Shape:")
    print(fused_features.shape)
else:
    print(f"File not found: {pt_file_path}")