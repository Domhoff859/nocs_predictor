import torch
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from star_dash.src.destar import DestarRepresentation
import json
from model import Autoencoder as ae


show_plot = False
number_plots = 10


obj_id = '1'
dataset_path = '/home/domin/Documents/Datasets/tless/xyz_data_360_notilt'

# Specify the path to the saved weights file
weights_path = '/home/domin/Documents/Datasets/Weights Baseline/tless/weights/' + obj_id + "/" +"generator_epoch_50.pth"

# Create an instance of your model
generator = ae(input_resolution=128)
# generator.to(torch.device('cuda'))

# Load the weights
generator.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
generator.eval()

# Use the loaded model for inference or further processing
rgb_folder_path = os.path.join(dataset_path, obj_id, 'rgb')
nocs_folder_path = os.path.join(dataset_path, obj_id, 'nocs')
mask_folder_path = os.path.join(dataset_path, obj_id, 'mask')

# Pick one random image from the star_path
nocs_files = os.listdir(nocs_folder_path)
#random_files: str = np.random.choice(nocs_files, size=number_plots)
random_files = nocs_files

if show_plot:
    f, ax = plt.subplots(number_plots, 3, figsize=(50, 50))
    
dataset_mse = 0

for i, random_file in enumerate(random_files):
    rgb_path = os.path.join(rgb_folder_path, random_file)
    nocs_path = os.path.join(nocs_folder_path, random_file)
    mask_path = os.path.join(mask_folder_path, random_file)

    rgb_image = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
    nocs_image = np.array(Image.open(nocs_path), dtype=np.uint8)
    mask_image = np.array(Image.open(mask_path), dtype=np.float64)#[...,np.newaxis]

    #rgb_image = np.where(mask_image > 1.0, rgb_image, [0, 0, 0])
    rgb_img_array = (rgb_image / 127.5 - 1.0).astype(np.float32)
    rgb_img_array = np.transpose(rgb_img_array, (2, 0, 1))
    rgb_img_array = torch.from_numpy(rgb_img_array)#.to(torch.device('cuda'))

    estimated_nocs = generator(rgb_img_array[np.newaxis, ...])
    
    cpu_estimated_nocs = estimated_nocs.detach().cpu().numpy().squeeze(0)
    cpu_estimated_nocs = cpu_estimated_nocs.transpose(1, 2, 0) / 2 + 0.5
    
    # Mean(MSE(nocs_image, cpu_estimated_nocs))
    
    dataset_mse += np.mean((nocs_image - cpu_estimated_nocs) ** 2)
    
    # Plot some iamges
    if show_plot:
        ax[i, 0].set_title('RGB')
        ax[i, 0].imshow(rgb_image)
        ax[i, 1].set_title('GT Nocs')
        ax[i, 1].imshow(nocs_image)
        ax[i, 2].set_title('GEN Nocs')
        ax[i, 2].imshow(cpu_estimated_nocs)
        
if show_plot:
    plt.tight_layout()
    plt.show()

print("MSE: ", dataset_mse / len(random_files))
