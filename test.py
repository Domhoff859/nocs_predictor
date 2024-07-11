import torch
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from star_dash.src.destar import DestarRepresentation
import json
from model import Autoencoder as ae

# Create an instance of your model
generator = ae(input_resolution=128)
# generator.to(torch.device('cuda'))

# Specify the path to the saved weights file
weights_path = './weights/1/generator_epoch_50.pth'

# Load the weights
generator.load_state_dict(torch.load(weights_path))

# Set the model to evaluation mode
generator.eval()

# Use the loaded model for inference or further processing
obj_id = '1'
dataset_path = '/home/domin/Documents/Datasets/tless/'

rgb_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'rgb')
star_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'star')
dash_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'dash')
nocs_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'nocs')
mask_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'mask')
train_R_folder_path = os.path.join(dataset_path, 'xyz_data', obj_id, 'cam_R_m2c')

# Pick one random image from the star_path
star_files = os.listdir(star_folder_path)
random_file: str = np.random.choice(star_files)
# random_file = "000001_000000_000000.png"

rgb_path = os.path.join(rgb_folder_path, random_file)
star_path = os.path.join(star_folder_path, random_file)
dash_path = os.path.join(dash_folder_path, random_file)
nocs_path = os.path.join(nocs_folder_path, random_file)
mask_path = os.path.join(mask_folder_path, random_file)
train_R_path = os.path.join(train_R_folder_path, random_file.replace('.png', '.npy'))
model_info_path = os.path.join(dataset_path, 'xyz_data', 'models_info.json')

rgb_image = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
star_image = np.array(Image.open(star_path), dtype=np.uint8)[:,:,::-1]
dash_image = np.array(Image.open(dash_path), dtype=np.uint8)[:,:,::-1]
nocs_image = np.array(Image.open(nocs_path), dtype=np.uint8)
mask_image = np.array(Image.open(mask_path), dtype=np.float64)[...,np.newaxis]
train_R = np.load(train_R_path)

with open(model_info_path, 'r') as f:
    model_info = json.load(f)

rgb_img_array = (rgb_image / 127.5 - 1.0).astype(np.float32)
rgb_img_array = np.transpose(rgb_img_array, (2, 0, 1))
rgb_img_array = torch.from_numpy(rgb_img_array)#.to(torch.device('cuda'))

estimated_star, estimated_dash, estimated_mask = generator(rgb_img_array[np.newaxis, ...])

cpu_estimated_star = estimated_star[0].detach().cpu().numpy().transpose(1, 2, 0)
cpu_estimated_dash = estimated_dash[0].detach().cpu().numpy().transpose(1, 2, 0)
cpu_estimated_mask = estimated_mask[0].detach().cpu().numpy().transpose(1, 2, 0)

cpu_estimated_star = np.array((cpu_estimated_star + 1.0) * 127.5, dtype=np.uint8)
cpu_estimated_dash = np.array((cpu_estimated_dash + 1.0) * 127.5, dtype=np.uint8)
cpu_estimated_mask = np.where(cpu_estimated_mask > 0.2, 255.0, 0.0)

destar = DestarRepresentation(model_info=model_info[obj_id])
cpu_estimated_nocs = destar.calculate(star=cpu_estimated_star[np.newaxis,...], dash=cpu_estimated_dash[np.newaxis,...], isvalid=cpu_estimated_mask, train_R=train_R[np.newaxis, ...])
cpu_estimated_nocs = cpu_estimated_nocs.squeeze(axis=0)

f, ax = plt.subplots(2, 5, figsize=(20, 10))
ax[0, 0].imshow(rgb_image)
ax[0, 1].set_title('Star')
ax[0, 1].imshow(star_image)
ax[0, 2].set_title('Dash')
ax[0, 2].imshow(dash_image)
ax[0, 3].set_title('Nocs')
ax[0, 3].imshow(nocs_image)
ax[0, 4].set_title('Mask')
ax[0, 4].imshow(mask_image)
ax[1, 0].axis('off')
ax[1, 1].imshow(cpu_estimated_star)
ax[1, 2].imshow(cpu_estimated_dash)
ax[1, 3].imshow(cpu_estimated_nocs)
ax[1, 4].imshow(cpu_estimated_mask)
plt.tight_layout()
plt.show()

