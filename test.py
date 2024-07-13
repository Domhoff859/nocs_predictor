import torch
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from star_dash.src.destar import DestarRepresentation
import json
from model import Autoencoder as ae

number_plots = 10


obj_id = '1'
dataset_path = '/home/domin/Documents/Datasets/tless/xyz_data_test'

# Specify the path to the saved weights file
weights_path = '/home/domin/Documents/Datasets/Weights StarDash/tless/weights/' + obj_id + "/" +"generator_epoch_50.pth"

# Create an instance of your model
generator = ae(input_resolution=128)
# generator.to(torch.device('cuda'))

# Load the weights
generator.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

# Set the model to evaluation mode
generator.eval()

# Use the loaded model for inference or further processing
rgb_folder_path = os.path.join(dataset_path, obj_id, 'rgb')
star_folder_path = os.path.join(dataset_path, obj_id, 'star')
dash_folder_path = os.path.join(dataset_path, obj_id, 'dash')
nocs_folder_path = os.path.join(dataset_path, obj_id, 'nocs')
mask_folder_path = os.path.join(dataset_path, obj_id, 'mask')
train_R_folder_path = os.path.join(dataset_path, obj_id, 'cam_R_m2c')

# Pick one random image from the star_path
star_files = os.listdir(star_folder_path)
random_files: str = np.random.choice(star_files, size=number_plots)

f, ax = plt.subplots(number_plots, 9, figsize=(50, 50))

for i, random_file in enumerate(random_files):
    print(f'Processing {random_file}')

    rgb_path = os.path.join(rgb_folder_path, random_file)
    star_path = os.path.join(star_folder_path, random_file)
    dash_path = os.path.join(dash_folder_path, random_file)
    nocs_path = os.path.join(nocs_folder_path, random_file)
    mask_path = os.path.join(mask_folder_path, random_file)
    train_R_path = os.path.join(train_R_folder_path, random_file.replace('.png', '.npy'))
    model_info_path = os.path.join(dataset_path, 'models_info.json')

    rgb_image = np.array(Image.open(rgb_path).convert('RGB'), dtype=np.uint8)
    star_image = np.array(Image.open(star_path), dtype=np.uint8)[:,:,::-1]
    dash_image = np.array(Image.open(dash_path), dtype=np.uint8)[:,:,::-1]
    nocs_image = np.array(Image.open(nocs_path), dtype=np.uint8)
    mask_image = np.array(Image.open(mask_path), dtype=np.float64)[...,np.newaxis]
    train_R = np.load(train_R_path)

    with open(model_info_path, 'r') as file:
        model_info = json.load(file)

    rgb_image = np.where(mask_image > 1.0, rgb_image, [0, 0, 0])
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


    # Mean(MSE(nocs_image, cpu_estimated_nocs))
    ax[i, 0].set_title('RGB')
    ax[i, 0].imshow(rgb_image)
    ax[i, 1].set_title('GT Nocs')
    ax[i, 1].imshow(nocs_image)
    ax[i, 2].set_title('GEN Nocs')
    ax[i, 2].imshow(cpu_estimated_nocs)
    ax[i, 3].set_title('GT Star')
    ax[i, 3].imshow(star_image)
    ax[i, 4].set_title('GEN Star')
    ax[i, 4].imshow(cpu_estimated_star)
    ax[i, 5].set_title('GT Dash')
    ax[i, 5].imshow(dash_image)
    ax[i, 6].set_title('GEN Dash')
    ax[i, 6].imshow(cpu_estimated_dash)
    ax[i, 7].set_title('GT Mask')
    ax[i, 7].imshow(mask_image)
    ax[i, 8].set_title('GEN Mask')
    ax[i, 8].imshow(cpu_estimated_mask)
plt.tight_layout()
plt.show()

