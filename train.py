# code adapted from: https://github.com/kirumang/Pix2Pose

import os
import sys
from PIL import Image
ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

import time
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import Autoencoder as ae
from model import TransformerLoss as transformerLoss

from nocs_dataset import NOCSTrain

from star_dash.src.destar import DestarRepresentation

def setup_environment():
    if len(sys.argv) != 4:
        print("Usage: python3 train.py <gpu_id> <obj_id> </path/to/dataset>")
        sys.exit()

    if sys.argv[1] == '-1':
        sys.argv[1] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
     
def load_destar_model_info(model_path: str , obj_id: str):
    model_info: dict = {}
    
    with open(model_path, 'r') as f:
        temp: dict = json.load(f)
    
    # Check if the object ID is present in the model info file    
    assert obj_id in temp.keys(), "Object ID not found in the model info file"
    
    # Load the discrete symmetries if present
    if "symmetries_discrete" in temp[obj_id].keys():
        model_info['symmetries_discrete'] = [np.array(_).reshape((4,4)) for _ in temp[obj_id]["symmetries_discrete"]]
    else:
        model_info['symmetries_discrete'] = []
        
    # Check if continuous symmetries are present
    model_info["symmetries_continuous"] = "symmetries_continuous" in temp[obj_id]
        
    return model_info
     

def main():
    setup_environment()
    
    max_epochs = 50 + 1
    batch_size = 16

    augmentation_prob=0.5
    imsize = 128
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    
    obj_id = sys.argv[2]
    dataset_path = sys.argv[3]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = NOCSTrain(data_root=dataset_path, size=imsize, augmentation_prob=augmentation_prob, obj_id=obj_id, crop_object=True, fraction=1.0, augment=True)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )

    weight_dir = "./weights"
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    obj_weight_dir = weight_dir + "/" + str(obj_id)
    if not(os.path.exists(obj_weight_dir)):
            os.makedirs(obj_weight_dir)    

    val_img_dir = "./val_img/"
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)

    obj_val_img_dir = val_img_dir + "/" + str(obj_id)
    if not(os.path.exists(obj_val_img_dir)):
        os.makedirs(obj_val_img_dir)    

    generator = ae(input_resolution=128)
    generator.to(device)

    mse_loss_star = torch.nn.MSELoss()
    mse_loss_dash = torch.nn.MSELoss()
    mse_loss_mask = torch.nn.MSELoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epochs, eta_min=1e-7)

    generator.train()

    epoch = 0
    iteration = 0

    total_iterations = len(train_dataloader)
    print("total iterations per epoch: ", total_iterations)
    start_time_global = time.time()
    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()
            rgb_images_gt = batch["rgb"].to(device)
            xyz_images_gt = batch["nocs"].to(device)
            star_image_gt = batch["star"].to(device)
            dash_image_gt = batch["dash"].to(device)
            mask_image_gt = batch["mask"].to(device)
            cam_R_m2c_gt = batch["cam_R_m2c"]
            
            estimated_star, estimated_dash, estimated_mask = generator(rgb_images_gt)
            
            output_star = mse_loss_star(estimated_star, star_image_gt)
            output_dash = mse_loss_dash(estimated_dash, dash_image_gt)
            output_mask = mse_loss_mask(estimated_mask, mask_image_gt)
            
            output = output_star + output_dash + output_mask

            optimizer_generator.zero_grad()
            output.backward()
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

            lr_current = optimizer_generator.param_groups[0]['lr']
            print("Epoch {:02d}, Iteration {:03d}/{:03d}, Recon Loss: {:.4f}, lr_gen: {:.6f}, Time per Iteration: {:.4f} seconds".format(epoch, iteration + 1, total_iterations, output, lr_current, elapsed_time_iteration))

            iteration += 1

        iteration = 0
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))
        
        """
        ############################################################################################################
        ############################################################################################################
        ####################             TRAINING END -> TESTING START             #################################
        ############################################################################################################
        ############################################################################################################
        """

        imgfn = obj_val_img_dir + "/{:03d}.jpg".format(epoch)

        # Get the generated Star and Dash representation
        gen_star, gen_dash, gen_mask = generator(rgb_images_gt)
        
        # Move everything from the GPU to the CPU
        cpu_rgb = rgb_images_gt.detach().cpu().numpy()
        cpu_gt_nocs = xyz_images_gt.detach().cpu().numpy()
        cpu_gen_star = gen_star.detach().cpu().numpy()
        cpu_gen_dash = gen_dash.detach().cpu().numpy()
        cpu_gt_star = star_image_gt.detach().cpu().numpy()
        cpu_gt_dash = dash_image_gt.detach().cpu().numpy()
        cpu_gt_mask = mask_image_gt.detach().cpu().numpy()
        cpu_gen_mask = gen_mask.detach().cpu().numpy()
        
        # Calculate the uint8 form of star and dash for destarring
        cpu_gt_nocs = (cpu_gt_nocs + 1.0) * 127.5
        cpu_gt_nocs = np.array(cpu_gt_nocs, dtype=np.uint8)
        
        cpu_gt_star = (cpu_gt_star + 1.0) * 127.5
        cpu_gt_star = np.array(cpu_gt_star, dtype=np.uint8)
        
        cpu_gt_dash = (cpu_gt_dash + 1.0) * 127.5
        cpu_gt_dash = np.array(cpu_gt_dash, dtype=np.uint8)
        
        cpu_gen_star = (cpu_gen_star + 1.0) * 127.5
        cpu_gen_star = np.array(cpu_gen_star, dtype=np.uint8)
        
        cpu_gen_dash = (cpu_gen_dash + 1.0) * 127.5
        cpu_gen_dash = np.array(cpu_gen_dash, dtype=np.uint8)
        
        cpu_gen_mask = np.where(cpu_gen_mask > 0.3, 255.0, 0.0)
        cpu_gen_mask = np.array(cpu_gen_mask, dtype=np.float32)

        # Calculate the DESTAR representation
        destar_model_info = load_destar_model_info(os.path.join(dataset_path, 'models_info.json'), obj_id)
        destar = DestarRepresentation(model_info=destar_model_info)
        
        f,ax = plt.subplots(10,9,figsize=(20,40))

        for i in range(10):
            cpu_rgb_image = cpu_rgb[i].transpose(1, 2, 0)
            cpu_gt_nocs_image = cpu_gt_nocs[i].transpose(1, 2, 0)
            cpu_gt_star_image = cpu_gt_star[i].transpose(1, 2, 0)
            cpu_gen_star_image = cpu_gen_star[i].transpose(1, 2, 0)
            cpu_gt_dash_image = cpu_gt_dash[i].transpose(1, 2, 0)
            cpu_gen_dash_image = cpu_gen_dash[i].transpose(1, 2, 0)
            cpu_gt_mask_image = cpu_gt_mask[i].transpose(1, 2, 0)
            cpu_gen_mask_image = cpu_gen_mask[i].transpose(1, 2, 0)
            
            # print(f'GT Star Shape: {cpu_gt_star_image.shape}')
            # print(f'Gen Star Shape: {cpu_gen_star_image.shape}')
            # print(f'GT Dash Shape: {cpu_gt_dash_image.shape}')
            # print(f'Gen Dash Shape: {cpu_gen_dash_image.shape}')
            # print(f'GT Mask Shape: {cpu_gt_mask_image.shape}')
            # print(f'Gen Mask Shape: {cpu_gen_mask_image.shape}')
            # print(f'GT Star Type: {cpu_gt_star_image.dtype}')
            # print(f'Gen Star Type: {cpu_gen_star_image.dtype}')
            # print(f'GT Dash Type: {cpu_gt_dash_image.dtype}')
            # print(f'Gen Dash Type: {cpu_gen_dash_image.dtype}')
            # print(f'GT Mask Type: {cpu_gt_mask_image.dtype}')
            # print(f'Gen Mask Type: {cpu_gen_mask_image.dtype}')
            
            cpu_gen_nocs_image = destar.calculate(star=cpu_gen_star_image[np.newaxis, ...], dash=cpu_gen_dash_image[np.newaxis, ...], isvalid=cpu_gen_mask_image, train_R=cam_R_m2c_gt[i][np.newaxis, ...])
            cpu_gen_nocs_image = cpu_gen_nocs_image.squeeze(0)
            
            
            # print(f'GT NOCS Max: {cpu_gt_nocs_image.max()}')
            # print(f'GT NOCS Min: {cpu_gt_nocs_image.min()}')
            
            # print(f'Gen NOCS Max: {cpu_gen_nocs_image.max()}')
            # print(f'Gen NOCS Min: {cpu_gen_nocs_image.min()}')
            
            # print(f'GT Star Max: {cpu_gt_star_image.max()}')
            # print(f'GT Star Min: {cpu_gt_star_image.min()}')
            
            # print(f'Gen Star Max: {cpu_gen_star_image.max()}')
            # print(f'Gen Star Min: {cpu_gen_star_image.min()}')
            
            # print(f'GT Dash Max: {cpu_gt_dash_image.max()}')
            # print(f'GT Dash Min: {cpu_gt_dash_image.min()}')
            
            # print(f'Gen Dash Max: {cpu_gen_dash_image.max()}')
            # print(f'Gen Dash Min: {cpu_gen_dash_image.min()}')
            
            # print(f'GT Mask Max: {cpu_gt_mask_image.max()}')
            # print(f'GT Mask Min: {cpu_gt_mask_image.min()}')
            
            # print(f'Gen Mask Max: {cpu_gen_mask_image.max()}')
            # print(f'Gen Mask Min: {cpu_gen_mask_image.min()}')


            ax[i,0].set_title("RGB")
            ax[i,0].imshow((cpu_rgb_image+1)/2)
            ax[i,1].set_title("GT Nocs")
            ax[i,1].imshow(cpu_gt_nocs_image)
            ax[i,2].set_title("Gen Nocs")
            ax[i,2].imshow(cpu_gen_nocs_image)
            ax[i,3].set_title("GT Star")
            ax[i,3].imshow(cpu_gt_star_image)
            ax[i,4].set_title("Gen Star")
            ax[i,4].imshow(cpu_gen_star_image)
            ax[i,5].set_title("GT Dash")
            ax[i,5].imshow(cpu_gt_dash_image)
            ax[i,6].set_title("Gen Dash")
            ax[i,6].imshow(cpu_gen_dash_image)
            ax[i,7].set_title("GT Mask")
            ax[i,7].imshow(cpu_gt_mask_image)
            ax[i,8].set_title("Gen Mask")
            ax[i,8].imshow(cpu_gen_mask_image)
            
        plt.savefig(imgfn)
        plt.close()
        
        scheduler_generator.step()
        
        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(obj_weight_dir, f'generator_epoch_{epoch}.pth'))
        epoch += 1
        
    print("Total training time taken: {:.4f} seconds".format(time.time() - start_time_global))

# Run the main function
if __name__ == "__main__":
    main()