# code adapted from: https://github.com/kirumang/Pix2Pose

import os
import sys

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
    
    max_epochs = 50
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
        num_workers=8,
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

    # m_info = model_info['{}'.format(obj_id)]
    # keys = m_info.keys()

    # sym_pool=[]
    # sym_cont = False
    # sym_pool.append(np.eye(3))
    # if('symmetries_discrete' in keys):
    #     print(obj_id,"is symmetric_discrete")
    #     print("During the training, discrete transform will be properly handled by transformer loss")
    #     sym_poses = m_info['symmetries_discrete']
    #     print("List of the symmetric pose(s)")
    #     for sym_pose in sym_poses:
    #         sym_pose = np.array(sym_pose).reshape(4,4)
    #         print(sym_pose[:3,:3])
    #         sym_pool.append(sym_pose[:3,:3])
    # if('symmetries_continuous' in keys):
    #     sym_cont=True

    generator = ae(input_resolution=128)
    generator.to(device)

    #transformer_loss = transformerLoss(sym=sym_pool)

    mse_loss_star = torch.nn.MSELoss()
    mse_loss_dash = torch.nn.MSELoss()
    mse_loss_mask = torch.nn.MSELoss()
    mse_loss_destar = torch.nn.MSELoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epochs, eta_min=1e-7)

    generator.train()

    epoch = 0
    iteration = 0

    total_iterations = len(train_dataloader)
    print("total iterations per epoch: ", total_iterations)

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()
            rgb_images_gt = batch["rgb"].to(device)
            xyz_images_gt = batch["nocs"].to(device)
            star_image_gt = batch["star"].to(device)
            dash_image_gt = batch["dash"].to(device)
            mask_image_gt = batch["mask"].to(device)
            # cam_R_m2c_gt = batch["cam_R_m2c"].to(device)

            estimated_star, estimated_dash, estimated_mask = generator(rgb_images_gt)
            
            mask_image_gt = torch.stack([mask_image_gt], dim=1)
            output_mask = mse_loss_mask(estimated_mask, mask_image_gt)
            
            output_star = mse_loss_star(estimated_star*mask_image_gt, star_image_gt*mask_image_gt)
            output_dash = mse_loss_dash(estimated_dash*mask_image_gt, dash_image_gt*mask_image_gt)
            
            output = output_star + output_dash + output_mask
            
            # output = mse_loss(xyz_images_estimated, xyz_images_gt)
            # loss_transformer = transformer_loss([input, target])   -> needs to be modified

            optimizer_generator.zero_grad()
            output.backward()
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

            lr_current = optimizer_generator.param_groups[0]['lr']
            print("Epoch {:02d}, Iteration {:03d}/{:03d}, Recon Loss: {:.4f}, lr_gen: {:.6f}, Time per Iteration: {:.4f} seconds".format(epoch, iteration + 1, total_iterations, output, lr_current, elapsed_time_iteration))

            iteration += 1

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        imgfn = obj_val_img_dir + "/{:03d}.jpg".format(epoch)


        gen_star, gen_dash, gen_mask = generator(rgb_images_gt)
        
        # Calculate the DESTAR representation
        destar_model_info = load_destar_model_info(os.path.join(dataset_path, 'models_info.json'), obj_id)
        destar = DestarRepresentation(model_info=destar_model_info)
        
        f,ax = plt.subplots(10,6,figsize=(10,20))

        for i in range(10):
            cpu_gen_star = gen_star[i].detach().cpu().numpy().transpose(1, 2, 0)
            cpu_gen_dash = gen_dash[i].detach().cpu().numpy().transpose(1, 2, 0)
            cpu_gen_mask = gen_mask[i].detach().cpu().numpy().transpose(1, 2, 0)
            
            cpu_gen_nocs = destar.calculate(star=cpu_gen_star[np.newaxis, ...], dash=cpu_gen_dash[np.newaxis, ...], isvalid=cpu_gen_mask[np.newaxis, ...])
            cpu_gen_nocs = cpu_gen_nocs.squeeze(0)
            
            ax[i,0].set_title("RGB")
            ax[i,0].imshow( ( (rgb_images_gt[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0)  )
            ax[i,1].set_title("GT Nocs")
            ax[i,1].imshow( ( (xyz_images_gt[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
            ax[i,2].set_title("Gen Nocs")
            ax[i,2].imshow( ( (cpu_gen_nocs+1)/2))
            ax[i,3].set_title("Gen Star")
            ax[i,3].imshow( ( (cpu_gen_star+1)/2))
            ax[i,4].set_title("Gen Dash")
            ax[i,4].imshow( ( (cpu_gen_dash+1)/2))
            ax[i,5].set_title("Gen Mask")
            ax[i,5].imshow( ( (cpu_gen_mask+1)/2))
        plt.savefig(imgfn)
        plt.close()
        
        scheduler_generator.step()

        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(obj_weight_dir, f'generator_epoch_{epoch}.pth'))
        epoch += 1

# Run the main function
if __name__ == "__main__":    
    main()