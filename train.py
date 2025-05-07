import argparse
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models
import matplotlib.pyplot as plt
import json

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--val_dataset_dir', type=str, help="validation dataset directory")
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--plot_loss', action='store_true', help="plot training and validation losses")

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet

# Loss functions
class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def evaluate(net, val_dataloader, content_loss, criterion, syncnet=None):
    """Evaluate model on validation set"""
    net.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            imgs, labels, audio_feat = batch
            imgs = imgs.cuda()
            labels = labels.cuda()
            audio_feat = audio_feat.cuda()
            
            preds = net(imgs, audio_feat)
            
            if use_syncnet and syncnet is not None:
                y = torch.ones([preds.shape[0],1]).float().cuda()
                a, v = syncnet(preds, audio_feat)
                sync_loss = cosine_loss(a, v, y)
            
            loss_PerceptualLoss = content_loss.get_loss(preds, labels)
            loss_pixel = criterion(preds, labels)
            
            if use_syncnet and syncnet is not None:
                loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss
            else:
                loss = loss_pixel + loss_PerceptualLoss*0.01
                
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def plot_losses(train_losses, val_losses, save_dir):
    """Plot and save training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()
    
    # Save loss values to JSON for later analysis
    loss_data = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(os.path.join(save_dir, 'loss_data.json'), 'w') as f:
        json.dump(loss_data, f)

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")
            
        syncnet = SyncNet_color(args.asr).eval().cuda()
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint))
    else:
        syncnet = None
        
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    # Load training dataset
    train_dataset = MyDataset(args.dataset_dir, args.asr)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    
    # Load validation dataset if provided
    val_dataloader = None
    if args.val_dataset_dir:
        val_dataset = MyDataset(args.val_dataset_dir, args.asr)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()
    
    best_val_loss = float('inf')
    
    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    
    for e in range(epoch):
        net.train()
        epoch_train_loss = 0
        num_batches = 0
        
        with tqdm(total=len(train_dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                imgs, labels, audio_feat = batch
                imgs = imgs.cuda()
                labels = labels.cuda()
                audio_feat = audio_feat.cuda()
                
                preds = net(imgs, audio_feat)
                
                if use_syncnet and syncnet is not None:
                    y = torch.ones([preds.shape[0],1]).float().cuda()
                    a, v = syncnet(preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                
                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)
                
                if use_syncnet and syncnet is not None:
                    loss = loss_pixel + loss_PerceptualLoss*0.01 + 10*sync_loss
                else:
                    loss = loss_pixel + loss_PerceptualLoss*0.01
                
                epoch_train_loss += loss.item()
                num_batches += 1
                    
                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Evaluate on validation set
        if val_dataloader is not None:
            val_loss = evaluate(net, val_dataloader, content_loss, criterion, syncnet)
            val_losses.append(val_loss)
            print(f"Training Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), os.path.join(save_dir, 'best.pth'))
                print(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            print(f"Training Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if e % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, f'epoch_{e}.pth'))
            
        if args.see_res:
            net.eval()
            img_concat_T, img_real_T, audio_feat = train_dataset.__getitem__(random.randint(0, train_dataset.__len__()))
            img_concat_T = img_concat_T[None].cuda()
            audio_feat = audio_feat[None].cuda()
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+"_real.jpg", img_real)
    
    # Plot losses if requested
    if args.plot_loss:
        plot_losses(train_losses, val_losses, save_dir)

if __name__ == '__main__':
    net = Model(6, args.asr).cuda()
    train(net, args.epochs, args.batchsize, args.lr)