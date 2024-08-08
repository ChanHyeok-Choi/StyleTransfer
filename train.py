# import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np
from PIL import Image
import os

from models import StyleTransfer
from loss import ContentLoss, StyleLoss
from tqdm import tqdm

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def pre_processing(image: Image.Image) -> torch.Tensor:
    # image resize 512 x 512
    # Image -> Tensor
    # Normalize
    preprocessing = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean, std) # lambda x: (x-mean) / std
    ]) # (c, h, w)
    
    # (1, c, h, w)
    image_tensor:torch.Tensor = preprocessing(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def post_processing(tensor: torch.Tensor) -> Image.Image:
    # shape 1,c,h,w
    image:np.ndarray = tensor.to('cpu').detach().numpy()
    # shape c,h,w
    image = image.squeeze()
    # shape h,w,c
    image = image.transpose(1, 2, 0)
    # de-normalize
    image = image * std + mean
    
    # clip
    image = image.clip(0.1) * 255
    # uint8
    image = image.astype(np.uint8)
    # numpy -> Image
    return Image.fromarray(image)
    

def train_main():
    # load data
    ## pre processing
    content_image = Image.open('./cat.jpg')
    style_image = Image.open('./gogh.jpg')
    content_image = pre_processing(content_image)
    style_image = pre_processing(style_image)

    # load model
    style_transfer = StyleTransfer().eval()
    
    # load loss
    content_loss = ContentLoss()
    style_loss = StyleLoss()

    # hyper-parameter
    alpha = 1
    beta = 1e6 # style image 마다 beta 값을 조정해줘야 이쁜 이미지가 나온다.
    lr = 1
    
    save_root = f'{alpha}_{beta}_{lr}_initContent'
    os.makedirs(save_root, exist_ok=True)
    
    # device setting
    device = 'cpu'
    if torch.backends.mps.is_built() and torch.backends.mps.is_available(): # only in macOS
        device = 'mps'
    if torch.cuda.is_available():
        device = 'cuda'
        
    style_transfer = style_transfer.to(device)
    content_image = content_image.to(device)
    style_image = style_image.to(device)
    
    # optimizer setting
    # x = torch.randn(1, 3, 512, 512).to(device)
    x = content_image.clone()
    x = x.requires_grad_(True)
    # optimizer = optim.Adam([x], lr=lr)
    optimizer = optim.LBFGS([x], lr=lr)
    
    def closure():
        optimizer.zero_grad()
        ## content representation (x, content_image)
        ## style representation (x, style_image)
        x_content_list = style_transfer(x, 'content')
        y_content_list = style_transfer(content_image, 'content')
        
        x_style_list = style_transfer(x, 'style')
        y_style_list = style_transfer(style_image, 'style')
        
        ## loss_content, loss_style
        loss_c = 0
        loss_s = 0
        loss_total = 0
        
        for x_content, y_content in zip(x_content_list, y_content_list):
            loss_c += content_loss(x_content, y_content)
        
        for x_style, y_style in zip(x_style_list, y_style_list):
            loss_s += style_loss(x_style, y_style)
        
        loss_total = alpha * loss_c + beta * loss_s
        loss_total.backward()
        return loss_total
    
    # train loop
    steps = 1000
    for step in tqdm(range(steps)):
        
        ## optimizer step
        optimizer.step(closure=closure)
        
        ## loss logging
        if step % 100 == 0:
            ## content representation (x, content_image)
            ## style representation (x, style_image)
            x_content_list = style_transfer(x, 'content')
            y_content_list = style_transfer(content_image, 'content')
            
            x_style_list = style_transfer(x, 'style')
            y_style_list = style_transfer(style_image, 'style')
            
            ## loss_content, loss_style
            loss_c = 0
            loss_s = 0
            loss_total = 0
            
            for x_content, y_content in zip(x_content_list, y_content_list):
                loss_c += content_loss(x_content, y_content)
            
            for x_style, y_style in zip(x_style_list, y_style_list):
                loss_s += style_loss(x_style, y_style)
            
            loss_total = alpha * loss_c + beta * loss_s
            print(f'loss_c: {loss_c.cpu()}')
            print(f'loss_s: {loss_s.cpu()}')
            print(f'loss_total: {loss_total.cpu()}')
            
            ## post processing
            gen_img: Image.Image = post_processing(x)
            ## image gen output save
            gen_img.save(os.path.join(save_root, f'{step}.jpg'))

if __name__ == '__main__':
    train_main()