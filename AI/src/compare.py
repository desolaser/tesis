from model.autoencoder import autoencoder
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets, transforms
import numpy as np
import glob
import torch
import os

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    return x

filenames = []
image_list = []
model = torch.load('./model/autoencoder.pth')
destination = '../compare/auto_output/'
trans = transforms.ToTensor()

for filename in glob.glob('../compare/inputs/*.png'): #assuming gif
    im = Image.open(filename)
    im = trans(im)
    im = im[np.newaxis, :, :, :].cuda()
    output, _ = model(im)
    output = to_img(output.cpu().data)
    file = os.path.basename(filename)
    save_image(output, destination+file)


