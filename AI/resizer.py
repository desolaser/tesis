from PIL import Image
import os, sys
from tqdm import *

path = "./training_set_min/train/"
dirs = os.listdir( path )

def resize():
    for item in tqdm(dirs):
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            imResize = im.resize((108,60), Image.ANTIALIAS)
            imResize.save(f + '.jpg', 'JPEG', quality=90)

resize()