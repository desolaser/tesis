import math
def layerCalculator(width, heigth):    
    """ Returns the dimensions of the data after having passed through the conv layers"""

    width, heigth = convSize(width, heigth, 8, 2) # conv 1
    print(width, heigth)
    width, heigth = convSize(width, heigth, 2, 2) # pool 1
    print(width, heigth)
    width, heigth = convSize(width, heigth, 4, 1) # conv 2
    print(width, heigth)
    width, heigth = convSize(width, heigth, 2, 2) # pool 1
    print(width, heigth)
    print(width * heigth * 32)
    return width * heigth * 32

def autoencoderLayerCalculator(width, heigth):    
    """ Returns the dimensions of the data after having passed through the conv layers"""

    width, heigth = convSize(width, heigth, 7, 1) # conv 1
    print(width, heigth)
    width, heigth = convSize(width, heigth, 2, 1) # pool 1
    print(width, heigth)
    width, heigth = convSize(width, heigth, 5, 2) # conv 2
    print(width, heigth)
    width, heigth = convSize(width, heigth, 3, 2) # conv 3
    print(width, heigth)
    width, heigth = convSize(width, heigth, 2, 2) # pool 1
    print(width, heigth)
    print("Conv layers output:", width * heigth * 32) 
    linear_input = width * heigth * 32
    print("Code size", 1024)
    width = 18
    heigth = 10     
    print("Conv layers output:", width * heigth * 32) 
    linear_output = width * heigth * 32   
    width, heigth = deconvSize(width, heigth, 3, 3, 3, 2) # deconv 1
    print(width, heigth)
    width, heigth = deconvSize(width, heigth, 5, 5, 2, 1) # deconv 2
    print(width, heigth)
    width, heigth = deconvSize(width, heigth, 8, 8, 1) # deconv 3
    print(width, heigth)

def convSize(width, heigth, kernel_size, stride=1, padding=0):
    width = ((width - kernel_size - (2 * padding)) / stride) + 1
    heigth = ((heigth - kernel_size - (2 * padding)) / stride) + 1
    width = math.trunc(width)
    heigth = math.trunc(heigth)
    return width, heigth

def deconvSize(width, heigth, kernel_heigth, kernel_width, stride=1, padding=0):
    width = ((width - 1) * stride) + kernel_width - (2 * padding)
    heigth = ((heigth - 1) * stride) + kernel_heigth - (2 * padding)
    width = math.trunc(width)
    heigth = math.trunc(heigth)
    return width, heigth
