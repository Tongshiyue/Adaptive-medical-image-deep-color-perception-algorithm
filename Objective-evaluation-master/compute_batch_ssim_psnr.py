# Compute the SSIM and PSNR of a batch of images with references

import os
from PIL import Image
from scipy.signal import convolve2d
from os.path import join as pjoin
import numpy as np 
import math
import matlab.engine
import pickle
import matplotlib.pyplot as plt

def load_data(path):
    images = []
    count = 0
    for i in os.listdir(path):
        image_dir = pjoin(path, i)
        image = Image.open(image_dir).convert('L')
        images.append(image)
        count = count+1
    
    return count, images

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape] 
    y,x = np.ogrid[-m:m+1,-n:n+1] 
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) ) 
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0 
    sumh = h.sum() 
    if sumh != 0:
        h /= sumh 
    return h

def filter2(x, kernel, mode='same'): 
    return convolve2d(x, np.rot90(kernel, 2), mode=mode) 

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape: 
        raise ValueError("Input Imagees must have the same dimensions") 
    
    if len(im1.shape) > 2: 
        raise ValueError("Please input the images with 1 channel") 
    
    M, N = im1.shape 
    C1 = (k1*L)**2 
    C2 = (k2*L)**2 
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5) 
    window = window/np.sum(np.sum(window)) 
    
    if im1.dtype == np.uint8: 
        im1 = np.double(im1) 
    
    if im2.dtype == np.uint8: 
        im2 = np.double(im2) 
    
    mu1 = filter2(im1, window, 'valid') 
    mu2 = filter2(im2, window, 'valid') 
    mu1_sq = mu1 * mu1 
    mu2_sq = mu2 * mu2 
    mu1_mu2 = mu1 * mu2 
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq 
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq 
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2 
    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2)) 
    
    return np.mean(np.mean(ssim_map)) 

def compute_psnr(img1, img2):
    if img1.dtype == np.uint8:
        img1 = np.double(img1)

    if img2.dtype == np.uint8:
        img2 = np.double(img2)

    mse = np.mean( (img1 - img2) ** 2 )
    
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == "__main__":
    data_dir = './n'
    files = os.listdir(data_dir)
    gen = pjoin(data_dir, files[0])
    raw = pjoin(data_dir, files[1])
    count, images1 = load_data(gen) #colored
    _, images2 = load_data(raw)
    SSIM = []
    PSNR = []
    FSIMc = []
    for i in range(count):
        im1 = images1[i]
        im2 = images2[i]
        ssim = compute_ssim(np.array(im1),np.array(im2))
        SSIM.append(ssim)
        psnr = compute_psnr(np.array(im1),np.array(im2))
        PSNR.append(psnr)
        eng = matlab.engine.start_matlab()
        im1_mat = matlab.double(np.array(im1).tolist())
        im2_mat = matlab.double(np.array(im2).tolist())
        fsimc = eng.FeatureSIM(im2_mat,im1_mat)
        FSIMc.append(fsimc)
    print('Average SSIM is %.3f' % np.mean(SSIM))
    print('std SSIM is %.3f' % np.std(SSIM))
    print('Average PSNR is %.3f' % np.mean(PSNR))
    print('std PSNR is %.3f' % np.std(PSNR))
    print('Average FSIMc is %.3f' % np.mean(FSIMc))
    print('std FSIMc is %.3f' % np.std(FSIMc))

  

