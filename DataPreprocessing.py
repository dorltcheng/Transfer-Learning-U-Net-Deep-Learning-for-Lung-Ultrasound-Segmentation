import os
import re
import numpy as np
import pandas as pd
import natsort
import matplotlib.pyplot as plt
import cv2
from tifffile import imsave
from PIL import Image
from tqdm import tqdm
from glob import glob
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Display image and masks example (single display)
def image_display(image_pathlist, mask_pathlist, index):
    # input: image and mask filepath and the index 
    # output matplotlib images 
    image = Image.open(image_pathlist[index])
    imagearray = np.array(image)
    print('Image shape: ', imagearray.shape)


    mask = Image.open(mask_pathlist[index])
    maskarray = np.array(mask)
    print('Mask shape: ', maskarray.shape)

    fig, ax = plt.subplots(3,figsize=(5,10))
    ax[0].imshow(imagearray, aspect='auto', cmap='gray')
    ax[1].imshow(maskarray, aspect='auto', cmap='gray')
    ax[2].imshow(imagearray, aspect='auto', cmap = 'gray')
    ax[2].imshow(maskarray, cmap = 'Reds', aspect='auto', alpha = 0.4)


# Create a list of filepaths of images and masks 
def train_filepath_list(image_path, mask_path):
    # input: image_path (filepath for images), mask_path (filepath for masks)
    # return: list of filepaths of images and masks
    img_filelist = os.listdir(image_path)
    img_filelist = natsort.natsorted(img_filelist,reverse=False)
    train_images = [image_path + '/' + x for x in img_filelist]
    
    mask_filelist = os.listdir(mask_path)
    mask_filelist = natsort.natsorted(mask_filelist,reverse=False)
    train_masks = [mask_path + '/' + x for x in mask_filelist]
    
    return train_images, train_masks


# Create extra copies of images using horizontal flip
def horizontal_flip(image):
    new_image = image.copy()
    return cv2.flip(new_image, 1)


# Generate new images and masks by calling horizontal_flip function
# Create a list of filepaths for generated images and masks
def gen_filepath_list(images_list, masks_list, gen_image_filepath, gen_mask_filepath):
    # input: original images and masks lists, and filepath for saving gen images and masks
    # return: a list of filepaths of the gen (flipped) images and gen (flipped) masks 
    gen_images = []
    gen_masks = []

    for (image_path, mask_path) in tqdm(zip(images_list, masks_list)):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name = mask_path.split('/')[-1].split('.')[0]

        gen_image_path = gen_image_filepath + '/' + image_name + '_flipped.tif'
        gen_mask_path = gen_mask_filepath + '/' + mask_name + '_flipped.tif'

        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        gen_img = horizontal_flip(img)
        gen_mask = horizontal_flip(mask)

        imsave(gen_image_path, gen_img)
        imsave(gen_mask_path, gen_mask)

        gen_images.append(gen_image_path)
        gen_masks.append(gen_mask_path)

    return gen_images, gen_masks
  
    
def create_copies(image_list, mask_list, img_copy_filepath, mask_copy_filepath):
    image_copy = []
    mask_copy = []
    
    for (image_path, mask_path) in tqdm(zip(image_list, mask_list)):
        image_name = image_path.split('/')[-1].split('.')[0]
        mask_name = mask_path.split('/')[-1].split('.')[0]

        image_copy_path = img_copy_filepath + '/' + image_name + '_copy.tif'
        mask_copy_path = mask_copy_filepath + '/' + mask_name + '_copy.tif'
        
        img = cv2.imread(image_path) 
        mask = cv2.imread(mask_path)
        
        new_img = img.copy()
        new_mask = mask.copy()
        
        imsave(image_copy_path, new_img)
        imsave(mask_copy_path, new_mask)
        
        image_copy.append(image_copy_path)
        mask_copy.append(mask_copy_path)
        
    return image_copy, mask_copy
        
    
# Create a pandas dataframe for dataset (images + masks) 
def train_dataframe(image_list, mask_list):
    df_ = pd.DataFrame(data={"filename": image_list, "mask": mask_list})
    df = df_.sample(frac=1).reset_index(drop=True)
    
    return df

    
# Normalising image pixel values to range 0-1 and convert masks pixels to 1 or 0 only (binarize)
def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

# Image generator for train dataset
def train_generator(data_frame, batch_size, train_path, aug_dict,
        save_img_dir, save_mask_dir,
        image_save_prefix, mask_save_prefix,
        save_format,
        target_size,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        seed=1):
    # return: generator type object
    '''
    Generate image and mask at the same time using the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same 
    ''' 
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_img_dir,
        save_prefix  = image_save_prefix,
        save_format = save_format, 
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_mask_dir,
        save_prefix  = mask_save_prefix,
        save_format = save_format, 
        seed = seed)

    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)
