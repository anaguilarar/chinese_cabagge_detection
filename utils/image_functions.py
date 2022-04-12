from PIL import Image, ImageOps, ImageFilter

from pathlib import Path
from itertools import product
from math import cos, sin, radians
import os
import numpy as np
import random
import math

import cv2



def start_points(size, split_size, overlap=0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(pt)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def split_image(image, nrows=None, ncols=None, overlap=0.0):

    img_height, img_width, channels = image.shape

    if nrows is None and ncols is None:
        nrows = 2
        ncols = 2

    width = math.ceil(img_width / ncols)
    height = math.ceil(img_height / nrows)

    row_off_list = start_points(img_height, height, overlap)
    col_off_list = start_points(img_width, width, overlap)
    offsets = product(col_off_list, row_off_list)

    imgtiles = []
    combs = []
    for col_off, row_off in offsets:
        imgtiles.append(image[row_off:(row_off + height), col_off:(col_off + width)])
        combs.append('{}_{}'.format(col_off,row_off))

    return imgtiles, combs


def from_array_2_jpg(arraydata,
                     ouputpath=None,
                     export_as_jpg=True,
                     size=None,
                     verbose=True):
    if ouputpath is None:
        ouputpath = "image.jpg"
        directory = ""
    else:
        directory = os.path.dirname(ouputpath)

    if arraydata.shape[0] == 3:
        arraydata = np.moveaxis(arraydata, 0, -1)

    image = Image.fromarray(arraydata.astype(np.uint8), 'RGB')
    if size is not None:
        image = image.resize(size)

    if export_as_jpg:
        Path(directory).mkdir(parents=True, exist_ok=True)

        if not ouputpath.endswith(".jpg"):
            ouputpath = ouputpath + ".jpg"

        image.save(ouputpath)

        if verbose:
            print("Image saved: {}".format(ouputpath))

    return image


def change_images_contrast(image,
                           alpha=1.0,
                           beta=0.0,
                           neg_brightness=False):
    """

    :param neg_brightness:
    :param nsamples:
    :param alpha: contrast contol value [1.0-3.0]
    :param beta: Brightness control [0-100]

    :return: list images transformed
    """
    if type(alpha) != list:
        alpha_values = [alpha]
    else:
        alpha_values = alpha

    if type(beta) != list:
        beta_values = [beta]
    else:
        beta_values = beta

    if neg_brightness:
        betshadow = []
        for i in beta_values:
            betshadow.append(i)
            betshadow.append(-1 * i)
        beta_values = betshadow

    ims_contrasted = []
    comb = []
    for alpha in alpha_values:
        for beta in beta_values:
            ims_contrasted.append(
                cv2.convertScaleAbs(image, alpha=alpha, beta=beta))

            comb.append('{}_{}'.format(alpha, beta))

    return ims_contrasted, comb


#### Histogram Equalization and Adaptive Histogram Equalization
## taken from https://github.com/albumentations-team/albumentations/blob/master/albumentations/augmentations/functional.py
def clahe_img(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    params:
     clip_limit: This is the threshold for contrast limiting
     tile_grid_size: Divides the input image into M x N tiles and then applies histogram equalization to each local tile
    """
    if type(clip_limit) == list:
        # pick angles at random
        clip_limit = random.choice(clip_limit)

    if img.dtype != np.uint8:
        raise TypeError("clahe supports only uint8 inputs")

    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2 or img.shape[2] == 1:
        img = clahe_mat.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe_mat.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img, [str(clip_limit)]

### hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
def shift_hsv(img, hue_shift, sat_shift, val_shift):

    if type(sat_shift) == list:
        # pick angles at random
        sat_shift = random.choice(sat_shift)

    if type(hue_shift) == list:
        # pick angles at random
        hue_shift = random.choice(hue_shift)

    if type(val_shift) == list:
        # pick angles at random
        val_shift = random.choice(val_shift)

    dtype = img.dtype
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(img)

    if hue_shift != 0:
        lut_hue = np.arange(0, 256, dtype=np.int16)
        lut_hue = np.mod(lut_hue + hue_shift, 180).astype(dtype)
        hue = cv2.LUT(hue, lut_hue)

    if sat_shift != 0:
        lut_sat = np.arange(0, 256, dtype=np.int16)
        lut_sat = np.clip(lut_sat + sat_shift, 0, 255).astype(dtype)
        sat = cv2.LUT(sat, lut_sat)

    if val_shift != 0:
        lut_val = np.arange(0, 256, dtype=np.int16)
        lut_val = np.clip(lut_val + val_shift, 0, 255).astype(dtype)
        val = cv2.LUT(val, lut_val)

    img = cv2.merge((hue, sat, val)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img, ['{}_{}_{}'.format(hue_shift, sat_shift, val_shift)]
## rotate image


def rotate_npimage(image, angle=[0]):

    if type(angle) == list:
        # pick angles at random
        angle = random.choice(angle)

    pil_image = Image.fromarray(image)

    img = pil_image.rotate(angle)

    return np.array(img), [str(angle)]


def rotate_xyxoords(x, y, anglerad, imgsize, xypercentage=True):
    center_x = imgsize[1] / 2
    center_y = imgsize[0] / 2

    xp = ((x - center_x) * cos(anglerad) - (y - center_y) * sin(anglerad) + center_x)
    yp = ((x - center_x) * sin(anglerad) + (y - center_y) * cos(anglerad) + center_y)

    if imgsize[0] != 0:
        if xp > imgsize[1]:
            xp = imgsize[1]
        if yp > imgsize[0]:
            yp = imgsize[0]

    if xypercentage:
        xp, yp = xp / imgsize[1], yp / imgsize[0]

    return xp, yp


def rotate_yolobb(yolo_bb, imgsize, angle):
    angclock = -1 * angle

    xc = float(yolo_bb[1]) * imgsize[1]
    yc = float(yolo_bb[2]) * imgsize[0]
    xr, yr = rotate_xyxoords(xc, yc, radians(angclock), imgsize)
    w_orig = yolo_bb[3]
    h_orig = yolo_bb[4]
    wr = np.abs(sin(radians(angclock))) * h_orig + np.abs(cos(radians(angclock)) * w_orig)
    hr = np.abs(cos(radians(angclock))) * h_orig + np.abs(sin(radians(angclock)) * w_orig)

    # l, r, t, b = from_yolo_toxy(origimgbb, (imgorig.shape[1],imgorig.shape[0]))
    # coords1 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords2 = rotate_xyxoords(r,b,radians(angclock),rotatedimg.shape)
    # coords3 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords4 = rotate_xyxoords(l,t,radians(angclock),rotatedimg.shape)
    # w = math.sqrt(math.pow((coords1[0] - coords2[0]),2)+math.pow((coords1[1] - coords2[1]),2))
    # h = math.sqrt(math.pow((coords3[0] - coords4[0]),2)+math.pow((coords3[1] - coords4[1]),2))
    return [yolo_bb[0], xr, yr, wr, hr]


### expand

def resize_npimage(image, newsize=(618, 618)):
    if len(newsize) == 3:
        newsize = [newsize[0],newsize[1]]

    pil_image = Image.fromarray(image)

    img = pil_image.resize(newsize, Image.ANTIALIAS)

    return np.array(img)


def expand_npimage(image, ratio=25, keep_size=True):
    if type(ratio) == list:
        # pick angles at random
        ratio = random.choice(ratio)

    pil_image = Image.fromarray(image)
    width = int(pil_image.size[0] * ratio / 100)
    height = int(pil_image.size[1] * ratio / 100)
    st = ImageOps.expand(pil_image, border=(width, height), fill='white')

    if keep_size:
        st = resize_npimage(np.array(st), image.shape)

    return np.array(st), [str(ratio)]


## blur image


def blur_image(image, radius=[0]):

    if type(radius) == list:
        # pick angles at random
        radius = random.choice(radius)

    pil_image = Image.fromarray(image)
    img = pil_image.filter(ImageFilter.GaussianBlur(radius))

    return np.array(img), [str(radius)]



def cartimg_topolar_transform(nparray, anglestep = 5, max_angle = 360, expand_ratio = 40, nathreshhold = 5):

    xsize = nparray.shape[1]
    ysize = nparray.shape[0]
    
    if expand_ratio is None:
        mayoraxisref = [xsize,ysize] if xsize > ysize else [ysize,xsize]
        expand_ratio = (mayoraxisref[0]/mayoraxisref[1] - 1)*100

    newwidth = int(xsize * expand_ratio / 100)
    newheight = int(ysize * expand_ratio / 100)

    # exand the image for not having problem whn one axis is bigger than other
    pil_imgeexpand = ImageOps.expand(Image.fromarray(nparray), 
                                     border=(newwidth, newheight), fill=np.nan)

    
    listacrossvalues = []
    distances = []
    # the image will be rotated, then the vertical values were be extracted with each new angle
    for angle in range(0, max_angle, anglestep):
        
        imgrotated = pil_imgeexpand.copy().rotate(angle)
        imgarray = np.array(imgrotated)
        cpointy = int(imgarray.shape[0]/2)
        cpointx = int(imgarray.shape[1]/2)

        valuesacrossy = []
        
        i=(cpointy+0)
        coordsacrossy = []
        # it is important to have non values as nan, if there are to many non values in a row it will stop
        countna = 0 
        while (countna<nathreshhold) and (i<(imgarray.shape[0]-1)):
            
            if np.isnan(imgarray[i,cpointx]):
                countna+=1
            else:
                coordsacrossy.append(i- cpointy)
                valuesacrossy.append(imgarray[i,cpointx])
                countna = 0
            i+=1

        distances.append(coordsacrossy)
        listacrossvalues.append(valuesacrossy)
    
    maxval = 0
    nrowid =0 
    for i in range(len(distances)):
        if maxval < len(distances[i]):
            maxval = len(distances[i])
            
            nrowid = distances[i][len(distances[i])-1]
    
    for i in range(len(listacrossvalues)):
        listacrossvalues[i] = listacrossvalues[i] + [np.nan for j in range((nrowid+1) - len(listacrossvalues[i]))]
    

    return [distances, np.array(listacrossvalues)]