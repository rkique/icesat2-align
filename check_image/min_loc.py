#This script matches jpgs from the atl_jpg and chirop_jpg folders.
#The width of the chiroptera lidar is about 490 meters.

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle

def min_loc(chirop_path, atl_path):
    #TODO: Fix file paths
    img = cv.imread(chirop_path, cv.IMREAD_GRAYSCALE)
    cw, ch = img.shape[::-1]
    #print(f'img shape: ({cw}, {ch})')
    template = cv.imread(atl_path, cv.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    img2 = img.copy()

    meth = 'cv.TM_SQDIFF_NORMED'

    img = img2.copy()
    method = eval(meth)

    #Template Matching
    res = cv.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    (x_min, y_min) = min_loc
    img[y_min:y_min+h, x_min:x_min+w] = template
    fig = plt.figure()
    plt.subplot(121)
    plt.subplot(121).add_patch(Rectangle(min_loc, 5, 5, fc ='none',  ec ='r', lw = 10) ) 
    plt.imshow(res,cmap = 'gray')
    plt.title('Correspondence Map'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Matched Images'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    #fig.savefig(f'match_{b1}_{b2}.png')

    print(f'minima found: {min_loc} with value of {min_val:.2f}')
    # im3 = Image.fromarray(img)
    # im3 = im3.convert("L")
    #im3.save(f'matched_{b1}_{b2}.jpg')
    return min_loc
