import os
import pandas as pd
import rasterio
import numpy as np

#First: test loading rasters on one row.
shifted = pd.read_csv('shifted_gt3r_seg_box.csv')

def check(p1, p2, base_array):
    """
    Uses the line defined by p1 and p2 to check array of 
    input indices against interpolated value

    Returns boolean array, with True inside and False outside of shape
    """
    idxs = np.indices(base_array.shape) # Create 3D array of indices

    p1 = p1.astype(float)
    p2 = p2.astype(float)

    # Calculate max column idx for each row idx based on interpolated line between two points
    max_col_idx = (idxs[0] - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1]) +  p1[1]    
    sign = np.sign(p2[0] - p1[0])
    return idxs[1] * sign <= max_col_idx * sign

#create polygon and reference x y for each one value 

def create_polygon(shape, vertices):
    """
    Creates np.array with dimensions defined by shape
    Fills polygon defined by vertices with ones, all other values zero"""
    base_array = np.zeros(shape, dtype=float)  # Initialize your array of zeros
    fill = np.ones(base_array.shape) * True  # Initialize boolean array defining shape fill
    # Create check array for each edge segment, combine into fill array
    for k in range(vertices.shape[0]):
        fill = np.all([fill, check(vertices[k-1], vertices[k], base_array)], axis=0)
    # Set all values inside polygon to one
    base_array[fill] = 1
    return base_array

#Inputs: an image and a coordinate
#Returns: a boolean for whether the coordinate is in the geotiff
def in_bbox(image, coord):
    bounds = image.bounds
    if  (coord[0] > bounds.left 
        and coord[0] < bounds.right 
        and coord[1] > bounds.bottom 
        and coord[1] < bounds.top):
        return True
    else:
        return False

def sample_images(images, pairs):
    l = np.array([0,0,0])
    if pairs == []:
        return l
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)

    for image in images:
        if not in_bbox(image, (min_x, min_y)) and not in_bbox(image, (max_x, max_y)):
            pass
        else:
            values = np.array(list(rasterio.sample.sample_gen(image, pairs)))
            red = sum(((value==[255,0,0]).all() for value in values))
            blue = sum(((value==[0,0,255]).all() for value in values))
            yellow = sum((value==[255, 255,0]).all() for value in values)
            if red != 0 or blue != 0 or yellow != 0:
                l += [red, blue, yellow]
    return l

images = []
for image in os.listdir('images'):
        images.append(rasterio.open(f'images/{image}'))

def sample_row(row, images):
    x1, y1, x2, y2, x3, y3, x4, y4 = row.x1, row.y1, row.x2, row.y2, row.x3, row.y3, row.x4, row.y4
    x_min = int(min(x1, x2, x3, x4))
    y_min = int(min(y1, y2, y3, y4))
    x1,x2,x3,x4 = x =  (np.array([x1,x2,x3,x4]) - x_min).astype(int)
    y1,y2,y3,y4 = y = (np.array([y1,y2,y3,y4]) - y_min).astype(int)
    #print(f'{x1},{y1}\n{x2},{y2}\n{x3},{y3}\n{x4},{y4}')
    arr = create_polygon([max(x) + 2,max(y) + 2], np.array([[x1, y1], [x4, y4], [x3, y3], [x2, y2]]))
    # print(f'{x} {y}: {min_x} {min_y} {max_x} {max_y}')
    pairs = list(zip(np.where(arr == 1)[0] + x_min, np.where(arr == 1)[1]+y_min))
    np.savetxt(f'pairs.txt', np.array(pairs))
    return sample_images(images, pairs)

