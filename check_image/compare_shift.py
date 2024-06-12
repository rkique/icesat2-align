import pandas as pd
import numpy as np
import math
import geopandas as gpd
from min_loc import min_loc

#Because the image matcher matches ATL and chiroptera images from the top left,
#it is necessary to geolocate the original top left corner of the Chiroptera.
#To do this, we unrotate the original Chiroptera LiDAR (which is at an angle), gets (xmin, ymax), and re-rotate it.

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def get_rotated_tl(df, ANGLE):
    ANGLE = np.deg2rad(ANGLE)
    center = ((min(df['x']) + max(df['x'])) / 2, (min(df['y']) + max(df['y'])) / 2)

    for index, row in df.iterrows():
        (x,y) = rotate(center, (row['x'], row['y']), ANGLE)
        row['x'], row['y'] = (x,y)

    (rx, ry) = (min(df['x']), max(df['y']))
    (urx, ury) = rotate(center, (rx, ry), -ANGLE)
    return (urx, ury)

def raster_bounds(path):
    gdf = gpd.read_file(path)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    return (xmin, ymin, xmax, ymax)

def get_offset(atl_df, chirop_df, atl_path, min_loc_val, ANGLE):
    x_min = min(chirop_df['x'])
    x_max = max(chirop_df['x'])
    y_min = min(chirop_df['y'])
    y_max = max(chirop_df['y'])

    ymin = raster_bounds(atl_path)[1]

    ATL_max_y = ymin + 4000
    ATL_min_y = ATL_max_y - 2000
    atl_df = atl_df[(atl_df['y'] >= ATL_min_y) & (atl_df['y'] <= ATL_max_y)]
    TL_POINT = get_rotated_tl(chirop_df, ANGLE)

    #revert ATL to Chirop TL
    x_max = max(atl_df['x'])
    y_max = max(atl_df['y'])

    return (TL_POINT[0] - x_max, TL_POINT[1] - y_max)


ANGLE = 17.88

def main():
    for i in range(0,8):
        b1 = i * 4
        b2 = b1 + 4
        atl_df = pd.read_csv('unshifted_gt3r_full.csv')
        atl_path = f'../make_image/atl 1C/atl json/atl_{b1}_{b2}.json'
        chirop_df = pd.read_pickle(f'../data/1C/10 1C/{b1}m_{b2}m.pkl')

        offset = get_offset(atl_df, chirop_df, atl_path, min_loc, ANGLE)
        atl_df['x'] = atl_df['x'] + offset[0]
        atl_df['y'] = atl_df['y'] + offset[1]

        atl_df.to_csv(f'debug_shifted_atl_{b1}_{b2}.csv', index=False)

        #At this point, the ATL has been matched to the top-left corner.
        #We now want to shift with the best fit (in original coordinates).
        
        chirop_path = f'../make_image/output_chirop/chirop_{b1}_{b2}.jpg'
        atl_path = f'../make_image/output_atl/atl_{b1}_{b2}.jpg'

        (best_x, best_y) = min_loc(chirop_path, atl_path)
        best_fit = (best_x, -best_y)
        best_fit = rotate((0,0), best_fit, -np.deg2rad(ANGLE))

        atl_df['x'] = atl_df['x'] + best_fit[0]
        atl_df['y'] = atl_df['y'] + best_fit[1]
        print(f'shift: {np.round(np.array(best_fit) + np.array(offset),2)}')
        atl_df.to_csv(f'debug_shifted_atl_{b1}_{b2}.csv', index=False)

main()