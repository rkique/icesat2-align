The code here comprises the ICESPLICE pipeline for the CV final project. There are three separate folders that do the bulk of the work, which are `make_image`, `match_image`, and `check_image`. They require LiDAR and altimetry data from a "data" directory, which has not been provided. The poster we presented is below.

The use of cv2.template_match() is primarily in "image_matcher.ipynb", but there are many preprocessing functions which take advantage of cv2 and other image libraries. In particular, in the make_image folder there are two critical scripts, "ATL_image.ipynb" which utilizes geopandas and shapely to rasterize ATLAS imagery, and "ATL_Chirop_image.ipynb" which utilizes pandas and ndimage to rasterize the Chiroptera LiDAR (provided in pkl format).

![Poster](visuals/poster.png)
