from skimage.filters import sobel
import numpy as np
import skimage.io as skio
from scipy import ndimage as ndi
from skimage import segmentation, morphology
import skimage.measure as skm

def count_dapi(filepath):
    dapi_img = skio.imread(filepath, plugin='tifffile')
    imggray = dapi_img[:, :, 2]  # selects greenchannel
    #Gets elevation map
    elevation_map = sobel(imggray)
    # Array indexing, selects only elements <10 and set to value 1
    markers = np.zeros_like(imggray)
    markers[imggray < 10] = 1
    markers[imggray > 50] = 2
    #Segment from elevation map
    segmentation_img = segmentation.watershed(elevation_map, mask=markers)
    #Label area of interests
    label_cleaned = morphology.remove_small_objects(segmentation_img, 10)
    label_cleaned = ndi.binary_fill_holes(label_cleaned)
    labeled_dapi, _ = ndi.label(label_cleaned)
    cell_label = skm.regionprops(labeled_dapi)
    cell_count = len(cell_label)

    return cell_count

def est_cellcount(imagecount):
    cellpersquarecm = imagecount /0.010593
    return int(cellpersquarecm)
