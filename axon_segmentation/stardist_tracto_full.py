import numpy as np
import dask.array as da
from stardist.models import StarDist2D
from tqdm import tqdm
import csbdeep
import skimage
import pickle
import scipy
import tifffile


# path to load image stack from
#TODO: handle loading non-zarr volumes
image_path = ""

#path to load fascicle masks from 
#TODO: handle loading non-tiff volumes
mask_path = ""

#name of stardist model to use
model_name = ""

#path to directory where stardist model is stored
model_dir = ""

#path to zarr file to save axon predictions
axon_path = ""

#path to pickle file to store axon centroids, areas, etc.
prop_path = ""

#path to pickle file to store kd trees of axon centroids for each slice
tree_path = ""


# 1. Load image volume and labels
volume = da.from_zarr(image_path)
volume = volume.rechunk((4,-1,-1))
labels =   tifffile.imread(mask_path)  

masked_cropped_vol = volume * labels[0:99]
    
# 2. Load model and generate predictions
model = StarDist2D(None, name=model_name, basedir=model_dir)
predictions = np.zeros(masked_cropped_vol.shape, dtype=int)
print('Generating axon predictions...')
for i, img in enumerate(tqdm(masked_cropped_vol)):
    img = masked_cropped_vol[i].compute()
    #img = csbdeep.utils.normalize(img, 1, 99.8)
    pred_labels, details = model.predict_instances(img)
    print(details)
    predictions[i,:,:] = pred_labels

predictions = da.from_array(predictions)
# 3. Save predictions
print('Saving axon predictions...')
try:
    da.to_zarr(predictions.rechunk((4,-1,-1)), axon_path)
except:
    #TODO add error handling
    print("Failed saving axon predicitons...")

#Generate regionprops and find centroids
#TODO: reject regions with size>2SD above mean size or irregular shape (maybe just save these properties for later processing)
pred_regionprops = [skimage.measure.regionprops(region.compute()) for region in predictions]
pred_centroids = []
print("Finding centroids...")
for pred in tqdm(pred_regionprops):
    pred_centroids.append([[np.array(region['centroid']), np.array(region['area']), np.array(region['perimeter'])] for region in pred])
print("Saving centroids...")
try:
    with open(prop_path, 'wb') as file:
        pickle.dump(pred_centroids, file)
except:
    #TODO add error handling
    print('Failed saving centroids!')

trees = [scipy.spatial.KDTree(slice_centroids) for slice_centroids in pred_centroids[0,:]]
print("Saving trees...")
try:
    with open(tree_path, 'wb') as file:
        pickle.dump(trees, file)
except:
    #TODO add error handling
    print('Failed saving trees!')