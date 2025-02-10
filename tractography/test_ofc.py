import optical_flow_centroid_class
import tractography_utils
import numpy as np
import matplotlib.pyplot as plt
import pickle

#Example script showing how to se the OpticalFlowCentroid class to generate a tractogram using
#axon tracking guided by optical flow


#pickle files will be saved into a folder in the directory where you run this script
#with structure ./pickle files/{sample_name}
sample_name = "ofc_meb"

#this is the path to the folder or zarr file containing your images
image_folder_path = r"C:\Users\ixm178\Desktop\code\meb_cropped.zarr"

#this is the path to the mask image for generating seed points
mask_image_path = r"C:\Users\ixm178\Desktop\code\meb_cropped.png"

mask_image = (plt.imread(mask_image_path)* 255).astype('uint8')
mask_image = mask_image.astype('uint8')
mask_image[mask_image > 0] = 255

#update the metadata dictionary with the values for the image you are loading
#image dimensions, pixel size, slice thickness, image type, etc.
metadata = {
	'num_images_to_read': 99,
	'y_size_pixels': 990,
	'image_type': 'zarr',
	'image_slice_thickness': 12,
	'pixel_size_xy': 0.9, 
}

affine = np.eye(4)
affine[0, 0] = metadata['pixel_size_xy']
affine[1, 1] = metadata['pixel_size_xy']
affine[2, 2] = metadata['image_slice_thickness']

#default window size = 50
window_size = 50

#default max level = 2
max_level = 2

#default seeds per pixel = 0.01
seeds_per_pixel = 0.0025

#default gaussian sigma = 2
gaussian_sigma = 2

#set to slice corresponding to mask image
start_slice_index = 0

#flag to track in forward direction
forward = True

#flag to track in the reverse direction
backward = True

#flag to manually set seeds and variable to manually pass seedpoints
seeded = True
with open(r"C:\Users\ixm178\Desktop\code\crop_pred_centroids.pkl", 'rb') as file:
    centroids = pickle.load(file)
seeds = centroids[0]

with open(r"C:\Users\ixm178\Desktop\code\crop_trees.pkl", 'rb') as file:
	trees = pickle.load(file)

test_tractogram = optical_flow_centroid_class.OpticFlowCentroidClass(image_folder_path, mask_image, affine, metadata, window_size, max_level, seeds_per_pixel, gaussian_sigma, start_slice_index, forward, backward, seeded, seeds, trees)

streamlines, colors = test_tractogram.get_streamlines_and_colors()

tractography_utils.export_streamlines_optical_flow(window_size, max_level, seeds_per_pixel, gaussian_sigma, sample_name, streamlines, colors)
