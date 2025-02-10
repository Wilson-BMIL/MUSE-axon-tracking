import cv2
import numpy as np
from skimage import measure
import datetime
import distinctipy
import sys
from tqdm import tqdm
import random
import pickle
import os
import scipy

def get_random_seedpoints(mask_image, seeds_per_pixel):

	random.seed(10)
	
	# Segment the fascicle image based on a simple threshold
	ret, thresh = cv2.threshold(mask_image, 20, 255, cv2.THRESH_BINARY)

	# Count total number of seeds 
	total_seed_count = int(np.floor(seeds_per_pixel * np.sum(thresh != 0)))

	# Variables to store seed point coordinates and respective fascicle number
	seed_point_coordinates = np.zeros((total_seed_count, 2), dtype = 'float32')
	fascicle_tracker_seed_points = np.empty((total_seed_count))

	# Assign unique label for each fascicle, enumerate those values
	labels = measure.label(thresh, background = 0)
	fascicles_unique_values = np.unique(labels)

	# Remove background value (0) from the unique values list
	fascicles_unique_values = fascicles_unique_values[fascicles_unique_values != 0]
	
	# Get distinct colors for each fascicle, make a color array that stores color for each seed
	color = np.zeros((total_seed_count, 3))
	distinct_colors = distinctipy.get_colors(len(fascicles_unique_values), [(0,0,0), (1, 1, 1)])

	# Counter to keep track of how many seeds have been found
	seeds_already_assigned = 0
	
	# Randomly sample seed points in the fascicles
	for i in np.arange(len(fascicles_unique_values)):
		
		# Make a binary image containing just the current fascicle
		current_fascicle_mask = (labels == fascicles_unique_values[i])

		num_pixels_in_current_fascicle = np.sum(current_fascicle_mask)
		num_seeds_in_current_fascicle = int(np.floor(seeds_per_pixel * num_pixels_in_current_fascicle))
		
		# Get x and y indices of all points in the fascicle
		horizontal_indices = np.where(np.any(current_fascicle_mask, axis=0))[0]
		vertical_indices = np.where(np.any(current_fascicle_mask, axis=1))[0]
		
		# Get limits (bounding box)
		x1, x2 = horizontal_indices[[0, -1]]
		y1, y2 = vertical_indices[[0, -1]]
		
		# Oversample within the bounding box and then remove seeds that fall outside the fascicle region  
		np.random.seed(42)
		random.seed(10)
  
		# oversampling_factor = 10
		# x_indices = np.random.uniform(low= x1, high = x2, size = num_seeds_in_current_fascicle * oversampling_factor)
		# y_indices = np.random.uniform(low= y1, high = y2, size = num_seeds_in_current_fascicle * oversampling_factor)

		x_range = np.arange(x1+0.5, x2 + 1, step=1)
		y_range = np.arange(y1+0.5, y2 + 1, step=1)
  
		xy_indices = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
		xy_indices = np.random.permutation(xy_indices)
  
		# Set a counter for how many seeds were found
		seed_counter = 0
		
		# For each randomly sampled point
		for j in np.arange(len(xy_indices)):
      
			# Once you reach the necessary number of seeds, break from this loop
			if seed_counter == num_seeds_in_current_fascicle:
				break			

			# Check if the current random point is inside the fascicle region
			if current_fascicle_mask[int(np.floor(xy_indices[j, 1])), int(np.floor(xy_indices[j,0]))]:
				seed_point_coordinates[seeds_already_assigned + seed_counter, :] = [xy_indices[j,0], xy_indices[j,1]]
				fascicle_tracker_seed_points[seeds_already_assigned + seed_counter] = fascicles_unique_values[i]
				seed_counter = seed_counter + 1

		# Assign color for this group of seeds
		color[seeds_already_assigned:seeds_already_assigned + seed_counter,:] = distinct_colors[i]
		seeds_already_assigned = seeds_already_assigned + seed_counter

		if seed_counter != num_seeds_in_current_fascicle:
			print('Error - did not find sufficient seeds inside fascicles. Try increasing oversampling_factor.')
			sys.exit(0)
	
	# Delete the last rows in the array that were not set (left at zero)
	seed_point_coordinates = seed_point_coordinates[~np.all(seed_point_coordinates == 0, axis=1)]
	color = color[~np.all(color == 0, axis=1)]
	
	return seed_point_coordinates, color

def get_random_aligned_seedpoints(mask_image, seeds_per_pixel):
	random.seed(10)
	
	# Segment the fascicle image based on a simple threshold
	ret, thresh = cv2.threshold(mask_image, 20, 255, cv2.THRESH_BINARY)

	# Count total number of seeds 
	total_seed_count = int(np.floor(seeds_per_pixel * np.sum(thresh != 0)))

	# Variables to store seed point coordinates and respective fascicle number
	seed_point_coordinates = np.zeros((total_seed_count, 2), dtype = 'float32')
	fascicle_tracker_seed_points = np.empty((total_seed_count))

	# Assign unique label for each fascicle, enumerate those values
	labels = measure.label(thresh, background = 0)
	fascicles_unique_values = np.unique(labels)

	# Remove background value (0) from the unique values list
	fascicles_unique_values = fascicles_unique_values[fascicles_unique_values != 0]
	
	# Get distinct colors for each fascicle, make a color array that stores color for each seed
	color = np.zeros((total_seed_count, 3))
	distinct_colors = distinctipy.get_colors(len(fascicles_unique_values), [(0,0,0), (1, 1, 1)])

	# Counter to keep track of how many seeds have been found
	seeds_already_assigned = 0
	
	# Randomly sample seed points in the fascicles
	for i in np.arange(len(fascicles_unique_values)):
		
		# Make a binary image containing just the current fascicle
		current_fascicle_mask = (labels == fascicles_unique_values[i])

		num_pixels_in_current_fascicle = np.sum(current_fascicle_mask)
		num_seeds_in_current_fascicle = int(np.floor(seeds_per_pixel * num_pixels_in_current_fascicle))
		
		# Get x and y indices of all points in the fascicle
		horizontal_indices = np.where(np.any(current_fascicle_mask, axis=0))[0]
		vertical_indices = np.where(np.any(current_fascicle_mask, axis=1))[0]
		
		# Get limits (bounding box)
		x1, x2 = horizontal_indices[[0, -1]]
		y1, y2 = vertical_indices[[0, -1]]
		
		# Oversample within the bounding box and then remove seeds that fall outside the fascicle region  
		np.random.seed(42)
		random.seed(10)
  
		# oversampling_factor = 10
		# x_indices = np.random.uniform(low= x1, high = x2, size = num_seeds_in_current_fascicle * oversampling_factor)
		# y_indices = np.random.uniform(low= y1, high = y2, size = num_seeds_in_current_fascicle * oversampling_factor)

		x_range = np.arange(x1+0.5, x2 + 1, step=1)
		y_range = np.arange(y1+0.5, y2 + 1, step=1)
  
		xy_indices = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)
		xy_indices = np.random.permutation(xy_indices)
  
		# Set a counter for how many seeds were found
		seed_counter = 0
		
		# For each randomly sampled point
		for j in np.arange(len(xy_indices)):
      
			# Once you reach the necessary number of seeds, break from this loop
			if seed_counter == num_seeds_in_current_fascicle:
				break			

			# Check if the current random point is inside the fascicle region
			if current_fascicle_mask[int(np.floor(xy_indices[j, 1])), int(np.floor(xy_indices[j,0]))]:
				seed_point_coordinates[seeds_already_assigned + seed_counter, :] = [np.floor(xy_indices[j,0]), np.floor(xy_indices[j,1])]
				fascicle_tracker_seed_points[seeds_already_assigned + seed_counter] = fascicles_unique_values[i]
				seed_counter = seed_counter + 1

		# Assign color for this group of seeds
		color[seeds_already_assigned:seeds_already_assigned + seed_counter,:] = distinct_colors[i]
		seeds_already_assigned = seeds_already_assigned + seed_counter

		if seed_counter != num_seeds_in_current_fascicle:
			print('Error - did not find sufficient seeds inside fascicles. Try increasing oversampling_factor.')
			sys.exit(0)
	
	# Delete the last rows in the array that were not set (left at zero)
	seed_point_coordinates = seed_point_coordinates[~np.all(seed_point_coordinates == 0, axis=1)]
	color = color[~np.all(color == 0, axis=1)]
	
	return seed_point_coordinates, color

# Get seed points from tractogram
def get_seedpoints_from_tractogram(streamlines, colors, affine, y_size_pixels, slice_index):
	
	# Get z slice spacing in physical dimensions (microns)
	z_spacing = affine[2,2]
 
	# Get the z coordinate from which streamline coordinates should be picked
	z_phys_coordinate = z_spacing * slice_index
 
	# Create a container for the seed points
	seed_point_coordinates = []
	color = []
 
	# Loop through the streamlines
	for k in range(len(streamlines)):
		current_streamline = streamlines[k]
  
		for i in range(current_streamline.shape[0]):
			if current_streamline[i,2] == z_phys_coordinate:
				seed_point_coordinates.append([current_streamline[i,0]/affine[0,0], y_size_pixels - (current_streamline[i,1]/affine[1,1])])
				color.append(colors[k,:])
	
	seed_point_coordinates = np.array(seed_point_coordinates)
	color = np.array(color)
	
	return seed_point_coordinates, color

def upsample_streamlines(streamlines, upsample):
	new_streamlines = []
	for streamline in streamlines:
		new_streamline = []
		for point in streamline:
			new_streamline.append((point[0]*upsample, point[1]*upsample, point[2]))
		new_streamlines.append(new_streamline)
	return new_streamlines

# Transform from pixel space to physical space
def tractogram(streamlines, affine, y_size_pixels):

	streamlines_phys_coords = [None] * len(streamlines)
	lin_T = affine[:3, :3].T.copy()
	offset = affine[:3, 3].copy()
	
	for streamline_index in tqdm(np.arange(len(streamlines_phys_coords))):
		
		current_streamline_points = streamlines[streamline_index]
		transformed_points = np.empty((len(current_streamline_points), 3))

		for i in np.arange(len(current_streamline_points)):
	  
			x = current_streamline_points[i][0]
			y = current_streamline_points[i][1]
			z = current_streamline_points[i][2]

			transformed_points[i, :] = np.dot([x, y_size_pixels - y, z], lin_T) + offset

		streamlines_phys_coords[streamline_index] = transformed_points
	#return streamlines
	new_streamlines = []
	for streamline in streamlines:
		new_streamlines.append(np.array(streamline))
	#return new_streamlines
	return streamlines_phys_coords

def export_streamlines_optical_flow(window_size, max_level, seeds_per_pixel, gaussian_sigma, sample_name, streamlines, colors):
	pickle_files_save_folder = 'pickle files\\' + sample_name
	if not os.path.exists(pickle_files_save_folder):
		os.makedirs(pickle_files_save_folder)
   
	streamlinesFileName = pickle_files_save_folder + '\\streamlines_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(window_size) + "_maxLev_" + str(max_level) + "_spp_" + str(seeds_per_pixel) + "_blur_" + str(gaussian_sigma) + ".pkl"
	colorsFileName = pickle_files_save_folder + '\\colors_lk_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_win_" + str(window_size) + "_maxLev_" + str(max_level) + "_spp_" + str(seeds_per_pixel) + "_blur_" + str(gaussian_sigma) + ".pkl"

	with open(streamlinesFileName, 'wb') as file:
		pickle.dump(streamlines, file)
		
	with open(colorsFileName, 'wb') as file:
		pickle.dump(colors, file)

def export_streamlines_structure_tensor(neighborhood_scale, noise_scale, seeds_per_pixel, downsample_factor, sample_name, streamlines, colors):
	 	
	pickle_files_save_folder = 'pickle files\\' + sample_name
	if not os.path.exists(pickle_files_save_folder):
		os.makedirs(pickle_files_save_folder)
   
	streamlinesFileName = pickle_files_save_folder + '\\streamlines_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhood_scale) + "_noise_scale_" + str(noise_scale) + "_spp_" + str(seeds_per_pixel) + "_ds_factor_" + str(downsample_factor) + ".pkl"
	colorsFileName = pickle_files_save_folder + '\\colors_st_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhood_scale) + "_noise_scale_" + str(noise_scale) + "_spp_" + str(seeds_per_pixel) + "_ds_factor_" + str(downsample_factor) + ".pkl"

	with open(streamlinesFileName, 'wb') as file:
		pickle.dump(streamlines, file)
		
	with open(colorsFileName, 'wb') as file:
		pickle.dump(colors, file)

def export_streamlines_dynamic_programming(neighborhood_scale, noise_scale, seeds_per_pixel, downsample_factor, sample_name, streamlines, colors):
	pickle_files_save_folder = 'pickle files\\' + sample_name
	if not os.path.exists(pickle_files_save_folder):
		os.makedirs(pickle_files_save_folder)
   
	streamlinesFileName = pickle_files_save_folder + '\\streamlines_dp_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhood_scale) + "_noise_scale_" + str(noise_scale) + "_spp_" + str(seeds_per_pixel) + "_ds_factor_" + str(downsample_factor) + ".pkl"
	colorsFileName = pickle_files_save_folder + '\\colors_dp_' + datetime.datetime.now().strftime("%H%M%S_%m%d%Y") + "_neighborscale_" + str(neighborhood_scale) + "_noise_scale_" + str(noise_scale) + "_spp_" + str(seeds_per_pixel) + "_ds_factor_" + str(downsample_factor) + ".pkl"

	with open(streamlinesFileName, 'wb') as file:
		pickle.dump(streamlines, file)
		
	with open(colorsFileName, 'wb') as file:
		pickle.dump(colors, file)

def get_colors(mask, seeds):
	labeled_img, num_labels = scipy.ndimage.label(mask)
	distinct_colors = distinctipy.get_colors(num_labels, [(0,0,0), (1, 1, 1)])
	colors = np.zeros((seeds.shape[0],3))
	print(len(distinct_colors))
	for i in range(seeds.shape[0]):
		color = labeled_img[int(seeds[i][0]), int(seeds[i][1])] - 1
		colors[i] = distinct_colors[color]
	return colors