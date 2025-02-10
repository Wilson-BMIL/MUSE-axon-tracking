import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
import glob
from tqdm import tqdm

from tractography_utils import get_random_seedpoints, tractogram, get_seedpoints_from_tractogram, get_colors
import pickle
import zarr
#import dask.array as da

class OpticFlowCentroidClass():

	def __init__(self, image_folder_path, mask_image, affine, metadata, window_size, max_level, seeds_per_pixel, gaussian_sigma, start_slice_index=0, forward=True, backward=False, seeded=False, seeds=None, trees=None):
		self.image_folder_path = image_folder_path
		self.mask_image = mask_image
		self.affine = affine
		self.metadata = metadata
		self.window_size = window_size
		self.max_level = max_level
		self.seeds_per_pixel = seeds_per_pixel
		self.gaussian_sigma = gaussian_sigma
		self.start_slice_index = start_slice_index

		self.streamline_phys_coords = None
		self.color = None
		self.forward = forward
		self.backward = backward

		self.lk_params = {
						"winSize": (self.window_size, self.window_size), 
						"maxLevel": self.max_level, 
						"criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
						}

		if seeded == True:
			self.seedpoint_coords = np.array([[s[1],s[0]] for s in seeds])
			self.color = get_colors(self.mask_image, seeds)
		else:
			self.seedpoint_coords, self.color = get_random_seedpoints(self.mask_image, self.seeds_per_pixel)

		self.trees = trees

		self.generate_tractogram()

	def generate_tractogram(self):
		forward_streamlines = []
		backward_streamlines = []
  
		compute_forward = self.forward and (self.start_slice_index != self.metadata['num_images_to_read'] - 1)
		compute_backward = self.backward and (self.start_slice_index != 0)

		if compute_forward:
			forward_streamlines = self.generate_streamlines(self.start_slice_index, self.metadata['num_images_to_read'] - 1, 1)
		print(len(forward_streamlines[0]))
		if compute_backward:
			backward_streamlines = self.generate_streamlines(self.start_slice_index, 0, -1)

		for k in range(len(backward_streamlines)):
			backward_streamlines[k].reverse()

		if compute_forward and compute_backward:   
			for k in range(len(backward_streamlines)):
				backward_streamlines[k].extend(forward_streamlines[k][1:])
			
			self.streamlines_phys_coords = tractogram(backward_streamlines, self.affine, self.metadata['y_size_pixels'])

		elif not compute_forward and compute_backward:
			self.streamlines_phys_coords = tractogram(backward_streamlines, self.affine, self.metadata['y_size_pixels'])

		elif compute_forward and not compute_backward:
			self.streamlines_phys_coords = tractogram(forward_streamlines, self.affine, self.metadata['y_size_pixels'])

	def generate_streamlines(self, start_slice_index, stop_slice_index, direction):
		# Streamlines variable (image space coordinates)
		streamlines = [[] for _ in range(self.seedpoint_coords.shape[0])]
			
		# Streamline coordinates on the first slice are simply the seed point coordinates
		for k in range(self.seedpoint_coords.shape[0]):
			streamlines[k].append((self.seedpoint_coords[k,0], self.seedpoint_coords[k,1], start_slice_index))   

		# Tracking status (1 to continue tracking, 0 to stop)
		tracking_status = np.ones((len(streamlines)))
   
		# Get filenames
		if self.metadata['image_type'] == '.png':
			image_filelist = glob.glob(self.image_folder_path + "\\*.png")
		else:
			dataset = zarr.open(self.image_folder_path)
			muse_dataset = dataset#['muse/stitched']
			#muse_dataset = dataset['data/0']
			#muse_dataset = dataset['images']
			#muse_dataset = da.from_zarr(self.image_folder_path, component="data/0")
  
		# Angle threshold (75 degrees) in pixels
		angle_threshold_pixels = int(np.tan(np.deg2rad(75)) * self.metadata['image_slice_thickness'] / self.metadata['pixel_size_xy'])

		# Loop through each frame, update list of points to track, streamlines
		for i in tqdm(np.arange(start_slice_index, stop_slice_index, direction)):
			next_tree = self.trees[i+1]
			# Blur the images a bit to have a better gradient, else will be noisy
			if i == self.start_slice_index:
				try:
					if self.metadata['image_type'] == '.png':
						image = (plt.imread(image_filelist[self.start_slice_index])* 255).astype('uint8')
					else:
						image = np.squeeze(np.array(muse_dataset[self.start_slice_index, :, :]))#*255
						image = image.astype('uint8')
				except ValueError:
					print('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
					return None
				current_image = gaussian_filter(image, sigma = self.gaussian_sigma)
			else:
				current_image = next_image
			
			try:
				if self.metadata['image_type'] == '.png':
					image = (plt.imread(image_filelist[i+direction])* 255).astype('uint8')
				else:
					image = np.squeeze(np.array(muse_dataset[i+direction, :, :]))
					image = image.astype('uint8')					
			except ValueError:
				print('Could not read image files, possible error in metadata file for image size fields or images not present in specified path')
				return None
			
			next_image = gaussian_filter(image, sigma = self.gaussian_sigma)

			# OpenCV needs points in the form (n, 1, 2), just the XY coordinates
			points_to_track = np.float32([tr[-1] for tr in streamlines]).reshape(-1, 1, 3)[:,:,:2]
			
			# Calculate optical flow
			new_location_of_tracked_points, status_fw, err_fw = cv2.calcOpticalFlowPyrLK(current_image, next_image, points_to_track, None, **self.lk_params)
			
			new_tracks = []
			for tr, (x, y), status, index in zip(streamlines, new_location_of_tracked_points.reshape(-1, 2), status_fw, np.arange(len(streamlines))):
				if not status:
					tracking_status[index] = 0
					new_tracks.append(tr)
					continue
 
				if tracking_status[index]:
					diff_in_x_coordinate = abs(tr[-1][0] - x)
					diff_in_y_coordinate = abs(tr[-1][1] - y)    

					# If there is a big jump in one direction compared to other, then that would probably be at a stitching slice or cutting artifact.
					# Keep the same coordinates as previous
					if abs(diff_in_x_coordinate - diff_in_y_coordinate) > angle_threshold_pixels:
						d, m = next_tree.query((tr[-1][1], tr[-1][0]), distance_upper_bound=angle_threshold_pixels)
						if m == next_tree.n:
							tr.append((tr[-1][0], tr[-1][1], i + direction))
						else:
							tr.append((next_tree.data[m][1], next_tree.data[m][0], i + direction))

					# Check if the new points are within a 75 degree angle
					if (diff_in_x_coordinate < angle_threshold_pixels) and (diff_in_y_coordinate < angle_threshold_pixels):
						d, m = next_tree.query((y, x))
						tr.append((next_tree.data[m][1], next_tree.data[m][0], i + direction))
					else:
						tracking_status[index] = 0
					new_tracks.append(tr)
				else:
					new_tracks.append(tr)
     
			streamlines = new_tracks
		
		return streamlines

	def get_streamlines_and_colors(self):
		return self.streamlines_phys_coords, self.color
