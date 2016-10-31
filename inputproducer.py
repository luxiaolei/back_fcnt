
import os
import skimage

import numpy as np

from scipy.misc import imread, imresize
from skimage import color
from utils import gauss2d


class InputProducer:
	def __init__(self, imgs_path, gt_path, live=False):
		"""
		Class for handle input images. 
		
		Args:
			imgs_path: String, directory for specifying the path of images.
			gt_path: String, filename of groundtruth file, which contains target 
				bounding box info in each line.
			live: bool, indicator for live video # TODO, implement.
	
		"""
		self.imgs_path_list = [os.path.join(imgs_path, fn) for fn in sorted(os.listdir(imgs_path))]
		self.gts_list = self._gen_gts(gt_path)
		self.gen_img = self.get_image()

		self.roi_params = {
		'roi_size': 224, 
		'roi_scale': 3,
		'l_off': [0,0]
		}

	def get_image(self):
		"""Generator for retrieve images and groundtruth bbox.
		Yields:
			img: np.ndarray with shape [240, 320, 3]
			gt: list, specifying location of interested target.
				[tlx, tly, w, h]
			idx: int, index of current image.
		"""
		idx = -1
		for img_path, gt in zip(self.imgs_path_list, self.gts_list):
			img = imread(img_path, mode='RGB')

			assert min(img.shape[:2]) >= 224

			# Gray to color. RES??
			#if len(img.shape) < 3:
			#img = skimage.color.gray2rgb(img)
			assert len(img.shape) == 3

			idx += 1
			if idx == 0: 
				self.first_gt = gt
				self.first_img = img
			yield img, gt, idx


	def _gen_gts(self, gt_path):
		"""
		Parse location info from ground truth file.
		Each row in the ground-truth files represents the bounding box 
		of the target in that frame. (tl_x, tl_y, box-width, box-height)
		
		Args:
			gt_path: string.
		Returns:
			gts_list: a nested list.
		"""
		f = open(gt_path, 'r')
		lines = f.readlines()

		try:
			gts_list = [[int(p) for p in i[:-1].split(',')] 
			                   for i in lines]
		except Exception as e:
			gts_list = [[int(p) for p in i[:-1].split('\t')] 
			                   for i in lines]
		return gts_list

	def extract_roi(self, img, gt):
		"""Extract ROI from img with target region centered.

		Args:
			img: np.ndarray, origin image.
			gt: list, locations for top-left x, top-left y, width, height.
		Returns:
		    roi: tensor,
		    roi_pos: list of params for roi_pos, [tlx, tly, w, h].
			resize_factor: float, translational scalling factor from 
				original image space to roi space. 
				>1 for enlarger, <1 for ensmaller.
		"""
		roi_size  = self.roi_params['roi_size']
		assert max(gt[2:]) <= roi_size

		# Construct an padded img first.
		convas = np.zeros([img.shape[0]+2*roi_size, img.shape[1]+2*roi_size, 3])
		convas[roi_size:-roi_size, roi_size:-roi_size] = img

		# Compute target center location in convas
		tlx_convas, tly_convas = gt[0]+roi_size, gt[1]+roi_size
		cx = tlx_convas + int(0.5 * gt[2])
		cy = tly_convas + int(0.5 * gt[3])

		# Crop an roi_size region centered at cx, cy
		scale_sz = max(gt[2:]) * self.roi_params['roi_scale']
		half = scale_sz // 2
		roi = convas[cy-half:cy+half, cx-half:cx+half, :]

		# compute new target pos in roi window
		new_cx, new_cy = [int(i*0.5) for i in roi.shape[:-1]]
		new_x = new_cx - gt[2] // 2
		new_y = new_cx - gt[3] // 2
	    
		roi_resized = imresize(roi, (roi_size, roi_size))
		resize_factor = roi_size / roi.shape[0]
		return roi_resized, [new_x, new_y, gt[2], gt[3]], resize_factor


	def gen_mask(self, fea_sz=(224,224)):
		"""
		Generates a 2D guassian masked convas with shape same as 
		fea_sz. This method should only called on the first frame.

		Args:
			fea_sz: 2 elements tuple, to be identical with the 
				Output of sel-CNN net.
		Returns:
			convas: np.ndarray, fea_sz shape with 1 channel. The central 
				region is an 2D gaussian.
		"""
		im_sz = self.first_img.shape
		x, y, w, h = self.first_gt
		convas = np.zeros(im_sz[:2])

		# Generates 2D gaussian mask
		scale = min([w,h]) / 3 # To be consistence with the paper
		mask = gauss2d([h, w], sigma=scale)
		#print(mask.max(), 'max of mask')

		# bottom right coordinate
		x2 = x + w - 1
		y2 = y + h - 1

		# Detects wether the location has out of the img or not
		clip = min(x, y, im_sz[0]-y2, im_sz[1]-x2)
		pad = 0
		if clip <= 0:
			pad = abs(clip) + 1
			convas = np.zeros((im_sz[0] + 2*pad, im_sz[1] + 2*pad))
			x += pad
			y += pad
			x2 += pad
			y2 += pad

		# Overwrite central arear of convas with mask;
		convas[y-1:y2, x-1:x2] = mask
		if clip <= 0:
			# Remove pad
			convas = convas[pad:-pad, pad, -pad]

		if len(convas.shape) < 3:
			convas = color.gray2rgb(convas)
		assert len(convas.shape) == 3

		# Extrac ROI and resize bicubicly
		convas, _, _  = self.extract_roi(convas, self.first_gt)
		#print(convas.shape)
		convas = imresize(convas[...,0], fea_sz[:2], interp='bicubic')

		# Swap back, and normalize
		convas = convas / convas.max()

		return convas

	def gen_batches(self, img, gt, n_samples=5000, batch_sz=10, pos_ratio=0.7, scale_factors=None):
		""" 
		Returns batched trainning examples, with which's target location and 
		width/height ratio are randomly distored.
		
		Args:
			img: np.ndarry with shape (240, 320, 3)
			gt: list, [tlx, tly, w, h] location for target.
			n_samples: int, total number of samples would like to generate.
			batch_sz: int, number of examples in a batch. 
			pos_ratio: float, in range (0, 1], portion of postitive samples
				in total samples.
			scale_factors: list, specifying scale factors when extracting target
				from image in the process of generating random postive samples. 
		Returns:
			sample_batches: list of roi batches. with each roi batch has shape
				[batch_size, 224, 224, 3].
			target_batches: list of target batches. with each target batch jas shape
				[batch_size, 224, 224, 3].
		"""

		# Gen n_pos number of scaled samples 
		n_pos = int(n_samples/pos_ratio)
		if scale_factors is None: scale_factors = np.arange(0.2, 5., 0.5)
		samples = []
		targets = []
		for pos_idx in range(n_pos):
			sf_idx = pos_idx % len(scale_factors)
			self.roi_params['roi_scale'] = scale_factors[sf_idx]
			roi, _, _ = self.extract_roi(img, gt)
			gt_M = self.gen_mask((224,224)).astype(np.float32)
			samples += [roi]
			targets += [gt_M]

		# Gen negative samples with random scale factor
		gt_M_neg = np.zeros((224,224), dtype=np.float32)
		for _ in range(n_samples - n_pos):
			lb, up = scale_factors[0], scale_factors[-1]
			self.roi_params['roi_scale'] = np.random.uniform(lb, up)
			roi = self._gen_neg_samples(img, gt)
			samples += [roi]
			targets += [gt_M_neg]

		# Random shuffeling 
		rand_idx = np.random.permutation(len(samples))
		samples = np.array(samples)[rand_idx]
		targets = np.array(targets)[rand_idx]

		# Batching
		sample_batches = [samples[i:i+batch_sz] for i in range(len(samples)) if i % batch_sz==0]
		target_batches = [targets[i:i+batch_sz] for i in range(len(targets)) if i % batch_sz==0]
		return sample_batches, target_batches

	def _gen_neg_samples(self, img, gt_1):
		"""
		Private method for generating negative samples,
		i.e., randomly extract a non-target region from image.

		Args:
			img: np.ndarray. 3D array
			gt_1: list, groundtruth for the target to avoid.
		Returns:
			roi_rand: np.ndarry, same shape as img. genrated negative sample.
		"""

		delta = 30 
		img = img.copy()
		w, h = gt_1[2:]
		tl_x, tl_y = gt_1[:2]
		tr_x, tr_y = tl_x + w, tl_y 
		dl_x, dl_y = tl_x, tl_y + h
		dr_x, dr_y = tl_x + w, tl_y +h
		img[tl_y:dr_y+delta,tl_x:dr_x+delta] = img.mean()
		
		# randomly extract an arear specified by gt_1
		x = np.random.randint(0, 224-w)
		y = np.random.randint(0, 224-h)
		roi_rand,_,_ = self.extract_roi(img, [y,x, w,h])
		return roi_rand





