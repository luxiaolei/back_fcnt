"""
Main script for FCNT tracker. 
"""
#%%
# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from selcnn import SelCNN
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval

import numpy as np 
import tensorflow as tf
import matplotlib.pylab as plt

import os
import time
#%%
tf.app.flags.DEFINE_integer('iter_step_sel', 200,
                          """Number of steps for trainning"""
                          """selCNN networks.""")
tf.app.flags.DEFINE_integer('iter_step_sg', 50,
                          """Number of steps for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('num_sel', 384,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_integer('iter_max', 200,
							"""Max iter times through imgs""")

FLAGS = tf.app.flags.FLAGS

## Define varies path
DATA_ROOT = 'data/Dog1'
PRE_ROOT = os.path.join(DATA_ROOT, 'img_loc')
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'#%%



print('Reading the first image...')
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = InputProducer(IMG_PATH, GT_PATH)
img, gt, s  = next(inputProducer.gen_img)
roi_t0, _, _ = inputProducer.extract_roi(img, gt)

# Predicts the first img.
print('Classify it with a pre-trained Vgg16 model.')
t_start = time.time()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
vgg.print_prob(roi_t0, sess)
print('Forwarding the vgg net cost : %.2f s'%(time.time() - t_start))## At t=0. Perform the following:
# 1. Train selCNN network for both local and gloabl feature maps
# 2. Train G and S networks.


def visual_selected_maps(s_sel_maps):
	return (s_sel_maps.sum(axis=0).sum(axis=-1)/s_sel_maps.shape[-1])



tracker = TrackerVanilla(gt)

# Iter imgs
gt_last = gt 
print("Total time cost for initialization : %.2f s"%(time.time() - t_start))
for i in range(FLAGS.iter_max):
	i += 1
	t_inter = time.time()
	print('Step: ', i)
	# Gnerates next frame infos
	img, gt_cur, s  = next(inputProducer.gen_img)

	## Crop a rectangle ROI region centered at last target location.
	roi, _, resize_factor = inputProducer.extract_roi(img, gt_last)
	
	## Perform Target localiation predicted by GNet
	# Get heat map predicted by GNet
	feed_dict_vgg = {vgg.imgs : [roi]}

	pre_M_g = sess.run(vgg.conv5_3, feed_dict={vgg.imgs: [roi]})
	pre_M_avg = visual_selected_maps(pre_M_g)[np.newaxis,...,np.newaxis]


	# Localize target with monte carlo sampling.
	tracker.draw_particles(gt_last)
	pre_loc = tracker.predict_location(pre_M_avg, gt_last, resize_factor, i, 224)
	#print('At time {0}, the most confident value is {1}'.format(t, tracker.cur_best_conf))


	gt_last = pre_loc

	# Draw bbox on image. And print associated IoU score.
	img_bbox = img_with_bbox(img, pre_loc)
	file_name = inputProducer.imgs_path_list[i-1].split('/')[-1]
	file_name = os.path.join(PRE_ROOT, file_name)
	plt.imsave(file_name, img_bbox)