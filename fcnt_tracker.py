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

import sys
import os
import time
#%%
tf.app.flags.DEFINE_integer('iter_step_sel', 200,
                          """Number of steps for trainning"""
                          """selCNN networks.""")
tf.app.flags.DEFINE_integer('iter_step_sg', 50,
                          """Number of steps for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('sel_num', 354,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_integer('iter_max', 1349,
							"""Max iter times through imgs""")
tf.app.flags.DEFINE_integer('S_adp_steps', 20,
							"""Steps to finetue Snet adaptively""")
FLAGS = tf.app.flags.FLAGS

## Define varies path
DATA_ROOT = 'data/Dog1'
PRE_ROOT = os.path.join(DATA_ROOT, 'img_loc')
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'#%%


def init_vgg():
	print('Classify it with a pre-trained Vgg16 model.')
	t_start = time.time()
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
	vgg.print_prob(roi_t0, sess)
	print('Forwarding the vgg net cost : %.2f s'%(time.time() - t_start))
	return sess, vgg





def train_SGNets(sess, vgg, snet, gnet, inputProducer):
    loss = gnet.loss() + snet.loss()
    vars_train = vgg.variables + gnet.variables + snet.variables

    # Backprop using SGD and updates vgg variables and sgnets variables
    global_step = tf.Variable(0, trainable=False)
    lr_exp = tf.train.exponential_decay(
            1e-3, # Initial learning rate 
            global_step, 
            1000, # Decay steps 
            0.9, # Decay rate 
            name='sg_lr')
    optimizer = tf.train.GradientDescentOptimizer(lr_exp)
    train_op = optimizer.minimize(loss, var_list= vars_train, global_step=global_step)
    sess.run(tf.initialize_variables(snet.variables + gnet.variables + [global_step]))

    losses = []
	num_epoch = 10
    sample_batches, target_batches = inputProducer.gen_batches(img, gt)
    print('Start training the SGNets........ for %s epochs'%num_epoch)
    for ep in range(num_epoch):
        step = 1
        for roi, target in zip(sample_batches, target_batches):
            t = time.time()
            fd = {vgg.imgs: roi, gnet.gt_M: target, snet.gt_M: target}
            pre_M_g, pre_M_s, l, _ = sess.run([gnet.pre_M, snet.pre_M, loss, train_op], feed_dict=fd)
            losses += [l]
            step += 1
            if step % 50 == 0:
                print('Epoch: ', ep+1, 'Step: ', (ep+1)*step, 'Loss : %.2f'%l, 'Speed: %2.f s/batch'%(time.time()-t))


# Deprecated! 
def train_SGNets_dep(sess, vgg, gnet, snet, gt_M, fd):
	# Initialize sgnets variables untill loss <= 1
	global_step = tf.Variable(0, trainable=False)
	sgnet_vars = snet.variables + gnet.variables
	total_loss_tensor = snet.loss(gt_M) + gnet.loss(gt_M)

	print('Initializing vars in SGNets........Entering while loop.')
	loss, s = 10, 0
	while loss > 1:
		sess.run(tf.initialize_variables(sgnet_vars+[global_step]))
		loss = sess.run(total_loss_tensor, feed_dict=fd)
		s += 1
		print('Current loss: ', loss, ' in loop steps: ', s)
	print('Final initiation Loss value for SGNets: ', loss)

	# Backprop using SGD and updates vgg variables and sgnets variables
	lr_exp = tf.train.exponential_decay(
			1e-3, # Initial learning rate 
			global_step, 
			600, # Decay steps 
			.5 , # Decay rate 
			name='sg_lr')
	optimizer = tf.train.GradientDescentOptimizer(lr_exp)
	train_op = optimizer.minimize(total_loss_tensor, global_step= global_step ,var_list= sgnet_vars) # TODO,debug updates vgg.variables
	print('Start training the SGNets........')
	for step in range(200):
		cur_lr, loss, pre_M_g, pre_M_s, _ = sess.run([lr_exp, total_loss_tensor, gnet.pre_M, snet.pre_M, train_op], feed_dict=fd)
		print('Step: ', step,' Total loss for SGnet is : %.5f'%loss, ' Current learning rate: ', cur_lr)
	return pre_M_g, pre_M_s




print('Reading the first image...')
t_start = time.time()
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = InputProducer(IMG_PATH, GT_PATH)
img, gt, s  = next(inputProducer.gen_img)
roi_t0, _, rz_factor = inputProducer.extract_roi(img, gt)

# Predicts the first img.
sess, vgg = init_vgg()
fd = {vgg.imgs: [roi_t0]}
gt_M = inputProducer.gen_mask((224,224)) # rank2 array


## At t=0. Perform the following:
# 1. feature maps selection
# 2. Train G and S networks.
idx_c4 = SelCNN.select_fms(sess, vgg.conv4_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
idx_c5 = SelCNN.select_fms(sess, vgg.conv5_3_norm, gt, rz_factor, fd, FLAGS.sel_num)

# Instainate SGNets with conv tensors and training.
snet = SNet('SNet', vgg.conv4_3_norm, idx_c4)
gnet = GNet('GNet', vgg.conv5_3_norm, idx_c5)
train_SGNets(sess, vgg, snet, gnet, inputProducer)

inputProducer.roi_params['roi_scale'] = 3
tracker = TrackerVanilla(gt)
# Iter imgs
gt_last = gt 
conf_list, roi_list = [], []
print("Total time cost for initialization : %.2f s"%(time.time() - t_start))
for i in range(FLAGS.iter_max):
	i += 1
	t_enter = time.time()
	# Gnerates next frame infos
	img, gt_cur, s  = next(inputProducer.gen_img)

	## Crop a rectangle ROI region centered at last target location.
	roi, _, rz_factor = inputProducer.extract_roi(img, gt_last)
	
	## Perform Target localiation predicted by GNet
	# Get heat map predicted by GNet
	fd = {vgg.imgs : [roi]}
	pre_M_g, _ = sess.run([gnet.pre_M, snet.pre_M], feed_dict=fd)
	
	# Adaptive fine tune SNet, 
	if 3 % FLAGS.S_adp_steps == 0:
		img_best, loc_best = tracker.gen_best_records(inter_steps=FLAGS.S_adp_steps)
		roi_best, _, _ = inputProducer.extract_roi(img_best, loc_best)
		fd_s_adp = {vgg.imgs: [roi_best]}
		snet.adaptive_finetune(sess, gt_M, fd_s_adp, lr=1e-6)

	# Performs distracter detecion.
	# Untested!
	if False: #tracker.distracted():
		# if detects distracters, then update 
		# SNet using descrimtive loss.
		# gen mask
		print("Distractor detected! ")
		t_ds = time.time()
		phi = gen_mask_phi(roi.shape, pre_loc)
		snet.descrimtive_finetune(sess, s_sel_maps_t0, gt_M, s_sel_maps, pre_M_g, phi)
		pre_M_s = sess.run(snet.pre_M, feed_dict=feed_dict)

		# Use location predicted by SNet.
		pre_M_s = np.transpose(pre_M_s, [0,2,1,3])
		pre_loc = tracker.predict_location(pre_M_s, gt_last, resize_factor, t, 224)
		print('Descrimtive finetune SNet costs : %.2f s'%(time.time() - t_ds))


	# Localize target with monte carlo sampling.
	tracker.draw_particles()
	pre_loc = tracker.predict_location(pre_M_g[0], gt_last, rz_factor, img)
	print('At frame {0}, the most confident value is {1}'.format(i, tracker.cur_best_conf))
	print('Time consumed : %.2f s'%(time.time() - t_enter))
	gt_last = pre_loc

	# Draw bbox on image. And print associated IoU score.
	img_bbox = img_with_bbox(img, pre_loc,c=1)
	#img_bbox = img_with_bbox(img_bbox, gt_cur, c=0)
	file_name = inputProducer.imgs_path_list[i-1].split('/')[-1]
	file_name = os.path.join(PRE_ROOT, file_name)
	plt.imsave(file_name, img_bbox)