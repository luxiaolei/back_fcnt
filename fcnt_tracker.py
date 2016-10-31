"""
Main script for FCNT tracker. 
"""

# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval

import numpy as np 
import tensorflow as tf
import matplotlib.pylab as plt

import sys
import os
import time

tf.app.flags.DEFINE_integer('iter_epoch_sg', 5,
                          """Number of epoches for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('n_samples_per_batch', 5000,
                          """Number of samples per batch for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('iter_max', 1349,
							"""Max iter times through imgs""")

tf.app.flags.DEFINE_string('model_name', 'all_vars_lr_1e3_posratio_7',
						"""true for train, false for eval""")
FLAGS = tf.app.flags.FLAGS

## Define varies pathes
DATA_ROOT = 'data/Dog1'
PRE_ROOT = os.path.join(DATA_ROOT, 'img_loc')
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'

if not os.path.isdir(PRE_ROOT):
    os.mkdir(PRE_ROOT)


TB_SUMMARY = os.path.join('tb_summary', FLAGS.model_name)
if not os.path.isdir('tb_summary'):
    os.mkdir('tb_summary')
if not os.path.isdir(TB_SUMMARY):
    os.mkdir(TB_SUMMARY)

CKPT_PATH = 'checkpoint'
if not os.path.isdir(CKPT_PATH):
    os.mkdir(CKPT_PATH)

model_name = FLAGS.model_name+'.ckpt'
CKPT_MODEL = os.path.join(CKPT_PATH, model_name)

def init_vgg(roi_t0):
	"""
	Initialize a tf.Session and a vgg16 graph. Followed
	by forwarding the vgg net once to predict top5 class labels
	for image generated in the first frame.

	Args:
		roi_t0: np.ndarray with shape (224x224x3), extracted roi in the first frame.
	Returns:
		sess: tf.Session object.
		vgg: Vgg16 class instance.
	"""
	print('Classify it with a pre-trained Vgg16 model.')
	t_start = time.time()
	sess = tf.Session()
	sess.run(tf.initialize_all_variables())
	vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
	vgg.print_prob(roi_t0, sess)
	print('Forwarding the vgg net cost : %.2f s'%(time.time() - t_start))
	return sess, vgg



def train_SGNets(sess, img, gt, vgg, snet, gnet, inputProducer):
	"""
	Train SGnets' variables by minimizing a composite L2 regression losses.

	Args:
		sess: tf.Session object.
		vgg: Vgg16 class instance.
		snet: SNet class instance.
		gnet:  GNet class instance.
		inputProducer: InputProducer class instance.
	"""

	loss = gnet.loss() + snet.loss()
	tf.scalar_summary('loss', total_loss)
	writer = tf.train.SummaryWriter(TB_SUMMARY, sess.graph)

	vars_train = vgg.variables + gnet.variables + snet.variables

	# Backprop using SGD and updates vgg variables and sgnets variables
	global_step = tf.Variable(0, trainable=False)
	lr_exp = tf.train.exponential_decay(
			1e-3, # Initial learning rate 
			global_step, 
			1000, # Decay steps 
			0.9, # Decay rate 
			name='sg_lr')

	tf.scalar_summary('Learning rate', lr_exp)
	merged = tf.merge_all_summaries()
	optimizer = tf.train.GradientDescentOptimizer(lr_exp)
	train_op = optimizer.minimize(loss, var_list= vars_train, global_step=global_step)
	sess.run(tf.initialize_variables(snet.variables + gnet.variables + [global_step]))


	sample_batches, target_batches = inputProducer.gen_batches(img, gt, n_samples=FLAGS.n_samples_per_batch, batch_sz=10, pos_ratio=0.7, scale_factors=np.arange(0.2, 5., 0.5))
	print('Start training the SGNets........ for %s epoches'%FLAGS.iter_epoch_sg)
	saver = tf.train.Saver()
	for ep in range(FLAGS.iter_epoch_sg):
		step = 0
		print('Total samples in each epoch: ', len(sample_batches))
		for roi, target in zip(sample_batches, target_batches):
			t = time.time()
			fd = {vgg.imgs: roi, gnet.gt_M: target, snet.gt_M: target}
			pre_M_g, pre_M_s, l, _, lr = sess.run([gnet.pre_M, snet.pre_M, loss, train_op, lr_exp], feed_dict=fd)

            if step % 20 == 0:
                summary_img_g = tf.image_summary('pre_M', pre_M_g)
				summary_img_s = tf.image_summary('pre_M', pre_M_s)
                summary, img_summary_g, img_summary_s = sess.run([merged, summary_img_g, summary_img_s], feed_dict=fd)
                
                writer.add_summary(summary, global_step=step)
                writer.add_summary(img_summary_g, global_step=step)
				writer.add_summary(img_summary_s, global_step=step)
			
			if step % 50 == 0:
				print('Epoch: ', ep+1, 'Step: ', (ep+1)*step, 'Loss : %.2f'%l, \
					'Speed: %2.f second/batch'%(time.time()-t), 'Lr: ', lr)
				saver.save(sess, CKPT_MODEL)
			step += 1


def main(args):
	print('Reading the first image...')
	t_start = time.time()
	## Instantiate inputProducer and retrive the first img
	# with associated ground truth. 
	inputProducer = InputProducer(IMG_PATH, GT_PATH)
	img, gt, s  = next(inputProducer.gen_img)
	roi_t0, _, rz_factor = inputProducer.extract_roi(img, gt)

	# Predicts the first img.
	sess, vgg = init_vgg(roi_t0)
	fd = {vgg.imgs: [roi_t0]}
	gt_M = inputProducer.gen_mask((224,224)) # rank2 array


	## At t=0. Train S and G Nets 
	# Instainate SGNets with conv tensors and training.
	snet = SNet('SNet', vgg.conv4_3_norm)
	gnet = GNet('GNet', vgg.conv5_3_norm)
	train_SGNets(sess, img, gt, vgg, snet, gnet, inputProducer)

	
	## Instainate a tracker object, set apoproaite initial parameters.
	tracker = TrackerVanilla(gt)
	inputProducer.roi_params['roi_scale'] = 3
	norm_roi_scale_w = inputProducer.roi_params['roi_scale'] / gt[2]
	tracker.params['aff_sig'] = [10, 10, 0.6, 0.6]
	tracker.params['p_num'] = 800
	tracker.params['particle_scales'] = [1,1]
	print("Total time cost for initialization : %.2f s"%(time.time() - t_start))
				
	# Iter imgs
	gt_last = gt 
	for i in range(FLAGS.iter_max):
		i += 1
		if i % 30 == 0:
			print(len(gt_list))
			v = np.diff([loc[2] for loc in gt_list[-30:]])
			ac = np.diff(v)
			acv = ac[-30:].sum() / 30
			print(acv)
			if acv >= 0.8:
				tracker.params['particle_scales'] = particle_scale*np.array([1.1, 1.2, 1.3, 1.4]) 
			elif acv < -0.8:
				tracker.params['particle_scales'] = particle_scale*np.array([0.9, 0.8, 0.7, 0.6])
		
		
		t_enter = time.time()
		# Gnerates next frame infos
		img, gt_cur, s  = next(inputProducer.gen_img)

		## Crop a rectangle ROI region centered at last target location.
		roi, _, rz_factor = inputProducer.extract_roi(img, gt_last)

		## Perform Target localiation predicted by GNet
		# Get heat map predicted by GNet
		fd = {vgg.imgs : [roi]}
		pre_M_g, _ = sess.run([gnet.pre_M, snet.pre_M], feed_dict=fd)

		# Localize target with monte carlo sampling.
		pre_loc = tracker.predict_location(pre_M_g, gt_last, rz_factor, img)

		if i % 100 == 0:
			print('At frame {0}, the most confident value is {1}'.format(i, tracker.cur_best_conf))
			print('Time consumed : %.2f s'%(time.time() - t_enter))

		gt_last = pre_loc
		print('pre: ', pre_loc, 'actual: ', gt_cur)
		# Draw bbox on image. And print associated IoU score.
		img_bbox = img_with_bbox(img, pre_loc,c=1)
		file_name = inputProducer.imgs_path_list[i-1].split('/')[-1]
		file_name = os.path.join(PRE_ROOT, file_name)
		plt.imsave(file_name, img_bbox)

if __name__=='__main__':
	tf.app.run()