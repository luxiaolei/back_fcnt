"""
Main script for FCNT tracker. 
"""

# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerVanilla
from vgg16 import Vgg16
from sgnet import GNet, SNet
from utils import img_with_bbox, IOU_eval, select_fms

import numpy as np 
import tensorflow as tf
import matplotlib.pylab as plt

from scipy.misc import imresize
from subprocess import call
import sys
import os
import time

tf.app.flags.DEFINE_integer('iter_epoch_sg', 5,
                          """Number of epoches for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('batch_size', 35,
                          """Batch size for SGNet trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('n_samples_per_batch', 5000,
                          """Number of samples per batch for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('iter_max', 1349,
							"""Max iter times through imgs""")
tf.app.flags.DEFINE_integer('sel_num', 354,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_string('model_name', 'lr015_early_stop_sf05_02_gauss_wd05_bz35_GLoss',
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
        roi_t0: np.ndarray with shape (28x28x3), extracted roi in the first frame.
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

def gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5):
    """Returns selected c4 and c5 maps"""
    if len(roi.shape) == 3: roi = [roi]
    fd = {vgg.imgs : roi}
    c4_arr, c5_arr = sess.run([vgg.conv4_3_norm, vgg.conv5_3_norm], feed_dict=fd)
    c4_maps = c4_arr[...,idx_c4]
    c5_maps = c5_arr[...,idx_c5]
    return c4_maps, c5_maps


def train_SGNets(sess, img, gt, vgg, snet, gnet, inputProducer, idx_c4, idx_c5):
    """
    Train SGnets' variables by minimizing a composite L2 regression losses.

    Args:
        sess: tf.Session object.
        vgg: Vgg16 class instance.
        snet: SNet class instance.
        gnet:  GNet class instance.
        inputProducer: InputProducer class instance.
    """
    gnet.params['wd'] = 0.5
    gloss, sloss = gnet.loss, snet.loss
    loss = gloss #+ 0.005*sloss
    tf.scalar_summary('loss', loss)
    writer = tf.train.SummaryWriter(TB_SUMMARY, sess.graph)
    
    vars_train = gnet.variables + snet.variables

    # Backprop using SGD and updates vgg variables and sgnets variables
    global_step = tf.Variable(0, trainable=False)
    lr_exp = tf.train.exponential_decay(
            0.15, # Initial learning rate 
            global_step, 
            1000, # Decay steps 
            0.8, # Decay rate 
            name='sg_lr')

    tf.scalar_summary('Learning rate', lr_exp)
    optimizer = tf.train.GradientDescentOptimizer(lr_exp)
    train_op = optimizer.minimize(loss, var_list= vars_train, global_step=global_step)
    merged = tf.merge_all_summaries()

    sample_batches, target_batches = inputProducer.gen_batches(img, gt, n_samples=FLAGS.n_samples_per_batch, batch_sz=FLAGS.batch_size, pos_ratio=0.5, scale_factors=np.arange(0.5, 5., 0.2)) #np.array([1]))#
    print('Start training the SGNets........ for %s epoches'%FLAGS.iter_epoch_sg)
    saver = tf.train.Saver()
    step = 1
    loss_list = []
    stop_flag = False
    for ep in range(FLAGS.iter_epoch_sg):
        print('Total batches in each epoch: ', len(sample_batches))
        if stop_flag:
            break
        for roi, target in zip(sample_batches, target_batches):
            #roi[roi>0] = 1 # neglect gaussian..set to 1 for target arear
            
            t = time.time()
            c4_maps, c5_maps = gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5)
            
            fd = {gnet.input_maps: c5_maps, gnet.gt_M: target, 
                  snet.input_maps: c4_maps, snet.gt_M: target}
            
            # Initialization 
            if step == 1:
                loss_g = 10
                init_s = 0
                while loss_g > 1.5:
                    init_s += 1
                    sess.run(tf.initialize_variables(gnet.variables))
                    loss_g = sess.run(gloss, feed_dict=fd)
                    print('Initial Gnet Loss: ', loss_g, 'In steps: ', init_s)
                sess.run(tf.initialize_variables(snet.variables + [global_step]))
                
            
            pre_M_g, l, _, lr = sess.run([gnet.pre_M, loss, train_op, lr_exp], feed_dict=fd)
            
            loss_list += [l]
            if l <= 0.3:
                stop_flag = True
                print('break learning!')
                break
            if step % 20 == 0:
                
                loss_ac = np.diff(np.diff(loss_list[-19:]))
                loss_ac_summary = tf.scalar_summary('Loss acceleration', loss_ac.mean())
                
                
                summary_img_g = tf.image_summary('pre_M', 
                                                 np.repeat(pre_M_g[...,np.newaxis], 3, axis=-1), name='GMap')

                summary, img_summary_g, ac_loss_summary = sess.run([merged, summary_img_g, loss_ac_summary], feed_dict=fd)

                writer.add_summary(summary, global_step=step)
                writer.add_summary(img_summary_g, global_step=step)
                writer.add_summary(ac_loss_summary, global_step=step)
                
                loss_std = np.std(loss_list[-19:])
                if loss_std <= 0.007:
                    stop_flag = True
                    print('Stop learning! Last 10 batches Loss Std: ', loss_std)
                    break

            #if step % 20 == 0:
                print('Epoch: ', ep+1, 'Step: ', (ep+1)*step, 'Loss : %.2f'%l, \
                    'Speed: %.2f second/batch'%(time.time()-t), 'Lr: ', lr)
                #saver.save(sess, CKPT_MODEL)
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
    gt_M = inputProducer.gen_mask((28,28)) # rank2 array


    ## At t=0. Train S and G Nets 
    # Instainate SGNets with conv tensors and training.
    # 1. feature maps selection
    # 2. Train G and S networks.
    idx_c4 = select_fms(sess, vgg.conv4_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
    idx_c5 = select_fms(sess, vgg.conv5_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
    snet = SNet('SNet', FLAGS.sel_num)
    gnet = GNet('GNet', FLAGS.sel_num)
    train_SGNets(sess, img, gt, vgg, snet, gnet, inputProducer, idx_c4, idx_c5)



    ## Instainate a tracker object, set apoproaite initial parameters.
    tracker = TrackerVanilla(gt)
    inputProducer.roi_params['roi_scale'] = 2.5
    norm_roi_scale_w = inputProducer.roi_params['roi_scale'] / gt[2]
    tracker.params['aff_sig'] = [10, 10, 0.6, 0.6]
    tracker.params['p_num'] = 800
    tracker.params['particle_scales'] = [1,1]
    print("Total time cost for initialization : %.2f s"%(time.time() - t_start))

    # Iter imgs
    gt_last = gt 
    gt_list = []
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

        roi[roi==0] = roi.mean()

        ## Perform Target localiation predicted by GNet
        # Get heat map predicted by GNet
        c4_maps, c5_maps = gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5)
        fd = {gnet.input_maps: c5_maps, snet.input_maps: c4_maps}

        pre_M_g, _ = sess.run([gnet.pre_M, snet.pre_M], feed_dict=fd)


        pre_M_g[pre_M_g<=0.7] = 0
        pre_M_g = imresize(pre_M_g, (224,224))


        # Localize target with monte carlo sampling.
        pre_loc = tracker.predict_location(pre_M_g, gt_last, rz_factor, img)

        if i % 100 == 0:
            print('At frame {0}, the most confident value is {1}'.format(i, tracker.cur_best_conf))
            print('Time consumed : %.2f s'%(time.time() - t_enter))

        gt_last = pre_loc
        gt_list += [pre_loc]
        print('Step: ',i-1, 'pre - actual : ', np.array(pre_loc) - np.array(gt_cur))

        # Draw bbox on image. And print associated IoU score.
        img_bbox = img_with_bbox(img, pre_loc,c=1)
        file_name = FLAGS.model_name + inputProducer.imgs_path_list[i-1].split('/')[-1]
        file_name = os.path.join(PRE_ROOT, file_name)
        plt.imsave(file_name, img_bbox)

    vid_path_prefix = os.path.join(PRE_ROOT, FLAGS.model_name) 
    os.system('ffmpeg -framerate 25 -i %s%%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p %s.mp4'\
              %(vid_path_prefix, FLAGS.model_name))
    
if __name__=="__main__":
    tf.app.run()