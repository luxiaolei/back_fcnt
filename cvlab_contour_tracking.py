"""
Main script for FCNT tracker. 
"""

# Import custom class and functions
from inputproducer import InputProducer
from tracker import TrackerContour
from vgg16 import Vgg16
from sgnet import GNet, SNet
from utils import select_fms, refPt_2_gt, gen_sel_maps, IOU_eval

import cv2
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
tf.app.flags.DEFINE_integer('batch_size', 15,
                          """Batch size for SGNet trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('n_samples_per_batch', 8000,
                          """Number of samples per batch for trainning"""
                          """SGnet works""")
tf.app.flags.DEFINE_integer('sel_num', 354,
                          """Number of feature maps selected.""")
tf.app.flags.DEFINE_string('use_net', 'g',
						"""true for train, false for eval""")
tf.app.flags.DEFINE_string('model_name', 'first_Car1-large-train-Gonly2',
						"""true for train, false for eval""")
FLAGS = tf.app.flags.FLAGS

## Define varies pathes
DATA_ROOT = 'data/Suv'
PRE_ROOT = os.path.join(DATA_ROOT, 'img_loc')
IMG_PATH = os.path.join(DATA_ROOT, 'img')
GT_PATH = os.path.join(DATA_ROOT, 'groundtruth_rect.txt')
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'

if not os.path.isdir(PRE_ROOT):
    os.mkdir(PRE_ROOT)

FLAGS.model_name += '-with-%snet'%FLAGS.use_net

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


def init_vgg(sess, roi_t0=None, predict=True):
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

    #sess.run(tf.initialize_all_variables())
    vgg = Vgg16(VGG_WEIGHTS_PATH, sess)
    if predict and roi_t0 is not None:
        vgg.print_prob(roi_t0, sess)
    print('Forwarding the vgg net cost : %.2f s'%(time.time() - t_start))
    return vgg


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
    gnet.params['wd'] = 0.05
    if FLAGS.use_net == 's':
        loss = snet.loss
        vars_train = snet.variables
    if FLAGS.use_net == 'g':
        loss = gnet.loss
        vars_train = gnet.variables 
    else: 
        loss = gnet.loss + snet.loss
        vars_train = gnet.variables + snet.variables

    
    # Backprop using SGD and updates vgg variables and sgnets variables
    global_step = tf.Variable(0, trainable=False)
    lr_exp = tf.train.exponential_decay(
            0.25, # Initial learning rate 
            global_step, 
            1000, # Decay steps 
            0.8, # Decay rate 
            name='sg_lr')
    optimizer = tf.train.GradientDescentOptimizer(lr_exp)
    train_op = optimizer.minimize(loss, var_list= vars_train, global_step=global_step)
    
    
    tf.scalar_summary('Learning rate', lr_exp)
    tf.scalar_summary('loss', loss)
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter(TB_SUMMARY, sess.graph)
    saver = tf.train.Saver()

    print('Generating batches from img size:%s  for trainning.'%str(img.shape))
    sample_batches, target_batches = inputProducer.gen_batches(img, gt,
                                    n_samples=FLAGS.n_samples_per_batch, 
                                    batch_sz=FLAGS.batch_size, 
                                    pos_ratio=1., 
                                    scale_factors=np.arange(1., 2., 0.5),
                                    random_brightness=False) #np.array([1]))#

    print('Start training the SGNets........ for %s epoches'%FLAGS.iter_epoch_sg)
    
    step = 1
    loss_list = []
    for ep in range(FLAGS.iter_epoch_sg):
        print('Total batches in each epoch: ', len(sample_batches))
        for roi, target in zip(sample_batches, target_batches):     
            t = time.time()
            c4_maps, c5_maps = gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5)         
            fd = {gnet.input_maps: c5_maps, gnet.gt_M: target, 
                  snet.input_maps: c4_maps, snet.gt_M: target}
            # Initialization 
            if step == 1:
                loss_g = 10
                init_s = 0
                while loss_g > 1.2:
                    init_s += 1
                    sess.run(tf.initialize_variables(gnet.variables))
                    loss_g = sess.run(gnet.loss, feed_dict=fd)
                    print('Initial Gnet Loss: ', loss_g, 'In steps: ', init_s)
                sess.run(tf.initialize_variables(snet.variables + [global_step]))
                           
            pre_M_g, l, _, lr = sess.run([gnet.pre_M, loss, train_op, lr_exp], feed_dict=fd)
            
            loss_list += [l]
            #if l <= 0.3:
                #print('break learning!?')
                #break
            if step % 20 == 0: 
                loss_ac = np.diff(np.diff(loss_list[-19:]))
                loss_ac_summary = tf.scalar_summary('Loss acceleration', loss_ac.mean())
                
                summary_img_g = tf.image_summary('pre_M_g', 
                                                 np.repeat(pre_M_g[...,np.newaxis], 3, axis=-1), 
                                                 name='GMap')

                summary, img_summary_g, ac_loss_summary = sess.run([merged, summary_img_g,\
                                                                    loss_ac_summary], feed_dict=fd)
                writer.add_summary(summary, global_step=step)
                writer.add_summary(img_summary_g, global_step=step)
                writer.add_summary(ac_loss_summary, global_step=step)
                
                loss_std = np.std(loss_list[-19:])
                if loss_std <= 0.007:
                    print('Stop learning??! Last 10 batches Loss Std: ', loss_std)

                print('Epoch: ', ep+1, 'Step: ', (ep+1)*step, 'Loss : %.2f'%l, \
                    'Speed: %.2f second/batch'%(time.time()-t), 'Lr: ', lr)
                #saver.save(sess, CKPT_MODEL)
            step += 1



print('Reading the first image...')
t_start = time.time()
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = InputProducer(IMG_PATH, GT_PATH)
image, gt, s  = next(inputProducer.gen_img)
roi_t0, _, rz_factor = inputProducer.extract_roi(image, gt)

tracker = TrackerContour()
inputProducer.roi_params['roi_scale'] = 1.5

sess = tf.Session()
snet = SNet('SNet', FLAGS.sel_num)
gnet = GNet('GNet', FLAGS.sel_num)
vgg = init_vgg(sess, roi_t0)
fd = {vgg.imgs: [roi_t0]}
gt_M = inputProducer.gen_mask((28,28)) # rank2 array

saver = tf.train.Saver()
saved_ckpt = os.path.join('checkpoint', FLAGS.model_name.split('_')[-1]+'.ckpt')

## At t=0. Train S and G Nets 
# Instainate SGNets with conv tensors and training.
idx_c4 = select_fms(sess, vgg.conv4_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
idx_c5 = select_fms(sess, vgg.conv5_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
tracker.init_first_img(sess, image, roi_t0, vgg, idx_c4, idx_c5, gt_M, gt)


if os.path.exists(saved_ckpt):
    print('Found saved model %s, restoring! '%saved_ckpt)
    saver.restore(sess, saved_ckpt)
else: 
    print('Not found saved model %s. Trainning! '%saved_ckpt)
    train_SGNets(sess, image, gt, vgg, snet, gnet, inputProducer, idx_c4, idx_c5)
    saver.save(sess, saved_ckpt)
    

trackerTLD = cv2.Tracker_create("TLD")
ok = trackerTLD.init(image, tuple(gt))


w0_half, h0_half = np.array(gt[2:])/2
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(inputProducer.imgs_path_list)-1):
    
    t_s = time.time()

    image, gt_cur, s  = next(inputProducer.gen_img)
    img = image.copy()# tracker.adjust_brightness(image)

    # @inputproducer, remove low level pixel
    try:
        #img = inputProducer.Ajust_brighteness(img, gt_last)
        roi, _, rz_factor = inputProducer.extract_roi(img, tracker.gt_last)
        roi = tracker.preporcess_roi(roi)
    except Exception:
        pass
        
    ## Perform Target localiation predicted by GNet
    # Get heat map predicted by GNet
    c4_maps, c5_maps = gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5)
    fd = {gnet.input_maps: c5_maps, snet.input_maps: c4_maps}
    pre_M_g, pre_M_s = sess.run([gnet.pre_M, snet.pre_M], feed_dict=fd)

    pre_M = tracker.preporcess_heatmaps(pre_M_g, pre_M_s, uses='g', resize=(224,224))
    pre_loc, pre_loc_roi = tracker.predict_location(pre_M ,rz_factor,threshold=np.arange(0.3, 0.9, 0.05))
    
    #Update SNet
    #snet.fine_finetune(sess, tracker, roi, pre_loc_roi, vgg, idx_c4, idx_c5, last_fames_n=1)


    _, pre_loc_tld = trackerTLD.update(image)
    tracker.gt_last = pre_loc_tld
    p1_tld = (int(pre_loc_tld[0]), int(pre_loc_tld[1]))
    p2_tld = (int(pre_loc_tld[0] + pre_loc_tld[2]), int(pre_loc_tld[1] + pre_loc_tld[3]))
    # visualizing
    x,y,w,h = pre_loc
    if w < w0_half: 
        w = int(w0_half)
        pre_loc = [w, y, w, h]
    if h < h0_half: 
        h = int(h0_half)
        pre_loc = [w, y, w, h]
    if (x+w) > image.shape[1]:
        w =  image.shape[1] - x - 1
    if (y+h) > image.shape[0]:
        h =  image.shape[0] - y - 1
    xr, yr, wr, hr = pre_loc_roi
    #cv2.rectangle(image,p1_tld, p2_tld,(0,225,0),1)    
    cv2.rectangle(image,(x,y),(x+w,y+h),(225,0,0),2)
    cv2.rectangle(pre_M,(xr,yr),(xr+wr,yr+hr),(225,0,0),2)
    cv2.putText(image, 'Frame: %s IOU score: %.2f'%(i, IOU_eval(img, gt_cur, pre_loc)),(5,50), font, 0.6,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(image, 'frames/sec : %.1f'%(60*(time.time()-t_s)) ,(5,20), font, 0.6,(255,0,0),1,cv2.LINE_AA)
    cv2.putText(pre_M, 'Conf Score : %.3f'%(tracker.conf_scores[-1]) ,(5,20), font, 0.6,(255,0,0),1,cv2.LINE_AA)

    #cv2.imshow("pre_M_g", imresize(np.repeat(pre_M_g[...,np.newaxis], 3, axis=-1), (224,224)))
    #cv2.imshow("pre_M_s", imresize(np.repeat(pre_M_s[...,np.newaxis], 3, axis=-1), (224,224)))
    cv2.imshow("pre_M", np.repeat(pre_M[...,np.newaxis], 3, axis=-1))
    cv2.imshow("ROI", roi)
    # Finetune SNet
    
    print('Tracking done in step: %s'%tracker.step)
    
        
    cv2.imshow("image", image)
    cv2.waitKey(1)

    file_name = FLAGS.model_name + inputProducer.imgs_path_list[i-1].split('/')[-1]
    file_name = os.path.join(PRE_ROOT, file_name)
    plt.imsave(file_name, image)

vid_path_prefix = os.path.join(PRE_ROOT, FLAGS.model_name) 
os.system('ffmpeg -framerate 25 -i %s%%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p %s.mp4'\
            %(vid_path_prefix, FLAGS.model_name))

# close all open windows
cv2.destroyAllWindows()