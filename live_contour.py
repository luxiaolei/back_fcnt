"""
Main script for FCNT tracker. 
"""

# Import custom class and functions
from inputproducer import LiveInput
from tracker import TrackerContour
from vgg16 import Vgg16
from sgnet import GNet, SNet
from utils import select_fms, refPt_2_gt, gen_sel_maps

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
tf.app.flags.DEFINE_string('use_net', 'sg',
						"""true for train, false for eval""")
tf.app.flags.DEFINE_string('model_name', 'first_LiveFaceXLOfficesgmult3',
						"""true for train, false for eval""")
FLAGS = tf.app.flags.FLAGS

## Define varies pathes
VGG_WEIGHTS_PATH = 'vgg16_weights.npz'

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
            0.15, # Initial learning rate 
            global_step, 
            1500, # Decay steps 
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
                                    scale_factors=np.arange(1., 2., 0.2),
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
            #if l <= 0.1:
                #print('break learning!')
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



refPt = []
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

print('Reading the first image...')
t_start = time.time()
## Instantiate inputProducer and retrive the first img
# with associated ground truth. 
inputProducer = LiveInput()
tracker = TrackerContour()
inputProducer.roi_params['roi_scale'] = 1.5


cap = cv2.VideoCapture(0)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)


sess = tf.Session()
snet = SNet('SNet', FLAGS.sel_num)
gnet = GNet('GNet', FLAGS.sel_num)
saver = tf.train.Saver()
saved_ckpt = os.path.join('checkpoint', FLAGS.model_name.split('_')[-1]+'.ckpt')

if os.path.exists(saved_ckpt):
    print('Found saved model %s, restoring! '%saved_ckpt)
    saver.restore(sess, saved_ckpt)
    TrackReady = True
else: 
    TrackReady = False

trackerTLD = cv2.Tracker_create("TLD")


PosReady = False
font = cv2.FONT_HERSHEY_SIMPLEX

kalman = cv2.KalmanFilter(4, 2)

kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.03

#bgs = cv2.createBackgroundSubtractorMOG2(varThreshold=1.0)#history,nGauss,bgThresh,noise)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()

    # load the image, clone it, and setup the mouse callback function
    clone = image.copy()
    
    key = cv2.waitKey(1) & 0xFF
    # keep looping until the 'q' key is pressed
    if key == ord('q'):
        break

    # if there are two reference points, then crop the region of interest
    # from the image and display it
    if len(refPt) == 2 and key==ord("c"):      
        roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
        cv2.imshow("CroppedROI", roi)
        gt = refPt_2_gt(refPt)
        img = image
        print(gt, 'gt in first!')
        #inputProducer.save_fist_roi_mean(img, gt)

    # train
    #if key == ord('t'):
        roi_t0, _, rz_factor = inputProducer.extract_roi(img, gt, first_frame=True)
        
        # Predicts the first img.
        
        vgg = init_vgg(sess, roi_t0)
        fd = {vgg.imgs: [roi_t0]}
        gt_M = inputProducer.gen_mask((28,28)) # rank2 array

        ## At t=0. Train S and G Nets 
        # Instainate SGNets with conv tensors and training.
        idx_c4 = select_fms(sess, vgg.conv4_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
        idx_c5 = select_fms(sess, vgg.conv5_3_norm, gt, rz_factor, fd, FLAGS.sel_num)
        tracker.init_first_img(sess, image, roi_t0, vgg, idx_c4, idx_c5, gt_M, gt)


        if not TrackReady:
            train_SGNets(sess, img, gt, vgg, snet, gnet, inputProducer, idx_c4, idx_c5)
            saver.save(sess, saved_ckpt)
            TrackReady = True
    
    # Records the first position
    if key == ord('s'):
        tracker.gt_last = refPt_2_gt(refPt)
        ok = trackerTLD.init(image, tuple(tracker.gt_last))
        w0_half, h0_half = np.array(tracker.gt_last[2:])/2
        print(tracker.gt_last, 'gt in start~!')
        PosReady = True
        
        
    # Start tracking
    if PosReady and TrackReady:
        if tracker.step <= 1:
            print('Everything is ready, start tracking! ')

        t_s = time.time()

        img = image.copy()#img = tracker.adjust_brightness(image)
        
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
        #pre_M = bgs.apply(pre_M)

        pre_loc, pre_loc_roi = tracker.predict_location(pre_M, rz_factor, threshold=np.arange(0.3, 0.9, 0.05))
        
        #Update SNet
        if tracker.step % 2000 == 0:
            snet.fine_finetune(sess, tracker, roi, pre_loc_roi, vgg, idx_c4, idx_c5, last_fames_n=1)

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


        _, pre_loc_tld= trackerTLD.update(image)#pre_loc
        mp = np.array([[np.float32(pre_loc_tld[0])],[np.float32(pre_loc_tld[1])]])
        kalman.correct(mp)
        tp = kalman.predict()
        pre_loc_tld= [tp[0], tp[1], pre_loc_tld[2], pre_loc_tld[3]]
        
        cur_velocity = [tp[2], tp[3]]
        
        
        if tracker.step > 10 :
            cur_speed = (cur_velocity[0]**2 + cur_velocity[1]**2)**0.5
            last_speed = (tracker.velocity[0]**2 + tracker.velocity[1]**2)**0.5
            if cur_speed <= 20.5* last_speed:
                    
                tracker.gt_last = pre_loc_tld

                tracker.velocity = cur_velocity
            else:
                print('cur_speed', cur_speed)
                print('last speed', last_speed)
                #time.sleep(5)
                pre_loc = tracker.gt_last
        else:
            tracker.velocity = cur_velocity
        

        #_, pre_loc_tld = trackerTLD.update(image)
        #pre_loc_tld= pre_loc
        p1_tld = (int(pre_loc_tld[0]), int(pre_loc_tld[1]))
        p2_tld = (int(pre_loc_tld[0] + pre_loc_tld[2]), int(pre_loc_tld[1] + pre_loc_tld[3]))


        cv2.rectangle(image,p1_tld, p2_tld,(225,0,0),2)

        #cv2.rectangle(image,(x,y),(x+w,y+h),(225,0,0),2)
        cv2.rectangle(pre_M,(xr,yr),(xr+wr,yr+hr),(225,0,0),2)
        cv2.putText(image, 'frames/sec : %.1f'%(60*(time.time()-t_s)) ,(5,20), font, 0.6,(255,0,0),1,cv2.LINE_AA)
        cv2.putText(pre_M, 'Conf Score : %.3f'%(tracker.conf_scores[-1]) ,(5,20), font, 0.6,(255,0,0),1,cv2.LINE_AA)

        cv2.imshow("pre_M_g", imresize(np.repeat(pre_M_g[...,np.newaxis], 3, axis=-1), (224,224)))
        cv2.imshow("pre_M_s", imresize(np.repeat(pre_M_s[...,np.newaxis], 3, axis=-1), (224,224)))
        cv2.imshow("pre_M", np.repeat(pre_M[...,np.newaxis], 3, axis=-1))
        cv2.imshow("ROI", roi)
        # Finetune SNet
        
        #print('Tracking done in step: %s'%tracker.step)
        
        
    cv2.imshow("image", image)
    cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()