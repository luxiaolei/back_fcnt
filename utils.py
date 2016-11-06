
import numpy as np
import tensorflow as tf
import skimage
import cv2

from skimage import draw
from scipy.misc import imresize
from scipy.linalg import norm




def variable_on_cpu(scope, name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensora

    """
    dtype = tf.float32
    with tf.variable_scope(scope) as scope:
        with tf.device('/cpu:0'):
            variable = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return variable

def variable_with_weight_decay(scope, name, shape, stddev=1e-3, wd=None):
    """Helper to create an initialized Variable with weight decay

    Args:
        name: name of the variable
        shape: list of ints
        stddev: float, standard deviation of a truncated Gaussian for initial value
        wd: add L2loss weight decay multiplied by this float. If None, weight decay 
                is not added to this variable

    Returns:
        Variable: Tensor
    """
    dtype = tf.float32
    variable = variable_on_cpu(
                            scope,
                            name, 
                            shape,
                            initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(variable), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return variable

def select_fms(sess, conv_tensor, gt, rz_factor, fd, sel_num):
    """
    Select feature maps of vg conv layers, by computing
    within/overall score of target region inside a map

    Args:
        sess: tf.Session object.
        conv_tensor: tf.Tensor, shoudld be either conv4_3 or conv5_3 resized and normalized layer.
        gt: [x,y,w,h], groundtruth of target in original image.
        rz_factor: float, size in original img * rz_factor = size in extracted roi img.
        fd: dict, feed_dict for feeding a vgg network.
        sel_num: int, number of slected maps.

    Returns:
        idx: list with length sel_num
    """
    assert isinstance(conv_tensor,tf.Tensor)
    assert isinstance(conv_tensor,tf.Tensor)
    def compute_score(roi, gt, rz_factor):
        """Helper func for computing confidence"""
        roi = np.copy(roi)
        _,_,w,h = gt
        w_half = int(0.5*w*rz_factor)# resize_factor
        h_half = int(0.5*h*rz_factor)
        c = 224/2
        conf_i = roi[c-h_half:c+h_half, c-w_half:c+w_half].sum()
        conf_u = roi.sum()
        if conf_u == 0:
            return 0.0
        else:
            return conf_i / conf_u

    # Get values of conv layer
    conv_arr = sess.run(conv_tensor, feed_dict=fd)

    # Compute score
    scores = []
    for idx in range(512):
        scores += [compute_score(conv_arr[0,...,idx], gt, rz_factor)]
    selected_idx = sorted(range(len(scores)), key=lambda i: scores[i])[-sel_num:]
    return selected_idx

# draw on img
def img_with_bbox(img_origin, gt_1, c=1):  
    """
    Draw boundding box on given image.

    Args: 
        img_origin: np.ndarry with shape (224, 224, 3). Image to draw on.
        gt_1: list of [tlx, tly, w, h].
    Returns:
        img: Image array with same shape as `img_origin`, with bbox on it.
    """  
    img =np.copy(img_origin)
    maxh, maxw = img.shape[:2]
    gt_1 = [int(i) for i in gt_1]
    tl_x, tl_y, w, h = gt_1

    if tl_x+w >= maxw:
        w = maxw - tl_x -1
    if tl_y+h >= maxh:
        h = maxh - tl_y -1

    tr_x, tr_y = tl_x + w, tl_y 
    dl_x, dl_y = tl_x, tl_y + h
    dr_x, dr_y = tl_x + w, tl_y +h

    rr1, cc1 = draw.line( tl_y,tl_x, tr_y, tr_x)
    rr2, cc2 = draw.line( tl_y,tl_x, dl_y, dl_x)
    rr3, cc3 = draw.line( dr_y,dr_x, tr_y, tr_x)
    rr4, cc4 = draw.line( dr_y,dr_x, dl_y, dl_x)
    img[rr1, cc1, :] = c
    img[rr2, cc2, :] = c
    img[rr3, cc3, :] = c
    img[rr4, cc4, :] = c
    return img

def gauss2d(shape=(6,6),sigma=1.):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    Args:
        shape: 2 element tuple. specifying the output shape.
        sigma: float, variance factor.
    Returns:
        h: a 2D gaussian with shape as `shape`.
    """
    # Implements 2D gaussian formula
    sigma *= 1000
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh

    # Normalize
    #h = h / h.max()
    return h

def calcEntropy(img):
    #hist,_ = np.histogram(img, np.arange(0, 256), normed=True)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist.ravel()/hist.sum()
    #logs = np.nan_to_num(np.log2(hist))
    logs = np.log2(hist+0.00001)
    #hist_loghist = hist * logs
    entropy = -1 * (hist*logs).sum()
    return entropy


def compare_images(img1, img2):
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = abs(diff).sum()  # Manhattan norm
    z_norm = norm(diff.ravel(), 0)  # Zero norm
    return (m_norm, z_norm)


def IOU_eval(img, gt_cur, pre_loc):
    """
    Returns:
        iou: scaler
    """
    convas = np.zeros(img.shape)
    xg, yg, wg, hg = gt_cur
    xp, yp, wp, hp = pre_loc
    convas[yg:yg+hg, xg:xg+wg, :] += 1
    convas[yp:yp+hp, xp:xp+wp, :] += 1
    intersection = convas[convas==2].sum()/2
    convas[convas>0] = 1
    union = convas[convas>0].sum()
    return intersection/union

def refPt_2_gt(refPt):
    p1, p2 = refPt
    x1, y1 = p1
    x2, y2 = p2
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)

def gen_sel_maps(sess, roi, vgg, idx_c4, idx_c5):
    """Returns selected c4 and c5 maps"""
    if len(roi.shape) == 3: roi = [roi]
    fd = {vgg.imgs : roi}
    c4_arr, c5_arr = sess.run([vgg.conv4_3_norm, vgg.conv5_3_norm], feed_dict=fd)
    c4_maps = c4_arr[...,idx_c4]
    c5_maps = c5_arr[...,idx_c5]
    return c4_maps, c5_maps