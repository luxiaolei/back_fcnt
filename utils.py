
import numpy as np
import tensorflow as tf
import skimage

from skimage import draw
from scipy.misc import imresize


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

def gauss2d(shape=(6,6),sigma=1000.):
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




def IOU_eval(groud_truth_box, predicted_box):
	"""
	Returns:
		iou: scaler
	"""

	pass

