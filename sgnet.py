
import tensorflow as tf
import numpy as np 

from scipy.misc import imresize

from utils import variable_on_cpu, variable_with_weight_decay



class SGNet:
	"""
	Network structure is as follow:
	A vgg's either conv4_3 or conv5_v3 layer is passed to 
	the first additional convolutional layer which has convolutional 
	kernels of size 9×9 and outputs 36 feature maps as the 
	input to the next layer. 
	The second additional convolutional layer has kernels of size 5 × 5
	and outputs the foreground heat map of the input image.
	ReLU is chosen as the nonlinearity for the first layer.
	"""
	def __init__(self, scope, conv_tensor):
		"""
		Args:
			scope: String, name of the network.
			conv_tensor: tf.Tensor, either vgg.conv4_3 or vgg.conv5_3
				with shape [batch_size, 224, 224, 512]
		"""
		self.scope = scope
		self.params = {'wd': 0.05} # L2 regulization coefficien
		
		self.variables = []
		with tf.variable_scope(scope) as scope:
			self.pre_M = self._build_graph(conv_tensors)
			self.gt_M = tf.placeholder(dtype=tf.float32, shape=(None,224,224),name='gt_M')


	def _build_graph(self, conv_tensors):
		"""
		Define network's structure. And returns
		the predicted heatmap which is a [batch_size, 224, 224, 1] tensor.
		"""
		self.variables = []
		self.kernel_weights = []
		sel_num = 512
		
		self.input_maps = conv_tensor        
		with tf.name_scope('conv1') as scope:
			kernel = tf.Variable(tf.truncated_normal([9,9,sel_num,36], dtype=tf.float32,stddev=1e-1), name='weights')

			conv = tf.nn.conv2d(self.input_maps, kernel, [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[36], dtype=tf.float32), name='biases')
			out = tf.nn.bias_add(conv, biases)
			conv1 = tf.nn.relu(out, name=scope)
			self.variables += [kernel, biases]
			self.kernel_weights += [kernel]


		with tf.name_scope('conv2') as scope:
			kernel = tf.Variable(tf.truncated_normal([5,5,36,1], dtype=tf.float32, stddev=1e-1), name='weights')
			conv = tf.nn.conv2d(conv1, kernel , [1, 1, 1, 1], padding='SAME')
			biases = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='biases')
			pre_M = tf.nn.bias_add(conv, biases)
			self.variables += [kernel, biases]
			self.kernel_weights += [kernel]

			# Turn pre_M to a rank2 tensor within range 0-1.
			pre_M = tf.squeeze(pre_M)
			pre_M /= tf.reduce_max(pre_M)
		return pre_M

	def loss(self):
		"""Returns Losses for the current network.
		Returns:
		    Loss: tf.Scalar tensor.
		"""

		with tf.name_scope(self.scope) as scope:
			beta = tf.constant(self.params['wd'], name='beta')
			loss_rms = tf.reduce_max(tf.squared_difference(self.gt_M, self.pre_M))
			loss_wd = [tf.reduce_mean(tf.square(w)) for w in self.kernel_weights]
			loss_wd = beta * tf.add_n(loss_wd)
			total_loss = loss_rms + loss_wd
		return total_loss

	@classmethod
	def eadge_RP():
		"""
		This method propose a series of ROI along eadges
		for a given frame. This should be called when particle 
		confidence below a critical value, which possibly accounts
		for object re-appearance.
		"""
		pass



class GNet(SGNet):
	"""
	A GNet model recieves vgg's conv5_3 layer as
	input tensor. It response for capturing general
	features.
	"""
	__doc__ = SGNet.__doc__ + __doc__
	def __init__(self, scope, conv_tensor):
		"""
		Register Network to current tf.Graph. 
		Args:
			scope: String, name of the network.
			conv_tensor: tf.Tensor, a normalized and resized
				vgg.conv5_3 with shape [batch_size, 224, 224, 512]
		"""
		super(GNet, self).__init__(scope, conv_tensor)



class SNet(SGNet):
	"""
	A SNet model recieves vgg's conv4_3 layer as
	input tensor. It response for capturing specific
	features.
	"""
	__doc__ = SGNet.__doc__ + __doc__
	def __init__(self, scope, conv_tensor):
		"""
		Register Network to current tf.Graph. 
		Args:
			scope: String, name of the network.
			conv_tensor: tf.Tensor, va normalized and resized
				vgg.conv4_3 with shape [batch_size, 224, 224, 512]
		
		"""
		super(SNet, self).__init__(scope, conv_tensor)

	# TODO, check contribution.
	def adaptive_finetune(self, sess, gt_M, fd_s_adp, lr=1e-6):
		"""Finetune SNet with best pre_M predicetd by gNet.
		
		Args:
			best_M: (1,14,14,1) shape array. gnet.pre_M 
		"""
		optimizer = tf.train.GradientDescentOptimizer(lr)
		loss = self.loss(gt_M)
		train_op = optimizer.minimize(loss, var_list=self.variables)
		print('SNet adaptive finetune')
		for step in range(20):
			loss_, _ = sess.run([train_op, loss], feed_dict = feed_dict_s)
			print('loss: ', loss_)

	# TODO, test!
	def descrimtive_finetune(self, sess, conv4_3_t0, sgt_M, conv4_3_t, pre_M_g, phi):
		# Type and shape check!
		# reshape pre_M_g 
		pre_M_g = imresize(pre_M_g[0,:,:,0], [28, 28], interp='bicubic')
		pre_M_g = tf.constant(pre_M_g[np.newaxis,:,:,np.newaxis], dtype=tf.float32)
		sgt_M = tf.constant(sgt_M, dtype=tf.float32)
		

		Loss_t0 = tf.reduce_mean(tf.squared_difference(sgt_M, self.pre_M))
		feed_dict_t0 = {self.input_maps: conv4_3_t0}
		train_op_t0 = SNet.optimizer.minimize(Loss_t0, var_list=self.variables)
		

		Loss_t =  tf.reduce_sum((1-phi) * tf.reduce_mean(tf.squared_difference(pre_M_g, self.pre_M)))
		feed_dict_t = {self.input_maps: conv4_3_t}
		train_op_t = SNet.optimizer.minimize(Loss_t0, var_list=self.variables)
		
		for step in range(20):
			_, loss_0 = sess.run([train_op_t0, Loss_t0], feed_dict_t0)
			_, loss_t = sess.run([train_op_t, Loss_t], feed_dict_t)
			print('Loss 0:', loss_0, 'Loss t:', loss_t)
		
