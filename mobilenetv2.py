"""MobileNet v2 model for TFLearn.
	# Reference
		[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
		(https://arxiv.org/abs/1801.04381)
	_author__ = "Ishrat Badami, badami@nevisq.com"
"""

from tflearn.layers.conv import conv_2d, grouped_conv_2d, global_avg_pool
from tflearn.layers.core import dropout, reshape, input_data
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import batch_normalization


def _conv_block(input_net, filters, kernel, strides):
	"""Convolution Block
	This function defines a 2D convolution operation with BN and relu6.

	Parameters
	----------
		input_net: Tensor, input tensor of convolution layer.
		filters: Integer, the dimensionality of the output space.
		kernel: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		strides: An integer or tuple/list of 2 integers,
			specifying the strides of the convolution along the width and height.
			Can be a single integer to specify the same value for
			all spatial dimensions.
	# Returns
		Output tensor.
	"""

	net = conv_2d(input_net, filters, kernel, strides, activation='relu6', weights_init='xavier')
	net = batch_normalization(net)
	return net


def _bottleneck(input_net, filters, kernel, t, s, r=False):
	"""Bottleneck
	This function defines a basic bottleneck structure.

	Parameters
	----------
		input_net: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		t: Integer, expansion factor.
			t is always applied to the input size.
		s: An integer or tuple/list of 2 integers,specifying the strides
			of the convolution along the width and height.Can be a single
			integer to specify the same value for all spatial dimensions.
		r: Boolean, Whether to use the residuals.
	# Returns
		Output tensor.
	"""
	t_channel = input_net.shape[3] * t  # channel expansion

	net = _conv_block(input_net, t_channel, (1, 1), (1, 1))

	net = grouped_conv_2d(net, channel_multiplier=1, filter_size=kernel, strides=(s, s), padding='same',
							activation='relu6', weights_init='xavier')
	net = batch_normalization(net)

	net = conv_2d(net, filters, (1, 1), strides=(1, 1), padding='same')
	net = batch_normalization(net)

	if r:
		net = merge([net, input_net], 'elemwise_sum')
	return net


def _inverted_residual_block(input_net, filters, kernel, t, strides, n):
	"""Inverted Residual Block
	This function defines a sequence of 1 or more identical layers.

	Parameters
	----------
		input_net: Tensor, input tensor of conv layer.
		filters: Integer, the dimensionality of the output space.
		kernel: An integer or tuple/list of 2 integers, specifying the
			width and height of the 2D convolution window.
		t: Integer, expansion factor.
			t is always applied to the input size.
		s: An integer or tuple/list of 2 integers,specifying the strides
			of the convolution along the width and height.Can be a single
			integer to specify the same value for all spatial dimensions.
		n: Integer, layer repeat times.
	# Returns
		Output tensor.
	"""

	net = _bottleneck(input_net, filters, kernel, t, strides)

	for i in range(1, n):
		net = _bottleneck(net, filters, kernel, t, 1, True)

	return net


def mobile_net_v2(input_shape, n_classes, img_prep=None, img_aug=None):
	"""MobileNetv2
	This function defines a MobileNetv2 architectures.

	Parameters
	----------
		input_shape: An integer or tuple/list of 3 integers, shape
			of input tensor.
		n_classes: Number of classes.
		img_prep: Function handle for image pre-processing
		img_aug: Function handle for image augmentation

	# Returns
		MobileNetv2 model.
	"""
	inputs = input_data(shape=input_shape, data_preprocessing=img_prep, data_augmentation=img_aug)
	x = reshape(inputs, [-1, input_shape[0], input_shape[1], 1])
	x = _conv_block(x, 32, (3, 3), strides=(2, 2))

	x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
	x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
	x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
	x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
	x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
	x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
	x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
	x = global_avg_pool(x)
	x = reshape(x, [-1, 1, 1, 1280])
	x = dropout(x, 0.3, name='Dropout')
	x = conv_2d(x, n_classes, (1, 1), padding='same', activation='softmax', weights_init='xavier')

	output = reshape(x, [-1, n_classes])
	return output
