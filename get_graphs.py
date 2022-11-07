# Keep conv2d for the conveniences of testing on CIFAR images
import numpy as np
import tensorflow as tf
import os
import ipdb

regularizer = tf.keras.regularizers.l2(l=0.01)
# regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
# initializer = tf.glorot_uniform_initializer()
initializer = tf.keras.initializers.he_normal(seed=458)


# initializer = tf.contrib.layers.xavier_initializer()

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization).
	https://www.tensorflow.org/guide/summaries_and_tensorboard"""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


def get_variables_from_graph(layer, *args):
	'''get the variables from the graph for further visualization
	param:
		layer: the output of the layer
		args: keyword arguments
				variable_names: '/kernel:0', '/weights:0', '/bias:0'

	e.g.  all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, layer_scope.name)
		'''
	for arg in args:
		variable_name = os.path.split(layer.name)[0] + '/' + arg + ':0'
		variable = tf.compat.v1.get_default_graph().get_tensor_by_name(variable_name)
		if len(variable.shape.as_list()) == 4:
			# to tf.image_summary format [batch_size, height, width, channels]
			variable = tf.transpose(variable, [3, 0, 1, 2])
			
			tf.summary.image(variable_name, variable)
	
	return variable


def post_process(logits, labels_hot, args):
	"""
	POst process the predictions
	:param logits: 2D array, prob distri from the output of the model
	:param args:
	:return: aggregated prob distribution of all the num_segs
	"""
	if args.num_segs > 1:
		reshape_lb = tf.reshape(logits, [-1, args.num_segs, args.num_classes])
		post_pred_hot = tf.reduce_sum(reshape_lb, axis=1)
		post_pred_int = tf.argmax(post_pred_hot, axis=1)
	else:
		post_pred_hot = logits
		post_pred_int = tf.argmax(post_pred_hot, axis=1)
	
	labels_int = tf.argmax(labels_hot, axis=1)
	
	return post_pred_hot, post_pred_int, labels_int


def get_loss(args, logits, labels_hot):
	"""
	Get loss given loss type
	:param args:
	:param logits: tensor, [batch, num_classes], output of the network
	:param labels_hot: tensor, [batch, num_classes], one_hot encoded labels
	:return: loss
	"""
	loss_type = args.loss_type
	if loss_type == "mse":
		loss = tf.reduce_sum(tf.reduce_mean((logits - labels_hot) ** 2, axis=1))
	if loss_type == "rmse":
		loss = tf.reduce_sum(tf.reduce_mean(tf.abs(logits - labels_hot), axis=1))
	if loss_type == "cross_entropy":
		loss = tf.reduce_sum(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels_hot))
	return loss


def get_num_correct(logits, labels_hot):
	"""
	Get the number of correct predicted labels for future average
	:param logits: tensor, [batch, num_classes], output of the network
	:param labels_hot: tensor, [batch, num_classes], one_hot encoded labels
	:return:
	"""
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels_hot, 1))
	num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
	return num_correct


def get_train_op(args, loss, learning_rate):
	optimizer_type = args.optimizer_name
	# lr = args.learning_rate
	update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
	global_step = tf.compat.v1.train.get_or_create_global_step()
	with tf.control_dependencies(update_ops):
		if optimizer_type == "adam":
			optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate,
			                                             beta1=0.9,
			                                             beta2=0.999,
			                                             epsilon=1e-08).minimize(loss, global_step=global_step)
		if optimizer_type == "rmsprop":
			optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)
		if optimizer_type == "sgd":
			optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss,
			                                                                                global_step=global_step)
	return optimizer


def get_graph(inputs, args):
	"""Model function defining the graph operations.
	params:
		inputs: dict, input tensors (features, labels)
		args: Params contains hyper parameters for the experiment including the model params
		is_train: bool, whether update the weights. True-update during training, False-no update, reuse for evaluation

	return:
		graph: dict, contains the graph operations or nodes needed for training/ testing"""
	
	# reuse = not is_train
	model_aspect = inputs
	if args.test_only:
		features, labels_hot, filenames = model_aspect["test_features"], model_aspect["test_labels"], model_aspect[
			"test_filenames"]
		model_aspect["test_filenames"] = filenames
	else:
		features, labels_hot = model_aspect["test_features"], model_aspect["test_labels"]
	if args.if_spectrum:
		features = tf.squeeze(features, axis=1)  # this is size 1 dimension (num_segs)
	# -----------------construct theb4softmax model--------------------
	outputs = construct_model(features, args, is_train=False)
	
	# -----------------------------------------------------------
	# METRICS AND SUMMARIES
	post_pred_logits, post_pred_int, labels_int = post_process(outputs["logits"], labels_hot, args)
	model_aspect["test_pred_int"] = post_pred_int
	model_aspect["test_pred_logits"] = post_pred_logits
	model_aspect["test_loss"] = get_loss(args, post_pred_logits, labels_hot)
	model_aspect["test_num_correct"] = get_num_correct(post_pred_logits, labels_hot)
	model_aspect["test_confusion"] = tf.compat.v1.confusion_matrix(labels_int, post_pred_int,
	                                                               num_classes=args.num_classes, name='confusion')
	model_aspect["test_batch_size"] = tf.shape(model_aspect["test_pred_int"])[0]
	if "cam" in args.model_name:
		model_aspect["test_conv"] = outputs["conv"]
		model_aspect["test_gap_w"] = outputs["gap_w"]
	model_aspect["test_b4softmax"] = outputs["b4softmax"]
	model_aspect["total_trainables"] = outputs["total_trainables"]
	
	if not args.test_only:
		features, labels_hot = model_aspect["train_features"], model_aspect["train_labels"]
		if args.if_spectrum:
			features = tf.squeeze(features, axis=1)  # this is size 1 dimension (num_segs)
		outputs = construct_model(features, args, is_train=True)
		post_pred_logits, post_pred_int, labels_int = post_process(outputs["logits"], labels_hot, args)
		if "cam" in args.model_name:
			model_aspect["train_conv"] = outputs["conv"]
			model_aspect["train_gap_w"] = outputs["gap_w"]
			model_aspect["train_kernels"] = outputs["kernels"]
		model_aspect["train_b4softmax"] = outputs["b4softmax"]
		model_aspect["train_pred_int"] = post_pred_int
		model_aspect["train_pred_logits"] = post_pred_logits
		model_aspect["train_loss"] = get_loss(args, post_pred_logits, labels_hot)
		model_aspect["train_lr_op"] = tf.compat.v1.placeholder(tf.float32, [], name='learning_rate')
		model_aspect["train_op"] = get_train_op(args, model_aspect["train_loss"], model_aspect["train_lr_op"])
		# model_aspect['train_op'] = get_train_op(args, model_aspect["train_loss"] )
		model_aspect["train_num_correct"] = get_num_correct(post_pred_logits, labels_hot)
		model_aspect["train_confusion"] = tf.compat.v1.confusion_matrix(labels_int, post_pred_int,
		                                                                num_classes=args.num_classes, name='confusion')
		model_aspect["train_batch_size"] = tf.shape(model_aspect["train_pred_logits"])[0]
	
	return model_aspect


def construct_model(features, args, is_train=False):
	"""Compute logits of the model (output distribution)
	Args:
		inputs: (dict) contains the inputs of the graph (features, labels...)
				this can be `tf.placeholder` or outputs of `tf.data`
		args: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
	Returns:
		output: (tf.Tensor) output of the model"""
	
	if args.model_name == "cnn_resi_ecg":
		outputs = construct_cnn_resi_ecg(features, args, is_train=is_train)
	elif args.model_name == "cnn_resi_cam_ecg":
		outputs = construct_cnn_resi_cam_ecg(features, args, is_train=is_train)
	elif args.model_name == "fnn":
		outputs = construct_fnn(features, args, is_train=is_train)
	else:
		print("No corresponsing construct function is foind!")
	
	return outputs


def single_cnn(x, out_channel, filter_size, pool_size, strides,
               drop, activity, is_train=True, layer_id='0',
               if_maxpool=True):
	"""
	Construct single cnn layer
	:param x: input to this layer
	:param out_channel: num_filters to use
	:param filter_size: the filter size
	:param pool_size: the pooling window
	:param drop: the rate to drop
	:param is_train: boolean, whether is training
	:param layer_id: layer id to variable scope
	:return:
	"""
	with tf.compat.v1.variable_scope('conv{}'.format(layer_id), reuse=tf.compat.v1.AUTO_REUSE) as scope:
		net = tf.compat.v1.layers.conv2d(
				inputs=x,
				filters=out_channel,
				kernel_size=filter_size,
				padding='SAME',
				kernel_regularizer=regularizer,
				kernel_initializer=initializer,
				activation=None
		)
		net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
		net = tf.nn.relu(net)
		activity['conv{}'.format(layer_id)] = net
		if if_maxpool:
			net = tf.compat.v1.layers.max_pooling2d(net, pool_size=pool_size, strides=strides, padding='SAME')
		net = tf.compat.v1.layers.dropout(net, rate=drop, training=is_train)
		print(scope.name + "shape", net.shape.as_list())
	return net, activity


def highway_block_cnn(x, filter_size=[9, 1], out_channels=[8], block_id=0):
	"""
	highway CNN block
	https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32
	:param x: batch*seq_len*width*channels
	:param filter_size: [height, width], kernel size to use in CNN
	:param output_channels: list, with all the output channels used in one block.
	:param block_id: int, the id of the res block
	:return:
	"""
	assert len(x.shape) > 2, (
			'Should input image-like shape')  ## to do conv using batch_size * height * width * channel
	
	with tf.compat.v1.variable_scope("highway_block" + str(block_id), reuse=tf.compat.v1.AUTO_REUSE):
		H = tf.compat.v1.layers.conv2d(
				inputs=x,
				filters=out_channels,
				kernel_size=filter_size,
				padding='same',
				activation=tf.nn.relu)
		T = tf.compat.v1.layers.conv2d(
				inputs=x,
				filters=out_channels,
				kernel_size=filter_size,
				# We initialize with a negative bias to push the network to use the skip connection
				padding='same',
				biases_initializer=tf.constant_initializer(-1.0),
				activation=tf.nn.sigmoid)
		# output = tf.add(tf.multiply(H, T), tf.multiply(x, 1 - T), name='y')
		output = H * T + x * (1.0 - T)
		return output


def build_res_block(x, out_channel, filter_size, num_layers, activity, layer_id=0, is_train=True):
	"""
	https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
	:param out_channels: int, the filters to use in this block
	:param filter_size: [height, width], the kernel size
	:param num_layers: int, how many cov layers in one resi block. inp--> (conv -->...--> conv) -->+inp-->
	:param layer_id: int, the layer id
	:return:
	"""
	net = x
	with tf.compat.v1.variable_scope("res_block" + str(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
		for layer in range(num_layers):
			net = tf.compat.v1.layers.conv2d(
					inputs=net,
					filters=out_channel,
					kernel_size=filter_size,
					padding='SAME',
					kernel_initializer=initializer,
					# kernel_regularizer=regularizer,
					activation=None
			)
			net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
			# net = tf.nn.sigmoid(net)
			net = tf.nn.relu(net)
			activity['res_block{}'.format(layer_id)] = net
		shortcut = tf.compat.v1.layers.conv2d(
				inputs=x,
				filters=out_channel,
				kernel_size=filter_size,
				padding='SAME',
				kernel_initializer=initializer,
				# kernel_regularizer=regularizer,
				activation=None
		)
		shortcut = tf.compat.v1.layers.batch_normalization(shortcut, training=is_train)
		# output = tf.nn.sigmoid(shortcut + net)
		output = tf.nn.relu(shortcut + net)
		print("ResiBlock{}-output pooling shape".format(layer_id), net.shape.as_list())
		return output


def build_res_block_ecg_1st(x, out_channel, filter_size, pool_size, stride, activity, drop=0.2, layer_id='0',
                            is_train=True, if_skip=True):
	"""
	https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
	:param out_channels: int, the filters to use in this block
	:param filter_size: [height, width], the kernel size
	:param num_layers: int, how many cov layers in one resi block. inp--> (conv -->...--> conv) -->+inp-->
	:param layer_id: int, the layer id
	:return: Conv bn relu drop conv
	"""
	net = x
	
	with tf.compat.v1.variable_scope("res_block" + str(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
		net = tf.compat.v1.layers.conv2d(
				inputs=net,
				filters=out_channel,
				kernel_size=filter_size,
				strides=[stride, 1],
				padding='SAME',
				kernel_initializer=initializer,
				kernel_regularizer=regularizer,
				activation=None
		)
		# net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
		net = tf.nn.relu(net)
		activity["res_block{}_conv1".format(str(layer_id))] = net
		net = tf.compat.v1.layers.dropout(net, drop, training=is_train)
		net = tf.compat.v1.layers.conv2d(
				inputs=net,
				filters=out_channel,
				kernel_size=filter_size,
				padding='SAME',
				kernel_initializer=initializer,
				kernel_regularizer=regularizer,
				activation=None
		)
		if if_skip:
			shortcut = tf.compat.v1.layers.max_pooling2d(x, pool_size=pool_size,
			                                             strides=stride,
			                                             padding='same')
			output = tf.nn.relu(shortcut + net)
		else:
			output = tf.nn.relu(net)
		activity["res_block{}_conv2".format(str(layer_id))] = net
		print("ResiBlock{}-pooling".format(layer_id), net.shape.as_list())
		return output, activity


def build_res_block_ecg(x, out_channel, filter_size, pool_size, stride, activity, drop=0.2, layer_id=0, is_train=True,
                        if_skip=True):
	"""
	https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
	:param out_channels: int, the filters to use in this block
	:param filter_size: [height, width], the kernel size
	:param num_layers: int, how many cov layers in one resi block. inp--> (conv -->...--> conv) -->+inp-->
	:param layer_id: int, the layer id
	:return: bn relu  conv bn relu drop conv
	"""
	net = x
	
	if layer_id % 4 == 0 and layer_id > 0:  # only every 4 blocks increase the number of channels
		zeros_x = tf.zeros_like(x)
		# concat_long = tf.concat([x, zeros_x], axis=1)
		# zeros2_x = tf.zeros_like(concat_long)
		concat_long_ch = tf.concat([x, zeros_x], axis=3)
		x = concat_long_ch
	
	# if stride == 2:
	#     temp = tf.zeros_like(x)
	#     concat_long = tf.concat([x, temp], axis=1)
	#     x = concat_long
	
	with tf.compat.v1.variable_scope("res_block" + str(layer_id), reuse=tf.compat.v1.AUTO_REUSE):
		for j in range(2):  # there are two conv layers in one block
			if not (layer_id == 0 and j == 0):
				net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
				net = tf.nn.relu(net)
				drop = drop if j > 0 else 0
				net = tf.compat.v1.layers.dropout(net, drop, training=is_train)
			
			net = tf.compat.v1.layers.conv2d(
					inputs=net,
					filters=out_channel,
					kernel_size=filter_size,
					padding='SAME',
					strides=[stride, 1] if j == 0 else [1, 1],
					kernel_initializer=initializer,
					kernel_regularizer=regularizer,
					activation=None
			)
			activity["res_block{}_conv{}".format(str(layer_id), j)] = net
		if if_skip:
			shortcut = tf.compat.v1.layers.max_pooling2d(x, pool_size=[stride, 1], strides=[stride, 1], padding='same')
			output = tf.nn.relu(shortcut + net)
			activity["res_block{}_comb".format(str(layer_id))] = net
		else:
			output = tf.nn.relu(net)
		print("ResiBlock{}-pooling".format(layer_id), net.shape.as_list())
		return output, activity


def construct_cnn_resi(features, args, iffusion=True, is_train=False):
	"""construct the CNN with residual connections given params"""
	# x = tf.reshape(features, [-1, args.height, args.width, args.channels])  ###
	if len(features.get_shape().as_list()) < 3:
		net = tf.reshape(features, [-1, args.height, args.width, args.channels])
	else:
		net = tf.expand_dims(features, axis=3)
	outputs = {}
	activities = {}
	reuse = not is_train
	layer_ids = np.arange(len(args.out_channels))
	for layer_id, out_channel, filter_size, drop in zip(layer_ids, args.out_channels, args.filter_sizes,
	                                                    args.drop_rates):
		net = build_res_block(net, out_channel,
		                      filter_size,
		                      args.num_layers_in_res,
		                      layer_id=layer_id, is_train=is_train)
		net = tf.compat.v1.layers.max_pooling2d(net, pool_size=args.pool_size,
		                                        strides=args.strides, padding='SAME')
		net = tf.compat.v1.layers.dropout(net, rate=drop, training=is_train)
		print("ResiBlock{} pooling shape".format(layer_id), net.shape.as_list())
	
	# net = tf.compat.v1.layers.average_pooling2d(net, [2, 1], [2, 1], padding='same')
	print("Average pooling shape", net.shape.as_list())
	with tf.compat.v1.variable_scope('fully_connected', reuse=tf.compat.v1.AUTO_REUSE) as scope:
		net = tf.reshape(net, [-1, net.shape[1] * net.shape[2] * net.shape[3]])
		print(scope.name + "shape", net.shape.as_list())
		for unit, drop in zip(args.fc, args.fc_drop_rates):
			net = tf.compat.v1.layers.dense(inputs=net,
			                                units=unit,
			                                kernel_regularizer=regularizer,
			                                kernel_initializer=initializer,
			                                activation=None)
			net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
			net = tf.nn.leaky_relu(net)
			net = tf.compat.v1.layers.dropout(net, rate=drop, training=is_train)
			print(scope.name + "shape", net.shape.as_list())
	outputs["b4softmax"] = net
	kernels = {}  #### implement attention
	logits = tf.compat.v1.layers.dense(
			inputs=net,
			units=args.num_classes,
			activation=tf.nn.softmax,
			kernel_regularizer=regularizer,
			name=scope.name, reuse=tf.compat.v1.AUTO_REUSE)
	
	##### track all variables
	all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
	
	for var in all_trainable_vars:
		if 'kernel' in var.name:
			kernels[var.name] = var
	outputs["logits"] = logits
	outputs["kernels"] = kernels
	outputs["activities"] = activities
	
	return outputs


def construct_cnn_resi_ecg(features, args, is_train=False):
	"""construct the CNN with residual connections given params like in
	https://www.nature.com/articles/s41591-018-0268-3.pdf
	"""
	# x = tf.reshape(features, [-1, args.height, args.width, args.channels])  ###
	if len(features.get_shape().as_list()) < 3:
		net = tf.reshape(features, [-1, args.height, args.width, args.channels])
	else:
		net = tf.expand_dims(features, axis=3)
	outputs = {}
	activities = {}
	reuse = not is_train
	channel_start = args.out_channel  # start with this number and increase 16*2^k (k=0 and increase 1 every residual blocks)
	out_channel = channel_start
	k = 0
	strides = [1 if i % 2 == 0 else 2 for i in range(args.num_res_blocks)]
	layer_ids = np.arange(args.num_res_blocks)
	
	# start CNN
	net, activities = single_cnn(net, channel_start,
	                             args.filter_size, args.pool_size,
	                             args.stride, 0, activities, is_train=is_train,
	                             layer_id="start", if_maxpool=False)
	
	net, activities = build_res_block_ecg_1st(net, args.out_channel,
	                                          args.filter_size, args.pool_size,
	                                          1, activities, drop=0.2, layer_id="00",
	                                          is_train=is_train)
	
	for layer_id, stride in zip(layer_ids, strides):
		if layer_id % 4 == 0 and layer_id > 0:
			k += 1
			out_channel = channel_start * np.power(2, k)
		
		net, activities = build_res_block_ecg(net, out_channel,
		                                      args.filter_size, args.pool_size,
		                                      stride, activities, drop=args.drop_rate,
		                                      is_train=is_train, layer_id=layer_id)
	
	with tf.compat.v1.variable_scope('fully_connected', reuse=tf.compat.v1.AUTO_REUSE) as scope:
		net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
		net = tf.nn.relu(net)
		net = tf.reshape(net, [-1, net.shape[1] * net.shape[2] * net.shape[3]])
	
	kernels = {}
	outputs["b4softmax"] = net
	logits = tf.compat.v1.layers.dense(
			inputs=net,
			units=args.num_classes,
			activation=tf.nn.softmax,
			kernel_regularizer=regularizer,
			name=scope.name, reuse=tf.compat.v1.AUTO_REUSE)
	
	##### track all variables
	all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
	outputs["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
	for var in all_trainable_vars:
		if 'kernel' in var.name:
			kernels[var.name] = var
	outputs["logits"] = logits
	outputs["kernels"] = kernels
	
	return outputs


def construct_cnn_resi_cam_ecg(features, args, is_train=False):
	"""construct the CNN with residual connections given params like in
	https://www.nature.com/articles/s41591-018-0268-3.pdf
	total trainables = 4200048
	"""
	net = tf.reshape(features, [-1, args.height, args.width, args.channels])  ###
	outputs = {}
	activities = {}
	channel_start = args.out_channel  # start with this number and increase 16*2^k (k=0 and increase 1 every residual blocks)
	out_channel = channel_start
	k = 0
	strides = [1 if i % 2 == 0 else 2 for i in range(args.num_res_blocks)]
	layer_ids = np.arange(args.num_res_blocks)
	
	# start CNN
	net, activities = single_cnn(net, channel_start,
	                             args.filter_size, args.pool_size,
	                             args.stride, 0, activities, is_train=is_train,
	                             layer_id="start", if_maxpool=False)
	
	net, activities = build_res_block_ecg_1st(net, args.out_channel,
	                                          args.filter_size, args.pool_size,
	                                          1, activities, drop=0.2, layer_id="00",
	                                          is_train=is_train)
	
	for layer_id, stride in zip(layer_ids, strides):
		if layer_id % 4 == 0 and layer_id > 0:
			k += 1
			out_channel = channel_start * np.power(2, k)
		
		net, activities = build_res_block_ecg(net, out_channel,
		                                      args.filter_size, args.pool_size,
		                                      stride, activities, drop=args.drop_rate,
		                                      is_train=is_train, layer_id=layer_id)
	
	outputs["b4softmax"] = net
	# GAP layer - global average pooling
	with tf.compat.v1.variable_scope('GAP', reuse=tf.compat.v1.AUTO_REUSE) as scope:
		net_gap = tf.reduce_mean(net, (1, 2))  # get the mean of axis 1 and 2 resulting in shape [batch_size, filters]
		print("gap shape", net_gap.shape.as_list())
		
		if args.class_mode == "regression":
			gap_w = tf.compat.v1.get_variable('W_gap',
			                                  shape=[net_gap.shape[-1],
			                                         1],
			                                  initializer=tf.random_normal_initializer(
					                                  0., 0.01))
		else:
			gap_w = tf.compat.v1.get_variable('W_gap',
			                                  shape=[net_gap.shape[-1], args.num_classes],
			                                  initializer=tf.random_normal_initializer(0., 0.01))
		logits = tf.nn.softmax(tf.matmul(net_gap, gap_w))
	
	kernels = {}  #### implement attention
	##### track all variables
	all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
	outputs["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
	print("total_trainables", outputs["total_trainables"])
	
	for var in all_trainable_vars:
		if 'kernel' in var.name:
			kernels[var.name] = var
	outputs["logits"] = logits
	outputs["kernels"] = kernels
	outputs["activities"] = activities
	outputs["conv"] = net
	outputs["gap_w"] = gap_w
	
	return outputs


def construct_fnn(x, args, is_train=True):
	"""
	COnstruct a fnn given parameters
	:param x:
	:param args:
	:param num_classes:
	:param is_train:
	:return:
	"""
	net = tf.compat.v1.layers.flatten(x)
	print("Input shape", net.shape.as_list())
	outputs = {}
	for layer_id, num_outputs in zip(np.arange(len(args.layer_dims)), args.layer_dims):  ## avoid the code repetation
		with tf.compat.v1.variable_scope('fc_{}'.format(layer_id), reuse=tf.compat.v1.AUTO_REUSE) as scope:
			net = tf.compat.v1.layers.dense(
					net,
					num_outputs,
					kernel_initializer=initializer,
					activation=tf.nn.leaky_relu,
					kernel_regularizer=regularizer
			)
			net = tf.compat.v1.layers.batch_normalization(net, training=is_train)
			net = tf.compat.v1.layers.dropout(inputs=net, rate=args.droprate, training=is_train)
			print(scope.name + "shape", net.shape.as_list())
	outputs["b4softmax"] = net
	
	with tf.compat.v1.variable_scope('fc_out', reuse=tf.compat.v1.AUTO_REUSE) as scope:
		if args.class_mode == "regression":
			logits = tf.compat.v1.layers.dense(
					net,
					1,  # regression on the remaining days until the end of EPG
					kernel_initializer=initializer,
					activation=tf.nn.softmax,
					kernel_regularizer=regularizer
			)
		else:
			logits = tf.compat.v1.layers.dense(
					net,
					args.num_classes,
					kernel_initializer=initializer,
					activation=tf.nn.softmax,
					kernel_regularizer=regularizer
			)
	
	kernels = {}  #### implement attention
	##### track all variables
	all_trainable_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
	outputs["total_trainables"] = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in all_trainable_vars])
	print("total_trainables", outputs["total_trainables"])
	outputs["logits"] = logits
	return outputs
