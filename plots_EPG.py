import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf

from textwrap import wrap

import matplotlib.pylab as pylab

base = 20
args = {'legend.fontsize': base - 4,
        'figure.figsize': (15, 10.5),
        'axes.labelsize': base - 4,
        'axes.titlesize': base,
        'xtick.labelsize': base - 10,
        'ytick.labelsize': base - 10}
pylab.rcParams.update(args)


def save_plots(data_dic, epoch, args, acc=0.5):
	"""
	Save the accuracy and loss plots
	:param data_dic: dict, contains the needed data
	:param epoch: int, No. of current training epoch
	:param args: Param object, contains hyperparams
	:return:
	"""
	if args.test_only:
		plot_learning_curve([data_dic["test_accuracy"]],
		                    hlines=[1.0 / args.num_classes],
		                    xlabel='training epochs/{}'.format(args.test_freq),
		                    ylabel='accuracy', title='Accuracy',
		                    colors=['g'], ylim=[0.0, 1.005],
		                    labels=['validation'],
		                    save_name=args.results_dir + '/{}-learning_curve_epoch_{}_acc_{:.4f}'.format(
			                    args.data_source, epoch, acc))
		
		plot_learning_curve([data_dic["test_loss"]], hlines=[],
		                    xlabel='training epochs/{}'.format(args.test_freq),
		                    ylabel='loss', title='Loss', colors=['limegreen'],
		                    labels=['validation'],
		                    save_name=args.results_dir + '/{}-loss_epoch_{}_acc_{:.4f}'.format(args.data_source, epoch,
		                                                                                       acc))
		plot_confusion_matrix(args,
		                      data_dic["test_confusion"],
		                      normalize=False, postfix=acc)
	else:
		plot_learning_curve([
				data_dic["train_accuracy"],
				data_dic["test_accuracy"]],
				hlines=[1.0 / args.num_classes],
				xlabel='training epochs/{}'.format(args.test_freq),
				ylabel='accuracy', title='Accuracy',
				colors=['darkviolet', 'g'], ylim=[0.0, 1.005],
				labels=['train', 'validation'],
				save_name=args.results_dir + '/{}-learning_curve_epoch_{}'.format(args.data_source, epoch))
		
		plot_learning_curve([
				data_dic["train_loss"],
				data_dic["test_loss"]], hlines=[],
				xlabel='training epochs/{}'.format(args.test_freq),
				ylabel='loss', title='Loss',
				colors=['blueviolet', 'limegreen'],
				labels=['train', 'validation'],
				save_name=args.results_dir + '/loss_epoch_{}'.format(epoch))


def plot_learning_curve(values, hlines=[0.8, 0.85], ylim=[0, 1.05],
                        colors=['mediumslateblue', 'limegreen'],
                        xlabel='training epoch / 2',
                        ylabel='accuracy',
                        title='Loss during training',
                        labels=['accuracy_train'],
                        save_name="loss"):
	'''plot a smooth version of noisy data with mean and std as shadow
	data: a list of variables values, shape: (batches, )
	color: list of prefered colors
	'''
	plt.figure()
	for ind, data in enumerate(values):
		plt.plot(data, '*-', linewidth=2, color=colors[ind], label=labels[ind])
	if hlines is not None:
		for hline in hlines:
			plt.hlines(hline, 0, np.array(data).size, linestyle='--', colors='salmon', linewidth=1.5)
	
	highest = np.array(data).max()
	plt.plot([np.argmax(np.array(data)), 0], [highest, highest], '-.k')
	plt.text(np.argmax(np.array(data)), highest, "%.3f" % (highest), horizontalalignment="center", color="k",
	         size=18)
	
	plt.ylabel(ylabel)
	plt.xlabel(xlabel)
	if ylim is not None:
		plt.ylim(ylim)
	plt.legend(loc="best")
	plt.title(title)
	plt.savefig(save_name + '.png', format="png")
	plt.close()


def plot_confusion_matrix(args, confm, title='Confusion matrix', cmap='Blues', normalize=False,
                          postfix=0.5):  # plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True.
	"""
	print("Confusion matrix: \n", confm)
	# confm = np.reshape(confm, [args.num_classes, args.num_classes])
	plt.figure()
	if normalize:
		cm = (confm * 1.0 / confm.sum(axis=1)[:, np.newaxis]) * 1.0
		# cm = cm.astype('float16') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
			if cm[i, j] != 0:
				plt.text(j, i, np.int(cm[i, j] * 100) / 100.0, horizontalalignment="center", color="orangered",
				         fontsize=base)
	
	else:
		cm = np.array(confm).astype(np.int)
		plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if cm[i, j] != 0:
				plt.text(j, i, np.int(cm[i, j]), horizontalalignment="center", color="orangered",
				         fontsize=base)
		print('Confusion matrix, without normalization')
	plt.title(title)
	tick_marks = np.arange(args.num_classes)
	plt.xticks(tick_marks, args.class_names, rotation=45)
	plt.yticks(tick_marks, args.class_names, rotation=45)
	plt.colorbar()
	
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(os.path.join(args.results_dir, 'certain_uncertain',
	                         '{}-val-confusion-acc-{}.png'.format(args.data_source, postfix)), format='png')
	# np.savetxt(os.path.join(args.results_dir,'certain_uncertain', '{}-val-confusion-matrix-acc-{:.3f}.csv'.format(args.data_source, acc)), np.array(confm), fmt="%d", delimiter=',')
	plt.close()


def plot_train_samples(samples, true_labels, args, postfix="samples"):
	"""plot the trainin samples"""
	plt.figure()
	if args.if_spectrum:
		samples = np.squeeze(samples, axis=1)  # this is size 1 dimension (num_segs)
		for ii in range(9):
			ax1 = plt.subplot(3, 3, ii + 1)
			plt.imshow(samples[ii].T, interpolation="nearest", cmap='viridis', aspect="auto", origin='lower')
			plt.xlabel("label: " + np.str(true_labels[ii]))
			plt.colorbar()
	else:
		samples = np.reshape(samples, [-1, args.secs_per_samp * args.num_segs * args.sr])
		for ii in range(min(20, samples.shape[0])):
			ax1 = plt.subplot(5, 4, ii + 1)
			plt.plot(np.arange(samples.shape[1]) / np.float(args.sr), samples[ii, :])
			plt.xlim([0, samples.shape[1] / np.float(args.sr)])
			plt.xlabel("label: " + np.str(true_labels[ii]))
			# plt.setp(ax1.get_xticklabels(), visible=False)
			# plt.setp(ax1.get_yticklabels(), visible=False)
	plt.tight_layout()
	plt.savefig(args.results_dir + '/samples-{}.png'.format(postfix), format='png')
	plt.close()


def get_class_map(labels, conv_out, weights, height, width):
	"""
	Get the map for a specific sample with class. Some samples can be assigned to different classes.
	:param labels_int: 1d array, [batch_size,], the int labels in one batch
	:param conv_out: list, list of arry that contains the output from the conv_layer from the network
	:param weights: weights of the last GAP feature map
	:param height: weights of the last GAP feature map
	:param im_width: The length of the input sample
	:return: class map: list
	"""
	channels = conv_out.shape[-1]
	classmaps = []
	if len(labels.shape) > 1:
		labels_int = np.argmax(labels, axis=1).astype(np.int32)
	else:
		labels_int = labels.astype(np.int32)
	conv_resized = tf.compat.v1.image.resize_nearest_neighbor(conv_out, [height, width])
	for ind, lb in enumerate(labels_int):
		label_w = tf.gather(tf.transpose(weights), lb)
		label_w = tf.reshape(label_w, [-1, channels, 1])
		resized = tf.cast(tf.reshape(conv_resized[ind], [-1, height * width, channels]), tf.float32)
		classmap = tf.matmul(resized, label_w)
		classmaps.append(tf.reshape(classmap, [-1, height, width]))
	return classmaps


def plot_class_activation_map(sess, class_activation_map,
                              features_test, labels_test,
                              pred_int_test,
                              args, postfix=""):
	"""
	https://github.com/philipperemy/tensorflow-class-activation-mapping
	Plot the class activation
	:param sess:
	:param class_activation_map: list
	:param features_test: 2d array of the features
	:param labels_test: 1d, int labels
	:param pred_int: 2d, output logits
	:param acc:
	:param num_samples:
	:param args:
	:return:
	"""
	num_samples = len(features_test)
	# Save all the CAM for furture plotting, 1000 samples
	classmap_answer = sess.run(class_activation_map)
	colors = pylab.cm.Dark2(np.linspace(0, 1, args.num_classes))
	labels = labels_test
	pred_int = pred_int_test
	
	if args.if_spectrum:
		plots_per_fig = 5
		counts = np.arange(plots_per_fig)
		for class_id in range(args.num_classes):
			need_inds = np.where(labels_test == class_id)[0]
			if len(need_inds) > 0:
				images_plot = features_test[need_inds]
				labels_plot = labels[need_inds]
				pred_plot = pred_int[need_inds]
				
				classmap_high = list(map(lambda x:
				                         ((x - x.min()) / (x.max() - x.min())),
				                         np.array(classmap_answer)[need_inds]))
				classmap_high = np.array(classmap_high)
				for jj in range(1):
					plt.figure()
					for count, vis_h, ori, label, pred_lb in \
							zip(counts,
							    classmap_high[jj * plots_per_fig: (jj + 1) * plots_per_fig],
							    images_plot[jj * plots_per_fig: (jj + 1) * plots_per_fig],
							    labels_plot[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))],
							    pred_plot[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))]):
						
						plt.subplot(2, plots_per_fig, count + 1)
						plt.imshow(ori[0].T, cmap='viridis', interpolation='nearest',
						           aspect='auto', origin='lower')
						plt.colorbar()
						if count == 0:
							plt.ylabel("original")
						plt.subplot(2, plots_per_fig, plots_per_fig + count + 1)
						plt.imshow(vis_h[0].T, cmap='jet', interpolation='nearest',
						           aspect='auto', origin='lower', alpha=0.95)
						plt.colorbar()
						plt.xlim([0, ori[0].shape[0]])
						plt.xlabel("label: {} - pred: {}".format(
								np.str(labels_plot[jj * plots_per_fig + count]),
								np.str(pred_plot[jj * plots_per_fig + count])))
						if count == 0:
							plt.ylabel("attention map")
						if count == plots_per_fig - 1:
							plt.xlabel("label: {} - pred: {}".format(
									np.str(labels_plot[jj * plots_per_fig + count]),
									np.str(pred_plot[jj * plots_per_fig + count])))
					
					plt.tight_layout()
					plt.subplots_adjust(top=0.90)
					plt.savefig(args.results_dir +
					            '/attention_maps/class_activity_map--count{}.png'
					            .format(postfix, jj), format='png')
					plt.close()  #
	
	else:
		row, col = 10, 6
		plots_per_fig = row * col
		counts = np.arange(plots_per_fig)
		for class_id in range(args.num_classes):  # plot examples from the same class in one plot
			need_inds = np.where(labels == class_id)[0]
			if len(need_inds) > 1:
				images_plot = features_test[need_inds]
				labels_plot = labels[need_inds]
				pred_plot = pred_int[need_inds]
				
				classmap_high = list(
						map(lambda x:
						    (np.where(np.squeeze(x) > np.percentile(np.squeeze(x), 80))[0]),
						    np.array(classmap_answer)[need_inds]))
				classmap_high = np.array(classmap_high)
				
				for jj in range(len(need_inds) // (plots_per_fig) + 1):
					fig = plt.figure()
					for count, vis_h, ori, label, pred_lb \
							in zip(counts,
							       classmap_high[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))],
							       images_plot[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))],
							       labels_plot[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))],
							       pred_plot[jj * plots_per_fig: min((jj + 1) * plots_per_fig, len(need_inds))]):
						
						ax = plt.subplot(row, col, count + 1)
						plt.plot(np.arange(ori.size) / np.float(args.sr), ori, color=colors[np.int(pred_lb)])
						
						att_indices = collect_and_plot_atten(vis_h, args.sr,
						                                     np.max(ori))  # collect the attention part indices
						plt.setp(ax.get_yticklabels(), visible=False)
						
						# Plot attention EEG segment in different color
						for ind, xc in enumerate(att_indices):
							if ind % 2 == 0 and len(att_indices) > 1:
								plt.plot(vis_h[att_indices[ind]: att_indices[
									ind + 1]] / np.float(args.sr), ori[vis_h[att_indices[ind]:
								                                             att_indices[
									                                             ind + 1]]],
								         color='lawngreen', linewidth=1)
						
						plt.setp(ax.get_yticklabels(), visible=False)
						
						if count == 0:
							for c in range(args.num_classes):  # just to show all the colors
								plt.plot(np.arange(3) / np.float(args.sr),
								         np.random.randn(3), color=colors[c], label=args.class_names[c])
							plt.plot(np.arange(3) / np.float(args.sr),
							         np.random.randn(3), color='lawngreen', label="attention")
							plt.legend(bbox_to_anchor=(0.1, 1.05, 2, 0.1), frameon=False,
							           ncol=min(5, args.num_classes + 1))
						plt.setp(ax.get_yticklabels(), visible=False)
					fig.text(0.5, 0.025, 'time [s]', fontsize=base, horizontalalignment="center")
					fig.text(0.5, 0.90, 'CAMs for class {}'.format(
							args.class_names[class_id]), fontsize=base, horizontalalignment="center")
					fig.text(0.025, 0.5, 'Normalized amplitude',
					         rotation=90, fontsize=base,
					         verticalalignment='center')
					plt.subplots_adjust(hspace=0, wspace=0., top=0.85, bottom=0.12, left=0.065, right=0.95)
					
					save_name = args.results_dir + '/attention_maps/cam-trueclass_{}-fig{}-{}-{}'.format(class_id, jj,
					                                                                                     postfix,
					                                                                                     args.data_source)
					plt.savefig(save_name + ".png", format='png')
					plt.savefig(save_name + ".pdf", format='pdf')
					plt.close()  #


def plot_cer_uncer_samples(labels_int, pred_int, features, args, acc=0, num_figs=1, postfix="something"):
	"""
	Plot the most certain and uncertain examples
	:param indices: list of indices of interest
	:param features: 2d array, of the whole features
	:param labels_int: 2d array, one-hot encoding of the whole labels
	:return:
	"""
	colors = pylab.cm.Dark2(np.linspace(0, 1, args.num_classes))
	num_classes = args.num_classes
	alphas = [0.5, 1.0]  # when the certain label is wrong alpha=0.5
	class_names = args.class_names
	
	for class_id in range(num_classes):
		row, col = 10, 6
		plot_inds = np.where(labels_int == class_id)[0]
		if len(plot_inds) > 0:
			# features_plot = result_data["test_certain_features"][plot_inds]
			features_plot = features[plot_inds]
			num_figs = min(num_figs, features_plot.shape[0] // (row * col))
			
			if len(labels_int[labels_int == class_id]) != 0:
				for ind in range(num_figs):
					f, axs = plt.subplots(row, col, sharex=True)
					plt.suptitle("{}-Very certain examples of class {}".format(args.data_source, class_names[class_id]),
					             x=0.5, y=0.925, fontsize=base)
					if not args.if_spectrum:
						
						for j in range(ind * row * col, min((ind + 1) * row * col, features_plot.shape[0] - 1)):
							color = colors[class_id] if alphas[np.int(labels_int[j] == pred_int[j])] == 1.0 else "r"
							axs[(j - ind * row * col) // col, np.mod(j, col)].plot(
								np.arange(features_plot.shape[1]) / np.float(args.sr), features_plot[j, :], color=color)
							plt.setp(axs[(j - ind * row * col) // col, np.mod(j, col)].get_yticklabels(), visible=False)
						f.subplots_adjust(hspace=0),
						f.subplots_adjust(wspace=0),
						f.text(0.5, 0.05, 'time [s]', fontsize=base),
						f.text(0.1, 0.5, 'Normalized amplitude',
						       rotation=90, fontsize=base,
						       verticalalignment='center'),
						f.savefig(
								args.results_dir +
								'/certain_uncertain/{}-certain_examples_of_class{}_acc_{:.3f}_fig_{}-{}-{}.png'.format(
									args.data_source, class_id, acc, ind, args.class_mode, postfix))
						f.savefig(
								args.results_dir +
								'/certain_uncertain/{}-certain_examples_of_class{}_acc_{:.3f}_fig_{}-{}-{}.pdf'.format(
									args.data_source, class_id, acc, ind, args.class_mode, postfix), format="pdf")
						plt.close()
					else:
						for j in range(ind * row * col, min((ind + 1) * row * col, features_plot.shape[0] - 1)):
							axs[(j - ind * row * col) // col, np.mod(j, col)].imshow(features_plot[j, ...].T,
							                                                         interpolation="nearest",
							                                                         cmap='viridis', aspect="auto",
							                                                         origin='lower')
							plt.xlabel("label {} - pred {}".format(labels_int[j], pred_int[j]))
						plt.tight_layout()
						f.subplots_adjust(hspace=0)
						f.subplots_adjust(wspace=0.05)
						f.text(0.5, 0.05, 'time [s]', fontsize=base)
						f.text(0.1, 0.5, 'Normalized amplitude',
						       rotation=90, fontsize=base,
						       verticalalignment='center')
						plt.savefig(
							args.results_dir + '/certain_uncertain/Most_certain_samples_acc_{:.3f}_fig_{}-{}.png'.format(
								acc, ind, args.class_mode), format='png')
						plt.savefig(
							args.results_dir + '/certain_uncertain/Most_certain_samples_acc_{:.3f}_fig_{}-{}.pdf'.format(
								acc, ind, args.class_mode), format='pdf')
						plt.close()
	print("Saved most certain examples")


def plot_roc_curve(args, data, acc=0):
	"""
	
	:param args:
	:param data:
	:param acc:
	:return:
	"""
	if args.num_classes == 2:
		plot_bin_roc_curve(args, data, acc=acc)
	elif args.num_classes > 2:
		plot_multiclass_roc(args, data, acc=acc)


def plot_bin_roc_curve(args, data, acc=0.5, postfix="seg-level-AUC"):
	"""
	Plot ROC curve for binary class
	:param args:
	:param data:
	:return:
	"""
	plt.figure(figsize=[10, 8])
	fpr, tpr, _ = metrics.roc_curve(np.argmax(data["test_labels"], 1),
	                                data["test_pred_logits"][:, 1])  # input the positive label's prob distribution
	auc = metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr, label="auc={0:.4f}".format(auc))
	plt.legend(loc=4)
	plt.xlabel("false positive rate")
	plt.ylabel("true positive rate")
	plt.savefig(
		args.results_dir + '/certain_uncertain/{}-AUC-curve-acc-{:.3f}-auc-{:.4f}-{}.png'.format(args.data_source, acc,
		                                                                                         auc, postfix),
		format="png")
	plt.savefig(
		args.results_dir + '/certain_uncertain/{}-AUC-curve-acc-{:.3f}-auc-{:.4f}-{}.eps'.format(args.data_source, acc,
		                                                                                         auc, postfix),
		format="eps")
	plt.close()
	# np.savetxt(args.results_dir + '/certain_uncertain/{}-{}-acc-{:.3f}-auc-{:.4}.csv'.format(args.data_source, postfix, acc, auc), np.array(np.vstack((fpr, tpr))).T, header='fpr,tpr', fmt="%.4f", delimiter=",")
	return auc


def plot_multiclass_roc(args, data, acc=0.5, postfix="seg-level-AUC"):
	"""
	Plot ROC curve for multiclasses
	https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
	A macro-average (independently add up / total number): Pr=(0.5+0.1+0.5+0.5) / 4=0.4
	A micro-average (aggregated ): Pr=(1+10+1+1) / (2+100+2+2)=0.123
	:param args:
	:param data: dict: "test_labels"-2d array, "test_pred_logits"-2d array
	:param acc:
	:return:
	"""
	# Compute ROC curve and ROC area for each class
	fpr = dict()  # class as key is for macro, micro is for micro
	tpr = dict()
	roc_auc = dict()
	for i in range(args.num_classes):
		fpr[i], tpr[i], _ = metrics.roc_curve(
				data["test_labels"][:, i], data["test_pred_logits"][:, i])
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])
	
	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = metrics.roc_curve(data["test_labels"].ravel(), data["test_pred_logits"].ravel())
	roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
	
	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(args.num_classes)]))
	
	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(args.num_classes):
		mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
	
	# Finally average it and compute AUC
	mean_tpr /= args.num_classes
	
	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
	
	# Plot all ROC curves
	plt.figure()
	colors = pylab.cm.Dark2(np.linspace(0, 1, args.num_classes))
	# plt.plot(fpr["micro"], tpr["micro"],
	#          label='micro-average ROC curve (area = {0:0.2f})'
	#                ''.format(roc_auc["micro"]),
	#          color='deeppink', linestyle=':', linewidth=2)
	#
	# plt.plot(fpr["macro"], tpr["macro"],
	#          label='macro-average ROC curve (area = {0:0.2f})'
	#                ''.format(roc_auc["macro"]),
	#          color='navy', linestyle=':', linewidth=2)
	#
	# colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(args.num_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
		         label='class {0} (area = {1:0.2f})'
		               ''.format(args.class_names[i], roc_auc[i]))
	
	plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
	plt.xlim([-0.01, 1.01])
	plt.ylim([-0.01, 1.01])
	plt.xlabel("false positive rate")
	plt.ylabel("true positive rate")
	plt.title('ROC curve for Multiclass')
	plt.legend(loc="lower right")
	plt.savefig(
		args.results_dir + '/certain_uncertain/{}-{}-acc-{:.3f}-auc-{:.4}.png'.format(args.data_source, postfix, acc,
		                                                                              roc_auc["micro"]))
	plt.close()
	
	return roc_auc["micro"]


def factorization(n):
	for i in range(int(np.sqrt(float(n))), 0, -1):
		if n % i == 0:
			if i == 1:
				print('Who would enter a prime number of filters')
			return (i, int(n / i))


def plot_sample_w_recon(data, recon=None, sr=512, num_figs=5, smp_names=None, ifpdf=False, step=0, save_dir="results/",
                        postfix="train"):
	"""
	:param data:
	:param diff:
	:param recon:
	:param num_figs:
	:param step:
	:param save_dir:
	:return:
	"""
	row, col = 6, 4
	rand_inds = np.random.choice(data.shape[0], row * col * num_figs)
	data_plot = data[rand_inds, :]
	recon_plot = recon[rand_inds, :]
	if smp_names is not None:
		names_plot = smp_names[rand_inds]
	colors = ["royalblue", "darkviolet"]
	counts = (np.arange(row * col * num_figs).reshape(-1, col)[np.arange(0, row, 2)]).reshape(-1)
	for ind in range(num_figs):
		f, axs = plt.subplots(row, col, sharex=True)
		plt.suptitle("\n".join(wrap("Samples and Reconstructions\n{}".format(postfix), 60)), x=0.5, y=0.98,
		             fontsize=base)
		
		for j, ori, recon in zip(np.arange(row * col), data_plot[ind * row * col: (ind + 1) * row * col],
		                         recon_plot[ind * row * col: (ind + 1) * row * col]):
			# axs[j // col, np.mod(j, col)].plot(np.arange(ori.size), ori, "m", label='original'),
			# axs[j // col + 1, np.mod(j, col)].plot(np.arange(recon.size), recon, "c", label="reconstruction", linewidth=0.8),
			axs[j // col, np.mod(j, col)].plot(np.arange(ori.size) / sr, ori, "m", label='original')
			axs[j // col, np.mod(j, col)].plot(np.arange(recon.size) / sr, recon, "c", label="reconstruction",
			                                   linewidth=0.8)
			axs[0, 0].legend(bbox_to_anchor=(0.15, 1.05, 2, 0.1), loc="lower left", mode="expand", frameon=False,
			                 borderaxespad=0, ncol=2)
			if np.mod(j, col) > 0:
				# plt.setp(axs[j // col + 1, np.mod(j, col)].get_yticklabels(), visible=False)
				plt.setp(axs[j // col, np.mod(j, col)].get_yticklabels(), visible=False)
		f.text(0.5, 0.02, 'time', fontsize=base),
		f.text(0.02, 0.5, 'Normalized amplitude', rotation=90, verticalalignment='center', fontsize=base),
		f.subplots_adjust(hspace=0, wspace=0)
		file_name = os.path.join(save_dir, '{}_recon_diff_ep{}_{}'.format(postfix, step, ind))
		plt.savefig('{}.png'.format(file_name))
		if smp_names is not None:
			names_this_figure = np.array(names_plot)[ind * row * col: (ind + 1) * row * col]
			fnames = ["{}-{}".format(ii + 1, names_this_figure[ii]) for ii in range(len(names_this_figure))]
			np.savetxt(os.path.join(save_dir, '{}_recon_diff_ep{}_fns_{}x{}.csv'.format(postfix, step, row, col)),
			           np.array(fnames).reshape(-1, 1), fmt="%s", delimiter=",")
		if ifpdf:
			plt.savefig('{}.pdf'.format(file_name), format="pdf")
		plt.close(f)


'''
colormaps:
dict_keys(['Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap', 'GnBu', 'Greens', 'Greys', 'OrRd', 'Oranges', 'PRGn', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy', 'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Spectral', 'Wistia', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone', 'brg', 'bwr', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag', 'gist_earth', 'gist_gray', 'gist_heat', 'gist_ncar', 'gist_rainbow', 'gist_stern', 'gist_yarg', 'gnuplot', 'gnuplot2', 'gray', 'hot', 'hsv', 'jet', 'nipy_spectral', 'ocean', 'pink', 'prism', 'rainbow', 'seismic', 'spring', 'summer', 'terrain', 'winter', 'Accent', 'Dark2', 'Paired', 'Pastel1', 'Pastel2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'Blues_r', 'BrBG_r', 'BuGn_r', 'BuPu_r', 'CMRmap_r', 'GnBu_r', 'Greens_r', 'Greys_r', 'OrRd_r', 'Oranges_r', 'PRGn_r', 'PiYG_r', 'PuBu_r', 'PuBuGn_r', 'PuOr_r', 'PuRd_r', 'Purples_r', 'RdBu_r', 'RdGy_r', 'RdPu_r', 'RdYlBu_r', 'RdYlGn_r', 'Reds_r', 'Spectral_r', 'Wistia_r', 'YlGn_r', 'YlGnBu_r', 'YlOrBr_r', 'YlOrRd_r', 'afmhot_r', 'autumn_r', 'binary_r', 'bone_r', 'brg_r', 'bwr_r', 'cool_r', 'coolwarm_r', 'copper_r', 'cubehelix_r', 'flag_r', 'gist_earth_r', 'gist_gray_r', 'gist_heat_r', 'gist_ncar_r', 'gist_rainbow_r', 'gist_stern_r', 'gist_yarg_r', 'gnuplot_r', 'gnuplot2_r', 'gray_r', 'hot_r', 'hsv_r', 'jet_r', 'nipy_spectral_r', 'ocean_r', 'pink_r', 'prism_r', 'rainbow_r', 'seismic_r', 'spring_r', 'summer_r', 'terrain_r', 'winter_r', 'Accent_r', 'Dark2_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'Set1_r', 'Set2_r', 'Set3_r', 'tab10_r', 'tab20_r', 'tab20b_r', 'tab20c_r', 'magma', 'magma_r', 'inferno', 'inferno_r', 'plasma', 'plasma_r', 'viridis', 'viridis_r', 'cividis', 'cividis_r'])'''
