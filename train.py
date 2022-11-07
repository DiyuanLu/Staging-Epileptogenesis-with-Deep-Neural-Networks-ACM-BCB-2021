import logging
import time
import pickle

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from dataio_EPG import check_if_save_model, load_model, save_data_to_csvs
import plots_EPG as Plot


def initialize_ops(sess, init_ops):
    """
    Initialize variables used in training or testing
    :param sess: tf.Session()
    :param init_ops: tensors or ops need to be intialized
    :return:
    """
    sess.run(init_ops)


def condition(end, results, epoch, number_of_epochs):
    """
    Condition to end the training. if epochs is finished or the accuracy decrease
     for 5 consecutive training intervals, then perform early-stopping
    :param end:
    :param results: results of the training
    :param epoch: current epoch
    :param number_of_epochs: total training epochs
    :return: False if the termination condition is fulfilled
    """
    if end:
        return False
    if epoch > number_of_epochs:
        return False
    if len(results["test_accuracy"]) < 1 or number_of_epochs != -1:  # just start training, keep training
        return True
    else:
        best_accuracy = max(results["test_accuracy"])
        c = (np.array(results["test_accuracy"])[-5:] < best_accuracy)
        if c:
            logging.info("Termination condition fulfilled")
        return not c


def reduce_data_mean(ret, key="loss"):
    if len(ret) > 1:
        n = sum([b["batch_size"] for b in ret])
        mean_out = sum([b[key] for b in ret]) / n
    else:
        mean_out = ret[key] / ret["batch_size"]
    return mean_out


def get_epoch_confusion(ret, key='confusion'):
    confusion = sum([b[key] for b in ret])
    return confusion


def concat_data(ret, key='labels'):
    """
    Concat the corresponding data
    :param ret: dict
    :param key:
    :return:
    """
    if len(ret[0][key].shape) >= 2:
        shape = np.array(ret[0][key].shape)
        shape[0] = 0
        concat = np.empty((shape))
        for b in ret:
            concat = np.vstack((concat, b[key]))
    else:
        concat = np.empty(0)
        for b in ret:
            concat = np.append(concat, b[key])
    return np.array(concat)


def reduce_mean_loss_accuracy(ret):
    n = sum([b["batch_sizes"] for b in ret])
    loss = sum([b["loss_sum"] for b in ret]) / n
    accuracy = sum([b["ncorrect"] for b in ret]) / n
    return loss, accuracy


# Processes the output of compute (cf See also) to calculate the sum of confusion matrices
# @param ret dictionary containing the keys "ncorrect", "loss_sum" and "batch_size"
# @param N number of examples computed by compute
# @see compute
def sum_confusion(ret):
    """
    Compute the confusion matrix for one epoch
    :param ret: dict, with the confu matrix for all batches
    :return: 2d array, (num_classes, num_classes)
    """
    confusion = sum([b["confusion"] for b in ret])
    return confusion


def compute(sess, fetches, compute_batches=100, lr=0.0005, if_get_certain=False):
    """
    Compute the interested tensors and ops
    :param sess:
    :param fetches:
    :param compute_batches:
    :param lr:
    :param if_get_wrong:
    :param if_get_certain:
    :return:
    """
    
    results = {key: 0 for key, _ in fetches.items() if key != "train_op"}
    sum_keys = ["loss", "num_correct", "confusion", "batch_size"]

    if_check_cam = False
    if "conv" in fetches.keys():
        if_check_cam = True
        exp_keys = ["labels", "features", "pred_int",
                    "pred_logits", "conv", "gap_w"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels_int",
                         "certain_pred_int", "certain_conv"]
    else:
        exp_keys = ["labels", "features", "pred_int",
                    "pred_logits"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels_int",
                         "certain_pred_int"]

    for key in exp_keys:
        results[key] = []
        
    example_batches = np.random.choice(compute_batches,
                                       min(compute_batches, 20),
                                       replace=False)  # for randomly picking out samples for visualization

    for i in tqdm(range(compute_batches)):
        if "train_op" in fetches.keys():
            run_all = sess.run(fetches, feed_dict={fetches["lr_op"]: lr})
        else:
            run_all = sess.run(fetches)

        for _, key in enumerate(run_all.keys()):
            # Sum over all the sumable variables
            if key in sum_keys:
                if np.isnan(run_all[key]).any():
                    logging.info("{}-th batch, {} contains NaN".format(i, key))
                else:
                    results[key] = results[key]+run_all[key]
            # only take the last batch example variables for further plotting
            elif key in exp_keys:
                if np.isnan(run_all[key]).any():
                    logging.info("{}-th batch, {} contains NaN".format(i, key))
                else:
                    results[key] = run_all[key]

    return results


def check_nan(ret):
    """
    CHeck whether there is nan in values
    :param ret:
    :return:
    """
    for j, b in enumerate(ret):
        for i, key in enumerate(b.keys()):
            if key != "train_op":
                if np.isnan(b[key]):
                    logging.info("{}-th, {} contains NaN".format(j, key))


def compute_test_only(sess, fetches, args, compute_batches=100, if_get_certain=False):
    """
    Compute the interested tensors and ops
    :param sess:
    :param fetches: all interested tensors
    :param compute_batches: compute interested tensors for this number of batches
    :return:
    """
    collections = {}
    
    results = {key: 0 for key, _ in fetches.items()}
    sum_keys = ["loss", "num_correct", "confusion", "batch_size"]
    concat_keys = ["filenames", "pred_int", "pred_logits", "labels"]
    
    results.update({key: 0 for key in sum_keys})
    results.update({key: [] for key in concat_keys})
    
    if "conv" in fetches.keys():
        exp_keys = ["labels", "features", "pred_int",
                "pred_logits", "conv", "gap_w"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels_int",
                         "certain_pred_int", "certain_conv"]
    else:
        exp_keys = ["labels", "features", "pred_int",
                    "pred_logits"]
        if if_get_certain:
            exp_keys += ["certain_features", "certain_labels_int",
                         "certain_pred_int"]
            
    for key in exp_keys:
        collections[key] = []

    example_batches = np.random.choice(compute_batches,
                                       min(compute_batches, 20),
                                       replace=False)
       
    total_counts = 0
    for batch in tqdm(range(compute_batches)):
        ret = sess.run(fetches) # run all tensors
        # run_all.append(ret)  # run all tensors
        total_counts += ret["batch_size"]
        
        for _, key in enumerate(ret.keys()):
            # Sum over all the sumable variables
            if key in sum_keys:
                results[key] = results[key] + ret[key]
            elif key in concat_keys:
                if isinstance(ret[key][0], bytes):
                    ret[key] = [ele.decode("utf-8") for ele in ret[key]]
                    results[key] += list(ret[key])
                else:
                    results[key] += list(ret[key])
            elif key in exp_keys:
                results[key] = ret[key]
        
    logging.info("Saved data info collection")

    return results


def get_learning_rate(epoch):
    """
    Get the learning rate given epoch
    :param epoch:
    :return:
    """
    learning_rate = 0.001
    if epoch > 150:
        learning_rate *= np.power(0.5, 7)
    elif epoch > 120:
        learning_rate *= np.power(0.5, 6)
    elif epoch > 100:
        learning_rate *= np.power(0.5, 5)
    elif epoch > 80:
        learning_rate *= np.power(0.5, 4)
    elif epoch > 60:
        learning_rate *= np.power(0.5, 3)
    elif epoch > 40:
        learning_rate *= np.power(0.5, 2)
    elif epoch > 20:
        learning_rate *= 0.5
    return learning_rate


def get_batch_size(epoch):
    """
    Get the learning rate given epoch
    :param epoch:
    :return:
    """
    batch_size = 8
    if epoch > 512:
        batch_size *= np.power(2, 7)
    elif epoch > 256:
        batch_size *= np.power(2, 6)
    elif epoch > 128:
        batch_size *= np.power(2, 5)
    elif epoch > 64:
        batch_size *= np.power(2, 4)
    elif epoch > 32:
        batch_size *= 4
    return batch_size


def get_fetches(model_aspect, names, train_or_test='test'):
    """
    Get fetches given key-words
    :param model_aspect: with all the train and test attributes
    :param names: the short key word from the attributes
    :param train_or_test: str, indicate which phase it is in
    :return: fetches, dict
    """
    fetches = {}
    for key in names:
        if key == 'train_op':
            fetches[key] = model_aspect[key]
        else:
            fetches[key] = model_aspect["{}_".format(train_or_test)+key]
    return fetches


def get_returns(results, names, train_or_test='test'):
    """
    Get fetches given key-words
    :param results: dict, with all the train and test attributes
    :param names: the short key word from the attributes
    :param train_or_test: str, indicate which phase it is in
    :return: fetches, dict
    """
    ret = {}
    for k in names:
        if k == 'accuracy':
            ret["{}_accuracy".format(train_or_test)] = results["num_correct"] / results["batch_size"]
        elif k == 'loss':
            ret["{}_loss".format(train_or_test)] = results["loss"] / results["batch_size"]
        else:
            ret["{}_".format(train_or_test)+k] = results[k]
    return ret


def reduce_lr_on_plateu(lr, acc_history, factor=0.1, patience=4,
                        epsilon=1e-02, min_lr=10e-8):
    """
    Reduce learning rate by factor when it didn't increase for patience number of epochs
    :param lr:, float, the learing rate
    :param acc_history:lr, float, the learing rate
    :param factor: float, new_lr = lr * factor
    :param patience: number of epoch that can tolerant with no increase
    :param epsilon: only focus on significant changes
    :param min_lr: lower bound on the learning rate.
    :return:
    """
    # if there are patience epochs with a decreasing accuracy and the decrease is bigger than epsilon, then reduce
    if np.sum((acc_history[1:] - acc_history[0:-1]) <= 0) >= patience \
            and np.abs(np.mean((acc_history[1:] - acc_history[0:-1]))) > epsilon:
        if lr > min_lr:
            new_lr = lr * factor
            new_lr = max(new_lr, min_lr)
        else:
            new_lr = lr
    else:
        new_lr = lr
    return new_lr


def get_cam_examples(results, max_num=60):
    """
    Get CAM examples for further plot
    :param results: dict, with keys "wrong_BL", "wrong_EPG"
    :param max_num: ind,
    :return: features, labels, conv, and pred_logits of CAM examples
    
    class_maps = Plot.get_class_map(
                    result_data["test_labels"][rand_ind].astype(np.int),
                    result_data["test_conv"][rand_ind].astype(np.float32),
                    result_data["test_gap_w"],
                    args.height, args.width)
    Plot.plot_class_activation_map(
        sess, class_maps,
        result_data["test_features"][rand_ind],
        result_data["test_labels"][rand_ind],
        result_data["test_pred_int"][rand_ind],
        "only_test", result_data["test_accuracy"], args)
        
        
    """
    labels = results["labels"]
    pred_int = results["pred_int"]
    conv = results["conv"]
    features = results["features"]
    
    cam_inds = np.random.choice(pred_int.size, min(30, len(labels)), replace=False)
    data_len = features.shape[-1]
    conv_shape = conv.shape
    num_classes = labels.shape[-1]

    if len(results["cam_features"]) == 0:
        results["cam_features"] = np.empty((0, data_len))
        results["cam_conv"] = np.empty(conv_shape)
        results["cam_labels"] = np.empty((0, num_classes))
        results["cam_pred_int"] = np.empty(0)
        results["cam_features"] = np.vstack(
            (results["cam_features"], features[cam_inds]))
        results["cam_conv"] = np.vstack(
            (results["cam_conv"], conv[cam_inds]))
        results["cam_labels"] = np.vstack(
            (results["cam_labels"], labels[cam_inds])).astype(np.int)
        results["cam_pred_int"] = np.append(results["cam_pred_int"], pred_int[cam_inds]).astype(np.int)
    else:
        results["cam_features"] = np.vstack(
            (results["cam_features"], features[cam_inds]))
        results["cam_conv"] = np.vstack(
            (results["cam_conv"], conv[cam_inds]))
        results["cam_labels"] = np.vstack(
            (results["cam_labels"], labels[cam_inds])).astype(np.int)
        results["cam_pred_int"] = np.append(results["cam_pred_int"], pred_int[cam_inds]).astype(np.int)

    return results
#

def training(sess, model_aspect, args):
    """
    The whole training process. Train_sess on test_every batches and test. Collect the performance
    :param sess: tf.Session, current session
    :param model_aspect: all operations related to training or testing
    :param args: all operations related to training or testing
    :return: : dict, contains accuracy, loss and conf_matrix
    """

    best_saver = tf.compat.v1.train.Saver(max_to_keep=2, save_relative_paths=True)  # only keep 1 best checkpoint (best on eval)
    best_eval_acc = 0.0
    end = False
    trained_batches = 0
    epoch = 0
    lr = args.learning_rate
    args.test_every = model_aspect["tot_train_batches"] // args.test_freq  # how many times to test during one epoch

    result_data = {"train_accuracy": [], "train_loss": [],
                   "test_accuracy": [], "test_loss": [],
                   "test_confusion": []}

    while condition(end, result_data, epoch, args.epochs):
        # save training samples
        if trained_batches == 0:
            samples, labels = sess.run([model_aspect["train_features"], model_aspect["train_labels"]])
            Plot.plot_train_samples(samples, np.argmax(labels, axis=1), args,
                                    postfix="training_batches_{}".format(epoch))
            samples, labels = sess.run([model_aspect["test_features"], model_aspect["test_labels"]])
            Plot.plot_train_samples(samples, np.argmax(labels, axis=1), args,
                                    postfix="testing_batches_{}".format(epoch))

        # Training phase
        if len(result_data["test_accuracy"]) > args.patience + 1:
            lr = reduce_lr_on_plateu(
                lr,
                np.array(result_data["test_accuracy"][-args.patience-1:]),
                factor=0.5, patience=args.patience,
                epsilon=1e-04, min_lr=10e-8)

        logging.info("learning rate: ", lr, "num of trainables: ", model_aspect["total_trainables"])
        metrics_train = train_sess(sess, model_aspect,
                                   lr=lr,
                                   compute_batches=args.test_every,
                                   train_or_test='train')

        metrics_string = "train_accuracy: {}\ntrain_loss: {}\n" \
                         "train_confusion:\n {}"\
            .format(metrics_train["train_accuracy"],
                    metrics_train["train_loss"],
                    metrics_train["train_confusion"])
        logging.info("batch {}/{} - Train metrics:\n"
                     .format(trained_batches,
                             model_aspect["tot_train_batches"]) + metrics_string)

        # Validation phase
        metrics_test = validation_sess(sess, model_aspect,
                                       compute_batches=model_aspect["tot_test_batches"],
                                       if_check_cam=False, train_or_test='test')
        metrics_string = "test_acc: {}\ntest_loss: {}\n" \
                         "test_confusion: \n{}"\
            .format(metrics_test["test_accuracy"],
                    metrics_test["test_loss"],
                    metrics_test["test_confusion"])
        logging.info("epoch {} - Test metrics:\n".format(epoch) + metrics_string)
        #
        
        result_data["train_accuracy"].append(metrics_train["train_accuracy"])
        result_data["test_accuracy"].append(metrics_test["test_accuracy"])
        result_data["train_loss"].append(metrics_train["train_loss"])
        result_data["test_loss"].append(metrics_test["test_loss"])
        result_data["test_confusion"].append(metrics_test["test_confusion"])

        if epoch % args.plot_every == 0:
            # Save accuracy and loss plots
            Plot.save_plots(result_data, epoch, args, acc="{:.4f}".format(metrics_test["test_accuracy"]))
            
        if epoch % args.save_every == 0:
            # check_if_save_model(saver, sess, args.model_save_dir, epoch, save_every=args.save_every)
            save_data_to_csvs(result_data, epoch, args)

        # # Save best performance model
        eval_acc = metrics_test["test_accuracy"]
        if eval_acc > best_eval_acc:
            # Store new best accuracy
            best_eval_acc = eval_acc
            # Save weights
            best_save_path = args.model_save_dir
            # Save best model
            check_if_save_model(best_saver, sess, args.model_save_dir,
                                epoch, save_every=None,
                                name="best-acc-{}".format(eval_acc))
            logging.info("- Found new best accuracy: {}, saving in {}".format(best_eval_acc, best_save_path))
            
            # Inspect class activity maps
            if_check_cam = True if 'cam' in args.model_name else False
            metrics_test = validation_sess(
                sess, model_aspect,
                compute_batches=model_aspect["tot_test_batches"],
                if_check_cam=if_check_cam, train_or_test='test')
            logging.info("epoch {} - Test metrics:\n".format(epoch) + metrics_string)

            Plot.plot_roc_curve(args, metrics_test, acc=best_eval_acc)
            Plot.plot_confusion_matrix(args, metrics_test["test_confusion"], normalize=False, postfix=best_eval_acc)
            if "cam" in args.model_name:
                logging.info("------------Plotting activation maps-----------")
                num_samples = 20  # 10, 3
                rand_ind = np.random.choice(
                    np.arange(metrics_test["test_certain_labels_int"].shape[0]),
                    min(num_samples, metrics_test["test_certain_labels_int"].shape[0]))
                # Save all the certain examples for future plotting
                with open(args.results_dir + '/attention_maps/cams_of_certain_examples_of_{}_acc_{:.3f}.txt'.format(
                        args.data_source, best_eval_acc),
                          'wb') as f:
                    pickle.dump({"val_original_data":
                                     np.array(metrics_test["test_certain_features"]),
                                 "val_true_labels": np.array(metrics_test["test_certain_labels_int"]),
                                 "val_convs": np.array(metrics_test["test_certain_conv"]),
                                 "val_gap_w": np.array(metrics_test["test_gap_w"]),
                                 "val_pred_labels": np.array(metrics_test["test_certain_pred_int"])}, f)

                class_maps = Plot.get_class_map(
                    metrics_test["test_certain_labels_int"][rand_ind],
                    metrics_test["test_certain_conv"][rand_ind],
                    metrics_test["test_gap_w"],
                    args.height, args.width)
                Plot.plot_class_activation_map(
                    sess, class_maps,
                    metrics_test["test_certain_features"][rand_ind],
                    metrics_test["test_certain_labels_int"][rand_ind],
                    metrics_test["test_certain_pred_int"][rand_ind],
                    args, postfix="ep{}-acc-{:.3f}".format(epoch, best_eval_acc))

        trained_batches += args.test_every
        epoch = trained_batches // model_aspect["tot_train_batches"]
        logging.info("Epoch {}/{}".format(epoch, args.epochs))

    return result_data


def train_sess(sess, model_aspect, lr=0.005, compute_batches=100, train_or_test='train'):
    """
    session training
    :param sess: tf.Session(
    :param model_aspect: dict
    :param lr: float
    :param compute_batches: int, number of batches to compute
    :param train_or_test: str, indicate the phase
    :return:
    """
    logging.info("---------------start training sess------------------")
    names = ['loss', 'num_correct', 'confusion', 'batch_size', 'train_op', 'lr_op']
    fetches = get_fetches(model_aspect, names, train_or_test=train_or_test)
    t1 = time.time()
    results = compute(sess, fetches,
                      compute_batches=compute_batches,
                      lr=lr,
                      if_get_certain=False)
    logging.info("Time for computing {} batches: {}".format(compute_batches, time.time() - t1))
    
    return_names = ["accuracy", "loss", "confusion"]
    return get_returns(results, return_names, train_or_test=train_or_test)
    
        
def validation_sess(sess, model_aspect, compute_batches=100, if_check_cam=False, train_or_test='train'):
    """
    :param sess: tf.Session, current session
    :param model_aspect: all operations related to training or testing
    :param compute_batches: int, number of batches to train or test
    :param if_check_cam: int, epoch
    :param train_or_test: str, 'train', 'test'
    :return: metrics: dict, contains accuracy, loss and conf_matrix
    """
    # Initialize the dataset iterators
    init_ops = [model_aspect["test_iter_init"]]
    initialize_ops(sess, init_ops)
    t1 = time.time()
    if not if_check_cam:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'pred_int',
                 'pred_logits', 'features']
        fetches = get_fetches(model_aspect, names, train_or_test=train_or_test)

        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_certain=True)
        
        return_names = ["accuracy", "loss", "confusion",
                        "labels", "features", "pred_int",
                        "pred_logits", "certain_features",
                        "certain_labels_int", "certain_pred_int"
                        ]
        ret = get_returns(results, return_names, train_or_test=train_or_test)
    else:   # testing during training
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'pred_int',
                 'pred_logits', 'features', 'conv', 'gap_w']
        
        fetches = get_fetches(model_aspect, names, train_or_test=train_or_test)

        results = compute(sess, fetches,
                          compute_batches=compute_batches,
                          if_get_certain=True)

        return_names = ["accuracy", "loss", "confusion",
                        "labels", "features", "conv", "gap_w",
                        "pred_int", "pred_logits", "certain_features",
                        "certain_labels_int", "certain_pred_int",
                        "certain_conv"]
        ret = get_returns(results, return_names, train_or_test=train_or_test)

    logging.info("Time for computing {} batches: {}".format(compute_batches, time.time() - t1))

    return ret


def testing(sess, model_aspect, args, compute_batches=100, if_check_cam=False, train_or_test='test'):
    """
    :param sess: tf.Session, current session
    :param model_aspect: all operations related to training or testing
    :param compute_batches: int, number of batches to train or test
    :param if_check_cam: int, epoch
    :param train_or_test: str, indicate which phase, 'test', 'train'
    :return: metrics: dict, contains accuracy, loss and conf_matrix
    """
    init_ops = [model_aspect["test_iter_init"]]
    # Initialize the dataset iterators
    initialize_ops(sess, init_ops)
    if if_check_cam:
        names = ['loss', 'num_correct', 'confusion', 'batch_size',
                 'labels', 'pred_int', 'filenames',
                 'features', 'conv', 'pred_logits',
                 'gap_w']
        fetches = get_fetches(model_aspect, names, train_or_test=train_or_test)

        results = compute_test_only(sess,
                                    fetches, args,
                                    compute_batches=compute_batches,
                                    if_get_certain=True)
        ret = {"test_accuracy": results["num_correct"]/results["batch_size"],
                "test_loss": results["loss"]/results["batch_size"],
                "test_confusion": results["confusion"]/results["batch_size"],
               "test_labels": np.array(results["labels"]),
               "test_pred_int": np.array(results["pred_int"]),
               "test_filenames": np.array(results["filenames"]),
               "test_pred_logits": np.array(results["pred_logits"])
                }
    else:
        names = ['loss', 'num_correct', 'confusion',
                 'batch_size', 'labels', 'pred_int',
                 'features', 'filenames', 'pred_logits']
        fetches = get_fetches(model_aspect, names, train_or_test=train_or_test)
 
        results  = compute_test_only(sess,
                                     fetches, args,
                                     compute_batches=compute_batches,
                                     if_get_certain=True)  # for test, we want to get the whole concat data

        ret = {"test_accuracy": results["num_correct"]/results["batch_size"],
               "test_loss": results["loss"]/results["batch_size"],
               "test_filenames": np.array(results["filenames"]),
               "test_pred_logits": np.array(results["pred_logits"]),
               "test_pred_int": np.array(results["pred_int"]),
               "test_labels": np.array(results["labels"]),
               "test_confusion": results["confusion"]
               }
    metrics_string = "test_acc: {}\ntest_loss: {}\n" \
                                 "test_confusion: \n{}" \
        .format(ret["test_accuracy"],
                ret["test_loss"],
                ret["test_confusion"])
    logging.info("Test metrics:\n {}".format(metrics_string))
    # Save all the certain examples for future plotting
    logging.info("save test_indiv_hour_data")
    logging.info("ret-test_filenames: {}".format(ret["test_filenames"]))

    logging.info("----------Plot ROC AUC ----------")
    if args.num_classes > 2:
        Plot.plot_multiclass_roc(args, ret, acc=ret["test_accuracy"], postfix="seg-level-AUC")
    else:
        Plot.plot_bin_roc_curve(args, ret, acc=ret["test_accuracy"], postfix="seg-level-AUC")

    logging.info("----------Plot plot_test_indiv_hour----------")
    Plot.plot_test_indiv_hour(ret, args)
    logging.info(args.results_dir, "----------Done----------")

    return ret

# -----need------
def run(model_aspect, args):
    """
    Train the model and evaluate every args.test_every batches. To restore if specified.
    :param model_aspect: (dict) contains the graph operations or nodes needed for training
    :param args: (Params) contains hyperparameters of the model.
    :return:
    """
    saver = tf.compat.v1.train.Saver(max_to_keep=2, save_relative_paths=True)
    with tf.compat.v1.Session() as sess:
        if args.restore_from:
            logging.info("Restoring parameters from {}".format(args.restore_from))
            load_model(saver, sess, args.restore_from)
        
        if args.test_only:
            init_ops = [model_aspect["test_iter_init"]]
            # Initialize the dataset iterators
            initialize_ops(sess, init_ops)
            if_check_cam = True if 'cam' in args.model_name else False
            testing(sess,
                    model_aspect, args,
                    compute_batches=model_aspect["tot_test_batches"],
                    if_check_cam=if_check_cam)

        else:
            init_ops = [model_aspect["train_iter_init"],
                        model_aspect["test_iter_init"],
                        tf.compat.v1.global_variables_initializer(),
                        tf.compat.v1.local_variables_initializer()]
            # Initialize the data set iterators
            initialize_ops(sess, init_ops)
            training(sess, model_aspect, args)

        logging.info("Done")
