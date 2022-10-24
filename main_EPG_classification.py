# use basic network to do classification
# from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')
import logging
import sys
import os

import numpy as np
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tensorflow.python.client import device_lib

from arguments import get_basic_EEG_params, get_proj_args
import dataio_EPG as data_io
import get_graphs as mod
from train import run
tf.compat.v1.disable_v2_behavior()
# tf.enable_eager_execution()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    logging.info("-------------------Available GPU-----------------------")
    logging.info([x.name for x in local_device_protos if x.device_type == 'GPU'])

    
if __name__ == '__main__':
    # Get all the params about experiment and the model
    print("GPU usage -------------------------------------")
    get_available_gpus()
    logging.info("Tensorflow version: ", tf.__version__)
    params = get_basic_EEG_params("exp_params.json")  # dir should start with the __main__ file

    params = get_proj_args(params, proj_dir="Classification_EPG", model_json="model_params.json")
    logging.info("params.results_dir", params.results_dir)

    # Check all the dirs and make new one if needed
    params = data_io.check_make_dirs(params)
    
    # output the terminal output to a file
    f = open(os.path.join(params.model_save_dir, 'train.out'), 'w')
    sys.stdout = f

    # specify params.model_save_dir. TODO make model config params and save results using the config params
    data_io.set_logger(os.path.join(params.model_save_dir, 'train.log'))
    data_io.save_command_line(params)

    # Set the random seed
    if params.seed is not None:
        np.random.seed(seed=params.seed)
        tf.compat.v1.set_random_seed(params.seed)
    else:
        params.seed = np.random.choice(np.arange(1, 9999), 1)[0]
        np.random.seed(seed=params.seed)
        tf.compat.v1.set_random_seed(params.seed)

    # Create the input data pipeline
    logging.info("Creating the datasets for training and testing...")
    if params.test_only:
        data_tensors, params = data_io.get_test_only_data_tensors(params,
                                                                  if_shuffle=False,
                                                                  if_repeat=True)
        # Construct the models (2 different set of nodes that share weights for train and eval)
        logging.info("Creating the model...")
        graph_dir = params.graph_dir
        sys.path.append(graph_dir)
        
        model_aspect = mod.get_graph(data_tensors, params)  #
        logging.info(params.results_dir)
    else:
        data_tensors, params = data_io.get_data_tensors(params,
                                                        if_shuffle_train=True,
                                                        if_shuffle_test=True,
                                                        if_repeat_train=True,
                                                        if_repeat_test=True)
        # Construct the models (2 different set of nodes that share weights for train and eval)
        logging.info("Creating the model...")
    
        model_aspect = mod.get_graph(data_tensors, params)  #
        logging.info(params.results_dir)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.epochs))
    params.save(os.path.join(params.model_save_dir, "parameters.json"))  # Save parameters into json
    run(model_aspect, params)

    f.close()
