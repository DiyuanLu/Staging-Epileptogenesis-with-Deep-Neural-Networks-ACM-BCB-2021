import argparse
import os
import sys
from dataio_EPG import get_timestamp_from_file

# jointly store the  params for experiment and the network in params


def get_proj_args(params, proj_dir="Classification_EPG", model_json="model_params.json"):
    """
    after arguments_for_all, here load project specific args
    :param proj_dir:
    :return:
    """
    if "CURE" in params.class_mode:
        params.sr = 1000
    else:
        params.sr = 512
    
    params.height = params.secs_per_samp * params.sr
    params.postfix = '{}h-{}seg-{}'.format(params.num_hours_per_class, params.num_segs, params.data_source)
    if params.test_only:
        #
        milestone_st = 1596123540.0  ##before data-2020-07-30T17-39-05 src is not divided into sub-projects
        time_stamp = get_timestamp_from_file("-".join(
            ["data"] + os.path.basename(os.path.dirname(params.restore_from)).split(
                "-")[0:5]))
        if time_stamp > milestone_st:
            full_model_json = os.path.join("src", proj_dir, model_json)
        else:
            full_model_json = os.path.join("src", model_json)
        
        params.graph_dir = os.path.join(params.restore_from, "src")
        json_path = os.path.join(params.restore_from, full_model_json)
        assert os.path.isfile(
            json_path), "No json file found at {}, run build_vocab.py".format(
            json_path)
        params.update(json_path,
                      model_key=params.model_name)  # update params with the model configuration
        params.ctrl_pps_ratio = 1
        
        pretrain_model = os.path.basename(os.path.dirname(params.restore_from)).split("-")[6]
        assert pretrain_model == params.model_name, "Pretrained on {}, but asked for {}, The model name doesn't match!!!".format(pretrain_model, params.model_name)
    else:
        json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                 model_json)  # 'model_params.json'
        assert os.path.isfile(
            json_path), "No json file found at {}, run build_vocab.py".format(
            json_path)
        params.update(json_path,
                      model_key=params.model_name)  # update params with the model configuration
        if 'clf' in params.model_name:  # when do EPG-clf and BL-clf, grouping two groups.
            params.ctrl_pps_ratio = 3

    return params


def get_basic_EEG_params(params):
    """
    this file shared by all projects
    :param exp_json_name:
    :param model_json_name:
    :param restore_from_model_json:
    :return:
    """
    
    # specify some params
    params.seq_len = np.int(params.secs_per_samp)
    params.height = np.int(params.secs_per_samp * params.sr)
    # params.restore_from = args.restore_from
    # params.test_only = args.restore_from is not None and args.keep_training == False
    params.test_ratio = 1.0 if params.test_only else params.test_ratio
    params.days4train = None if params.test_only else params.days4train
    
    params.possible_class_modes = {"BL+1EPG": (2, ["BL", "EPG"]),
                                   "BL+2EPG": (3, ["BL", "2.1EPG", "2.2EPG"])
                                   }
    
    if "EPG" in params.class_mode or "BL" in params.class_mode:
        params.sr = 512
        params.start_EPG_date = {
                "1227": "EPG-2014-10-03T06-34",
                "1237": "EPG-2014-10-04T01-29",
                "1243": "EPG-2014-09-13T15-12",
                "1270": "EPG-2015-01-28T23-17",
                "1275": "EPG-2015-03-19T23-52",
                "1276": "EPG-2015-05-06T08-31",
                "32140": "EPG-2017-04-14T11-59",
                "32141": "EPG-2017-06-26T23-59",
                "3263": "EPG-2016-03-19T04-59",
                "3266": "EPG-2016-04-02T22-59",
                "3267": "EPG-2016-04-10T20-59"
        }
        params.end_EPG_date = {
                "1227": "EPG-2014-11-02T22-01",
                "1237": "EPG-2014-10-28T11-30",
                "1243": "EPG-2014-09-22T12-25",
                "1270": "EPG-2015-02-14T14-16",
                "1275": "EPG-2015-03-25T18-52",
                "1276": "EPG-2015-06-21T04-35",
                "32140": "EPG-2017-04-26T03-59",
                "32141": "EPG-2017-07-23T22-59",
                "3263": "EPG-2016-06-21T12-59",
                "3266": "EPG-2016-06-02T12-59",
                "3267": "EPG-2016-07-07T09-00"
        }
    
    params.num_classes = params.possible_class_modes[params.class_mode][0]
    params.class_names = params.possible_class_modes[params.class_mode][1]
    
    if params.if_spectrum and 'cnn' in params.model_name:
        params.output_channels = [8, 16, 16]
        params.pool_size = [2, 2]
        params.strides = [2, 2]
        params.filter_size = [[3, 2], [3, 2]]
        params.fc = [200]
        params.drop_rate = [0.5, 0.5, 0.5]
        params.seq_len = params.secs_per_samp  # for shuffle buffer # compute
    
    return params
