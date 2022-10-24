## @package dataio
#  This package handles the interface with the hard drive.
#
#  It can in particular read and write matlab or python matrices.
import sys
sys.path.insert(0, "../")
import os
import fnmatch
import datetime
import tracemalloc
import linecache
import gc

import numpy as np
import pandas as pd
import tensorflow as tf


def find_files(args):
    """
    Find all the files in one directory with pattern in the filenames and perform train_test_split, and save file names seperately.
    :param args.data_dir: str, the directory of the files
    :param args.class_mode: str, "1EPG (BL-EPG)", "3EPG (BL-earlyEPG-middleEPG-lateEPG)"
    :param args.test_ratio: the ratio of whole data used for testing
    :param args.num_hours_per_class: how many hours to choose for training
    :param save_dir:
    :return: test_files, list all the testing files
    :return: train_files, list all the training files
    """
    train_files_labels = []
    test_files_labels = []
    ## get the number of files in foldersnum_EPGs = args.num_classes
    class_start = 0
    num_EPGs = args.num_classes
    
    print("data_io args.data_dir: {}".format(args.data_dirs))
    for folder in args.data_dirs:
        for root, dirnames, fnames in os.walk(folder):
            # if BL is in classification, get them
            
            if "BL+" in args.class_mode:
                if os.path.basename(root) == "BL":
                    rat_id = os.path.basename(os.path.dirname(root))
                    label = 0
                    num_EPGs = args.num_classes - 1
                    fnames = fnmatch.filter(fnames, args.file_pattern)
                    class_start = 1
                    print("{}, label-{}".format("BL", label))
                    train_files_labels, test_files_labels = \
                        get_train_test_files_split(root, fnames, args.test_ratio,
                                                   train_files_labels,
                                                   test_files_labels, rat_id=rat_id, label=label,
                                                   num2use=args.num_hours_per_class)
            if args.class_mode == "BL+1EPG":
                assert args.class_names == ['BL', "EPG"]
                assert args.num_classes == 2
                if os.path.basename(root) == "EPG":
                    # folder's name should be strictly the class names
                    print("mode {} files is found under {}".format(args.class_mode, root))
                    fnames = fnmatch.filter(fnames, args.file_pattern)
                    label = 1
                    rat_id = os.path.basename(os.path.dirname(root))
                    train_files_labels, test_files_labels = \
                        get_train_test_files_split(root, fnames, args.test_ratio,
                                                   train_files_labels,
                                                   test_files_labels, rat_id=rat_id, label=label,
                                                   num2use=args.num_hours_per_class)
            elif args.class_mode == "BL+2EPG":
                args.class_names = ["BL","early EPG","late EPG"]
                assert args.num_classes == 3
                num_EPGs = 2  # Even it is the earlyEPG vs. lateEPG, but by default it is 3, there is a middle EPG
                if os.path.basename(root) == "EPG" or os.path.basename(root) == "STIM":
                    print("mode {} files is found under {}".format(args.class_mode, root))
                    rat_id = os.path.basename(os.path.dirname(root))
                    fnames = fnmatch.filter(fnames, args.file_pattern)

                    if len(fnames) > 1:
                        split_stages = get_stage_timestamps(fnames, num_stages=num_EPGs, days4train=args.days4train, last_day=args.end_EPG_date[rat_id])
                        for label, files in zip([1, 2], [split_stages["2.1EPG"], split_stages["2.2EPG"]]):
                            train_files_labels, test_files_labels = \
                                get_train_test_files_split(root, files, args.test_ratio,
                                                           train_files_labels,
                                                           test_files_labels, rat_id=rat_id, label=label,
                                                           num2use=args.num_hours_per_class)

    test_files_labels = np.array(test_files_labels)
    if args.test_only:
        time_stamps = np.array([get_timestamp_from_file(os.path.basename(fn), year_ind=args.year_ind) for fn in np.array(test_files_labels)[:, 0]])
        sort_temp = test_files_labels[np.argsort(time_stamps)]
        test_files_labels = sort_temp
    np.savetxt(os.path.join(args.results_dir,
                            "test_files-{}.txt".format(args.data_source)),
               np.array(test_files_labels), fmt="%s", delimiter=",")
    # assert np.sum((time_stamps[1:] - time_stamps[0:-1]) < 0) == 0, "The test files are not properly sorted"
    if args.test_ratio != 1:   # when it is not in test_only case, there are training files
        np.savetxt(os.path.join(args.results_dir, "train_files-{}.txt".format(args.data_source)), np.sort(np.array(train_files_labels)), fmt="%s", delimiter=",")
        np.random.shuffle(np.array(train_files_labels))
    
    return train_files_labels, test_files_labels


def find_only_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files


# -----need------
def match_find_files(directory, pattern='*.csv'):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    return files

# -----need------
def load_from_file_list(args):
    """
    When keep_training is true, then load the training and testing file names from the pre-trained model dir
    :param args:
    :return:
    """
    train_fn = match_find_files(os.path.dirname(args.restore_from), pattern="train_files*.txt")
    test_fn = match_find_files(os.path.dirname(args.restore_from), pattern="test_files*.txt")
    training_fns = pd.read_csv(train_fn[0], header=None).values
    testing_fns = pd.read_csv(test_fn[0], header=None).values

    np.savetxt(os.path.join(args.results_dir,
                            "test_files-{}.txt".format(args.data_source)),
               np.array(testing_fns), fmt="%s", delimiter=",")
    np.random.shuffle(np.array(testing_fns))
    if args.test_ratio != 1:  # when it is not in test_only case, there are training files
        np.savetxt(os.path.join(args.results_dir,
                                "train_files-{}.txt".format(args.data_source)),
                   np.array(training_fns), fmt="%s", delimiter=",")
        np.random.shuffle(np.array(training_fns))
        
    return training_fns, testing_fns


def get_specific_days(fnames, first_day=None, year_ind=1, whichday4train=None, last_day=None):
    """
    Get the timestamps for each stage and get files in corresponding stage
    :param fnames: list
    :param num_stages: int
    :param whichday4train: int or NOne, days. which day to use 1st, 2nd, 3rd, -1=last day, NOne=all
    :param test_only: boolean, if in training, get the first and the last subseg days, otherwise evenly split
    :return:
    """
    
    sorted_fns = sorted(fn for fn in fnames)
    if first_day is None:
        first_day = sorted_fns[0]

    start_timestamp = get_timestamp_from_file(first_day, year_ind=year_ind)
    
    # last_day = sorted_fns[-1]
    last_timestamp = get_timestamp_from_file(last_day, year_ind=year_ind)
    stage_files = []
    if whichday4train is not None:
        if whichday4train == -1:  # get the last day(24 hours)
            edge_1 = last_timestamp
            for fn in sorted_fns:
                stamp = get_timestamp_from_file(fn, year_ind=year_ind)
                if (edge_1 - stamp) / (24 * 3600 * 1.0)  <= 1:
                    stage_files.append(fn)
        else:
            whichday4train = min(whichday4train, (last_timestamp - start_timestamp) // (24 * 3600))
            edge_1 = start_timestamp + 24 * 3600 * (whichday4train - 1)
            edge_2 = start_timestamp + 24 * 3600 * whichday4train
            for fn in sorted_fns:
                stamp = get_timestamp_from_file(fn, year_ind=year_ind)
                if stamp >= edge_1 and stamp < edge_2:
                    stage_files.append(fn)
    else:
        stage_files = fnames   # get ALL files
        
    print("Stage segment Done, whichday4train:{}".format(whichday4train))
    
    return stage_files


def get_stage_timestamps(fnames, num_stages, first_day=None, days4train=None, last_day=None):
    """
    Get the timestamps for each stage and get files in corresponding stage
    :param fnames: list
    :param num_stages: int
    :param days4train: int, days. how many days at the beginning and the end to use for training
    :param test_only: boolean, if in training, get the first and the last subseg days, otherwise evenly split
    :return:
    """
    
    sorted_fns = sorted(fn for fn in fnames)
    if first_day is None:
        first_day = sorted_fns[0]

    start_timestamp = get_timestamp_from_file(first_day)
    
    # last_day = sorted_fns[-1]
    last_timestamp = get_timestamp_from_file(last_day)
    if days4train is None: # equally divide the phase
        interval = ((last_timestamp - start_timestamp) / num_stages) + 1  #make sure the last file in the last stage
        # get the first and the last day
        
        stage_files = {"{}.{}EPG".format(num_stages, i + 1): [] for i in range(num_stages)}
        for fn in sorted_fns:
            stamp = get_timestamp_from_file(fn)
            phase = np.int((stamp - start_timestamp) // interval)  # +1 to avoid the first day is not recognized
            name = "{}.{}EPG".format(num_stages, phase + 1)
            stage_files[name].append(fn)
        print("Stage segment Done, days4train: {}".format(days4train))
    else:
        days4train = min(days4train, (last_timestamp - start_timestamp) // (2 * 24 * 3600))
        
        edge1 = start_timestamp + 24 * 3600 * days4train
        edge2 = last_timestamp - 24 * 3600 * days4train
        stage_files = {"{}.{}EPG".format(num_stages, i + 1): [] for i in range(num_stages)}
        for fn in sorted_fns:
            stamp = get_timestamp_from_file(fn)
            if stamp < edge1:
                name = "{}.{}EPG".format(num_stages, 1)
                stage_files[name].append(fn)
            elif stamp > edge2:
                name = "{}.{}EPG".format(num_stages, 2)
                stage_files[name].append(fn)
            
        print("Stage segment Done, days4train: {}".format(days4train))
    
    return stage_files


def get_VT_files_timestamps(fnames, until_which=None, mode="2sides", first_day=None, last_day=None):
    """
    Get the timestamps for each stage and get files in corresponding stage
    :param fnames: list
    :param num_stages: int
    :param which_day: int, from the 1st to which_day.
    :param test_only: boolean, if in training, get the first and the last subseg days, otherwise evenly split
    :return:
    """
    
    sorted_fns = sorted(fn for fn in fnames)
    # first_day = sorted_fns[0]
    start_timestamp = get_timestamp_from_file(first_day)
    last_timestamp = get_timestamp_from_file(last_day)

    need_files = []
    if until_which is None:
        if "sides" in mode:
            interval = (last_timestamp - start_timestamp) / ( 24 * 3600)
        elif "stages" in mode: # divide into two stages
            interval = (last_timestamp - start_timestamp) / ( 2 * 24 * 3600)
    else:
        if "sides" in mode:
            interval = until_which
        elif "stages" in mode:
            interval = until_which
    #
    for fn in sorted_fns:
        stamp = get_timestamp_from_file(fn)
        if "sides" in mode:
            if (stamp - start_timestamp) // (24*3600) < interval:
                if "-S-" in fn:
                    need_files.append((fn, 1))
                elif "-NS-" in fn:
                    need_files.append((fn, 0))
        elif "stages" in mode:
            if (stamp - start_timestamp) // (24*3600) < interval and "-S-" in fn:
                need_files.append((fn, 0))
            elif (last_timestamp - stamp) // (24*3600) < interval and "-S-" in fn:
                need_files.append((fn, 1))
    
    # for fn in sorted_fns:
    #     stamp = get_timestamp_from_file(fn)
    #     if "sides" in mode:
    #         if (stamp - start_timestamp) // (24*3600) < interval:
    #             if "-S-" in fn:
    #                 need_files.append(fn)
    #             elif "-NS-" in fn:
    #                 need_files.append(fn)
    #     elif "stages" in mode:
    #         if (stamp - start_timestamp) // (24*3600) < interval and "-S-" in fn:
    #             need_files.append(fn)
    #         elif (last_timestamp - stamp) // (24*3600) < interval and "-S-" in fn:
    #             need_files.append(fn)
    
    return need_files


def get_timestamp_from_file(fn, year_ind=1):
    # year = np.int(fn.split("T")[-2].split("-")[-3])
    # mon = np.int(fn.split("T")[-2].split("-")[-2])
    # day = np.int(fn.split("T")[-2].split("-")[-1])
    # hour = np.int(fn.split("T")[-1].split("-")[0])
    # min = np.int(fn.split("T")[-1].split("-")[1])

    print(fn)
    year = np.int(fn.split("-")[year_ind])
    mon = np.int(fn.split("-")[year_ind+1])
    day = np.int(fn.split("-")[year_ind+2].split("T")[0])
    hour = np.int(fn.split("-")[year_ind+2].split("T")[1])
    min = np.int(fn.split("-")[year_ind+3])
    timestamp = datetime.datetime(year, mon, day, hour, min).timestamp()
    return timestamp


def get_train_test_files_split(root, fns, ratio, train_list, test_list, year_ind=1, rat_id="1227", label=0, num2use=100):
    """
    Get equal number of files for testing from each folder
    :param fns: list, all file names from the folder
    :param ratio: float, the test file ratio.
    :param train_list: the list for training files
    :param test_list: the list for testing files
    :param label: int or list, the label need to be assigned to the file
    :param num2use: int, the number of files that you want to use(randomize file selection)
    :return: lists, editted train and test file lists
    """
    rand_inds = np.arange(len(fns)).astype(np.int)
    if ratio != 1: # test_only mode, don't shuffle
        np.random.shuffle(rand_inds)
    if isinstance(label, int):
        labels = np.repeat(label, len(fns))
        rand_fns = np.array(fns)[rand_inds]
    elif isinstance(label, list):
        labels = np.array(label)[rand_inds]
        rand_fns = np.array(fns)[rand_inds]
    
    num_files_need = min(len(rand_fns), num2use)

    num_test_files = np.ceil(ratio * num_files_need).astype(np.int)

    current_test_files = []
    current_train_files = []
    for ind, f, lb in zip(np.arange(num_files_need), rand_fns[0:num_files_need], labels):
        num_rows = os.path.basename(f).split('-')[-2]
        # num_rows = os.path.basename(f).split('-')[-1].split('.')[0]
        if ind < num_test_files:
            current_test_files.append((os.path.join(root, f), lb, num_rows, rat_id))
        else:
            current_train_files.append((os.path.join(root, f), lb, num_rows, rat_id))
            
    # sort the test files
    time_stamps = np.array(
        [get_timestamp_from_file(os.path.basename(fn), year_ind=year_ind) for fn in np.array(current_test_files)[:, 0]])
    print("time_stamps", time_stamps)
    print("np.argsort(time_stamps", np.argsort(time_stamps))
    test_files_labels = np.array(current_test_files)[np.argsort(time_stamps)]

    test_list += list(test_files_labels)
    train_list += current_train_files
    
    return train_list, test_list


def get_regression_train_test_files_split(root, fns, ratio, train_list, test_list, num2use=100, endofEPG=2654478.):
    """
    Get equal number of files for testing from each folder
    :param fns: list, all file names from the folder
    :param ratio: float, the test file ratio.
    :param train_list: the list for training files
    :param test_list: the list for testing files
    :param label: int, the label need to be assigned to the file
    :param num2use: int, the number of files that you want to use(randomize file selection)
    :return: lists, editted train and test file lists
    """
    np.random.shuffle(fns)
    num_files = min(len(fns), num2use)

    num_test_files = np.ceil(ratio * num_files).astype(np.int)

    train_within_folder = []

    for ind, f in enumerate(fns[0:num_files]):
        num_rows = os.path.basename(f).split('-')[-2]

        day_until_end = (endofEPG - get_timestamp_from_file(f)) / (
                    3600. * 24)
        if ind < num_test_files:
            test_list.append((os.path.join(root, f), day_until_end, num_rows))
        else:
            train_list.append((os.path.join(root, f), day_until_end, num_rows))
            train_within_folder.append((os.path.join(root, f), day_until_end, num_rows))

    return train_list, test_list


def parse_function(filename, label, args):
    """
    parse the file. It does following things:
    1. init a TextLineDataset to read line in the files
    2. decode each line and group args.secs_per_samp*args.num_segs rows together as one sample
    3. repeat the label for each long chunk
    4. return the transformed dataset
    :param filename: str, file name
    :param label: int, label of the file
    :param num_rows: int, the number of rows in the file (since they are artifacts free)
    :param args: Param object, contains hyperparams
    :return: transformed dataset with the label of the file assigned to each batch of data from the file
    """
    skip = 0

    decode_ds = tf.compat.v1.data.TextLineDataset(filename).skip(skip).map(lambda line: decode_csvfile(line, args=args))
    # decode_ds = decode_ds.map(lambda feature: decode_label_fn(feature, assign_label=label, assign_fn=filename))
    decode_ds = decode_ds.map(lambda ft, lb, fn: decode_label_fn(ft, lb, fn, assign_label=label))
    
    # decode_ds = decode_ds.map(lambda fn, lb, feat: decode_mod_label(fn, lb, feat, assign_label=label))
    # decode_ds = decode_ds.apply(tf.contrib.data.batch_and_drop_remainder(args.secs_per_samp*args.num_segs))
    # decode_ds = decode_ds.batch(args.secs_per_samp * args.num_segs)

    if args.if_spectrum:
        decode_ds = decode_ds.map(lambda feature, label, fn:
                                  get_spectrum(feature, label, fn, args=args))
    else:
        decode_ds = decode_ds.map(scale_to_zscore)  # zscore norm the data

    return decode_ds


def decode_csvfile(line, args=None):
    # Map function to decode the .csv file in TextLineDataset
    # @param line object in TextLineDataset
    # @return: zipped Dataset, (features, labels)
    defaults = [['']] + [[0.0]] * (args.sr * args.secs_per_row + 1)  # there are 5 sec in one row
    csv_row = tf.compat.v1.decode_csv(line, record_defaults=defaults)

    filename = tf.cast(csv_row[0], tf.string)
    label = tf.cast(csv_row[1], tf.int32)  # given the label
    # features = tf.cast(tf.stack(csv_row), tf.float32)
    features = tf.cast(tf.stack(csv_row[2:]), tf.float32)

    # return features
    return features, label, filename


def decode_label_fn(features, label, filename, assign_label=0):
    """
    Modify the label for each sample given different data_mode. E.g., EPG by default is 1, but i EPG_id mode,
    control_EPG is 0, and pps_EPG is 1
    :param features:
    :param label:
    :param fn:
    :return:
    """

    return features, assign_label, filename


def scale_to_zscore(data, label, filename):
    """
    zscore normalize the features
    :param data: 2d-array, batch_size, seq_len
    :param label: 1d-array, batch_size,
    :param filename: 1d-array, batch_size, seq_len
    :return: normalized data
    """
    # ret = tf.nn.moments(data, 0)
    mean = tf.reduce_mean(data)
    std = tf.compat.v1.math.reduce_std(data)
    zscore = (data - mean) / (std + 1e-13)

    return zscore, label, filename


def get_spectrum(features, label, filename, id, args=None):
    """
    Transform function on already flattened features to get spectrum.
    Matlab: spectrogram(CSV(1:512*20), 64, 16, 512);
    :param feature: element in the dataset
    :param label:
    :param filename:
    :param args:
    :return:
    """
    # Get stft, shape: [batch_size, ?, fft_unique_bins] where fft_bins = fft_length // 2 + 1
    try:  # for some segments, the remaining secs are not enough for a segment
        flat_features = tf.reshape(features, [-1, args.secs_per_samp * args.sr])
        stfts = tf.compat.v1.contrib.signal.stft(flat_features, frame_length=512, frame_step=32, fft_length=128)
        power_spec = tf.compat.v1.real(stfts * tf.compat.v1.conj(stfts))  # A float32 Tensor of shape [batch_size, time_bins, fft_bins].
        power_spec = tf.compat.v1.log(power_spec + 1e-13)  # log the spectrum
        return power_spec, label, filename
    except:
        print("Too short for a segment")
    pass


def creat_data_tensors(dataset, data_tensors, filenames_w_lb, args, batch_size=32, prefix='test'):
    """
    Create the data tensors for test or train
    :param dataset:
    :param data_tensors:
    :param filenames_w_lb:
    :param args:
    :param batch_size:
    :param prefix:
    :return:
    """
    num_rows = np.array(filenames_w_lb)[:, 2].astype(np.int)
    if prefix == 'test':
        print("Test files: \n{}".format(np.array(filenames_w_lb)[:, 0:2]))

    iter = dataset.make_initializable_iterator()
    batch_ds = iter.get_next()  # test contains features and label
    data_tensors["{}_iter_init".format(prefix)] = iter.initializer

    if args.if_spectrum:
        data_tensors["{}_features".format(prefix)] = batch_ds[0]  # shape=[bs, num_seg, time_bins, freq_bins]
        args.height = batch_ds[0][0].get_shape().as_list()[2]  # [bs, 1, time_bins, 129]
        args.width = batch_ds[0][0].get_shape().as_list()[3]
    else:
        data_tensors["{}_features".format(prefix)] = tf.reshape(batch_ds[0][0], [-1, args.sr * args.secs_per_samp])
        
    if args.class_mode == "regression":
        data_tensors["{}_labels".format(prefix)] = tf.cast(tf.repeat(batch_ds[0][1],
                                                                     repeats=args.secs_per_row//args.secs_per_samp, axis=0),
                                                           dtype=tf.float32)
    else:
        # labels = tf.repeat(batch_ds[0][1], repeats=args.secs_per_row//args.secs_per_samp, axis=0)
        data_tensors["{}_labels".format(prefix)] = tf.one_hot(tf.repeat(batch_ds[0][1],
                                                                        repeats=args.secs_per_row//args.secs_per_samp, axis=0),
                                                              args.num_classes,
                                                              dtype=tf.int32)
    # data_tensors["{}_filenames".format(prefix)] = tf.cast(batch_ds[0][2], dtype=tf.string)
    # data_tensors["{}_ids".format(prefix)] = tf.cast(batch_ds[1], dtype=tf.string)
    data_tensors["{}_filenames".format(prefix)] = tf.cast(tf.repeat(batch_ds[0][2],
                                                                    repeats=args.secs_per_row//args.secs_per_samp, axis=0),
                                                          dtype=tf.string)
    data_tensors["{}_ids".format(prefix)] = tf.cast(tf.repeat(batch_ds[1],
                                                              repeats=args.secs_per_row//args.secs_per_samp, axis=0),
                                                    dtype=tf.string)
    data_tensors["tot_{}_batches".format(prefix)] = np.int((np.sum(num_rows)*args.secs_per_row//args.secs_per_samp) // data_tensors["{}_features".format(prefix)].get_shape().as_list()[0])  # when use non-5 sec as length, the batchsize changes

    return data_tensors


def create_dataset(filenames_w_lb, args, batch_size=32, if_shuffle=True, if_repeat=True):
    """

    :param filenames_w_lb:
    :param batch_size:
    :param if_shuffle:
    :return:
    """
    if if_shuffle:
        inds = np.arange(len(filenames_w_lb))
        np.random.shuffle(inds)
    else:
        inds = np.arange(len(filenames_w_lb))
        # train_list.append((os.path.join(root, f), lb, num_rows, rat_id))
    labels = np.array(filenames_w_lb)[:, 1][inds].astype(np.int32)
        
    filenames = np.array(filenames_w_lb)[:, 0][inds].astype(np.str)
    num_rows = np.array(filenames_w_lb)[:, 2][inds].astype(np.int32)  # in the filename, it also indicates how many
    # rows in the file
    file_ids = list(np.array(filenames_w_lb)[:, 3][inds].astype(np.str))  # baseline (BL) and epileptogenesis (EPG)
    # classes are saved in separate folders

    dataset = tf.compat.v1.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.flat_map(lambda fname, lbs: parse_function(fname, lbs, args=args))
    
    rat_ids = []
    for id, num in zip(file_ids, num_rows):
        rat_ids += [id]*np.int(num)
    ds_rat_ids = tf.compat.v1.data.Dataset.from_tensor_slices((rat_ids))  # up to now, each row is one element in the dataset
    
    comb_ds = tf.compat.v1.data.Dataset.zip((dataset, ds_rat_ids))
    if if_shuffle:
        comb_ds = comb_ds.shuffle(buffer_size=5000)  # fn_lb: filename and label
    if if_repeat:
        comb_ds = comb_ds.repeat()  # fn_lb: filename and label
    comb_ds = comb_ds.batch(batch_size, drop_remainder=True)

    # TODO: it would be great if ther is a seperate function data, labels = get_data(batch_size)
    comb_ds = comb_ds.prefetch(2)

    return comb_ds

# -----need------
def get_data_tensors(args, if_shuffle_train=True, if_shuffle_test=True,
                     if_repeat_train=True, if_repeat_test=True):
    """
    :param args: contrain hyperparams
    :return: train_data: dict, contains 'features', 'labels'
    :return: test_data, dict, contains 'features', 'labels'
    :return: num_samples, dict, contains 'num_train', 'num_test'
    """
    data_tensors = {}
    
    if args.keep_training:
        train_f_with_l, test_f_with_l = load_from_file_list(args)
    else:
        train_f_with_l, test_f_with_l = find_files(args)
        
    test_ds = create_dataset(test_f_with_l, args,
                             batch_size=args.test_bs,
                             if_shuffle=if_shuffle_test,
                             if_repeat=if_repeat_test)

    data_tensors = creat_data_tensors(test_ds, data_tensors,
                                      test_f_with_l, args,
                                      batch_size=args.test_bs,
                                      prefix='test')

    if not args.test_only:
        train_ds = create_dataset(train_f_with_l, args,
                                  batch_size=args.batch_size,
                                  if_shuffle=if_shuffle_train,
                                  if_repeat=if_repeat_train)

        data_tensors = creat_data_tensors(train_ds, data_tensors,
                                          train_f_with_l, args,
                                          batch_size=args.batch_size,
                                          prefix='train')

    print("Finish reading the data tensors")
    return data_tensors, args


def get_test_only_data_tensors(args, if_shuffle=True, if_repeat=False):
    """
    Automate the test only process. select ransom number of hours and get one label for each file
    :param args: contrain hyperparams
    :return: train_data: dict, contains 'features', 'labels'
    :return: test_data, dict, contains 'features', 'labels'
    :return: num_samples, dict, contains 'num_train', 'num_test'
    """
    data_tensors = {}
    train_f_with_l, test_f_with_l = find_files(args)
    test_ds = create_dataset(test_f_with_l, args,
                             batch_size=args.test_bs,
                             if_shuffle=if_shuffle,
                             if_repeat=if_repeat)

    data_tensors = creat_data_tensors(test_ds, data_tensors,
                                      test_f_with_l, args,
                                      batch_size=args.test_bs,
                                      prefix='test')

    print("Finish reading the data tensors")

    return data_tensors, args


## -------------------------------------------------------------------------

def v2_create_dataset(filenames, args, batch_size=32, shuffle=True, n_sec_per_sample=1, sr=512, name="train"):
    def decode_csv(line):
        # Map function to decode the .csv file in TextLineDataset
        # @param line object in TextLineDataset
        # @return: zipped Dataset, (features, (label, filename, rat_id))
        """
        The defaults I copied from one of your previous emails.
        As I don't work with arguments like sampling rate and seconds per row,
        I simply put them in manually for my case (sampling rate 512 with 5 seconds
        per row.
        """
        
        defaults = [['']] + [[0.0]] * (512 * 5 + 1)  # there are 5 sec in one row
        csv_row = tf.io.decode_csv(line, record_defaults=defaults)
        
        filename = tf.cast(csv_row[0], tf.string)
        label = tf.cast(csv_row[1], tf.int32)  # given the label
        features = tf.stack(csv_row[2:])
        
        # why do we need the rat_id as a number?
        rat_id = tf.cast(tf.strings.split(filename, sep="-")[0], tf.string)
        # rat_id = tf.strings.to_number(tf.strings.substr(filename, 1, 2), out_type=tf.dtypes.int32)
        # Apply the zscore transformation
        mean = tf.reduce_mean(features)
        std = tf.math.reduce_std(features)
        zscore = (features - mean) / (std + 1e-13)
        
        return zscore, label, filename, rat_id
        
        # reshape the sample to 1 second
    
    def reshape_to_k_sec(feature, label, filename, rat_id, n_sec=1, sr=512):
        reshaped_x = tf.reshape(feature[:(5 // n_sec) * n_sec * sr], (5 // n_sec, np.int(n_sec * sr), 1))
        filename = tf.cast(tf.repeat(filename, repeats=5 // n_sec, axis=0), dtype=tf.string)
        label = tf.repeat(label, repeats=5 // n_sec, axis=0)
        rat_id = tf.cast(tf.repeat(rat_id, repeats=5 // n_sec, axis=0), dtype=tf.string)
        return reshaped_x, label, filename, rat_id
    
    def flat_map_reshaped(feature, label, filename, rat_id):
        label = tf.data.Dataset.from_tensor_slices(label)
        feature = tf.data.Dataset.from_tensor_slices(feature)
        filename = tf.data.Dataset.from_tensor_slices(filename)
        rat_id = tf.data.Dataset.from_tensor_slices(rat_id)
        return tf.data.Dataset.zip((feature, label, filename, rat_id))
    
    #########################################################################
    tot_rows = np.sum(np.array(filenames)[:, 2].astype(np.int))
    tot_batches = (tot_rows * 5 / n_sec_per_sample) // batch_size
    args.tot_batches[name] = np.int(tot_batches)
    
    # create dataset
    dataset = tf.data.Dataset.list_files(filenames[:, 0])
    
    # Apply the transformation method to all lines
    if shuffle:
        dataset = dataset.interleave(lambda fn: tf.data.TextLineDataset(fn), cycle_length=8, num_parallel_calls=8)
        dataset = dataset.shuffle(8000).repeat()
    else:
        dataset = tf.data.TextLineDataset(dataset)
    
    # zscore, label, filename, rat_id
    dataset = dataset.map(decode_csv)
    dataset = dataset.map(
        map_func=lambda x, lb, fn, rat_id: reshape_to_k_sec(x, lb, fn, rat_id, n_sec=n_sec_per_sample, sr=sr))
    dataset = dataset.flat_map(lambda x, lb, fn, rat_id: flat_map_reshaped(x, lb, fn, rat_id))
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset.prefetch(1), args


### -------------------------------------------------------------------------------------
def check_make_dirs(args):          #picked-300h-11seg-1237
    time_str = '{0:%Y-%m-%dT%H-%M-%S-}'.format(datetime.datetime.now())

    train_or_test = "test" if args.test_only else "train"

    if train_or_test == "test":
        short_time_str = '{0:%Y%m%dT%H%M}'.format(datetime.datetime.now())
        pretrained_dir = os.path.dirname(args.restore_from)
        args.results_dir = os.path.join(os.path.dirname(pretrained_dir), "-".join(os.path.basename(pretrained_dir).split("-"))) + '-{}-{}h-{}-test'.format(short_time_str, args.num_hours_per_class, args.data_source)
    else:
        args.results_dir = os.path.join(args.results_root, time_str + args.class_mode + '-'  + args.model_name + '-'  + args.postfix + "-{}".format(train_or_test))
    args.model_save_dir = os.path.join(args.results_dir, 'model')

    # Set the logger. Copy param file with arg in the model results dir
    plot_dirs = [os.path.join(args.results_dir, x) for x in args.sub_plots_dirs]
    for name_dir in plot_dirs:  #make the dirs for the plots
        if not os.path.exists(name_dir):
            os.makedirs(name_dir)
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    args.is_overwritten_training = args.model_save_dir != args.restore_from

    return args


def save_command_line(args):
    cmd = " ".join(sys.argv[:])
    with open(args.model_save_dir + "/command_line.txt", 'w') as f:
        f.write(cmd)


def save_model(saver, sess, save_dir, step, name=None):
    """
    Save the model under current step into save_dir
    :param saver: tf.Saver
    :param sess: tf.Session
    :param save_dir: str, directory to save the model
    :param step: int, current training step
    :param name: if specify a name, then save with this name
    :return:
    """
    model_name = '.ckpt'
    if not name:
        checkpoint_path = os.path.join(save_dir, model_name)
    else:
        checkpoint_path = os.path.join(save_dir, name + model_name)
    print('Saving checkpoint to {} ...'.format(save_dir))
    sys.stdout.flush()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('Done.')


def check_if_save_model(saver, sess, save_dir, step, save_every=None, name=None):
    """best_saver, sess, args.model_save_dir, epoch, save_every=None, name='best-acc-{}'.format(eval_acc)
    :param saver: tf.Saver
    :param sess: tf.Session
    :param save_dir: str, model save dir
    :param step: int epoch count
    :param save_every: int or None, save every once in a while. If it is None, then save immediately
    :param name: str, save name
    :return: NOne
    """
    if not save_every:
        save_model(saver, sess, save_dir, step, name=name)
    else:
        if step % save_every == 0:
            save_model(saver, sess, save_dir, step, name=name)
        else:
            pass


def load_model(saver, sess, model_dir):
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt:
        print('Checkpoint found: {}'.format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print('  Global step was: {}'.format(global_step))
        print('  Restoring...')

        saver.restore(sess, os.path.join(os.path.dirname(model_dir), "model", os.path.basename(ckpt.model_checkpoint_path)))
        print(' Done.')
        return global_step
    else:
        print(' No checkpoint found.')
        return None


def save_data_to_csv(data, header='data', save_name="save_data"):
    '''save data into a .csv file
    data: list of data that need to be saved, (x1, x2, x3...)
    header: String that will be written at the beginning of the file.'''
    np.savetxt(save_name, data, header=header, delimiter=',', fmt='%10.5f', comments='')


def save_data_to_csvs(data_dic, epoch, args):
    """
    Save resulst data dict into csv for future plots
    :param data_dic: dict, contains all the interesting results data
    :param epoch: int, No. of current training epoch
    :param args: Param object, contains hyperparams
    :return:
    """
    save_data_to_csv((data_dic["train_accuracy"], data_dic["test_accuracy"], data_dic["train_loss"], data_dic["test_loss"]), header='acc_train, acc_test, loss_train, loss_test', save_name=args.results_dir + '/performance_scalars-{}.csv'.format(args.data_source))
    # save_data_to_csv((np.reshape(np.array(data_dic["conf_flat_train"]) * 1.0, [-1, ]),
    #                   np.reshape(np.array(data_dic["conf_flat_test"]) * 1.0, [-1, ])),
    #                  header='conf_matrix_train, conf_matrix_test',
    #                  save_name=args.results_dir + '/performance_conf_matrix_classes{}_epoch{}.csv'.format(
    #                      args.num_classes, epoch))


def copy_save_all_files(args):
    """
    Copy and save all files related to model directory
    :param args:
    :return:
    """
    # src_dir = '../../src'
    import shutil
            
    save_dir = os.path.join(args.results_dir, 'model', 'src')
    if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(save_dir)
        
    for item in os.listdir(args.src_dir):
        s = os.path.join(args.src_dir, item)
        d = os.path.join(save_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, ignore=shutil.ignore_patterns('*.pyc', 'tmp*', "*.h"))
        else:
            shutil.copy2(s, d)
    print('Done WithCopy File!')


def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def print_memory_usage(batch, e):
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.3f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.3f %s%s" % (num, 'Yi', suffix)
    print(
        "-------------------- show size of var. Epoch {}-b-{} -------------".format(
            e, batch))
    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)
    for name, size in sorted(
            ((name, sys.getsizeof(v)) for name, v in locals().items()),
            key=lambda x: -x[1])[0:5]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    gc.collect()
    print("------------------- show size of var. ------------------------")
    

def set_random_seed(args):
    if args.seed is not None:
        np.random.seed(seed=args.seed)
        tf.compat.v1.set_random_seed(args.seed)
    else:
        args.seed = np.random.choice(np.arange(1, 9999), 1)[0]
        np.random.seed(seed=args.seed)
        tf.compat.v1.set_random_seed(args.seed)
    return args



def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
