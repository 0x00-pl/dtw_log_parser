import itertools
import os
import random
from multiprocessing import Pool

import dtw
import numpy as np
import pydub
import python_speech_features
import tqdm
import webrtcvad
from pydub import effects

import roc


def read_all_features(f):
    ret = []
    current_name = 'noname'
    current_feature_list = []

    def push_current():
        nonlocal current_name
        nonlocal current_feature_list

        if len(current_feature_list) == 0:
            return

        ret.append([current_name, current_feature_list])
        current_name = 'noname'
        current_feature_list = []

    for line in f:
        if line.startswith('>>>'):
            push_current()
            current_name = line[3:].strip()
        else:
            try:
                feature = [float(i) for i in line.split(', ') if i.strip() != '']
                current_feature_list.append(feature)
            except:
                pass

    push_current()

    return ret


def read_file(filename):
    with open(filename, 'r') as fi:
        return read_all_features(fi)


def read_file_pos_neg(filename):
    with open(filename, 'r') as fi:
        return read_all_features(fi)


def get_tag_from_filename(filename):
    try:
        return int(filename.split('/')[1].split('\\')[0])
    except:
        return float('NaN')


def cal_dtw(feature_list1, feature_list2):
    d, cost_matrix, acc_cost_matrix, path = dtw.accelerated_dtw(feature_list1, feature_list2, dist='cosine', warp=1)
    return d


def cal_all_dtw(name_features_list):
    ret_dist_tag_tag_list = []
    length = len(name_features_list)
    for i in range(length - 1):
        print(i, length)
        tag1 = get_tag_from_filename(name_features_list[i][0])
        for j in range(i + 1, length):
            tag2 = get_tag_from_filename(name_features_list[j][0])
            dist = cal_dtw(name_features_list[i][1], name_features_list[j][1])
            ret_dist_tag_tag_list.append([dist, tag1, tag2])

    return ret_dist_tag_tag_list


def cal_all_dtw_multiprocessing_step(args):
    features_list1, features_list2, tag1, tag2 = args
    dist = cal_dtw(features_list1, features_list2)
    return [dist, tag1, tag2]


def cal_all_dtw_multiprocessing(name_features_list, limit=None):
    tasks = []
    length = len(name_features_list)
    for i in range(length - 1):
        tag1 = get_tag_from_filename(name_features_list[i][0])
        for j in range(i + 1, length):
            tag2 = get_tag_from_filename(name_features_list[j][0])
            tasks.append([name_features_list[i][1], name_features_list[j][1], tag1, tag2])

    if limit is not None:
        tasks = random.choices(tasks, k=limit)

    with Pool(processes=8) as pool:
        return list(tqdm.tqdm(pool.imap(cal_all_dtw_multiprocessing_step, tasks), total=len(tasks)))


def cal_all_dtw_pos_neg_multiprocessing(pos_features_list, neg_features_list, limit=None):
    tasks = []
    length = len(pos_features_list)
    for i in range(len(pos_features_list)):
        feature1 = pos_features_list[i]
        for j in range(i + 1, length):
            tasks.append([feature1, pos_features_list[j], 1, 1])
        for feature2 in neg_features_list:
            tasks.append([feature1, feature2, 1, 2])

    if limit is not None and len(tasks) > limit:
        tasks = random.choices(tasks, k=limit)

    with Pool() as pool:
        return list(tqdm.tqdm(pool.imap(cal_all_dtw_multiprocessing_step, tasks, chunksize=10), total=len(tasks)))


def load_file(filename, file_format, frame_rate=16000):
    sound = pydub.AudioSegment.from_file(filename, file_format)
    sound = sound.set_frame_rate(frame_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)
    sound = sound.remove_dc_offset()
    sound = effects.normalize(sound)
    signal = np.array(sound.get_array_of_samples())

    # preemph = 0.97
    # signal = python_speech_features.sigproc.preemphasis(signal, preemph)

    vader = webrtcvad.Vad()
    vader.set_mode(1)
    frames = python_speech_features.sigproc.framesig(signal, 320, 160)
    frames = np.array([i for i in frames if vader.is_speech(i.astype('int16').tobytes(), 16000)])
    signal = frames.flatten()

    # ret = python_speech_features.sigproc.powspec(frames, 320)
    # ret = python_speech_features.fbank(signal, winlen=0.02, winstep=0.02, winfunc=lambda x: np.hamming(x))[0]
    ret = python_speech_features.mfcc(signal, numcep=13, nfilt=26, winlen=0.02, winstep=0.02, lowfreq=100,
                                      winfunc=lambda x: np.hamming(x))
    ret = ret - np.mean(ret, axis=0)
    ret = ret / np.var(ret, axis=0)

    ret_delta = python_speech_features.delta(ret, 1)
    # ret_delta = ret_delta - np.mean(ret_delta, axis=0)
    # ret_delta = ret_delta / np.var(ret_delta, axis=0)
    ret_delta2 = python_speech_features.delta(ret_delta, 1)
    # ret_delta2 = ret_delta2 - np.mean(ret_delta2, axis=0)
    # ret_delta2 = ret_delta2 / np.var(ret_delta2, axis=0)

    ret_acc = np.array(list(itertools.accumulate(ret, (lambda prev, cur: prev / 2 + cur))))
    ret_acc = ret_acc - np.mean(ret_acc, axis=0)
    ret_acc = ret_acc / np.var(ret_acc, axis=0)

    # # ret[ret <= 1e-30] = 1e-30
    return np.concatenate((ret, ret_delta, ret_delta2, ret_acc), axis=1)
    # ret = np.add.accumulate(ret)
    # ret_delta = np.add.accumulate(ret_delta)
    # ret_delta2 = np.add.accumulate(ret_delta2)
    # return np.concatenate((ret, ret_delta, ret_delta2), axis=1)


def dataset_read_all(root_path='kanzhitongxue'):
    ret = []
    for prefix_path, dirs, files in os.walk(root_path):
        for filename in files:
            file_path = os.path.join(prefix_path, filename)
            feature_list = load_file(file_path, 'wav')
            ret.append([file_path, feature_list])

    return ret


def dtw_local_pos_neg_main():
    pos_features_list = [i[1] for i in dataset_read_all('kanzhitongxue/pos')]
    neg_features_list = [i[1] for i in dataset_read_all('kanzhitongxue/other_text')]
    dist_tag_tag_list = cal_all_dtw_pos_neg_multiprocessing(pos_features_list, neg_features_list)
    dist_trueneg_falsepos = roc.roc_from_dist_tag_tag(dist_tag_tag_list, 1000)
    roc.print_roc(dist_trueneg_falsepos, 'roc_dtw')


def dtw_local_main():
    name_features_list = dataset_read_all('dataset')
    dist_tag_tag_list = cal_all_dtw_multiprocessing(name_features_list, limit=10000)
    dist_trueneg_falsepos = roc.roc_from_dist_tag_tag(dist_tag_tag_list, 1000)
    roc.print_roc(dist_trueneg_falsepos, 'roc_dtw')


def pos_neg_main():
    name_features_list = read_file_pos_neg('result/dump_feature_pn.txt')
    pos_features_list = [i[1] for i in name_features_list if get_tag_from_filename(i[0]) == 1]
    neg_features_list = [i[1] for i in name_features_list if get_tag_from_filename(i[0]) == 2]
    dist_tag_tag_list = cal_all_dtw_pos_neg_multiprocessing(pos_features_list, neg_features_list)
    dist_trueneg_falsepos = roc.roc_from_dist_tag_tag(dist_tag_tag_list, 1000)
    roc.print_roc(dist_trueneg_falsepos, 'roc_dtw')


def main():
    name_features_list = read_file('result/dump_feature.txt')
    dist_tag_tag_list = cal_all_dtw_multiprocessing(name_features_list, limit=10000)
    dist_trueneg_falsepos = roc.roc_from_dist_tag_tag(dist_tag_tag_list, 1000)
    roc.print_roc(dist_trueneg_falsepos, 'roc_dtw')


if __name__ == '__main__':
    pos_neg_main()
