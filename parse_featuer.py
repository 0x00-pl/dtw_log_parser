import dtw
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
            feature = [float(i) for i in line.split(', ') if i.strip() != '']
            current_feature_list.append(feature)

    push_current()

    return ret


def read_file(filename):
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
        tag1 = get_tag_from_filename(name_features_list[i][0])
        for j in range(i + 1, length):
            tag2 = get_tag_from_filename(name_features_list[j][0])
            dist = cal_dtw(name_features_list[i][1], name_features_list[j][1])
            ret_dist_tag_tag_list.append([dist, tag1, tag2])

    return ret_dist_tag_tag_list


def main():
    name_features_list = read_file('result/dump_feature.txt')
    dist_tag_tag_list = cal_all_dtw(name_features_list)
    dist_trueneg_falsepos = roc.roc_from_dist_tag_tag(dist_tag_tag_list, 1000)
    roc.print_roc(dist_trueneg_falsepos, 'roc_dtw')


if __name__ == '__main__':
    main()
