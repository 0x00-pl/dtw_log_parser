def roc_from_dist_tag_tag(dist_tag_tag_list, limit=None):
    data = sorted(dist_tag_tag_list, key=(lambda x: x[0]))
    true_sum = sum(
        [1 if tag1 == tag2 else 0 for dist, tag1, tag2 in data])
    total_sum = len(data)
    print(true_sum, total_sum - true_sum)

    ret = []
    pos_count = 0
    true_pos = 0
    for dist, tag1, tag2 in data:
        pos_count = pos_count + 1

        true_pos = true_pos + (1 if tag1 == tag2 else 0)
        false_pos = pos_count - true_pos
        true_neg = true_sum - true_pos
        false_neg = total_sum - true_sum - false_pos
        if limit is None or int(round(pos_count / total_sum * limit)) != int(
                round((pos_count - 1) / total_sum * limit)):
            ret.append([dist, true_neg, false_pos])

    return ret


def print_roc(roc_list, name='roc'):
    print(name, '=', end='[')
    for trueneg, falsepos, threahold in roc_list:
        print(trueneg, falsepos, threahold, end=';')
    print('];')
