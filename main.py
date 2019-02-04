def get_tag_from_filename(filename):
    return filename[:-4].split('\\')[-1]


def parse_line(line: str):
    line = line.strip()
    tokens = line.split()
    if len(tokens) < 3:
        return None
    dis, name1, name2 = tokens[-3:]

    dis = float(dis)
    name1 = name1.replace(',', '')
    return dis, name1, name2


def read_file(filename):
    ret = []
    for line in open(filename, 'r'):
        ret.append(parse_line(line))

    ret = [i for i in ret if i is not None]
    return ret


def cal_roc(dist_name_name_list, limit=None):
    data = sorted(dist_name_name_list, key=(lambda x: x[0]))
    true_sum = sum(
        [1 if get_tag_from_filename(name1) == get_tag_from_filename(name2) else 0 for dist, name1, name2 in data])
    total_sum = len(data)
    print(true_sum, total_sum - true_sum)

    ret = []
    pos_count = 0
    true_pos = 0
    for dist, name1, name2 in data:
        pos_count = pos_count + 1

        true_pos = true_pos + (1 if get_tag_from_filename(name1) == get_tag_from_filename(name2) else 0)
        false_pos = pos_count - true_pos
        true_neg = true_sum - true_pos
        false_neg = total_sum - true_sum - false_pos
        if int(round(pos_count / total_sum * limit)) != int(round((pos_count - 1) / total_sum * limit)):
            ret.append([dist, true_neg, false_pos])

    return ret


def print_roc(roc_list, name='roc'):
    print(name, '=', end='[')
    for trueneg, falsepos, threahold in roc_list:
        print(trueneg, falsepos, threahold, end=';')
    print('];')


if __name__ == '__main__':
    _dist_name_name_list = read_file('result/dump.txt')
    _roc = cal_roc(_dist_name_name_list, 1000)
    print_roc(_roc)
