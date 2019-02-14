import zlib
from collections import defaultdict

HEADER = "timestamp;label;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;CG1;CG2;CG3;l1;l2;C11;C12".split(';')
COUNT_BASIC = 2
COUNT_CATEG_FIRST = 10
COUNT_GROUP = 3
COUNT_CATEG_SECOND = 2
COUNT_CATEGORIAL = COUNT_CATEG_FIRST+COUNT_CATEG_SECOND

OUT_HEADER = "timestamp;Label;I1;I2;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;C11;C12;C13;C14;C15;C16;C17;C18;C19;C20;C21;C22;C23;C24;C25;C26;C27;C28;C29;C30".split(';')

CG1_GROUPS = {"C13": [335, 99, 122, 58, 385, 268, 341, 382, 52, 416],
              "C14": [332, 76, 59, 96, 399, 273, 57, 53, 49, 437],
              "C15": [334, 155, 422, 130, 74, 357, 419, 276, 54, 343],
              "C16": [333, 154, 331, 151, 18, 279, 330, 449, 336, 269],
              "C17": [139, 412, 123, 124, 150, 205, 435, 277, 438, 88],
              "C18": []} #For other vals

CG2_GROUPS = {"C19": [16810, 20892, 25731, 8426, 8395, 29347, 18705, 2390, 8833, 823],
              "C20": [30009, 16444, 22582, 29432, 3864, 4378, 25900, 7326, 31328, 3326],
              "C21": [2293, 944, 19883, 29463, 6746, 2253, 16676, 7923, 516, 17179],
              "C22": [18714, 7636, 14322, 10328, 6619, 29768, 15650, 30235, 24843, 18543],
              "C23": [28695, 9793, 2254, 19225, 20755, 3746, 30749, 24338, 207, 18724],
              "C24": []}

CG3_GROUPS = {"C25": [46839, 49517, 49513, 48114, 44289, 57895, 17419, 11599, 56445, 7592],
              "C26": [38344, 46594, 6214, 49272, 45902, 46401, 24887, 3311, 49784, 37588],
              "C27": [45509, 75, 15769, 38346, 27636, 28676, 40461, 5178, 19563, 8894],
              "C28": [56597, 15529, 45501, 18336, 49966, 21144, 47396, 2340, 49518, 8650],
              "C29": [33701, 54846, 55412, 43844, 43845, 20076, 49962, 43666, 23339, 33789],
              "C30": []}


def ProcessGroup(group_name, stat, group_feature):
    #if group_feature == '':
    #    return ''
    #for val in group_feature.split(','):
    #    stat[group_name+"\t"+val] += 1
    vals_array = None
    if group_name == 'CG1':
        vals_array = CG1_GROUPS
    elif group_name == 'CG2':
        vals_array = CG2_GROUPS
    elif group_name == 'CG3':
        vals_array = CG3_GROUPS
    else:
        raise Exception("Unknown group")

    result = defaultdict(list)
    for val in group_feature.split(','):
        if val == '':
            continue
        else:
            val = int(val)

        for key, feature_vals in vals_array.items():
            if len(feature_vals) == 0: # orphan
                result[key].append(str(val))
                break
            if val in feature_vals:
                result[key].append(str(val))
                break
    for key in result:
        if len(result[key]) == 0:
            result[key] = ''
        else:
            result[key] = str(zlib.crc32((",".join(sorted(result[key]))).encode()))
        global_stat[key+"-"+result[key]] += 1
    return result

def GetFeatures(input_str, global_statistics):
    features = {}
    for feature_id, feature_str in enumerate(input_str.split(';')):
        feature_name = HEADER[feature_id]
        if feature_name == 'label':
            feature_name = 'Label'
        elif feature_name.startswith('CG'): # group
            features.update(ProcessGroup(feature_name, global_stat, feature_str))
        elif feature_name.startswith('l'): # counters
            feature_name = feature_name.replace('l', 'I')

        if not feature_name.startswith('CG'):
            features.update({feature_name: feature_str})
        if feature_name.startswith('C') and not feature_name.startswith('CG'):
            global_statistics[feature_name+"-"+feature_str] += 1

    return features

def CreateOutFeatures(input_str, statistics):
    features = GetFeatures(input_str, statistics)
    out = ""
    for feature_name in OUT_HEADER:
        feature_val = features.get(feature_name)
        if feature_val is None:
            feature_val = ''
        out += feature_val + ","
    return out[:-1]

def ProcessFile(filename_in, filename_out, global_statistics):
    with open(filename_in, 'r', encoding='utf-8') as input, \
            open(filename_out, 'w', encoding='utf-8') as output:
        output.write(','.join(OUT_HEADER)+'\n')
        for counter, line in enumerate(input):
            if line.startswith('timestamp'):
                continue

            line = line[:-1]
            out_line = CreateOutFeatures(line, global_statistics)
            output.write(out_line+'\n')
            if counter % 100000 == 0:
                print(filename_in, counter)
                #break

global_stat = defaultdict(int)
ProcessFile('data/train.dcsv', 'data/train_2.csv', global_stat)
ProcessFile('data/test.dcsv', 'data/test_2.csv', global_stat)

sorted_stat = sorted(global_stat.items(), key=lambda kv: -kv[1])
with open('data/stat_2.txt', 'w', encoding='utf-8') as stat_file:
    for stat_name, stat_count in sorted_stat:
        stat_file.write(stat_name+"\t"+str(stat_count)+"\n")