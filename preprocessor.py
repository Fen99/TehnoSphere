import zlib
from collections import defaultdict

HEADER = "timestamp;label;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;CG1;CG2;CG3;l1;l2;C11;C12".split(';')
COUNT_BASIC = 2
COUNT_CATEG_FIRST = 10
COUNT_GROUP = 3
COUNT_CATEG_SECOND = 2
COUNT_CATEGORIAL = COUNT_CATEG_FIRST+COUNT_CATEG_SECOND

OUT_HEADER = "timestamp;Label;I1;I2;C1;C2;C3;C4;C5;C6;C7;C8;C9;C10;C11;C12;C13;C14;C15".split(';')


def ProcessGroup(group_name, stat, group_feature):
    if group_feature == '':
        return ''
    for val in group_feature.split(','):
        stat[group_name+"\t"+val] += 1

    return str(zlib.crc32(group_feature.encode()))

def GetFeatures(input_str, global_statistics):
    features = {}
    group_processed = 1
    for feature_id, feature_str in enumerate(input_str.split(';')):
        feature_name = HEADER[feature_id]
        if feature_name == 'label':
            feature_name = 'Label'
        elif feature_name.startswith('CG'): # group
            feature_str = ProcessGroup(feature_name, global_stat, feature_str)
            feature_name = 'C'+str(COUNT_CATEGORIAL + group_processed)
            group_processed += 1
        elif feature_name.startswith('l'): # counters
            feature_name = feature_name.replace('l', 'I')

        features.update({feature_name: feature_str})
        if feature_name.startswith('C'):
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

global_stat = defaultdict(int)
ProcessFile('data/train.dcsv', 'data/traincsv', global_stat)
ProcessFile('data/test.dcsv', 'data/test.csv', global_stat)

sorted_stat = sorted(global_stat.items(), key=lambda kv: -kv[1])
with open('data/stat_1.txt', 'w', encoding='utf-8') as stat_file:
    for stat_name, stat_count in sorted_stat:
        stat_file.write(stat_name+"\t"+str(stat_count)+"\n")