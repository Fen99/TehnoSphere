#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

#These features are dense enough (they appear in the dataset more than 4 million times), so we include them in GBDT
#target_cat_feats = ['C9-a73ee510', 'C22-', 'C17-e5ba7672', 'C26-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53', 'C14-1adce6ef', 'C25-001f3601', 'C22-ad3062eb', 'C17-07c540c4', 'C6-', 'C23-423fab69', 'C17-d4bb7bd8', 'C2-38a947a1', 'C25-e8b83407', 'C9-7cc72ec2']

# V.1
#target_cat_feats = ['C15-', 'C6-995', 'C11-0', 'C9-15', 'C14-', 'C7-2', 'C7-0', 'C12-106', 'C12-105', 'C12-104', 'C5-30', 'C11-1', 'C10-821', 'C10-802', 'C5-88', 'C8-177', 'C5-10', 'C3-2252', 'C8-452', 'C12-103', 'C8-293', 'C8-419', 'C8-428', 'C10-49', 'C5-11', 'C5-54', 'C8-8', 'C8-341', 'C8-361']

# New features
target_cat_feats = ['C6-995', 'C11-0', 'C9-15', 'C7-2', 'C7-0', 'C12-106', 'C12-105', 'C12-104', 'C5-30', 'C11-1', 'C10-821', 'C17-2058081684', 'C10-802', 'C5-88', 'C8-177', 'C15-3071806138', 'C13-1757768508', 'C5-10', 'C3-2252', 'C8-452', 'C12-103', 'C16-337426089', 'C8-293', 'C14-1064159205', 'C8-419', 'C14-3781537013']

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for j in range(1, 3):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10 
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
        
        cat_feats = set()
        for j in range(1, 31):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
