# coding: utf-8
import argparse
from docreader import GetDocs

import index

#Параметры командной строки, соответствующие массиву COMPRESSION_OBJECTS
COMPRESSION_OBJECTS_NAMES = ['varbyte',  'simple9']

def parse_command_line():
    parser = argparse.ArgumentParser(description='compressed documents reader')
    parser.add_argument('compression',  nargs=1,  default='varbyte', 
                                 choices=['varbyte',  'simple9'], 
                                 help='Type of index compression ("varbyte" or "simple9"')
    parser.add_argument('files', nargs='+', help='Input files (.gz or plain) to process')
    args = parser.parse_args()
    return args.compression[0],  args.files #compression[0] - compression is an array ['varbyte'] or ['simple9']
    
##main
compression,  filenames = parse_command_line()
compression_object = index.COMPRESSION_OBJECTS[COMPRESSION_OBJECTS_NAMES.index(compression)]
docs = GetDocs(filenames)

##Сохраняем URL в порядке загрузки
urls_file = open('urls.txt',  'w')
idx = index.NewIndex(compression_object)
for docid,  doc in enumerate(docs):
    idx.IndexDocument(docid,  doc.text)
    urls_file.write(doc.url+'\n')
urls_file.close()
idx.SaveToFile('test.idx.bin',  'test.dic.bin')
