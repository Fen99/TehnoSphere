# coding: utf-8
import sys
import codecs

import search
import index

def ProcessString(query_str):
    #Распознаем запрос и исполняем дерево
    tokens = search.Tokenize(query_str)
    tree = search.GetQueryTree(tokens, idx_loaded)
    result_str = ""
    count_results = 0
    next_id = tree.Evaluate()
    while next_id < max_docid :
        result_str += urls[next_id]+'\n'
        next_id = tree.Evaluate()
        count_results += 1
        
    #Вывод
    print query_str.encode('utf-8')
    print count_results
    if count_results != 0:
        print result_str[:-1]

##main
#Читаем unicode
#UTF8Reader = codecs.getreader('utf8')
#sys.stdin = UTF8Reader(sys.stdin)

#Загрузка индекса и списка URL
idx_loaded = index.LoadedIndex('test.idx.bin',  'test.dic.bin')
urls_file = open('urls.txt',  'r')
urls = urls_file.read().split('\n')[:-1] #В конце-- тоже \n
max_docid = len(urls)

query_str = sys.stdin.readline()
while query_str != '':
    if query_str[-1] == '\n':
        query_str = query_str[:-1]
    if query_str == '':
        break
    ProcessString(query_str)
    query_str = sys.stdin.readline()

urls_file.close()
idx_loaded.Close()
