# coding: utf-8

import re
from copy import copy
import numpy as np

import index

INT32_MAX = np.iinfo(np.int32).max-1
class Node(object):
        #Возвращает следующий docid и INT32_MAX, если конец списка
        def Evaluate(self):
            pass
            
        #Нужна для быстрой работы Not
        #Пропуск всех id < docid (при следующем вызове Evaluate)
        def GoTo(self,  docid):
            pass

class BinaryOperator(Node):
    def __init__(self):
        #Левый и правый подузлы и последнее значение, которое они возвращали
        self.subnodes = [None,  None]
        self.last_values = np.full((2,  ),  -1,  dtype=np.int32)
        
    def GoTo(self,  docid):
        self.subnodes[0].GoTo(docid)
        self.subnodes[1].GoTo(docid)
        #Фиктивное, чтобы заставить оба узла сдвинуться
        self.last_values[0] = docid
        self.last_values[1] = docid

class Or(BinaryOperator):
    def __init__(self):
        super(Or, self).__init__()
        
    def Evaluate(self):
        if self.last_values[0] == self.last_values[1]: #Вернули одинаковые значения => продвинуть оба
            if self.last_values[0] == INT32_MAX: #Список исчерпан
                return INT32_MAX
            self.last_values[0] = self.subnodes[0].Evaluate()
            self.last_values[1] = self.subnodes[1].Evaluate()
        else:
            min_index = np.argmin(self.last_values) #Какой узел нужно продвинуть
            self.last_values[min_index] = self.subnodes[min_index].Evaluate()
        
        return np.min(self.last_values)

class And(BinaryOperator):
    def __init__(self):
        super(And, self).__init__()
        
    def Evaluate(self):
        #Всегда равны, т.к. &
        self.last_values[0] = self.subnodes[0].Evaluate()
        self.last_values[1] = self.subnodes[1].Evaluate()
        
        while self.last_values[0] != self.last_values[1]:
            #Сдвигаем меньший
            min_index = np.argmin(self.last_values)
            max_index = 0 if min_index == 1 else 1
            self.subnodes[min_index].GoTo(self.last_values[max_index])
            self.last_values[min_index] = self.subnodes[min_index].Evaluate()
        
        return self.last_values[0] # = self.last_values[1]
        
class Not(Node):
    def __init__(self):
        self.subnode = None
        self.last_value_node = -1
        self.last_not = -2
        
    def Evaluate(self):
        self.last_not += 1
        if self.last_not == INT32_MAX: #Было GoTo до конца
            return INT32_MAX
        
        while self.last_not == self.last_value_node:
            self.last_value_node = self.subnode.Evaluate()
            self.last_not += 1
        return self.last_not
        
    def GoTo(self,  docid):
        self.subnode.GoTo(docid)
        self.last_value_node = self.subnode.Evaluate()       
        self.last_not = docid - 1
        
class Term(Node):
    def __init__(self, docid_list):
        self.docid_list = docid_list
        if self.docid_list is None:
            self.docid_list = np.array([])
        self.position = -1
    
    def Evaluate(self):
        self.position += 1
        if self.position >= len(self.docid_list): #Все docid исчерпаны
            return INT32_MAX
        else:
            return self.docid_list[self.position]
    
    def GoTo(self,  docid):
        if self.position >= len(self.docid_list): #Список уже исчерпан
            return
            
        while self.docid_list[self.position] < docid:
            self.position += 1
            if self.position == len(self.docid_list):
                break
        self.position -= 1 #При вызове Evaluate будет выдано число > docid
        
# Словарь имеющихся операторов (<знак>: <приоритет>)
OPERATORS = {'|': 0,  '&': 1,  '!': 2}
#Словарь классов для создания операторов по приоритету
OPERATOR_TYPES = {0: Or,  1: And,  2: Not}

#Позволяет убрать вложенные списки из одного элемента, образующиеся в выражениях вида (((a)))
def ExtractNestedLists(tokens):
    while type(tokens[0]) == list and len(tokens) == 1:
        tokens = tokens[0]
    return tokens

#Заменяет скобки подмассивами токенов
def CombineBrackets(tokens_list):
    nested_level = 0
    start = -1
    pos = 0
    
    while True:
        if tokens_list[pos] == '(':
            if nested_level == 0:
                start = pos
            nested_level += 1
        if tokens_list[pos] == ')':
            nested_level -= 1
            if nested_level == 0:
                new_list = CombineBrackets(copy(tokens_list[start+1 : pos])) #Удаляем скобки
                new_list = ExtractNestedLists(new_list)
                del tokens_list[start : pos+1] #удаляем со скобками
                tokens_list.insert(start, new_list)
                pos = start #Т.к. вставили новый список, но в конце будет +1
        pos += 1
        if pos >= len(tokens_list):
            break
    
    if nested_level != 0:
        raise SyntaxError("Unpaired brackets!")
    return ExtractNestedLists(tokens_list)

#Разбирает строку utf-8 на подстроки (токены), заменяет скобки подмассивами токенов, строит карту приоритетов
def Tokenize(str):
    split_regexp = re.compile(r'\w+|[\(\)&\|!]', re.U)
    tokens = re.findall(split_regexp, str)
    tokens = map(lambda s: s.lower().encode('utf-8'), tokens)
    return CombineBrackets(tokens)

#Возвращает список приоритетов в списке токенов. Термы получают 5, скобки - 4
def GetPrioritiesList(tokens_list):
    result = np.zeros((len(tokens_list),  ),  dtype=np.uint8)
    for pos, token in enumerate(tokens_list):
        try:
            result[pos] = OPERATORS[token]
        except: #List
            if type(token) == list:
                result[pos] = 4
            else: #Term
                result[pos] = 5
    return result

#Строит дерево запроса
def GetQueryTree(tokens_list, index_object):
    if type(tokens_list) != list: #Один термин - строка
        tokens_list = [tokens_list]
    
    priorities = GetPrioritiesList(tokens_list)
    min_priority_pos = np.argmin(priorities)
    
    if priorities[min_priority_pos] == 5: #Term
        if len(tokens_list) != 1:
            raise SyntaxError("Two terms without operator!")
        return Term(index_object.GetIndexForTerm(tokens_list[0]))
    
    if priorities[min_priority_pos] == 4: #List (brackets)
        if len(tokens_list) != 1:
            raise SyntaxError("Brackets without operator!")
        return GetQueryTree(tokens_list[0],  index_object)
        
    if tokens_list[min_priority_pos] == '!': # !<term/list>
        if min_priority_pos != 0 or min_priority_pos == 1 or len(tokens_list) != 2:
            raise SyntaxError("Wrong syntax near '!'")
        result = Not()
        result.subnode = GetQueryTree(tokens_list[1],  index_object)
        return result
    
    # <...> &<or>| <...>
    if min_priority_pos == 0 or min_priority_pos == len(tokens_list)-1:
        raise SyntaxError("Only one operand for "+tokens_list[min_priority_pos])
    result = OPERATOR_TYPES[priorities[min_priority_pos]]() #And <or> Or
    result.subnodes[0] = GetQueryTree(tokens_list[0:min_priority_pos],  index_object)
    result.subnodes[1] = GetQueryTree(tokens_list[min_priority_pos+1 : None],  index_object)
    return result

